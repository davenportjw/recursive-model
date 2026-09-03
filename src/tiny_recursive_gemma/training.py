import os
import json
from typing import Optional, List, Tuple, Any
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as mx_utils
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers
from .continuous_model import get_transformer_layers, get_logits, ACTHaltingHead, rms_norm


def load_data(file_path: str):
    """Loads dataset from JSONL, supporting both messages format and prompt/solution format.
    Automatically extracts clean solution code and optional target_halt_step for continuous latent training.
    """
    import re
    dataset = []
    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            halt_step = obj.get("target_halt_step", obj.get("halt_step", None))
            if 'messages' in obj:
                user_part = obj['messages'][0]['content']
                code_part = obj['messages'][1]['content']
                
                # If training continuous model, strip XML thought tags if present
                code_match = re.search(r'<code_update>(.*?)</code_update>', code_part, re.DOTALL)
                if code_match:
                    code_part = code_match.group(1).strip()
                    
                dataset.append((user_part, code_part, halt_step))
            elif 'prompt' in obj and ('solution' in obj or 'code' in obj or 'canonical_solution' in obj):
                prompt = obj['prompt']
                solution = obj.get('solution', obj.get('code', obj.get('canonical_solution', '')))
                dataset.append((prompt, solution, halt_step))
    return dataset


class RecursiveModelWrapper(nn.Module):
    """Encapsulates the base LM and optional ACT halting head for joint optimization."""
    def __init__(self, model: nn.Module, halting_head: Optional[ACTHaltingHead] = None):
        super().__init__()
        self.model = model
        self.halting_head = halting_head


def loss_fn(
    model: Any, 
    transformer: Any, 
    lm: Any, 
    prompt_tokens: List[int], 
    target_tokens: List[int], 
    iterations: int = 3, 
    trm_mode: bool = True, 
    dual_latent: bool = True, 
    reasoning_steps: int = 3,
    halting_head: Optional[ACTHaltingHead] = None,
    target_halt_step: Optional[int] = None,
    act_loss_weight: float = 0.1,
    deep_supervision_decay: float = 1.0,
    normalize_latents: bool = True
) -> mx.array:
    """Computes recursive loss with detached multi-step deep supervision and ACT halting BCE."""
    if len(target_tokens) == 0:
        return mx.array(0.0)
        
    prompt_ids = mx.array(prompt_tokens)[None]
    prompt_embeds = transformer.embed_tokens(prompt_ids)
    
    # Autoregressive target inputs and labels
    if len(target_tokens) > 1:
        target_ids_input = mx.array(target_tokens[:-1])[None]
        target_embeds_input = transformer.embed_tokens(target_ids_input)
    else:
        # Edge case: single token target
        target_embeds_input = mx.zeros((1, 0, prompt_embeds.shape[-1]))
        
    target_labels = mx.array(target_tokens)[None]
    hidden_dim = prompt_embeds.shape[-1]
    
    # Compute step weights for deep supervision: w_t = t^gamma / sum(j^gamma)
    if deep_supervision_decay > 0.0 and iterations > 1:
        raw_w = [(t + 1) ** deep_supervision_decay for t in range(iterations)]
        sum_w = sum(raw_w)
        weights = [w / sum_w for w in raw_w]
    else:
        weights = [0.0] * (iterations - 1) + [1.0]

    total_loss = mx.array(0.0)
    eps = 1e-7

    if trm_mode:
        # --- Samsung TRM Mode ---
        if dual_latent:
            z = mx.zeros((1, 1, hidden_dim))
            y = mx.zeros((1, 1, hidden_dim))
            
            # 1. Run first T-1 steps with stopped gradients (O(1) memory scaling)
            if iterations > 1:
                for t in range(iterations - 1):
                    # Reasoning steps (n times)
                    for _ in range(reasoning_steps):
                        y_inj = rms_norm(y) if normalize_latents else y
                        z_inj = rms_norm(z) if normalize_latents else z
                        injected = mx.concatenate([prompt_embeds, y_inj, z_inj], axis=1)
                        _, hidden = get_logits(model, transformer, lm, injected)
                        z = mx.stop_gradient(hidden[:, -1:, :])
                    
                    # Solution step (1 time)
                    z_inj = rms_norm(z) if normalize_latents else z
                    y_inj = rms_norm(y) if normalize_latents else y
                    injected = mx.concatenate([prompt_embeds, z_inj, y_inj], axis=1)
                    _, hidden = get_logits(model, transformer, lm, injected)
                    y = mx.stop_gradient(hidden[:, -1:, :])
                    
                    # Intermediate deep supervision on detached state
                    if weights[t] > 0.0:
                        z_inj = rms_norm(z) if normalize_latents else z
                        y_inj = rms_norm(y) if normalize_latents else y
                        if target_embeds_input.shape[1] > 0:
                            step_embeds = mx.concatenate([prompt_embeds, z_inj, y_inj, target_embeds_input], axis=1)
                        else:
                            step_embeds = mx.concatenate([prompt_embeds, z_inj, y_inj], axis=1)
                        step_logits, _ = get_logits(model, transformer, lm, step_embeds)
                        prefix_len = prompt_embeds.shape[1] + 2
                        target_logits = step_logits[:, prefix_len - 1 : prefix_len - 1 + target_labels.shape[1], :]
                        step_ce = mx.mean(nn.losses.cross_entropy(target_logits, target_labels))
                        total_loss = total_loss + (weights[t] * step_ce)
                        
                    # ACT Halting supervision (BCE)
                    if halting_head is not None:
                        halt_prob = halting_head(y, z)
                        halt_target = 1.0 if (target_halt_step is not None and (t + 1) >= target_halt_step) else 0.0
                        bce = - (halt_target * mx.log(halt_prob + eps) + (1.0 - halt_target) * mx.log(1.0 - halt_prob + eps))
                        total_loss = total_loss + (act_loss_weight * mx.mean(bce))

            # 2. Run the final step with gradients tracked
            for _ in range(reasoning_steps):
                y_inj = rms_norm(y) if normalize_latents else y
                z_inj = rms_norm(z) if normalize_latents else z
                injected = mx.concatenate([prompt_embeds, y_inj, z_inj], axis=1)
                _, hidden = get_logits(model, transformer, lm, injected)
                z = hidden[:, -1:, :]
                
            z_inj = rms_norm(z) if normalize_latents else z
            y_inj = rms_norm(y) if normalize_latents else y
            injected = mx.concatenate([prompt_embeds, z_inj, y_inj], axis=1)
            _, hidden = get_logits(model, transformer, lm, injected)
            y_next = hidden[:, -1:, :]
            
            # Target prediction loss at the final step
            z_inj = rms_norm(z) if normalize_latents else z
            y_inj = rms_norm(y_next) if normalize_latents else y_next
            if target_embeds_input.shape[1] > 0:
                full_embeds = mx.concatenate([prompt_embeds, z_inj, y_inj, target_embeds_input], axis=1)
            else:
                full_embeds = mx.concatenate([prompt_embeds, z_inj, y_inj], axis=1)
                
            logits, _ = get_logits(model, transformer, lm, full_embeds)
            
            prefix_len = prompt_embeds.shape[1] + 2
            target_logits = logits[:, prefix_len - 1 : prefix_len - 1 + target_labels.shape[1], :]
            ce_loss = mx.mean(nn.losses.cross_entropy(target_logits, target_labels))
            total_loss = total_loss + (weights[-1] * ce_loss)
            
            if halting_head is not None:
                halt_prob = halting_head(y_next, z)
                halt_target = 1.0 if (target_halt_step is not None and iterations >= target_halt_step) else 0.0
                bce = - (halt_target * mx.log(halt_prob + eps) + (1.0 - halt_target) * mx.log(1.0 - halt_prob + eps))
                total_loss = total_loss + (act_loss_weight * mx.mean(bce))
                
            return total_loss
        else:
            # --- Single Latent TRM Mode ---
            z = mx.zeros((1, 1, hidden_dim))
            if iterations > 1:
                for t in range(iterations - 1):
                    z_inj = rms_norm(z) if normalize_latents else z
                    injected = mx.concatenate([prompt_embeds, z_inj], axis=1)
                    _, hidden = get_logits(model, transformer, lm, injected)
                    z = mx.stop_gradient(hidden[:, -1:, :])
                    
                    if weights[t] > 0.0:
                        z_inj = rms_norm(z) if normalize_latents else z
                        if target_embeds_input.shape[1] > 0:
                            step_embeds = mx.concatenate([prompt_embeds, z_inj, target_embeds_input], axis=1)
                        else:
                            step_embeds = mx.concatenate([prompt_embeds, z_inj], axis=1)
                        step_logits, _ = get_logits(model, transformer, lm, step_embeds)
                        prefix_len = prompt_embeds.shape[1] + 1
                        target_logits = step_logits[:, prefix_len - 1 : prefix_len - 1 + target_labels.shape[1], :]
                        step_ce = mx.mean(nn.losses.cross_entropy(target_logits, target_labels))
                        total_loss = total_loss + (weights[t] * step_ce)
            
            z_inj = rms_norm(z) if normalize_latents else z
            injected = mx.concatenate([prompt_embeds, z_inj], axis=1)
            _, hidden = get_logits(model, transformer, lm, injected)
            z_next = hidden[:, -1:, :]
            
            z_inj = rms_norm(z_next) if normalize_latents else z_next
            if target_embeds_input.shape[1] > 0:
                full_embeds = mx.concatenate([prompt_embeds, z_inj, target_embeds_input], axis=1)
            else:
                full_embeds = mx.concatenate([prompt_embeds, z_inj], axis=1)
                
            logits, _ = get_logits(model, transformer, lm, full_embeds)
            
            prefix_len = prompt_embeds.shape[1] + 1
            target_logits = logits[:, prefix_len - 1 : prefix_len - 1 + target_labels.shape[1], :]
            ce_loss = mx.mean(nn.losses.cross_entropy(target_logits, target_labels))
            total_loss = total_loss + (weights[-1] * ce_loss)
            return total_loss
    else:
        # --- Standard Full Unrolled Mode ---
        if dual_latent:
            z = mx.zeros((1, 1, hidden_dim))
            y = mx.zeros((1, 1, hidden_dim))
            total_loss = mx.array(0.0)
            
            for t in range(iterations):
                for _ in range(reasoning_steps):
                    y_inj = rms_norm(y) if normalize_latents else y
                    z_inj = rms_norm(z) if normalize_latents else z
                    injected = mx.concatenate([prompt_embeds, y_inj, z_inj], axis=1)
                    _, hidden = get_logits(model, transformer, lm, injected)
                    z = hidden[:, -1:, :]
                    
                z_inj = rms_norm(z) if normalize_latents else z
                y_inj = rms_norm(y) if normalize_latents else y
                injected = mx.concatenate([prompt_embeds, z_inj, y_inj], axis=1)
                _, hidden = get_logits(model, transformer, lm, injected)
                y = hidden[:, -1:, :]
                
                z_inj = rms_norm(z) if normalize_latents else z
                y_inj = rms_norm(y) if normalize_latents else y
                if target_embeds_input.shape[1] > 0:
                    full_embeds = mx.concatenate([prompt_embeds, z_inj, y_inj, target_embeds_input], axis=1)
                else:
                    full_embeds = mx.concatenate([prompt_embeds, z_inj, y_inj], axis=1)
                    
                logits, _ = get_logits(model, transformer, lm, full_embeds)
                
                prefix_len = prompt_embeds.shape[1] + 2
                target_logits = logits[:, prefix_len - 1 : prefix_len - 1 + target_labels.shape[1], :]
                ce_loss = nn.losses.cross_entropy(target_logits, target_labels)
                total_loss = total_loss + (weights[t] * mx.mean(ce_loss))
                
                if halting_head is not None:
                    halt_prob = halting_head(y, z)
                    halt_target = 1.0 if (target_halt_step is not None and (t + 1) >= target_halt_step) else 0.0
                    bce = - (halt_target * mx.log(halt_prob + eps) + (1.0 - halt_target) * mx.log(1.0 - halt_prob + eps))
                    total_loss = total_loss + (act_loss_weight * mx.mean(bce))
            return total_loss
        else:
            # --- Single Latent Full Unrolled Mode ---
            z = mx.zeros((1, 1, hidden_dim))
            total_loss = mx.array(0.0)
            for t in range(iterations):
                z_inj = rms_norm(z) if normalize_latents else z
                injected = mx.concatenate([prompt_embeds, z_inj], axis=1)
                _, hidden = get_logits(model, transformer, lm, injected)
                z_next = hidden[:, -1:, :]
                
                z_inj = rms_norm(z_next) if normalize_latents else z_next
                if target_embeds_input.shape[1] > 0:
                    full_embeds = mx.concatenate([prompt_embeds, z_inj, target_embeds_input], axis=1)
                else:
                    full_embeds = mx.concatenate([prompt_embeds, z_inj], axis=1)
                    
                logits, _ = get_logits(model, transformer, lm, full_embeds)
                
                prefix_len = prompt_embeds.shape[1] + 1
                target_logits = logits[:, prefix_len - 1 : prefix_len - 1 + target_labels.shape[1], :]
                ce_loss = nn.losses.cross_entropy(target_logits, target_labels)
                total_loss = total_loss + (weights[t] * mx.mean(ce_loss))
                z = z_next
            return total_loss


def train_continuous_model(
    model_path: str,
    data_path: str,
    iters: int,
    output_path: str,
    lora_layers: int = 2,
    lora_rank: int = 4,
    trm_mode: bool = True,
    recursive_iters: int = 3,
    dual_latent: bool = True,
    reasoning_steps: int = 3,
    use_ema: bool = True,
    ema_beta: float = 0.99,
    enable_act: bool = False,
    act_loss_weight: float = 0.1,
    deep_supervision_decay: float = 1.0,
    normalize_latents: bool = True,
):
    """Trains a continuous latent model with unrolled recurrence, ACT halting, and Weight EMA smoothing.
    
    Args:
        model_path: Path to the base model (e.g. google/gemma-4-E2B-it-qat-q4_0-unquantized).
        data_path: Path to the JSONL training data.
        iters: Number of training steps to optimize.
        output_path: Output safetensors file path for trained weights.
        lora_layers: Number of transformer layers from the end to apply LoRA to.
        lora_rank: LoRA decomposition rank.
        trm_mode: Whether to use Samsung TRM style gradient-free recurrence for prefixes.
        recursive_iters: Number of recursive steps (T).
        dual_latent: Whether to use separate reasoning (z) and solution (y) states.
        reasoning_steps: Number of reasoning inner steps (n) per iteration in dual mode.
        use_ema: Whether to stabilize training using Weight Exponential Moving Average (EMA).
        ema_beta: Weight EMA decay factor.
        enable_act: Whether to instantiate and train an Adaptive Computation Time halting head.
        act_loss_weight: Weight lambda for ACT BCE halting loss.
        deep_supervision_decay: Power gamma for detached multi-step deep supervision weighting.
        normalize_latents: Whether to apply RMSNorm to latent states before injection.
    """
    print(f"Loading model {model_path}...")
    model, tokenizer = load(model_path)
    model.freeze()
    
    print(f"Configuring LoRA (rank={lora_rank}) on the last {lora_layers} transformer layers...")
    lora_config = {"rank": lora_rank, "alpha": lora_rank * 2, "dropout": 0.0, "scale": 10.0}
    linear_to_lora_layers(model, num_layers=lora_layers, config=lora_config)
    
    transformer, lm = get_transformer_layers(model)
    hidden_dim = getattr(transformer, "embed_tokens").weight.shape[-1]
    
    halting_head = ACTHaltingHead(hidden_dim) if enable_act else None
    wrapper = RecursiveModelWrapper(model, halting_head)
    
    optimizer = optim.AdamW(learning_rate=1e-4)
    
    def compute_step_loss(wrap, p_tokens, t_tokens, halt_step=None):
        return loss_fn(
            wrap.model,
            transformer,
            lm,
            p_tokens,
            t_tokens,
            iterations=recursive_iters,
            trm_mode=trm_mode,
            dual_latent=dual_latent,
            reasoning_steps=reasoning_steps,
            halting_head=wrap.halting_head,
            target_halt_step=halt_step,
            act_loss_weight=act_loss_weight,
            deep_supervision_decay=deep_supervision_decay,
            normalize_latents=normalize_latents
        )
        
    loss_and_grad_fn = nn.value_and_grad(wrapper, compute_step_loss)
    
    dataset = load_data(data_path)
    if not dataset:
        print("Dataset empty or failed to parse.")
        return
        
    print(f"Loaded {len(dataset)} samples. Starting continuous training (TRM: {trm_mode}, ACT: {enable_act}, Decay: {deep_supervision_decay}, EMA: {use_ema})...")
    
    # Initialize EMA weights
    if use_ema:
        print(f"Initializing Weight EMA (decay beta={ema_beta})...")
        ema_weights = mx_utils.tree_map(lambda x: x, wrapper.trainable_parameters())
        state = [wrapper.state, optimizer.state, ema_weights]
    else:
        ema_weights = None
        state = [wrapper.state, optimizer.state]

    for i in range(iters):
        sample = dataset[i % len(dataset)]
        prompt_tokens = tokenizer.encode(sample[0])
        target_tokens = tokenizer.encode(sample[1])
        halt_step = sample[2] if len(sample) > 2 else None
        
        # MLX value and grad
        loss, grads = loss_and_grad_fn(wrapper, prompt_tokens, target_tokens, halt_step)
        
        optimizer.update(wrapper, grads)
        
        if use_ema:
            ema_weights = mx_utils.tree_map(
                lambda ema, param: ema_beta * ema + (1.0 - ema_beta) * param,
                ema_weights,
                wrapper.trainable_parameters()
            )
            state[2] = ema_weights
            
        mx.eval(state)
        
        print(f"Iter {i+1}/{iters} | Loss: {loss.item():.4f}")
        
    print("Training complete!")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Select weights to save
    if use_ema:
        print(f"Saving EMA-smoothed weights...")
        save_weights = ema_weights
    else:
        print("Saving raw trainable weights...")
        save_weights = wrapper.trainable_parameters()
        
    # Helper to flatten MLX trees
    def _flatten_dict(d, parent_key=''):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(_flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)

    flat_weights = _flatten_dict(save_weights)
    
    # Separate model LoRA weights from halting head weights
    model_lora_weights = {}
    head_weights = {}
    for k, v in flat_weights.items():
        if k.startswith("model."):
            model_lora_weights[k[len("model."):]] = v
        elif k.startswith("halting_head."):
            head_weights[k[len("halting_head."):]] = v
        else:
            model_lora_weights[k] = v
            
    mx.save_safetensors(output_path, model_lora_weights)
    print(f"Saved LoRA weights to {output_path}")
    
    if enable_act and halting_head is not None:
        act_output_path = os.path.join(output_dir if output_dir else ".", "act_head.safetensors")
        mx.save_safetensors(act_output_path, head_weights)
        print(f"Saved ACT halting head weights to {act_output_path}")

    # Also serialize adapter_config.json alongside the safetensors file
    config_path = os.path.join(output_dir if output_dir else ".", "adapter_config.json")
    adapter_config = {
        "model": model_path,
        "num_layers": lora_layers,
        "lora_parameters": lora_config,
        "recursive_iters": recursive_iters,
        "dual_latent": dual_latent,
        "reasoning_steps": reasoning_steps,
        "trm_mode": trm_mode,
        "use_ema": use_ema,
        "ema_beta": ema_beta,
        "enable_act": enable_act,
        "act_loss_weight": act_loss_weight,
        "deep_supervision_decay": deep_supervision_decay,
        "normalize_latents": normalize_latents
    }
    with open(config_path, "w") as f:
        json.dump(adapter_config, f, indent=4)
    print(f"Saved adapter configuration to {config_path}")
