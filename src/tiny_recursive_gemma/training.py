import os
import json
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as mx_utils
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers
from .continuous_model import get_transformer_layers, get_logits


def load_data(file_path):
    """Loads dataset from JSONL, supporting both messages format and prompt/solution format.
    Automatically extracts clean solution code for continuous latent training.
    """
    import re
    dataset = []
    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if 'messages' in obj:
                user_part = obj['messages'][0]['content']
                code_part = obj['messages'][1]['content']
                
                # If training continuous model, strip XML thought tags if present
                code_match = re.search(r'<code_update>(.*?)</code_update>', code_part, re.DOTALL)
                if code_match:
                    code_part = code_match.group(1).strip()
                    
                dataset.append((user_part, code_part))
            elif 'prompt' in obj and ('solution' in obj or 'code' in obj):
                prompt = obj['prompt']
                solution = obj.get('solution', obj.get('code', ''))
                dataset.append((prompt, solution))
    return dataset

def loss_fn(model, transformer, lm, prompt_tokens, target_tokens, iterations=3, trm_mode=True, dual_latent=True, reasoning_steps=3):
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
    
    if trm_mode:
        # --- Samsung TRM Mode ---
        if dual_latent:
            z = mx.zeros((1, 1, hidden_dim))
            y = mx.zeros((1, 1, hidden_dim))
            
            # 1. Run first T-1 steps without tracking gradients (using stop_gradient)
            if iterations > 1:
                for t in range(iterations - 1):
                    # Reasoning steps (n times)
                    for _ in range(reasoning_steps):
                        injected = mx.concatenate([prompt_embeds, y, z], axis=1)
                        _, hidden = get_logits(model, transformer, lm, injected)
                        z = mx.stop_gradient(hidden[:, -1:, :])
                    
                    # Solution step (1 time)
                    injected = mx.concatenate([prompt_embeds, z, y], axis=1)
                    _, hidden = get_logits(model, transformer, lm, injected)
                    y = mx.stop_gradient(hidden[:, -1:, :])
            
            # 2. Run the final step with gradients tracked
            for _ in range(reasoning_steps):
                injected = mx.concatenate([prompt_embeds, y, z], axis=1)
                _, hidden = get_logits(model, transformer, lm, injected)
                z = hidden[:, -1:, :]
                
            injected = mx.concatenate([prompt_embeds, z, y], axis=1)
            _, hidden = get_logits(model, transformer, lm, injected)
            y_next = hidden[:, -1:, :]
            
            # 3. Compute target prediction loss (Deep Supervision at the final step)
            if target_embeds_input.shape[1] > 0:
                full_embeds = mx.concatenate([prompt_embeds, z, y_next, target_embeds_input], axis=1)
            else:
                full_embeds = mx.concatenate([prompt_embeds, z, y_next], axis=1)
                
            logits, _ = get_logits(model, transformer, lm, full_embeds)
            
            prefix_len = prompt_embeds.shape[1] + 2
            target_logits = logits[:, prefix_len - 1 : prefix_len - 1 + target_labels.shape[1], :]
            ce_loss = nn.losses.cross_entropy(target_logits, target_labels)
            return mx.mean(ce_loss)
        else:
            # --- Single Latent TRM Mode ---
            z = mx.zeros((1, 1, hidden_dim))
            if iterations > 1:
                for t in range(iterations - 1):
                    injected = mx.concatenate([prompt_embeds, z], axis=1)
                    _, hidden = get_logits(model, transformer, lm, injected)
                    z = mx.stop_gradient(hidden[:, -1:, :])
            
            injected = mx.concatenate([prompt_embeds, z], axis=1)
            _, hidden = get_logits(model, transformer, lm, injected)
            z_next = hidden[:, -1:, :]
            
            if target_embeds_input.shape[1] > 0:
                full_embeds = mx.concatenate([prompt_embeds, z_next, target_embeds_input], axis=1)
            else:
                full_embeds = mx.concatenate([prompt_embeds, z_next], axis=1)
                
            logits, _ = get_logits(model, transformer, lm, full_embeds)
            
            prefix_len = prompt_embeds.shape[1] + 1
            target_logits = logits[:, prefix_len - 1 : prefix_len - 1 + target_labels.shape[1], :]
            ce_loss = nn.losses.cross_entropy(target_logits, target_labels)
            return mx.mean(ce_loss)
    else:
        # --- Standard Full Unrolled Mode ---
        if dual_latent:
            z = mx.zeros((1, 1, hidden_dim))
            y = mx.zeros((1, 1, hidden_dim))
            total_loss = 0.0
            
            for t in range(iterations):
                for _ in range(reasoning_steps):
                    injected = mx.concatenate([prompt_embeds, y, z], axis=1)
                    _, hidden = get_logits(model, transformer, lm, injected)
                    z = hidden[:, -1:, :]
                    
                injected = mx.concatenate([prompt_embeds, z, y], axis=1)
                _, hidden = get_logits(model, transformer, lm, injected)
                y = hidden[:, -1:, :]
                
                if target_embeds_input.shape[1] > 0:
                    full_embeds = mx.concatenate([prompt_embeds, z, y, target_embeds_input], axis=1)
                else:
                    full_embeds = mx.concatenate([prompt_embeds, z, y], axis=1)
                    
                logits, _ = get_logits(model, transformer, lm, full_embeds)
                
                prefix_len = prompt_embeds.shape[1] + 2
                target_logits = logits[:, prefix_len - 1 : prefix_len - 1 + target_labels.shape[1], :]
                ce_loss = nn.losses.cross_entropy(target_logits, target_labels)
                total_loss = total_loss + mx.mean(ce_loss)
            return total_loss / iterations
        else:
            # --- Single Latent Full Unrolled Mode ---
            z = mx.zeros((1, 1, hidden_dim))
            total_loss = 0.0
            for t in range(iterations):
                injected = mx.concatenate([prompt_embeds, z], axis=1)
                _, hidden = get_logits(model, transformer, lm, injected)
                z_next = hidden[:, -1:, :]
                
                if target_embeds_input.shape[1] > 0:
                    full_embeds = mx.concatenate([prompt_embeds, z_next, target_embeds_input], axis=1)
                else:
                    full_embeds = mx.concatenate([prompt_embeds, z_next], axis=1)
                    
                logits, _ = get_logits(model, transformer, lm, full_embeds)
                
                prefix_len = prompt_embeds.shape[1] + 1
                target_logits = logits[:, prefix_len - 1 : prefix_len - 1 + target_labels.shape[1], :]
                ce_loss = nn.losses.cross_entropy(target_logits, target_labels)
                total_loss = total_loss + mx.mean(ce_loss)
                z = z_next
            return total_loss / iterations

def train_continuous_model(
    model_path: str,
    data_path: str,
    iters: int,
    output_path: str,
    lora_layers: int = 2,
    trm_mode: bool = True,
    recursive_iters: int = 3,
    dual_latent: bool = True,
    reasoning_steps: int = 3,
    use_ema: bool = True,
    ema_beta: float = 0.99,
):
    """Trains a continuous latent model with unrolled recurrence and optional EMA smoothing.
    
    Args:
        model_path: Path to the base model (e.g. google/gemma-4-E2B-it-qat-q4_0-unquantized).
        data_path: Path to the JSONL training data.
        iters: Number of training steps to optimize.
        output_path: Output safetensors file path for trained weights.
        lora_layers: Number of transformer layers from the end to apply LoRA to.
        trm_mode: Whether to use Samsung TRM style gradient-free recurrence for prefixes.
        recursive_iters: Number of recursive steps (T).
        dual_latent: Whether to use separate reasoning (z) and solution (y) states.
        reasoning_steps: Number of reasoning inner steps (n) per iteration in dual mode.
        use_ema: Whether to stabilize training using Weight Exponential Moving Average (EMA).
        ema_beta: Weight EMA decay factor.
    """
    print(f"Loading model {model_path}...")
    model, tokenizer = load(model_path)
    model.freeze()
    
    print(f"Configuring LoRA on the last {lora_layers} transformer layers...")
    lora_config = {"rank": 4, "alpha": 8, "dropout": 0.0, "scale": 10.0}
    linear_to_lora_layers(model, num_layers=lora_layers, config=lora_config)
    
    transformer, lm = get_transformer_layers(model)
        
    optimizer = optim.AdamW(learning_rate=1e-4)
    loss_and_grad_fn = nn.value_and_grad(
        model, 
        lambda m, p, t: loss_fn(m, transformer, lm, p, t, iterations=recursive_iters, trm_mode=trm_mode, dual_latent=dual_latent, reasoning_steps=reasoning_steps)
    )
    
    dataset = load_data(data_path)
    if not dataset:
        print("Dataset empty or failed to parse.")
        return
        
    print(f"Loaded {len(dataset)} samples. Starting custom unrolled continuous training (TRM Mode: {trm_mode}, Dual Latent: {dual_latent}, EMA: {use_ema})...")
    
    # Initialize EMA weights
    if use_ema:
        print(f"Initializing Weight EMA (decay beta={ema_beta})...")
        ema_weights = mx_utils.tree_map(lambda x: x, model.trainable_parameters())
        state = [model.state, optimizer.state, ema_weights]
    else:
        ema_weights = None
        state = [model.state, optimizer.state]

    for i in range(iters):
        sample = dataset[i % len(dataset)]
        prompt_tokens = tokenizer.encode(sample[0])
        target_tokens = tokenizer.encode(sample[1])
        
        # MLX value and grad
        loss, grads = loss_and_grad_fn(model, prompt_tokens, target_tokens)
        
        optimizer.update(model, grads)
        
        if use_ema:
            # Update EMA weights and mutate state[2] so mx.eval evaluates current EMA graph
            ema_weights = mx_utils.tree_map(
                lambda ema, param: ema_beta * ema + (1.0 - ema_beta) * param,
                ema_weights,
                model.trainable_parameters()
            )
            state[2] = ema_weights
            
        mx.eval(state) # Evaluates and updates
        
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
        print("Saving raw LoRA weights...")
        save_weights = model.trainable_parameters()
        
    # Save parameters
    try:
        trainable_params = dict(mx_utils.tree_flatten(save_weights))
    except AttributeError:
        def _flatten_dict(d, parent_key=''):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(_flatten_dict(v, new_key).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        trainable_params = _flatten_dict(save_weights)
        
    mx.save_safetensors(output_path, trainable_params)
    print(f"Saved weights to {output_path}")

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
        "ema_beta": ema_beta
    }
    with open(config_path, "w") as f:
        json.dump(adapter_config, f, indent=4)
    print(f"Saved adapter configuration to {config_path}")
