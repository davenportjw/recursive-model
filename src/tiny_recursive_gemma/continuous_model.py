import os
import json
from typing import Tuple, Any, Optional, Dict, Union, List
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.tuner.utils import linear_to_lora_layers

from tiny_recursive_gemma.memory_guard import (
    init_memory_guardrails,
    flush_memory,
    guarded_memory_scope
)

_CACHED_MODEL: Optional[Any] = None
_CACHED_TOKENIZER: Optional[Any] = None
_CACHED_MODEL_KEY: Optional[Tuple[str, Optional[str]]] = None

def get_or_load_model(model_path: str, adapter_path: Optional[str] = None) -> Tuple[Any, Any]:
    """Retrieves or loads the model as a singleton to avoid holding multiple copies in unified memory."""
    global _CACHED_MODEL, _CACHED_TOKENIZER, _CACHED_MODEL_KEY
    key = (model_path, adapter_path)
    if _CACHED_MODEL is not None and _CACHED_MODEL_KEY == key:
        return _CACHED_MODEL, _CACHED_TOKENIZER
    
    init_memory_guardrails()
    flush_memory()
    print(f"Loading model {model_path} with memory guardrails...")
    model, tokenizer = load(model_path, adapter_path=adapter_path)
    _CACHED_MODEL = model
    _CACHED_TOKENIZER = tokenizer
    _CACHED_MODEL_KEY = key
    return model, tokenizer

class ACTHaltingHead(nn.Module):
    """Adaptive Computation Time (ACT) halting classifier head."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc = nn.Linear(hidden_dim * 2, 1)
        # Negative bias initialization to encourage multi-step recurrence early in training
        self.fc.bias = mx.full((1,), -2.0)

    def __call__(self, y: mx.array, z: mx.array) -> mx.array:
        """Computes halting probability given solution state y and reasoning state z."""
        concat = mx.concatenate([y, z], axis=-1)
        logits = self.fc(concat)
        return mx.sigmoid(logits)

def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    """Applies RMSNorm across the hidden dimension for latent stability."""
    variance = mx.mean(x ** 2, axis=-1, keepdims=True)
    return x * mx.rsqrt(variance + eps)

def get_transformer_layers(model: Any) -> Tuple[Any, Any]:
    """Helper to extract the base transformer and language model head.
    
    Args:
        model: The MLX LM model.
        
    Returns:
        A tuple of (transformer, language_model).
    """
    lm = getattr(model, 'language_model', model)
    return getattr(lm, 'model', lm), lm

def get_logits(
    model: Any, 
    transformer: Any, 
    lm: Any, 
    input_embeddings: mx.array, 
    cache: Optional[Any] = None
) -> Tuple[mx.array, mx.array]:
    """Helper to get logits directly from embeddings.
    
    Supports native input_embeddings arguments as well as direct layer-by-layer
    forward propagation fallback for models that require embedding bypass.
    
    Args:
        model: The MLX LM model.
        transformer: The transformer model.
        lm: The language model wrapper.
        input_embeddings: Input embedding tensor.
        cache: Optional attention cache.
        
    Returns:
        A tuple of (logits, hidden_states).
    """
    try:
        # Try native input_embeddings keyword argument (supported in Gemma 4)
        hidden_states = transformer(input_embeddings=input_embeddings, cache=cache)
    except TypeError:
        try:
            # Fallback 1: positional pass
            hidden_states = transformer(input_embeddings, cache=cache)
        except Exception:
            # Fallback 2: route through transformer layers directly
            h = input_embeddings
            if hasattr(transformer, "layers"):
                for layer in transformer.layers:
                    h = layer(h, cache=cache)
                if hasattr(transformer, "norm"):
                    h = transformer.norm(h)
            hidden_states = h
    
    if hasattr(model, 'lm_head') and model.lm_head is not None:
        logits = model.lm_head(hidden_states)
    elif hasattr(lm, 'lm_head') and lm.lm_head is not None:
        logits = lm.lm_head(hidden_states)
    elif hasattr(transformer, 'embed_tokens'):
        logits = hidden_states @ transformer.embed_tokens.weight.T
    else:
        logits = hidden_states
        
    return logits, hidden_states


def generate_continuous(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int = 512,
    iterations: int = 5,
    dual_latent: bool = True,
    reasoning_steps: int = 3,
    recurrent_layers: int = 2,
    halting_head: Optional[ACTHaltingHead] = None,
    halt_threshold: float = 0.85,
    min_iterations: int = 2,
    normalize_latents: bool = True,
    return_telemetry: bool = False,
) -> Union[str, Tuple[str, Dict[str, Any]]]:
    """Generates text from a continuous latent model with T iterations and optional ACT early stopping.
    
    Args:
        model: The base MLX LM model.
        tokenizer: The tokenizer compatible with the model.
        prompt: The input prompt string.
        max_tokens: Maximum number of tokens to generate.
        iterations: Maximum number of continuous latent recurrence iterations (T_max).
        dual_latent: Whether to use separate reasoning (z) and solution (y) states.
        reasoning_steps: Number of reasoning inner steps (n) per iteration in dual mode.
        recurrent_layers: Number of top transformer layers to recycle (Top-K layer recycling).
        halting_head: Optional ACTHaltingHead module for adaptive computation time halting.
        halt_threshold: Halting probability threshold to trigger early exit (tau).
        min_iterations: Minimum number of recurrence steps before allowing early exit.
        normalize_latents: Whether to apply RMSNorm to latent states before injection.
        return_telemetry: If True, returns (text, telemetry_dict).
        
    Returns:
        The generated text response, or (text, telemetry_dict) if return_telemetry is True.
    """
    init_memory_guardrails()
    transformer, lm = get_transformer_layers(model)
    
    prompt_ids = mx.array(tokenizer.encode(prompt))[None]
    prompt_embeds = transformer.embed_tokens(prompt_ids)
    
    hidden_dim = prompt_embeds.shape[-1]
    
    # Gemma 4 stopping tokens: check eos_token_id and Gemma end_of_turn (ID 107)
    stop_tokens = {getattr(tokenizer, "eos_token_id", None), 107}
    special_ids = getattr(tokenizer, "additional_special_tokens_ids", None)
    if special_ids:
        stop_tokens.update(special_ids)
    stop_tokens.discard(None)
    
    trajectory_distances: List[float] = []
    z_prev: Optional[mx.array] = None
    iterations_completed = 0
    halted_early = False
    
    if dual_latent:
        # --- Dual Latent Mode (Samsung TRM style) ---
        z = mx.zeros((1, 1, hidden_dim))
        y = mx.zeros((1, 1, hidden_dim))
        
        # Check if Top-K layer recycling can be used
        use_layer_recycling = (
            recurrent_layers > 0 
            and hasattr(transformer, "layers") 
            and 0 < recurrent_layers < len(transformer.layers)
        )
        if use_layer_recycling:
            base_layers = transformer.layers[:-recurrent_layers]
            top_layers = transformer.layers[-recurrent_layers:]
            h_base = prompt_embeds
            for layer in base_layers:
                h_base = layer(h_base)
        else:
            base_layers = None
            top_layers = None
            h_base = None

        for iter_idx in range(iterations):
            iterations_completed = iter_idx + 1
            
            # 1. Update reasoning state z (n times)
            for _ in range(reasoning_steps):
                y_inj = rms_norm(y) if normalize_latents else y
                z_inj = rms_norm(z) if normalize_latents else z
                if use_layer_recycling:
                    injected = mx.concatenate([h_base, y_inj, z_inj], axis=1)
                    h = injected
                    for layer in top_layers:
                        h = layer(h)
                    if hasattr(transformer, "norm") and transformer.norm is not None:
                        h = transformer.norm(h)
                    z = h[:, -1:, :]
                else:
                    injected = mx.concatenate([prompt_embeds, y_inj, z_inj], axis=1)
                    _, hidden_states = get_logits(model, transformer, lm, injected)
                    z = hidden_states[:, -1:, :]
            
            # 2. Update solution state y (1 time)
            z_inj = rms_norm(z) if normalize_latents else z
            y_inj = rms_norm(y) if normalize_latents else y
            if use_layer_recycling:
                injected = mx.concatenate([h_base, z_inj, y_inj], axis=1)
                h = injected
                for layer in top_layers:
                    h = layer(h)
                if hasattr(transformer, "norm") and transformer.norm is not None:
                    h = transformer.norm(h)
                y = h[:, -1:, :]
            else:
                injected = mx.concatenate([prompt_embeds, z_inj, y_inj], axis=1)
                _, hidden_states = get_logits(model, transformer, lm, injected)
                y = hidden_states[:, -1:, :]

            # Force evaluation and flush unused GPU buffers to protect Apple Silicon unified memory
            mx.eval(z, y)
            flush_memory()
            
            # Track latent trajectory distance d(z_t, z_{t-1})
            if z_prev is not None:
                z_flat = z.reshape(-1)
                z_prev_flat = z_prev.reshape(-1)
                denom = (mx.linalg.norm(z_flat) * mx.linalg.norm(z_prev_flat)).item()
                if denom > 1e-8:
                    cos_sim = (mx.sum(z_flat * z_prev_flat).item()) / denom
                    trajectory_distances.append(float(max(0.0, 1.0 - cos_sim)))
            z_prev = z
            
            # 3. Check ACT halting head
            if halting_head is not None:
                halt_prob = float(halting_head(y, z)[0, 0].item())
                if (iter_idx + 1) >= min_iterations and halt_prob >= halt_threshold:
                    halted_early = True
                    break
            
        # Prime the generation with prompt, final reasoning state z, and final solution state y
        z_inj = rms_norm(z) if normalize_latents else z
        y_inj = rms_norm(y) if normalize_latents else y
        injected_embeds = mx.concatenate([prompt_embeds, z_inj, y_inj], axis=1)
    else:
        # --- Single Latent Mode ---
        z = mx.zeros((1, 1, hidden_dim))
        for iter_idx in range(iterations):
            iterations_completed = iter_idx + 1
            z_inj = rms_norm(z) if normalize_latents else z
            injected_embeds = mx.concatenate([prompt_embeds, z_inj], axis=1)
            _, hidden_states = get_logits(model, transformer, lm, injected_embeds)
            z = hidden_states[:, -1:, :]

            mx.eval(z)
            flush_memory()
            
            if z_prev is not None:
                z_flat = z.reshape(-1)
                z_prev_flat = z_prev.reshape(-1)
                denom = (mx.linalg.norm(z_flat) * mx.linalg.norm(z_prev_flat)).item()
                if denom > 1e-8:
                    cos_sim = (mx.sum(z_flat * z_prev_flat).item()) / denom
                    trajectory_distances.append(float(max(0.0, 1.0 - cos_sim)))
            z_prev = z
            
        z_inj = rms_norm(z) if normalize_latents else z
        injected_embeds = mx.concatenate([prompt_embeds, z_inj], axis=1)
    
    # 2. Final pass to generate auto-regressively
    try:
        cache = make_prompt_cache(model)
    except Exception:
        cache = None
    
    # Process prefix
    logits, _ = get_logits(model, transformer, lm, injected_embeds, cache=cache)
    
    # Get the first generated token
    next_token = mx.argmax(logits[:, -1, :], axis=-1)
    
    generated_tokens = []
    if next_token.item() not in stop_tokens:
        generated_tokens.append(next_token.item())
    else:
        decoded_text = tokenizer.decode(generated_tokens)
        if return_telemetry:
            return decoded_text, {
                "iterations": iterations_completed,
                "halted_early": halted_early,
                "trajectory_distances": trajectory_distances,
                "generated_token_count": 0
            }
        return decoded_text
    
    # Auto-regressive loop
    for _ in range(max_tokens - 1):
        token_embeds = transformer.embed_tokens(next_token[None])
        logits, _ = get_logits(model, transformer, lm, token_embeds, cache=cache)
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        
        if next_token.item() in stop_tokens:
            break
            
        generated_tokens.append(next_token.item())
        
    decoded_text = tokenizer.decode(generated_tokens)
    flush_memory()
    if return_telemetry:
        return decoded_text, {
            "iterations": iterations_completed,
            "halted_early": halted_early,
            "trajectory_distances": trajectory_distances,
            "generated_token_count": len(generated_tokens)
        }
    return decoded_text

class ContinuousLatentPipeline:
    """A pipeline to easily interact with the continuous latent model."""
    def __init__(
        self, 
        model_path: Optional[str] = None, 
        adapter_path: Optional[str] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        lora_layers: int = 2,
        lora_config: Optional[dict] = None,
        enable_act: bool = False,
        recurrent_layers: int = 2
    ):
        init_memory_guardrails()
        self.recurrent_layers = recurrent_layers
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            target_path = model_path or os.getenv("BASE_MODEL", "google/gemma-4-E2B-it-qat-q4_0-unquantized")
            self.model, self.tokenizer = get_or_load_model(target_path, adapter_path=adapter_path)
        self.halting_head: Optional[ACTHaltingHead] = None
        
        if adapter_path:
            config_file = None
            weights_file = adapter_path
            
            if os.path.isdir(adapter_path):
                config_file = os.path.join(adapter_path, "adapter_config.json")
                weights_file = os.path.join(adapter_path, "continuous_weights.safetensors")
            else:
                candidate_config = os.path.join(os.path.dirname(adapter_path), "adapter_config.json")
                if os.path.exists(candidate_config):
                    config_file = candidate_config
                    
            # Parse saved adapter config if present
            if config_file and os.path.exists(config_file):
                try:
                    with open(config_file, "r") as f:
                        saved_cfg = json.load(f)
                        lora_layers = saved_cfg.get("num_layers", lora_layers)
                        lora_config = saved_cfg.get("lora_parameters", lora_config)
                        enable_act = saved_cfg.get("enable_act", enable_act)
                except Exception as e:
                    print(f"Warning: could not parse adapter config {config_file}: {e}")
                    
            if lora_config is None:
                lora_config = {"rank": 4, "alpha": 8, "dropout": 0.0, "scale": 10.0}
                
            print(f"Initializing LoRA layers on last {lora_layers} layers...")
            linear_to_lora_layers(self.model, num_layers=lora_layers, config=lora_config)
            
            if os.path.exists(weights_file):
                print(f"Loading continuous LoRA weights from {weights_file}...")
                self.model.load_weights(weights_file, strict=False)
            else:
                print(f"Warning: weights file {weights_file} not found. Running with initialized LoRA adapter.")
            
        self.transformer, self.lm = get_transformer_layers(self.model)
        
        if enable_act:
            hidden_dim = getattr(self.transformer, "embed_tokens").weight.shape[-1]
            self.halting_head = ACTHaltingHead(hidden_dim)
            if adapter_path:
                act_weights = os.path.join(adapter_path, "act_head.safetensors") if os.path.isdir(adapter_path) else None
                if act_weights and os.path.exists(act_weights):
                    print(f"Loading ACT halting head weights from {act_weights}...")
                    self.halting_head.load_weights(act_weights, strict=False)
        
    def __call__(
        self, 
        prompt: str, 
        max_tokens: int = 512, 
        iterations: int = 5, 
        dual_latent: bool = True, 
        reasoning_steps: int = 3,
        recurrent_layers: Optional[int] = None,
        halt_threshold: float = 0.85,
        min_iterations: int = 2,
        normalize_latents: bool = True,
        return_telemetry: bool = False
    ) -> Union[str, Tuple[str, Dict[str, Any]]]:
        rec_layers = self.recurrent_layers if recurrent_layers is None else recurrent_layers
        return generate_continuous(
            self.model, 
            self.tokenizer, 
            prompt, 
            max_tokens=max_tokens, 
            iterations=iterations, 
            dual_latent=dual_latent, 
            reasoning_steps=reasoning_steps,
            recurrent_layers=rec_layers,
            halting_head=self.halting_head,
            halt_threshold=halt_threshold,
            min_iterations=min_iterations,
            normalize_latents=normalize_latents,
            return_telemetry=return_telemetry
        )
