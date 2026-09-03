import os
import json
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.tuner.utils import linear_to_lora_layers
from typing import Tuple, Any, Optional

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
) -> str:
    """Generates text from a continuous latent model with T iterations.
    
    Args:
        model: The base MLX LM model.
        tokenizer: The tokenizer compatible with the model.
        prompt: The input prompt string.
        max_tokens: Maximum number of tokens to generate.
        iterations: Number of continuous latent recurrence iterations (T).
        dual_latent: Whether to use separate reasoning (z) and solution (y) states.
        reasoning_steps: Number of reasoning inner steps (n) per iteration in dual mode.
        
    Returns:
        The generated text response.
    """
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
    
    if dual_latent:
        # --- Dual Latent Mode (Samsung TRM style) ---
        # y: Solution state, z: Reasoning/thought state
        # Initialized with small magnitude for RMSNorm stability
        z = mx.zeros((1, 1, hidden_dim))
        y = mx.zeros((1, 1, hidden_dim))
        
        for _ in range(iterations):
            # 1. Update reasoning state z (n times)
            for _ in range(reasoning_steps):
                injected = mx.concatenate([prompt_embeds, y, z], axis=1)
                _, hidden_states = get_logits(model, transformer, lm, injected)
                z = hidden_states[:, -1:, :]
            
            # 2. Update solution state y (1 time)
            injected = mx.concatenate([prompt_embeds, z, y], axis=1)
            _, hidden_states = get_logits(model, transformer, lm, injected)
            y = hidden_states[:, -1:, :]
            
        # Prime the generation with prompt, final reasoning state z, and final solution state y
        injected_embeds = mx.concatenate([prompt_embeds, z, y], axis=1)
    else:
        # --- Single Latent Mode ---
        z = mx.zeros((1, 1, hidden_dim))
        for _ in range(iterations):
            injected_embeds = mx.concatenate([prompt_embeds, z], axis=1)
            _, hidden_states = get_logits(model, transformer, lm, injected_embeds)
            z = hidden_states[:, -1:, :]
        injected_embeds = mx.concatenate([prompt_embeds, z], axis=1)
    
    # 2. Final pass to generate auto-regressively
    cache = make_prompt_cache(model)
    
    # Process prefix
    logits, _ = get_logits(model, transformer, lm, injected_embeds, cache=cache)
    
    # Get the first generated token
    next_token = mx.argmax(logits[:, -1, :], axis=-1)
    
    generated_tokens = []
    if next_token.item() not in stop_tokens:
        generated_tokens.append(next_token.item())
    else:
        return tokenizer.decode(generated_tokens)
    
    # Auto-regressive loop
    for _ in range(max_tokens - 1):
        token_embeds = transformer.embed_tokens(next_token[None])
        logits, _ = get_logits(model, transformer, lm, token_embeds, cache=cache)
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        
        if next_token.item() in stop_tokens:
            break
            
        generated_tokens.append(next_token.item())
        
    return tokenizer.decode(generated_tokens)

class ContinuousLatentPipeline:
    """A pipeline to easily interact with the continuous latent model."""
    def __init__(
        self, 
        model_path: str, 
        adapter_path: Optional[str] = None,
        lora_layers: int = 2,
        lora_config: Optional[dict] = None
    ):
        print(f"Loading base model {model_path}...")
        self.model, self.tokenizer = load(model_path)
        
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
        
    def __call__(
        self, 
        prompt: str, 
        max_tokens: int = 512, 
        iterations: int = 5, 
        dual_latent: bool = True, 
        reasoning_steps: int = 3
    ) -> str:
        return generate_continuous(
            self.model, 
            self.tokenizer, 
            prompt, 
            max_tokens=max_tokens, 
            iterations=iterations, 
            dual_latent=dual_latent, 
            reasoning_steps=reasoning_steps
        )
