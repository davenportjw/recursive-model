import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
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
    
    Args:
        model: The MLX LM model.
        transformer: The transformer model.
        lm: The language model wrapper.
        input_embeddings: Input embedding tensor.
        cache: Optional attention cache.
        
    Returns:
        A tuple of (logits, hidden_states).
    """
    # Pass input_embeddings directly to the transformer to get hidden states
    hidden_states = transformer(input_embeddings=input_embeddings, cache=cache)
    
    if hasattr(model, 'lm_head'):
        logits = model.lm_head(hidden_states)
    elif hasattr(lm, 'lm_head'):
        logits = lm.lm_head(hidden_states)
    else:
        logits = hidden_states @ transformer.embed_tokens.weight.T
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
    
    if dual_latent:
        # --- Dual Latent Mode (Samsung TRM style) ---
        # y: Solution state, z: Reasoning/thought state
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
    
    generated_tokens = [next_token.item()]
    
    # Auto-regressive loop
    for _ in range(max_tokens - 1):
        if next_token.item() == getattr(tokenizer, "eos_token_id", None) or next_token.item() in getattr(tokenizer, "additional_special_tokens_ids", []):
            break
            
        token_embeds = transformer.embed_tokens(next_token[None])
        logits, _ = get_logits(model, transformer, lm, token_embeds, cache=cache)
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        generated_tokens.append(next_token.item())
        
    return tokenizer.decode(generated_tokens)

class ContinuousLatentPipeline:
    """A pipeline to easily interact with the continuous latent model."""
    def __init__(self, model_path: str, adapter_path: Optional[str] = None):

        self.model, self.tokenizer = load(model_path)
        
        if adapter_path:
            import os
            if os.path.isdir(adapter_path):
                adapter_path = os.path.join(adapter_path, "continuous_weights.safetensors")
            self.model.load_weights(adapter_path, strict=False)
            
        self.transformer, self.lm = get_transformer_layers(self.model)
        
    def __call__(self, prompt: str, max_tokens: int = 512, iterations: int = 5, dual_latent: bool = True, reasoning_steps: int = 3) -> str:
        return generate_continuous(
            self.model, 
            self.tokenizer, 
            prompt, 
            max_tokens=max_tokens, 
            iterations=iterations, 
            dual_latent=dual_latent, 
            reasoning_steps=reasoning_steps
        )
