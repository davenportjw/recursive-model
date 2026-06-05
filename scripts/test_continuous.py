import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

def test_continuous_injection():
    print("Loading model...")
    model, tokenizer = load("google/gemma-4-E2B-it-qat-q4_0-unquantized")
    
    prompt = "Hello, what is 2+2?"
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array(tokens)[None] # [1, L]
    
    print(f"Input IDs shape: {input_ids.shape}")
    
    # 1. Standard forward pass to get logits and hidden states
    print(f"Model attributes: {[k for k in dir(model) if not k.startswith('__')]}")
    
    # In MLX's gemma4, the actual text transformer is nested under `language_model.model`
    lm = getattr(model, 'language_model', model)
    transformer = getattr(lm, 'model', lm)
    print(f"Transformer type: {type(transformer)}")
    
    # Get embeddings
    embeddings = transformer.embed_tokens(input_ids)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Create latent vector z
    z = mx.mean(embeddings, axis=1, keepdims=True)
    
    # Draft embeddings
    draft = "Let me think."
    draft_ids = mx.array(tokenizer.encode(draft))[None]
    draft_embeddings = transformer.embed_tokens(draft_ids)
    
    injected_embeddings = mx.concatenate([z, draft_embeddings], axis=1)
    print(f"Injected embeddings shape: {injected_embeddings.shape}")
    
    import inspect
    sig = inspect.signature(transformer.__call__)
    print(f"Transformer __call__ signature: {sig}")
    
    try:
        print("Passing injected embeddings to transformer...")
        # Get hidden states
        hidden_states = transformer(input_embeddings=injected_embeddings)
        
        # Get logits
        if hasattr(model, 'lm_head'):
            logits = model.lm_head(hidden_states)
        elif hasattr(lm, 'lm_head'):
            logits = lm.lm_head(hidden_states)
        else:
            # Maybe it's tied to embed_tokens
            logits = hidden_states @ transformer.embed_tokens.weight.T
            
        print(f"Final output/logits shape: {logits.shape}")
        print("Continuous Injection SUCCESS!")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Injection FAILED: {e}")
        import traceback
        traceback.print_exc()
        print(f"Injection FAILED: {e}")
        
if __name__ == "__main__":
    test_continuous_injection()
