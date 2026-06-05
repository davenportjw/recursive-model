import argparse
import json
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
import math
from tqdm import tqdm

def load_data(file_path):
    # Load custom jsonl where we extract the 'user' prompt and 'model' response (the code part)
    # Our data/train.jsonl has the MLX conversational format: {"text": "<|im_start|>user...<|im_end|>\n<|im_start|>model\n..."}
    # Since we are abandoning text-thoughts, let's extract just the user prompt and the final code.
    dataset = []
    with open(file_path, 'r') as f:
        for line in f:
            obj = json.loads(line)
            if 'messages' in obj:
                user_part = obj['messages'][0]['content']
                code_part = obj['messages'][1]['content']
                dataset.append((user_part, code_part))
    return dataset

def get_transformer(model):
    lm = getattr(model, 'language_model', model)
    return getattr(lm, 'model', lm), lm

def get_logits(model, transformer, lm, input_embeddings):
    hidden_states = transformer(input_embeddings=input_embeddings)
    if hasattr(model, 'lm_head'):
        logits = model.lm_head(hidden_states)
    elif hasattr(lm, 'lm_head'):
        logits = lm.lm_head(hidden_states)
    else:
        logits = hidden_states @ transformer.embed_tokens.weight.T
    return logits, hidden_states

def loss_fn(model, transformer, lm, prompt_tokens, target_tokens):
    # Step 1: Forward pass on prompt to generate z
    prompt_ids = mx.array(prompt_tokens)[None] # [1, L]
    prompt_embeds = transformer.embed_tokens(prompt_ids) # [1, L, D]
    
    # Run through transformer to get hidden states
    _, hidden_states_1 = get_logits(model, transformer, lm, prompt_embeds)
    
    # Project to continuous latent bottleneck z
    # Pool over sequence length to get [1, 1, D]
    z = mx.mean(hidden_states_1, axis=1, keepdims=True)
    
    # Step 2: Inject z for final prediction
    # We want to predict target_tokens. 
    # For causal LM, input is prompt + target[:-1], label is target[1:]
    # Since we are injecting z, input_embeds = [z, prompt_embeds, target_embeds[:-1]]
    
    target_ids_input = mx.array(target_tokens[:-1])[None]
    target_embeds_input = transformer.embed_tokens(target_ids_input)
    
    injected_embeds = mx.concatenate([z, prompt_embeds, target_embeds_input], axis=1)
    
    logits, _ = get_logits(model, transformer, lm, injected_embeds)
    
    # The targets should align with the predictions for target_embeds
    # Injected sequence length: 1 (z) + L_prompt + L_target - 1
    # We only compute loss on the target portion.
    # The logits for the target start after index: 1 + L_prompt - 1 = L_prompt
    
    target_logits = logits[:, 1 + prompt_embeds.shape[1] - 1:]
    target_labels = mx.array(target_tokens)[None]
    
    # Cross entropy loss
    ce_loss = nn.losses.cross_entropy(target_logits, target_labels)
    return mx.mean(ce_loss)

def train(model_path, data_path, iters):
    print("Loading model...")
    model, tokenizer = load(model_path)
    model.freeze()
    
    # Unfreeze LoRA layers (we assume adapters are already loaded or we inject them)
    # For simplicity in this script, let's inject a simple LoRA into linear layers manually 
    # or just unfreeze the top 2 layers to save memory, as mlx_lm handles LoRA injection via CLI.
    # We will just unfreeze the last 2 layers for this custom script experiment
    transformer, lm = get_transformer(model)
    for l in transformer.layers[-2:]:
        l.unfreeze()
        
    optimizer = optim.AdamW(learning_rate=1e-4)
    loss_and_grad_fn = nn.value_and_grad(model, lambda m, p, t: loss_fn(m, transformer, lm, p, t))
    
    dataset = load_data(data_path)
    if not dataset:
        print("Dataset empty or failed to parse.")
        return
        
    print(f"Loaded {len(dataset)} samples. Starting custom unrolled continuous training...")
    
    state = [model.state, optimizer.state]
    
    for i in range(iters):
        # We just cycle through the dataset
        sample = dataset[i % len(dataset)]
        prompt_tokens = tokenizer.encode(sample[0])
        target_tokens = tokenizer.encode(sample[1])
        
        # MLX value and grad
        loss, grads = loss_and_grad_fn(model, prompt_tokens, target_tokens)
        
        optimizer.update(model, grads)
        mx.eval(state) # Evaluates and updates
        
        print(f"Iter {i+1}/{iters} | Loss: {loss.item():.4f}")
        
    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-4-E2B-it-qat-q4_0-unquantized")
    parser.add_argument("--data", default="data/train.jsonl")
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()
    
    train(args.model, args.data, args.iters)
