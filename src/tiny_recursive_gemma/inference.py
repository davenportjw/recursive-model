import re
from mlx_lm import load, generate

def run_recursive_inference(
    model_path: str,
    adapter_path: str,
    prompt: str,
    max_iters: int = 2,
    model = None,
    tokenizer = None,
):
    """
    Runs the model recursively, feeding its thoughts and code back into itself.
    If model and tokenizer are provided, they are used directly to avoid reloading.
    """
    if model is None or tokenizer is None:
        print(f"Loading model {model_path} with adapters from {adapter_path}...")
        model, tokenizer = load(model_path, adapter_path=adapter_path)

    
    current_code = ""
    current_thought = ""
    
    for i in range(max_iters):
        print(f"\n--- Iteration {i+1} ---")
        
        # Build prompt using the same format as training
        user_msg = f"Task:\n{prompt}\n\nCurrent Code:\n{current_code}\n\nPrevious Thought:\n{current_thought}\n\nAnalyze the current code and provide an updated thought process, followed by the refined code."
        
        # Format conversation using the tokenizer's chat template
        messages = [{"role": "user", "content": user_msg}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generate next step
        response = generate(
            model,
            tokenizer,
            prompt=formatted_prompt,
            max_tokens=1024,
            verbose=False
        )
        
        print(f"Raw Output:\n{response}\n")
        
        # Parse the output
        # First try custom XML tags we trained it on
        thought_match = re.search(r'<thought>(.*?)</thought>', response, re.DOTALL)
        code_match = re.search(r'<code_update>(.*?)</code_update>', response, re.DOTALL)
        
        if thought_match and code_match:
            current_thought = thought_match.group(1).strip()
            current_code = code_match.group(1).strip()
        else:
            # Fallback for under-trained models: Grab the last python code block
            print("Warning: XML tags not found. Falling back to markdown parsing.")
            code_blocks = re.findall(r'```python\n(.*?)\n```', response, re.DOTALL)
            if code_blocks:
                current_code = code_blocks[-1].strip()
            
            # Just grab all text before the code block as the thought
            thought_text = re.sub(r'```python\n.*?\n```', '', response, flags=re.DOTALL)
            current_thought = thought_text.strip()
            
        print(f"Extracted Code Update:\n{current_code}")
            
    print("\n=== FINAL RESULT ===")
    print(current_code)
    return current_code
