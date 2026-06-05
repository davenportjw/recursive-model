import argparse
import mlx.core as mx
from mlx_lm import load, generate

def run_recursive_inference(model_path, adapter_path, prompt, max_iters=2):
    """
    Runs the model recursively, feeding its thoughts and code back into itself.
    """
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
        import re
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-4-E2B-it-qat-q4_0-unquantized", help="Model path")
    parser.add_argument("--adapter", default="adapters", help="Path to LoRA adapters")
    parser.add_argument("--iters", type=int, default=2, help="Number of recursive improvements")
    parser.add_argument("--prompt", type=str, required=True, help="The coding task")
    args = parser.parse_args()
    
    run_recursive_inference(args.model, args.adapter, args.prompt, args.iters)
