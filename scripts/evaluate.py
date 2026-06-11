import argparse
import json
import os
import subprocess
import tempfile
import re
from tqdm import tqdm
import mlx.core as mx
from mlx_lm import load, generate
from datasets import load_dataset
from dotenv import load_dotenv
from tiny_recursive_gemma import run_recursive_inference, generate_continuous

def run_test(generated_code: str, test_code: str, entry_point: str) -> bool:
    """
    Executes the generated code combined with the HumanEval test suite.
    Uses 'uv run python' to execute the test file in the isolated environment.
    """
    # Clean up markdown if present
    code = generated_code
    if "```python" in code:
        blocks = re.findall(r'```python\n(.*?)\n```', code, re.DOTALL)
        if blocks:
            code = blocks[-1]
    
    # Combine code, test suite, and the check() call
    full_code = f"{code}\n\n{test_code}\n\ncheck({entry_point})\n"
    
    # Write to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_code)
        temp_file_path = f.name
        
    try:
        # Run using 'uv run python' with a timeout to prevent infinite loops
        result = subprocess.run(
            ["uv", "run", "python", temp_file_path],
            capture_output=True,
            text=True,
            timeout=10 # 10 second timeout per test
        )
        passed = (result.returncode == 0)
    except subprocess.TimeoutExpired:
        passed = False
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
    return passed

def evaluate_baseline(model, tokenizer, samples):
    """Zero-shot evaluation without recursion."""
    passed = 0
    total = len(samples)
    
    for item in tqdm(samples, desc="Evaluating Baseline"):
        prompt = item['prompt']
        
        # Simple completion
        messages = [{"role": "user", "content": f"Complete this Python function:\n{prompt}"}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        response = generate(model, tokenizer, prompt=formatted_prompt, max_tokens=512, verbose=False)
        
        if run_test(response, item['test'], item['entry_point']):
            passed += 1
            
    return passed, total

def evaluate_recursive(model_path, adapter_path, iters, samples):
    """Evaluates the recursive model."""
    passed = 0
    total = len(samples)
    
    print(f"Loading model {model_path} with adapters from {adapter_path}...")
    model, tokenizer = load(model_path, adapter_path=adapter_path)
    
    for item in tqdm(samples, desc="Evaluating Recursive"):
        prompt = item['prompt']
        # Pass the preloaded model and tokenizer to avoid reloading on every sample
        generated_code = run_recursive_inference(
            model_path, 
            adapter_path, 
            prompt, 
            max_iters=iters,
            model=model,
            tokenizer=tokenizer
        )
        
        if run_test(generated_code, item['test'], item['entry_point']):
            passed += 1
            
    return passed, total



def evaluate_continuous(model, tokenizer, samples, dual_latent=True, reasoning_steps=3):
    """Evaluates the continuous latent model."""
    passed = 0
    total = len(samples)
    
    for item in tqdm(samples, desc="Evaluating Continuous"):
        prompt = item['prompt']
        generated_code = generate_continuous(
            model, 
            tokenizer, 
            prompt, 
            dual_latent=dual_latent, 
            reasoning_steps=reasoning_steps
        )
        
        if run_test(generated_code, item['test'], item['entry_point']):
            passed += 1
            
    return passed, total

if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", 
        default=os.getenv("BASE_MODEL", "google/gemma-4-E2B-it-qat-q4_0-unquantized"), 
        help="Model path"
    )
    parser.add_argument(
        "--adapter", 
        default=os.getenv("ADAPTER_PATH", "adapters"), 
        help="Path to LoRA adapters (if testing recursive)"
    )
    parser.add_argument("--iters", type=int, default=2, help="Number of recursive improvements")
    parser.add_argument("--samples", type=int, default=10, help="Number of HumanEval samples to evaluate")
    parser.add_argument("--baseline", action="store_true", help="Run baseline evaluation instead of recursive")
    parser.add_argument("--continuous", action="store_true", help="Run continuous latent model evaluation")
    
    parser.add_argument("--no-dual-latent", dest="dual_latent", action="store_false", help="Disable Dual-Latent (y and z) reasoning")
    parser.add_argument("--reasoning-steps", type=int, default=3, help="Number of reasoning steps (n) per iteration in Dual-Latent mode")
    parser.set_defaults(dual_latent=True)
    
    args = parser.parse_args()
    
    print("Loading dataset...")
    ds = load_dataset("openai/openai_humaneval", split="test")
    samples = list(ds)[:args.samples]
    
    if args.baseline:
        print(f"Starting Baseline Zero-Shot Evaluation for {args.samples} samples...")
        model, tokenizer = load(args.model)
        passed, total = evaluate_baseline(model, tokenizer, samples)
    elif args.continuous:
        print(f"Starting Continuous Latent Evaluation for {args.samples} samples...")
        model, tokenizer = load(args.model)
        if args.adapter:
            adapter_path = args.adapter
            if os.path.isdir(adapter_path):
                adapter_path = os.path.join(adapter_path, "continuous_weights.safetensors")
            print(f"Loading continuous weights from {adapter_path}...")
            model.load_weights(adapter_path, strict=False)
        passed, total = evaluate_continuous(
            model, 
            tokenizer, 
            samples, 
            dual_latent=args.dual_latent, 
            reasoning_steps=args.reasoning_steps
        )
    else:
        print(f"Starting Recursive Evaluation for {args.samples} samples with {args.iters} iterations...")
        passed, total = evaluate_recursive(args.model, args.adapter, args.iters, samples)
        
    print("\n" + "="*30)
    print(f"Evaluation Results:")
    print(f"Total Samples: {total}")
    print(f"Passed: {passed}")
    print(f"Pass@1: {(passed/total)*100:.2f}%")
    print("="*30 + "\n")
