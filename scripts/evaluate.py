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
from datetime import datetime
from tiny_recursive_gemma import run_recursive_inference, generate_continuous, ContinuousLatentPipeline
from tiny_recursive_gemma.inference import extract_thought_and_code

def run_test(generated_code: str, test_code: str, entry_point: str) -> bool:
    """
    Executes the generated code combined with the test suite.
    Uses 'uv run python' to execute the test file in the isolated environment.
    """
    # Clean up code using the robust extraction hierarchy
    _, code = extract_thought_and_code(generated_code)
    if not code:
        code = generated_code
        
    # Additional cleanup for any trailing markdown backticks
    code = re.sub(r'^```(?:python)?\s*\n?', '', code.strip())
    code = re.sub(r'\n?```$', '', code.strip())
    
    # Combine code, test suite, and the check() call
    full_code = f"{code}\n\n{test_code}\n\ncheck({entry_point})\n"
    
    # Write to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_code)
        temp_file_path = f.name
        
    try:
        # Run using 'uv run python' with a timeout to prevent infinite loops
        # Fallback to sys.executable if uv is not in PATH
        cmd = ["uv", "run", "python", temp_file_path]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10 # 10 second timeout per test
            )
        except FileNotFoundError:
            import sys
            result = subprocess.run(
                [sys.executable, temp_file_path],
                capture_output=True,
                text=True,
                timeout=10
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

def evaluate_continuous(pipeline: ContinuousLatentPipeline, samples, dual_latent=True, reasoning_steps=3):
    """Evaluates the continuous latent model."""
    passed = 0
    total = len(samples)
    
    for item in tqdm(samples, desc="Evaluating Continuous"):
        prompt = item['prompt']
        generated_code = pipeline(
            prompt, 
            dual_latent=dual_latent, 
            reasoning_steps=reasoning_steps
        )
        
        if run_test(generated_code, item['test'], item['entry_point']):
            passed += 1
            
    return passed, total

def load_eval_dataset(dataset_name: str, max_samples: int):
    """Loads either OpenAI HumanEval or local complex tasks benchmark."""
    if dataset_name.lower() == "complex":
        path = "eval/complex_tasks.jsonl"
        print(f"Loading local complex tasks benchmark from {path}...")
        samples = []
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return samples[:max_samples]
    else:
        print("Loading HumanEval dataset...")
        ds = load_dataset("openai/openai_humaneval", split="test")
        return list(ds)[:max_samples]

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
    parser.add_argument("--dataset", choices=["humaneval", "complex"], default="humaneval", help="Benchmark dataset to run")
    parser.add_argument("--iters", type=int, default=2, help="Number of recursive improvements")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument("--baseline", action="store_true", help="Run baseline evaluation instead of recursive")
    parser.add_argument("--continuous", action="store_true", help="Run continuous latent model evaluation")
    parser.add_argument("--output-report", default="eval/results.json", help="Path to write evaluation report")
    
    parser.add_argument("--no-dual-latent", dest="dual_latent", action="store_false", help="Disable Dual-Latent (y and z) reasoning")
    parser.add_argument("--reasoning-steps", type=int, default=3, help="Number of reasoning steps (n) per iteration in Dual-Latent mode")
    parser.set_defaults(dual_latent=True)
    
    args = parser.parse_args()
    
    samples = load_eval_dataset(args.dataset, args.samples)
    total_samples = len(samples)
    
    if args.baseline:
        print(f"Starting Baseline Zero-Shot Evaluation on {args.dataset} ({total_samples} samples)...")
        model, tokenizer = load(args.model)
        passed, total = evaluate_baseline(model, tokenizer, samples)
        eval_mode = "baseline"
    elif args.continuous:
        print(f"Starting Continuous Latent Evaluation on {args.dataset} ({total_samples} samples)...")
        pipeline = ContinuousLatentPipeline(args.model, adapter_path=args.adapter)
        passed, total = evaluate_continuous(
            pipeline, 
            samples, 
            dual_latent=args.dual_latent, 
            reasoning_steps=args.reasoning_steps
        )
        eval_mode = "continuous"
    else:
        print(f"Starting Recursive Evaluation on {args.dataset} ({total_samples} samples) with {args.iters} iterations...")
        passed, total = evaluate_recursive(args.model, args.adapter, args.iters, samples)
        eval_mode = "recursive"
        
    pass_at_1 = (passed / total) * 100 if total > 0 else 0.0
    print("\n" + "="*30)
    print(f"Evaluation Results ({eval_mode} on {args.dataset}):")
    print(f"Total Samples: {total}")
    print(f"Passed: {passed}")
    print(f"Pass@1: {pass_at_1:.2f}%")
    print("="*30 + "\n")
    
    # Save structured results
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "mode": eval_mode,
        "dataset": args.dataset,
        "model": args.model,
        "total": total,
        "passed": passed,
        "pass_at_1": pass_at_1,
        "dual_latent": args.dual_latent if args.continuous else None,
        "reasoning_steps": args.reasoning_steps if args.continuous else None,
        "iters": args.iters if eval_mode == "recursive" else None
    }
    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    with open(args.output_report, "w") as f:
        json.dump(report, f, indent=4)
    print(f"Saved evaluation report to {args.output_report}")
