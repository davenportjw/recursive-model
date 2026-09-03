import json
import os
import re
from typing import Dict, Any, List
from datasets import load_dataset

def verify_task_execution(item: Dict[str, Any]) -> bool:
    """Verifies that the canonical solution passes its own unit tests."""
    prompt = item.get("prompt", "")
    solution = item.get("canonical_solution", "")
    test = item.get("test", "")
    entry_point = item.get("entry_point", "")
    
    # If the solution is an indented block following the prompt, concatenate directly
    if solution.startswith(" ") or solution.startswith("\t"):
        full_code = f"{prompt}{solution}"
    else:
        full_code = f"{prompt}\n{solution}"
        
    harness = f"{full_code}\n{test}\ncheck({entry_point})"
    try:
        exec_globals = {}
        exec(harness, exec_globals)
        return True
    except Exception as e:
        print(f"Validation failed for task {item.get('task_id')}: {e}")
        return False

def build_benchmark_suite(
    output_path: str = "eval/benchmark_suite_200.jsonl",
    num_mbpp: int = 33,
    complex_tasks_path: str = "eval/complex_tasks.jsonl"
) -> List[Dict[str, Any]]:
    """Builds a verified, non-gamed 200-task benchmark suite combining HumanEval, MBPP, and complex holdouts."""
    suite: List[Dict[str, Any]] = []
    
    # 1. Ingest HumanEval (164 tasks)
    print("Loading HumanEval (openai/openai_humaneval)...")
    he_dataset = load_dataset("openai/openai_humaneval", split="test")
    he_passed = 0
    for item in he_dataset:
        record = {
            "task_id": item["task_id"],
            "source": "HumanEval",
            "prompt": item["prompt"],
            "entry_point": item["entry_point"],
            "canonical_solution": item["canonical_solution"],
            "test": item["test"]
        }
        if verify_task_execution(record):
            suite.append(record)
            he_passed += 1
    print(f"Ingested & verified {he_passed} / {len(he_dataset)} HumanEval tasks.")
    
    # 2. Ingest MBPP Sanitized (33 tasks)
    print(f"Loading MBPP sanitized test tasks (target: {num_mbpp})...")
    mbpp_dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    mbpp_passed = 0
    for item in mbpp_dataset:
        if mbpp_passed >= num_mbpp:
            break
        code = item["code"]
        m = re.search(r"def\s+(\w+)\s*\(", code)
        if not m:
            continue
        entry_point = m.group(1)
        
        # Build prompt
        prompt = f'def {entry_point}(*args, **kwargs):\n    """\n    {item["prompt"].strip()}\n    """\n'
        
        # Build test harness
        test_body = "\n".join([f"    {t.replace(f'{entry_point}(', 'candidate(')}" for t in item["test_list"]])
        test_harness = f"def check(candidate):\n{test_body}\n"
        
        record = {
            "task_id": f"MBPP/{item['task_id']}",
            "source": "MBPP",
            "prompt": prompt,
            "entry_point": entry_point,
            "canonical_solution": code,
            "test": test_harness
        }
        
        if verify_task_execution(record):
            suite.append(record)
            mbpp_passed += 1
            
    print(f"Ingested & verified {mbpp_passed} MBPP tasks.")
    
    # 3. Ingest Complex Algorithmic Holdout Tasks
    if os.path.exists(complex_tasks_path):
        print(f"Loading complex algorithmic holdouts from {complex_tasks_path}...")
        complex_count = 0
        with open(complex_tasks_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                item["source"] = "ComplexAlgorithmic"
                if verify_task_execution(item):
                    suite.append(item)
                    complex_count += 1
        print(f"Ingested & verified {complex_count} complex algorithmic holdouts.")
        
    print(f"Total benchmark suite size: {len(suite)} verified tasks.")
    
    # Save benchmark suite
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for item in suite:
            f.write(json.dumps(item) + "\n")
    print(f"Saved benchmark suite to {output_path}")
    
    # Also write alias complex_tasks_200.jsonl
    alias_path = "eval/complex_tasks_200.jsonl"
    with open(alias_path, "w") as f:
        for item in suite:
            f.write(json.dumps(item) + "\n")
    print(f"Saved alias benchmark suite to {alias_path}")
    
    return suite

if __name__ == "__main__":
    build_benchmark_suite()
