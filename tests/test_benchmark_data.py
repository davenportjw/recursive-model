import json
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from eval.judge_evaluator import execute_unit_test

def test_benchmark_suite_integrity():
    suite_path = "eval/benchmark_suite_200.jsonl"
    assert os.path.exists(suite_path), f"{suite_path} must exist"
    
    with open(suite_path, "r") as f:
        tasks = [json.loads(line) for line in f if line.strip()]
        
    assert len(tasks) == 200, f"Expected 200 tasks in benchmark suite, got {len(tasks)}"
    
    required_keys = {"task_id", "prompt", "canonical_solution", "test", "entry_point"}
    for idx, task in enumerate(tasks):
        missing = required_keys - set(task.keys())
        assert not missing, f"Task {idx} ({task.get('task_id')}) missing keys: {missing}"
        assert task["prompt"].strip(), f"Task {idx} prompt is empty"
        assert task["canonical_solution"].strip(), f"Task {idx} canonical_solution is empty"
        assert task["test"].strip(), f"Task {idx} test is empty"
        assert task["entry_point"].strip(), f"Task {idx} entry_point is empty"

def test_continuous_train_augmented_integrity():
    train_path = "data/continuous_train_augmented.jsonl"
    assert os.path.exists(train_path), f"{train_path} must exist"
    
    with open(train_path, "r") as f:
        samples = [json.loads(line) for line in f if line.strip()]
        
    assert len(samples) == 200
    for idx, s in enumerate(samples):
        assert "prompt" in s and s["prompt"].strip()
        assert ("solution" in s and s["solution"].strip()) or ("code" in s and s["code"].strip())

def test_sample_execution_verification():
    """Executes a sample of tasks from the benchmark suite to verify executable harnesses."""
    suite_path = "eval/benchmark_suite_200.jsonl"
    with open(suite_path, "r") as f:
        tasks = [json.loads(line) for line in f if line.strip()]
        
    # Test first 3 tasks (HumanEval) and last task (Complex Holdout)
    sample_indices = [0, 1, 2, 199]
    for idx in sample_indices:
        t = tasks[idx]
        code = f"{t['prompt']}\n{t['canonical_solution']}"
        passed, err = execute_unit_test(code, t["test"], t["entry_point"])
        assert passed is True, f"Task {t['task_id']} failed execution test: {err}"
