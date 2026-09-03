import json
import pytest

def test_complex_tasks_canonical_solutions():
    """Verify that all canonical solutions in eval/complex_tasks.jsonl pass their check() test suites."""
    with open("eval/complex_tasks.jsonl", "r") as f:
        for line in f:
            if not line.strip():
                continue
            task = json.loads(line)
            task_id = task["task_id"]
            prompt = task["prompt"]
            solution = task["canonical_solution"]
            test_code = task["test"]
            entry_point = task["entry_point"]
            
            full_code = f"{prompt}\n{solution}\n{test_code}\ncheck({entry_point})\n"
            scope = {}
            try:
                exec(full_code, scope)
            except Exception as e:
                pytest.fail(f"Task {task_id} failed canonical test: {e}")
