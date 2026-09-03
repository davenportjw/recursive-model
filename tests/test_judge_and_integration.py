import json
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.abspath("."))

from eval.judge_evaluator import (
    execute_unit_test,
    mock_judge_evaluate,
    run_benchmark_comparison,
    JUDGE_RUBRIC_PROMPT
)

def test_judge_rubric_prompt_formatting():
    """Verify that the judge prompt formats without KeyError or invalid placeholders."""
    formatted = JUDGE_RUBRIC_PROMPT.format(
        task_id="Task_123",
        task_prompt="def add(a, b): return a + b",
        test_suite="assert add(1, 2) == 3",
        candidates_formatted="Candidate output"
    )
    assert "Task_123" in formatted
    assert "def add(a, b)" in formatted
    assert "Candidate output" in formatted
    assert "samsung_paper_alignment" in formatted

def test_mock_judge_schema_conformance():
    """Verify that the evaluation results match the strict JSON rubric schema."""
    candidates = {
        "baseline": "def foo(): return 1",
        "discrete": "<thought>thinking</thought><code_update>def foo(): return 1</code_update>",
        "continuous": "def foo(): return 1"
    }
    test_results = {"baseline": True, "discrete": True, "continuous": True}
    res = mock_judge_evaluate("Test/0", candidates, test_results)
    
    assert res["task_id"] == "Test/0"
    assert "evaluations" in res
    for model_key in ["baseline", "discrete", "continuous"]:
        eval_dict = res["evaluations"][model_key]
        for field in [
            "functional_correctness",
            "algorithmic_soundness",
            "recursive_progression",
            "token_efficiency",
            "hallucination_resistance",
            "total_score",
            "critique"
        ]:
            assert field in eval_dict
            
    assert "samsung_paper_alignment" in res
    assert "did_continuous_match_discrete" in res["samsung_paper_alignment"]
    assert "token_saving_factor" in res["samsung_paper_alignment"]

def test_unit_test_executor_success_and_failure():
    """Verify that execute_unit_test executes python code safely and captures test results."""
    # 1. Success case
    code_pass = "def add(a, b):\n    return a + b"
    test_code = "def check(candidate):\n    assert candidate(2, 3) == 5"
    passed, _ = execute_unit_test(code_pass, test_code, "add")
    assert passed is True

    # 2. Failure case (AssertionError)
    code_fail = "def add(a, b):\n    return a - b"
    passed_fail, _ = execute_unit_test(code_fail, test_code, "add")
    assert passed_fail is False

    # 3. Syntax error case
    code_syntax_err = "def broken(a:\n    return a"
    passed_syntax, _ = execute_unit_test(code_syntax_err, test_code, "broken")
    assert passed_syntax is False

def test_benchmark_comparison_end_to_end():
    """Verify that run_benchmark_comparison executes end-to-end and writes a valid report."""
    with tempfile.TemporaryDirectory() as tmpdir:
        report_file = os.path.join(tmpdir, "test_report.json")
        summary = run_benchmark_comparison(
            dataset_path="eval/complex_tasks.jsonl",
            output_report=report_file,
            mock_run=True,
            max_samples=2
        )
        
        assert os.path.exists(report_file)
        with open(report_file, "r") as f:
            data = json.load(f)
            
        assert data["total_tasks_evaluated"] == 2
        assert "pass_rates" in data
        assert "average_judge_scores" in data
        assert "samsung_trm_extrapolation" in data
        assert len(data["detailed_results"]) == 2
