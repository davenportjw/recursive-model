import pytest
from tiny_recursive_gemma.inference import extract_thought_and_code

def test_extract_xml_clean():
    text = (
        "<thought>\nLet's write a simple addition function.\n</thought>\n"
        "<code_update>\ndef add(a, b):\n    return a + b\n</code_update>"
    )
    thought, code = extract_thought_and_code(text)
    assert thought == "Let's write a simple addition function."
    assert code == "def add(a, b):\n    return a + b"

def test_extract_xml_with_nested_markdown():
    text = (
        "<thought>\nOptimizing sort.\n</thought>\n"
        "<code_update>\n```python\ndef quicksort(arr):\n    return sorted(arr)\n```\n</code_update>"
    )
    thought, code = extract_thought_and_code(text)
    assert thought == "Optimizing sort."
    assert code == "def quicksort(arr):\n    return sorted(arr)"

def test_extract_markdown_fallback():
    text = (
        "Here is the plan: we will count elements.\n\n"
        "```python\ndef count_elements(lst):\n    return len(lst)\n```"
    )
    thought, code = extract_thought_and_code(text)
    assert "Here is the plan" in thought
    assert code == "def count_elements(lst):\n    return len(lst)"

def test_extract_unclosed_tags():
    text = (
        "<thought>\nPartial thinking here...\n"
        "<code_update>\ndef incomplete(x):\n    return x * 2"
    )
    thought, code = extract_thought_and_code(text)
    assert "Partial thinking" in thought
    assert "def incomplete(x):" in code

def test_extract_raw_python_heuristic():
    text = (
        "I am writing code directly:\n"
        "def compute(n):\n"
        "    return n ** 2\n"
    )
    thought, code = extract_thought_and_code(text)
    assert "I am writing code directly:" in thought
    assert "def compute(n):" in code

def test_extract_empty():
    thought, code = extract_thought_and_code("")
    assert thought == ""
    assert code == ""
