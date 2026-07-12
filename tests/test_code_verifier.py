from code_agent import (
    CodeTask,
    TestCase as CodeTestCase,
    VerificationStatus,
    extract_python_code,
    score_verification,
    verify_code,
)


TASK = CodeTask(
    task_id="add-two",
    prompt="Implement add(a, b).",
    entry_point="add",
    tests=(
        CodeTestCase(args=(1, 2), expected=3),
        CodeTestCase(args=(5, 0), expected=5),
    ),
)


def test_extracts_fenced_code_after_thinking():
    response = "<think>reasoning</think>\n```python\ndef add(a, b):\n    return a + b\n```"
    assert extract_python_code(response).startswith("def add")


def test_verifier_accepts_correct_solution():
    result = verify_code("def add(a, b):\n    return a + b", TASK)
    assert result.status == VerificationStatus.PASSED
    assert result.pass_rate == 1.0
    assert score_verification(result).total == 1.0


def test_verifier_reports_partial_credit_and_feedback():
    result = verify_code("def add(a, b):\n    return a - b", TASK)
    assert result.status == VerificationStatus.WRONG_ANSWER
    assert result.passed_tests == 1
    assert "expected 3, got -1" in result.feedback
    assert score_verification(result).total == 0.4


def test_verifier_reports_syntax_error():
    result = verify_code("def add(a, b)\n    return a + b", TASK)
    assert result.status == VerificationStatus.SYNTAX_ERROR
    assert score_verification(result).total == -0.8


def test_verifier_blocks_unsafe_import():
    result = verify_code("import os\ndef add(a, b):\n    return a + b", TASK)
    assert result.status == VerificationStatus.POLICY_VIOLATION
    assert "import 'os' is not allowed" in result.feedback


def test_verifier_times_out_infinite_loop():
    result = verify_code("def add(a, b):\n    while True:\n        pass", TASK, timeout_seconds=0.2)
    assert result.status == VerificationStatus.TIMEOUT
    assert score_verification(result).total == -0.6
