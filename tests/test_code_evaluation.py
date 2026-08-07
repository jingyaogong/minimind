import pytest

from code_agent import (
    CodeTask,
    TestCase as CodeTestCase,
    VerificationStatus,
    aggregate_reports,
    estimate_pass_at_k,
    evaluate_candidates,
)


TASK = CodeTask(
    task_id="square",
    prompt="Implement square(x).",
    entry_point="square",
    tests=(
        CodeTestCase(args=(2,), expected=4),
        CodeTestCase(args=(-3,), expected=9),
    ),
)


def test_pass_at_k_estimator():
    assert estimate_pass_at_k(10, 0, 1) == 0.0
    assert estimate_pass_at_k(10, 1, 10) == 1.0
    assert estimate_pass_at_k(10, 2, 1) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        estimate_pass_at_k(2, 1, 3)


def test_evaluation_reports_execution_metrics():
    candidates = [
        "def square(x):\n    return x * x",
        "def square(x):\n    return x + x",
        "def square(x)\n    return x * x",
    ]
    report = evaluate_candidates(TASK, candidates, ks=(1, 3))

    assert report.correct_count == 1
    assert report.compile_rate == pytest.approx(2 / 3)
    assert report.execution_rate == pytest.approx(2 / 3)
    assert report.test_pass_rate == pytest.approx(0.5)
    assert report.pass_at_k["pass@1"] == pytest.approx(1 / 3)
    assert report.pass_at_k["pass@3"] == 1.0
    assert report.results[2].status == VerificationStatus.SYNTAX_ERROR

    summary = aggregate_reports([report])
    assert summary["candidate_count"] == 3
    assert summary["status_counts"] == {
        "passed": 1,
        "syntax_error": 1,
        "wrong_answer": 1,
    }
