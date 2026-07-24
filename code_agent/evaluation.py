"""Evaluation metrics for execution-verified code generation."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

from .verifier import CodeTask, VerificationResult, VerificationStatus, verify_code


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """Unbiased pass@k estimator used by common code-generation benchmarks."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if not 0 <= num_correct <= num_samples:
        raise ValueError("num_correct must be between 0 and num_samples")
    if not 1 <= k <= num_samples:
        raise ValueError("k must be between 1 and num_samples")
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - math.comb(num_samples - num_correct, k) / math.comb(num_samples, k)


@dataclass(frozen=True)
class EvaluationReport:
    task_id: str
    candidate_count: int
    correct_count: int
    compile_rate: float
    execution_rate: float
    test_pass_rate: float
    pass_at_k: dict[str, float]
    status_counts: dict[str, int]
    results: tuple[VerificationResult, ...]

    def to_dict(self, *, include_results: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        if include_results:
            payload["results"] = [result.to_dict() for result in self.results]
        else:
            payload.pop("results", None)
        return payload


_COMPILED_STATUSES = {
    VerificationStatus.PASSED,
    VerificationStatus.RUNTIME_ERROR,
    VerificationStatus.TIMEOUT,
    VerificationStatus.WRONG_ANSWER,
}
_EXECUTED_STATUSES = {
    VerificationStatus.PASSED,
    VerificationStatus.RUNTIME_ERROR,
    VerificationStatus.WRONG_ANSWER,
}


def evaluate_candidates(
    task: CodeTask,
    candidates: Sequence[str],
    *,
    ks: Iterable[int] = (1, 5, 10),
    timeout_seconds: float = 2.0,
    memory_mb: int = 256,
) -> EvaluationReport:
    if not candidates:
        raise ValueError("at least one candidate is required")
    results = tuple(
        verify_code(candidate, task, timeout_seconds=timeout_seconds, memory_mb=memory_mb)
        for candidate in candidates
    )
    count = len(results)
    correct = sum(result.status == VerificationStatus.PASSED for result in results)
    status_counts = Counter(result.status.value for result in results)
    total_tests = sum(result.total_tests for result in results)
    passed_tests = sum(result.passed_tests for result in results)
    requested_ks = sorted(set(int(k) for k in ks if 1 <= int(k) <= count))
    pass_at_k = {f"pass@{k}": estimate_pass_at_k(count, correct, k) for k in requested_ks}
    return EvaluationReport(
        task_id=task.task_id,
        candidate_count=count,
        correct_count=correct,
        compile_rate=sum(result.status in _COMPILED_STATUSES for result in results) / count,
        execution_rate=sum(result.status in _EXECUTED_STATUSES for result in results) / count,
        test_pass_rate=passed_tests / total_tests if total_tests else 0.0,
        pass_at_k=pass_at_k,
        status_counts=dict(sorted(status_counts.items())),
        results=results,
    )


def aggregate_reports(reports: Sequence[EvaluationReport]) -> dict[str, Any]:
    if not reports:
        raise ValueError("at least one evaluation report is required")
    status_counts: Counter[str] = Counter()
    for report in reports:
        status_counts.update(report.status_counts)
    metric_names = sorted({name for report in reports for name in report.pass_at_k})
    pass_at_k = {
        name: sum(report.pass_at_k[name] for report in reports if name in report.pass_at_k)
        / sum(name in report.pass_at_k for report in reports)
        for name in metric_names
    }
    candidate_count = sum(report.candidate_count for report in reports)
    return {
        "task_count": len(reports),
        "candidate_count": candidate_count,
        "correct_count": sum(report.correct_count for report in reports),
        "compile_rate": sum(report.compile_rate * report.candidate_count for report in reports) / candidate_count,
        "execution_rate": sum(report.execution_rate * report.candidate_count for report in reports) / candidate_count,
        "test_pass_rate": sum(report.test_pass_rate * report.candidate_count for report in reports) / candidate_count,
        "pass_at_k": pass_at_k,
        "status_counts": dict(sorted(status_counts.items())),
    }
