"""Verifiable rewards derived from deterministic code execution."""

from __future__ import annotations

from dataclasses import dataclass

from .verifier import VerificationResult, VerificationStatus


@dataclass(frozen=True)
class CodeReward:
    total: float
    correctness: float
    execution: float
    format: float
    reason: str


def score_verification(result: VerificationResult) -> CodeReward:
    """Map verifier outcomes to a transparent reward in [-1, 1]."""
    status = result.status
    if status == VerificationStatus.PASSED:
        return CodeReward(1.0, 0.8, 0.1, 0.1, "all tests passed")
    if status == VerificationStatus.WRONG_ANSWER:
        correctness = 0.8 * result.pass_rate
        total = -0.2 + correctness + 0.1 + 0.1
        return CodeReward(total, correctness, 0.1, 0.1, "partial test credit")
    if status == VerificationStatus.RUNTIME_ERROR:
        return CodeReward(-0.4, 0.0, -0.5, 0.1, "runtime error")
    if status == VerificationStatus.TIMEOUT:
        return CodeReward(-0.6, 0.0, -0.7, 0.1, "execution timeout")
    if status == VerificationStatus.SYNTAX_ERROR:
        return CodeReward(-0.8, 0.0, -0.8, 0.0, "syntax error")
    if status in {VerificationStatus.NO_CODE, VerificationStatus.POLICY_VIOLATION}:
        return CodeReward(-1.0, 0.0, 0.0, -1.0, status.value)
    return CodeReward(-1.0, 0.0, -1.0, 0.0, "verifier failure")
