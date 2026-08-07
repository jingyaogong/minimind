"""A minimal generate-execute-feedback-repair loop for algorithmic code tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

from .reward import CodeReward, score_verification
from .verifier import (
    CodeTask,
    VerificationResult,
    VerificationStatus,
    extract_python_code,
    verify_code,
)


@dataclass(frozen=True)
class AttemptRecord:
    attempt: int
    response: str
    code: str
    verification: VerificationResult
    reward: CodeReward
    feedback_to_model: str

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["verification"] = self.verification.to_dict()
        payload["reward"] = asdict(self.reward)
        return payload


@dataclass(frozen=True)
class AgentResult:
    task_id: str
    success: bool
    attempts: tuple[AttemptRecord, ...]

    @property
    def final(self) -> AttemptRecord:
        return self.attempts[-1]

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "attempt_count": len(self.attempts),
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }


def _feedback_for_model(result: VerificationResult, reveal_test_details: bool) -> str:
    if reveal_test_details:
        return result.feedback
    if result.status == VerificationStatus.WRONG_ANSWER:
        return (
            f"The solution passed {result.passed_tests}/{result.total_tests} tests. "
            "At least one hidden test failed. Re-check edge cases and algorithmic assumptions."
        )
    if result.status == VerificationStatus.RUNTIME_ERROR:
        last_line = next((line.strip() for line in reversed(result.stderr.splitlines()) if line.strip()), "")
        return f"The program raised a runtime error{': ' + last_line if last_line else ''}."
    return result.feedback


class ExecutionFeedbackAgent:
    """Iteratively ask a generator to repair code using verifier feedback.

    ``generate`` receives a plain prompt and returns a model response.  This
    keeps the control loop independent from a specific inference backend.
    """

    def __init__(
        self,
        generate: Callable[[str], str],
        *,
        max_attempts: int = 3,
        timeout_seconds: float = 2.0,
        memory_mb: int = 256,
        reveal_test_details: bool = False,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.generate = generate
        self.max_attempts = max_attempts
        self.timeout_seconds = timeout_seconds
        self.memory_mb = memory_mb
        self.reveal_test_details = reveal_test_details

    @staticmethod
    def _initial_prompt(task: CodeTask) -> str:
        return (
            "You are solving a Python algorithm task. Return exactly one ```python``` code block.\n"
            f"Required entry point: {task.entry_point}\n"
            f"Task:\n{task.prompt}\n"
            "Do not read files, access the network, spawn processes, or call unsafe builtins."
        )

    @staticmethod
    def _repair_prompt(task: CodeTask, previous_code: str, feedback: str, attempt: int) -> str:
        return (
            "Repair the previous Python solution using the execution feedback. "
            "Return the complete replacement in exactly one ```python``` code block.\n"
            f"Required entry point: {task.entry_point}\n"
            f"Task:\n{task.prompt}\n"
            f"Previous solution (attempt {attempt - 1}):\n```python\n{previous_code}\n```\n"
            f"Execution feedback:\n{feedback}"
        )

    def run(self, task: CodeTask) -> AgentResult:
        attempts: list[AttemptRecord] = []
        prompt = self._initial_prompt(task)
        previous_code = ""
        for attempt_number in range(1, self.max_attempts + 1):
            response = self.generate(prompt)
            code = extract_python_code(response)
            verification = verify_code(
                code,
                task,
                timeout_seconds=self.timeout_seconds,
                memory_mb=self.memory_mb,
            )
            reward = score_verification(verification)
            feedback = _feedback_for_model(verification, self.reveal_test_details)
            attempts.append(
                AttemptRecord(
                    attempt=attempt_number,
                    response=response,
                    code=code,
                    verification=verification,
                    reward=reward,
                    feedback_to_model=feedback,
                )
            )
            if verification.status == VerificationStatus.PASSED:
                break
            previous_code = code
            prompt = self._repair_prompt(task, previous_code, feedback, attempt_number + 1)
        return AgentResult(
            task_id=task.task_id,
            success=attempts[-1].verification.status == VerificationStatus.PASSED,
            attempts=tuple(attempts),
        )
