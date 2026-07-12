"""Deterministic verifier for function-style Python code generation tasks.

The local subprocess backend is intended for controlled experiments.  It uses
AST policy checks, a temporary working directory, an isolated Python process,
and a hard timeout, but it is not a production security boundary.  Run code
from untrusted users inside a dedicated container or micro-VM.
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class VerificationStatus(str, Enum):
    PASSED = "passed"
    NO_CODE = "no_code"
    SYNTAX_ERROR = "syntax_error"
    POLICY_VIOLATION = "policy_violation"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    WRONG_ANSWER = "wrong_answer"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True)
class TestCase:
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    expected: Any = None

    def to_json(self) -> dict[str, Any]:
        return {"args": list(self.args), "kwargs": self.kwargs, "expected": self.expected}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TestCase":
        return cls(
            args=tuple(payload.get("args", ())),
            kwargs=dict(payload.get("kwargs", {})),
            expected=payload.get("expected"),
        )


@dataclass(frozen=True)
class CodeTask:
    task_id: str
    prompt: str
    entry_point: str
    tests: tuple[TestCase, ...]

    def __post_init__(self) -> None:
        if not self.entry_point.isidentifier():
            raise ValueError(f"entry_point must be a Python identifier: {self.entry_point!r}")
        if not self.tests:
            raise ValueError("a code task must contain at least one test")
        # Fail early instead of discovering non-serializable tests in a worker.
        json.dumps([case.to_json() for case in self.tests], ensure_ascii=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "entry_point": self.entry_point,
            "tests": [case.to_json() for case in self.tests],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CodeTask":
        return cls(
            task_id=str(payload["task_id"]),
            prompt=str(payload["prompt"]),
            entry_point=str(payload["entry_point"]),
            tests=tuple(TestCase.from_dict(case) for case in payload["tests"]),
        )


@dataclass(frozen=True)
class VerificationResult:
    status: VerificationStatus
    passed_tests: int
    total_tests: int
    duration_ms: float
    feedback: str
    stdout: str = ""
    stderr: str = ""

    @property
    def pass_rate(self) -> float:
        return self.passed_tests / self.total_tests if self.total_tests else 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["pass_rate"] = self.pass_rate
        return payload


_FENCED_CODE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)

_ALLOWED_IMPORT_ROOTS = {
    "bisect",
    "collections",
    "functools",
    "heapq",
    "itertools",
    "math",
    "operator",
    "re",
    "string",
    "typing",
}
_BLOCKED_CALLS = {"breakpoint", "compile", "eval", "exec", "input", "open", "__import__"}
_BLOCKED_NAMES = {"os", "pathlib", "requests", "shutil", "socket", "subprocess", "sys"}


def extract_python_code(response: str) -> str:
    """Extract the first fenced Python block, falling back to plain text."""
    cleaned = _THINK_BLOCK.sub("", response or "").strip()
    matches = _FENCED_CODE.findall(cleaned)
    if matches:
        return textwrap.dedent(matches[0]).strip()
    return textwrap.dedent(cleaned).strip()


def _policy_errors(tree: ast.AST) -> list[str]:
    errors: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in _ALLOWED_IMPORT_ROOTS:
                    errors.append(f"import {alias.name!r} is not allowed")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root not in _ALLOWED_IMPORT_ROOTS:
                errors.append(f"import from {node.module!r} is not allowed")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _BLOCKED_CALLS:
                errors.append(f"call to {node.func.id!r} is not allowed")
        elif isinstance(node, ast.Name) and node.id in _BLOCKED_NAMES:
            errors.append(f"name {node.id!r} is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            errors.append(f"dunder attribute {node.attr!r} is not allowed")
    return sorted(set(errors))


def _limit_resources(timeout_seconds: float, memory_mb: int):
    if os.name == "nt":
        return None

    def apply_limits() -> None:
        import resource

        cpu_seconds = max(1, math.ceil(timeout_seconds))
        memory_bytes = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))

    return apply_limits


_RUNNER_SOURCE = r'''
import importlib.util
import json
import math
import traceback


def equivalent(actual, expected):
    if isinstance(actual, float) or isinstance(expected, float):
        try:
            return math.isclose(float(actual), float(expected), rel_tol=1e-7, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    if isinstance(actual, tuple):
        actual = list(actual)
    if isinstance(expected, tuple):
        expected = list(expected)
    return actual == expected


def main():
    spec = importlib.util.spec_from_file_location("candidate_solution", "solution.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with open("cases.json", "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    function = getattr(module, payload["entry_point"])
    passed = 0
    failure = None
    for index, case in enumerate(payload["tests"]):
        actual = function(*case["args"], **case["kwargs"])
        if equivalent(actual, case["expected"]):
            passed += 1
        elif failure is None:
            failure = {
                "index": index,
                "expected": repr(case["expected"]),
                "actual": repr(actual),
            }
    print("__MINIMIND_RESULT__=" + json.dumps({"passed": passed, "failure": failure}))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("__MINIMIND_EXCEPTION__=" + traceback.format_exc())
        raise
'''


def _clip_output(value: str, limit: int = 4000) -> str:
    return value if len(value) <= limit else value[-limit:]


def verify_code(
    response_or_code: str,
    task: CodeTask,
    *,
    timeout_seconds: float = 2.0,
    memory_mb: int = 256,
) -> VerificationResult:
    """Run a candidate function against deterministic JSON-serializable tests."""
    started = time.perf_counter()
    code = extract_python_code(response_or_code)
    if not code:
        return VerificationResult(
            VerificationStatus.NO_CODE, 0, len(task.tests), 0.0, "No Python code was found."
        )

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        duration = (time.perf_counter() - started) * 1000
        return VerificationResult(
            VerificationStatus.SYNTAX_ERROR,
            0,
            len(task.tests),
            duration,
            f"SyntaxError on line {exc.lineno}: {exc.msg}",
        )

    errors = _policy_errors(tree)
    if errors:
        duration = (time.perf_counter() - started) * 1000
        return VerificationResult(
            VerificationStatus.POLICY_VIOLATION,
            0,
            len(task.tests),
            duration,
            "; ".join(errors),
        )

    payload = {"entry_point": task.entry_point, "tests": [case.to_json() for case in task.tests]}
    env = {"PYTHONHASHSEED": "0", "PYTHONIOENCODING": "utf-8"}
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

    try:
        with tempfile.TemporaryDirectory(prefix="minimind-code-") as temp_dir:
            root = Path(temp_dir)
            (root / "solution.py").write_text(code, encoding="utf-8")
            (root / "cases.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            (root / "runner.py").write_text(_RUNNER_SOURCE, encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, "-I", "-S", "runner.py"],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                check=False,
                creationflags=creationflags,
                preexec_fn=_limit_resources(timeout_seconds, memory_mb),
            )
    except subprocess.TimeoutExpired as exc:
        duration = (time.perf_counter() - started) * 1000
        return VerificationResult(
            VerificationStatus.TIMEOUT,
            0,
            len(task.tests),
            duration,
            f"Execution exceeded the {timeout_seconds:.2f}s timeout.",
            _clip_output(exc.stdout or ""),
            _clip_output(exc.stderr or ""),
        )
    except Exception as exc:
        duration = (time.perf_counter() - started) * 1000
        return VerificationResult(
            VerificationStatus.INTERNAL_ERROR,
            0,
            len(task.tests),
            duration,
            f"Verifier failure: {type(exc).__name__}: {exc}",
        )

    duration = (time.perf_counter() - started) * 1000
    stdout, stderr = _clip_output(completed.stdout), _clip_output(completed.stderr)
    marker = "__MINIMIND_RESULT__="
    result_line = next((line for line in reversed(stdout.splitlines()) if line.startswith(marker)), None)
    if completed.returncode != 0 or result_line is None:
        exception_line = next(
            (line for line in reversed(stdout.splitlines()) if line.startswith("__MINIMIND_EXCEPTION__=")),
            None,
        )
        feedback = exception_line.split("=", 1)[1] if exception_line else (stderr.strip() or "Candidate raised an exception.")
        return VerificationResult(
            VerificationStatus.RUNTIME_ERROR, 0, len(task.tests), duration, feedback, stdout, stderr
        )

    try:
        worker_result = json.loads(result_line[len(marker):])
        passed = int(worker_result["passed"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return VerificationResult(
            VerificationStatus.INTERNAL_ERROR,
            0,
            len(task.tests),
            duration,
            f"Invalid verifier output: {exc}",
            stdout,
            stderr,
        )

    failure = worker_result.get("failure")
    if passed == len(task.tests):
        status = VerificationStatus.PASSED
        feedback = f"Passed all {passed}/{len(task.tests)} tests."
    else:
        status = VerificationStatus.WRONG_ANSWER
        feedback = (
            f"Passed {passed}/{len(task.tests)} tests. First failure at test {failure['index']}: "
            f"expected {failure['expected']}, got {failure['actual']}."
        )
    return VerificationResult(status, passed, len(task.tests), duration, feedback, stdout, stderr)
