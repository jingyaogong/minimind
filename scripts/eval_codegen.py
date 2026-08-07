"""Evaluate JSONL code-generation candidates with deterministic tests."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from code_agent import CodeTask, aggregate_reports, evaluate_candidates


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc


def main():
    parser = argparse.ArgumentParser(description="Execution-based evaluation for code generation")
    parser.add_argument("--tasks", required=True, help="JSONL file containing CodeTask records")
    parser.add_argument("--predictions", required=True, help="JSONL records with task_id and candidates")
    parser.add_argument("--output", help="Optional JSON report path")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10], help="pass@k values")
    parser.add_argument("--timeout", type=float, default=2.0, help="Timeout per candidate in seconds")
    parser.add_argument("--memory-mb", type=int, default=256, help="POSIX worker memory limit")
    args = parser.parse_args()

    tasks = {record["task_id"]: CodeTask.from_dict(record) for record in read_jsonl(args.tasks)}
    predictions = {record["task_id"]: record["candidates"] for record in read_jsonl(args.predictions)}
    missing = sorted(set(tasks) - set(predictions))
    unknown = sorted(set(predictions) - set(tasks))
    if missing or unknown:
        raise ValueError(f"Task/prediction mismatch: missing={missing}, unknown={unknown}")

    reports = [
        evaluate_candidates(
            task,
            predictions[task_id],
            ks=args.ks,
            timeout_seconds=args.timeout,
            memory_mb=args.memory_mb,
        )
        for task_id, task in tasks.items()
    ]
    output = {
        "summary": aggregate_reports(reports),
        "tasks": [report.to_dict() for report in reports],
        "safety_note": (
            "The local verifier is for controlled experiments, not hostile code. "
            "Use a dedicated container or micro-VM for untrusted submissions."
        ),
    }
    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
