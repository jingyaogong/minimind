"""Run the execution-feedback repair agent against an OpenAI-compatible model."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from code_agent import CodeTask, ExecutionFeedbackAgent, OpenAICompatibleGenerator


def read_tasks(path):
    tasks = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                tasks.append(CodeTask.from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid task at {path}:{line_number}: {exc}") from exc
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Generate, execute, and repair Python algorithm solutions")
    parser.add_argument("--tasks", required=True, help="CodeTask JSONL file")
    parser.add_argument("--task-id", help="Run only one task id")
    parser.add_argument("--output", help="Optional JSON output path")
    parser.add_argument("--base-url", default="http://localhost:8998/v1")
    parser.add_argument("--api-key", default="not-needed")
    parser.add_argument("--model", default="minimind")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--memory-mb", type=int, default=256)
    parser.add_argument("--open-thinking", action="store_true")
    parser.add_argument("--reveal-test-details", action="store_true")
    args = parser.parse_args()

    tasks = read_tasks(args.tasks)
    if args.task_id:
        tasks = [task for task in tasks if task.task_id == args.task_id]
        if not tasks:
            raise ValueError(f"Unknown task id: {args.task_id}")

    generator = OpenAICompatibleGenerator(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        open_thinking=args.open_thinking,
    )
    agent = ExecutionFeedbackAgent(
        generator,
        max_attempts=args.max_attempts,
        timeout_seconds=args.timeout,
        memory_mb=args.memory_mb,
        reveal_test_details=args.reveal_test_details,
    )
    results = [agent.run(task).to_dict() for task in tasks]
    output = {
        "task_count": len(results),
        "success_count": sum(result["success"] for result in results),
        "results": results,
    }
    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
