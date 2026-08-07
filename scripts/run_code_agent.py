"""Run the execution-feedback repair agent against an OpenAI-compatible model."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from code_agent import CodeTask, ExecutionFeedbackAgent, OpenAICompatibleGenerator


TASK_SEED_STRIDE = 1000


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


def read_existing_results(path):
    output_path = Path(path)
    if not output_path.exists():
        return []
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"Invalid agent output at {path}: missing results list")
    seen = set()
    for result in results:
        task_id = str(result.get("task_id", ""))
        if not task_id or task_id in seen:
            raise ValueError(f"Invalid agent output at {path}: duplicate or missing task_id")
        seen.add(task_id)
    return results


def build_output(results):
    return {
        "task_count": len(results),
        "success_count": sum(result["success"] for result in results),
        "results": results,
    }


def write_output(path, output):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


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
    parser.add_argument("--seed", type=int, default=42, help="Base seed for reproducible task-level generation")
    parser.add_argument("--resume", action="store_true", help="Resume completed tasks from --output")
    parser.add_argument("--open-thinking", action="store_true")
    parser.add_argument("--reveal-test-details", action="store_true")
    args = parser.parse_args()

    indexed_tasks = list(enumerate(read_tasks(args.tasks)))
    if args.task_id:
        indexed_tasks = [(index, task) for index, task in indexed_tasks if task.task_id == args.task_id]
        if not indexed_tasks:
            raise ValueError(f"Unknown task id: {args.task_id}")

    if args.resume and not args.output:
        parser.error("--resume requires --output")
    expected_task_ids = {task.task_id for _, task in indexed_tasks}
    existing_results = read_existing_results(args.output) if args.resume else []
    unexpected_task_ids = {result["task_id"] for result in existing_results} - expected_task_ids
    if unexpected_task_ids:
        raise ValueError(
            f"Existing output contains tasks not selected for this run: {sorted(unexpected_task_ids)}"
        )
    results_by_id = {result["task_id"]: result for result in existing_results}

    for position, (task_index, task) in enumerate(indexed_tasks, start=1):
        if task.task_id in results_by_id:
            print(f"[{position}/{len(indexed_tasks)}] {task.task_id}: already complete", flush=True)
            continue
        generator = OpenAICompatibleGenerator(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            open_thinking=args.open_thinking,
            seed=args.seed + task_index * TASK_SEED_STRIDE,
        )
        agent = ExecutionFeedbackAgent(
            generator,
            max_attempts=args.max_attempts,
            timeout_seconds=args.timeout,
            memory_mb=args.memory_mb,
            reveal_test_details=args.reveal_test_details,
        )
        result = agent.run(task).to_dict()
        results_by_id[task.task_id] = result
        ordered_results = [
            results_by_id[selected_task.task_id]
            for _, selected_task in indexed_tasks
            if selected_task.task_id in results_by_id
        ]
        if args.output:
            write_output(args.output, build_output(ordered_results))
        print(
            f"[{position}/{len(indexed_tasks)}] {task.task_id}: "
            f"success={result['success']} attempts={result['attempt_count']}",
            flush=True,
        )

    results = [results_by_id[task.task_id] for _, task in indexed_tasks]
    output = build_output(results)
    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        write_output(args.output, output)
    print(rendered)


if __name__ == "__main__":
    main()
