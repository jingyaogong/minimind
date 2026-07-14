"""Generate reproducible multi-sample candidates for execution-based evaluation."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from code_agent import CodeTask, OpenAICompatibleGenerator


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


def build_prompt(task):
    return (
        "You are solving a Python algorithm task. Return exactly one ```python``` code block.\n"
        f"Required entry point: {task.entry_point}\n"
        f"Task:\n{task.prompt}\n"
        "Do not read files, access the network, spawn processes, or call unsafe builtins."
    )


def generate_predictions(tasks, generator, samples_per_task):
    if samples_per_task < 1:
        raise ValueError("samples_per_task must be at least 1")
    return [
        {
            "task_id": task.task_id,
            "candidates": [generator(build_prompt(task)) for _ in range(samples_per_task)],
        }
        for task in tasks
    ]


def main():
    parser = argparse.ArgumentParser(description="Generate code candidates through an OpenAI-compatible API")
    parser.add_argument("--tasks", required=True, help="CodeTask JSONL file")
    parser.add_argument("--output", required=True, help="Prediction JSONL output path")
    parser.add_argument("--samples-per-task", type=int, default=10)
    parser.add_argument("--base-url", default="http://localhost:8998/v1")
    parser.add_argument("--api-key", default="not-needed")
    parser.add_argument("--model", default="minimind")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--open-thinking", action="store_true")
    args = parser.parse_args()

    tasks = read_tasks(args.tasks)
    generator = OpenAICompatibleGenerator(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        open_thinking=args.open_thinking,
        seed=args.seed,
    )
    records = generate_predictions(tasks, generator, args.samples_per_task)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Generated {len(tasks) * args.samples_per_task} candidates for {len(tasks)} tasks -> {output_path}")


if __name__ == "__main__":
    main()
