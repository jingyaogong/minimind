"""Convert compatible MBPP-sanitized assertions into MiniMind CodeTask splits."""

import argparse
import ast
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from code_agent import CodeTask, VerificationStatus, verify_code


def _literal(node):
    value = ast.literal_eval(node)
    json.dumps(value, ensure_ascii=False)
    return value


def parse_assertion(source):
    statement = ast.parse(source.strip()).body
    if len(statement) != 1 or not isinstance(statement[0], ast.Assert):
        raise ValueError("test is not a single assert statement")
    expression = statement[0].test

    if isinstance(expression, ast.Call):
        call, expected = expression, True
    elif isinstance(expression, ast.UnaryOp) and isinstance(expression.op, ast.Not) and isinstance(expression.operand, ast.Call):
        call, expected = expression.operand, False
    elif (
        isinstance(expression, ast.Compare)
        and len(expression.ops) == 1
        and isinstance(expression.ops[0], ast.Eq)
        and len(expression.comparators) == 1
    ):
        left, right = expression.left, expression.comparators[0]
        if isinstance(left, ast.Call):
            call, expected = left, _literal(right)
        elif isinstance(right, ast.Call):
            call, expected = right, _literal(left)
        else:
            raise ValueError("equality does not compare a function call")
    else:
        raise ValueError("unsupported assertion shape")

    if not isinstance(call.func, ast.Name):
        raise ValueError("entry point is not a direct function call")
    args = [_literal(argument) for argument in call.args]
    kwargs = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            raise ValueError("expanded keyword arguments are unsupported")
        kwargs[keyword.arg] = _literal(keyword.value)
    return call.func.id, {"args": args, "kwargs": kwargs, "expected": expected}


def convert_record(record):
    parsed = [parse_assertion(test) for test in record["test_list"]]
    entry_points = {entry_point for entry_point, _ in parsed}
    if len(entry_points) != 1:
        raise ValueError("tests use multiple entry points")
    payload = {
        "task_id": f"mbpp-{record['task_id']}",
        "prompt": record["prompt"].strip(),
        "entry_point": entry_points.pop(),
        "tests": [test for _, test in parsed],
    }
    task = CodeTask.from_dict(payload)
    verification = verify_code(record["code"], task)
    if verification.status != VerificationStatus.PASSED:
        raise ValueError(f"reference solution failed verifier: {verification.status.value}")
    return task.to_dict()


def build_sft_record(record, task):
    return {
        "conversations": [
            {
                "role": "user",
                "content": (
                    "Solve the Python algorithm task below. Return one complete ```python``` code block.\n"
                    f"Required entry point: {task['entry_point']}\n"
                    f"Task:\n{task['prompt']}"
                ),
            },
            {"role": "assistant", "content": f"```python\n{record['code'].strip()}\n```"},
        ]
    }


def convert_split(records):
    converted = []
    sft_records = []
    skipped = Counter()
    for record in records:
        try:
            task = convert_record(record)
            converted.append(task)
            sft_records.append(build_sft_record(record, task))
        except (KeyError, SyntaxError, TypeError, ValueError) as exc:
            skipped[str(exc)] += 1
    return converted, sft_records, skipped


def main():
    parser = argparse.ArgumentParser(description="Prepare verifier-compatible MBPP sanitized splits")
    parser.add_argument("--dataset", default="google-research-datasets/mbpp")
    parser.add_argument("--config", default="sanitized")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--output-dir", default="out/mbpp_sanitized")
    args = parser.parse_args()

    from datasets import load_dataset

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in args.splits:
        records = load_dataset(args.dataset, args.config, split=split)
        converted, sft_records, skipped = convert_split(records)
        output_path = output_dir / f"{split}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for record in converted:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        sft_output = None
        if split == "train":
            sft_output = output_dir / "train_sft.jsonl"
            with sft_output.open("w", encoding="utf-8") as handle:
                for record in sft_records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        summary[split] = {
            "source_count": len(records),
            "converted_count": len(converted),
            "skipped_count": len(records) - len(converted),
            "top_skip_reasons": skipped.most_common(10),
            "output": str(output_path),
            "sft_output": str(sft_output) if sft_output else None,
        }
        print(f"{split}: converted {len(converted)}/{len(records)} -> {output_path}")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
