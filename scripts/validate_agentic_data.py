import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agentic.data_analysis_env import score_agentic_trajectory
from dataset.agentic_dataset import load_agentic_jsonl


REQUIRED_FIELDS = ["id", "question", "tables", "documents", "tools", "expected_tools", "answer", "checks"]


def trajectory_to_turn_outputs(sample):
    turns = []
    for item in sample.get("expert_trajectory", []):
        if item.get("role") == "assistant" and item.get("tool_call"):
            turns.append("<tool_call>\n" + json.dumps(item["tool_call"], ensure_ascii=False) + "\n</tool_call>")
        elif item.get("role") == "assistant":
            turns.append(str(item.get("content", "")))
    return turns


def validate_sample(sample):
    errors = []
    for field in REQUIRED_FIELDS:
        if field not in sample:
            errors.append(f"missing field: {field}")
    for table in sample.get("tables", []):
        path = table.get("path", "")
        if not path:
            errors.append("table path is empty")
        elif not os.path.exists(path):
            errors.append(f"table path not found: {path}")
    if not sample.get("expert_trajectory"):
        errors.append("missing expert_trajectory")
    if not errors:
        reward, parts = score_agentic_trajectory(trajectory_to_turn_outputs(sample), sample)
        if parts.get("task_success", 0) < 1:
            errors.append(f"expert trajectory not successful: reward={reward:.3f}, parts={parts}")
    return errors


def parse_args():
    parser = argparse.ArgumentParser(description="Validate Agentic DataAnalysis JSONL")
    parser.add_argument("--data", required=True)
    parser.add_argument("--max_errors", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_agentic_jsonl(args.data)
    total_errors = 0
    for row in rows:
        errors = validate_sample(row)
        if errors:
            total_errors += len(errors)
            if total_errors <= args.max_errors:
                print(json.dumps({"id": row.get("id"), "errors": errors}, ensure_ascii=False))
    summary = {"data": args.data, "samples": len(rows), "errors": total_errors}
    print(json.dumps(summary, ensure_ascii=False))
    if total_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
