import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agentic.data_analysis_env import AGENTIC_SYSTEM_PROMPT, format_agentic_user_prompt, get_agentic_tools
from dataset.agentic_dataset import load_agentic_jsonl


def assistant_tool_message(tool_call):
    return {
        "role": "assistant",
        "content": "",
        "reasoning_content": "",
        "tools": "",
        "tool_calls": json.dumps(
            [
                {
                    "type": "function",
                    "function": {
                        "name": tool_call.get("name", ""),
                        "arguments": tool_call.get("arguments", {}),
                    },
                }
            ],
            ensure_ascii=False,
        ),
    }


def normal_message(role, content):
    return {
        "role": role,
        "content": content,
        "reasoning_content": "",
        "tools": "",
        "tool_calls": "",
    }


def convert_sample(sample):
    tools = get_agentic_tools(sample.get("tools"))
    conversations = [
        {
            "role": "system",
            "content": AGENTIC_SYSTEM_PROMPT,
            "reasoning_content": "",
            "tools": json.dumps(tools, ensure_ascii=False),
            "tool_calls": "",
        },
        normal_message("user", format_agentic_user_prompt(sample)),
    ]
    for item in sample.get("expert_trajectory", []):
        if item.get("role") == "assistant" and item.get("tool_call"):
            conversations.append(assistant_tool_message(item["tool_call"]))
        elif item.get("role") == "tool":
            conversations.append(normal_message("tool", json.dumps(item.get("content", {}), ensure_ascii=False)))
        elif item.get("role") == "assistant":
            conversations.append(normal_message("assistant", str(item.get("content", ""))))
    return {"id": sample.get("id", ""), "conversations": conversations}


def parse_args():
    parser = argparse.ArgumentParser(description="Build Agentic DataAnalysis SFT conversations")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_agentic_jsonl(args.input)
    if args.limit > 0:
        rows = rows[:args.limit]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for sample in rows:
            f.write(json.dumps(convert_sample(sample), ensure_ascii=False) + "\n")
    print(json.dumps({"input": args.input, "output": args.output, "samples": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
