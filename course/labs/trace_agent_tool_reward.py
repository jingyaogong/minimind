import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from trainer.train_agent import (  # noqa: E402
    TOOLS,
    calculate_rewards,
    execute_tool,
    parse_tool_calls,
    validate_gt_in_text,
)


def pick_tools(names):
    wanted = set(names)
    return [tool for tool in TOOLS if tool["function"]["name"] in wanted]


def main():
    tools = pick_tools(["calculate_math", "get_current_time"])
    prompts = [
        "<|im_start|>user\n"
        "帮我算一下 256 乘以 37 等于多少<|im_end|>\n"
        "<|im_start|>assistant\n"
    ]
    gt_batch = [["9472"]]
    completions = [
        '</think>\n\n<tool_call>{"name":"calculate_math","arguments":{"expression":"256 * 37"}}</tool_call>\n工具返回 9472，所以答案是 9472。',
        '</think>\n\n<tool_call>{"name":"calculate_math","arguments":{}}</tool_call>\n我没有拿到有效表达式。',
        '</think>\n\n<tool_call>{"name":"unknown_tool","arguments":{"expression":"256 * 37"}}</tool_call>\n答案可能是 9472。',
        '</think>\n\n<tool_call>{"name":"calculate_math","arguments":{"expression":"256 * 37"}}\n答案是 9472。',
    ]
    turn_outputs_batch = [[text] for text in completions]
    unfinished_batch = [False, False, False, False]

    rewards = calculate_rewards(
        prompts=prompts,
        completions=completions,
        gt_batch=gt_batch,
        tools_batch=[tools],
        num_gen=len(completions),
        reward_model=None,
        device="cpu",
        turn_outputs_batch=turn_outputs_batch,
        unfinished_batch=unfinished_batch,
    )

    print("[Agent tool reward trace]")
    print(f"available_tools={[tool['function']['name'] for tool in tools]}")
    print(f"gt={gt_batch[0]}")
    print()

    for idx, text in enumerate(completions):
        calls = parse_tool_calls(text)
        final_text = text.split("</tool_call>")[-1] if "</tool_call>" in text else text
        verified = validate_gt_in_text(final_text, gt_batch[0])
        executed = []
        for call in calls:
            raw_args = call.get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except Exception:
                    raw_args = {}
            executed.append(
                {
                    "name": call.get("name", ""),
                    "args": raw_args,
                    "result": execute_tool(call.get("name", ""), raw_args),
                }
            )

        print(f"case={idx}")
        print(f"parsed_calls={calls}")
        print(f"executed={executed}")
        print(f"verified_gt={sorted(verified)}")
        print(f"reward={rewards[idx].item():.4f}")
        print("-" * 80)

    grouped = rewards.view(1, len(completions))
    advantages = (rewards - grouped.mean(dim=1).repeat_interleave(len(completions))) / (
        grouped.std(dim=1, unbiased=False).repeat_interleave(len(completions)) + 1e-4
    )
    print("[GRPO-style group normalization]")
    print(f"rewards={rewards.tolist()}")
    print(f"advantages={advantages.tolist()}")
    print(f"adv_mean={advantages.mean().item():.6f}")


if __name__ == "__main__":
    main()
