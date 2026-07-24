"""Dataset adapter for execution-verifiable GRPO code tasks."""

from __future__ import annotations

import json
import random
from pathlib import Path

from torch.utils.data import Dataset

from .verifier import CodeTask


class CodeRLDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, thinking_ratio: float = 0.5):
        if not 0.0 <= thinking_ratio <= 1.0:
            raise ValueError("thinking_ratio must be between 0 and 1")
        self.tokenizer = tokenizer
        self.thinking_ratio = thinking_ratio
        self.tasks: list[CodeTask] = []
        with Path(jsonl_path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    self.tasks.append(CodeTask.from_dict(json.loads(line)))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Invalid code task at {jsonl_path}:{line_number}: {exc}") from exc
        if not self.tasks:
            raise ValueError("code RL dataset is empty")

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, index):
        task = self.tasks[index]
        message = {
            "role": "user",
            "content": (
                "Solve the Python algorithm task below. Return one complete ```python``` code block.\n"
                f"Required entry point: {task.entry_point}\n"
                f"Task:\n{task.prompt}"
            ),
        }
        prompt = self.tokenizer.apply_chat_template(
            [message],
            tokenize=False,
            open_thinking=random.random() < self.thinking_ratio,
            add_generation_prompt=True,
        )
        return {"prompt": prompt, "task": task}


def collate_code_rl(samples):
    return {
        "prompt": [sample["prompt"] for sample in samples],
        "tasks": [sample["task"] for sample in samples],
    }
