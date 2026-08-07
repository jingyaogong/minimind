import json

from code_agent import CodeRLDataset, collate_code_rl


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"] + f"\nthinking={kwargs['open_thinking']}"


def test_code_rl_dataset_builds_prompt_and_preserves_tests(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        json.dumps(
            {
                "task_id": "inc",
                "prompt": "Implement inc(x).",
                "entry_point": "inc",
                "tests": [{"args": [1], "expected": 2}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = CodeRLDataset(path, FakeTokenizer(), thinking_ratio=0.0)
    sample = dataset[0]

    assert "Required entry point: inc" in sample["prompt"]
    assert sample["task"].tests[0].expected == 2
    batch = collate_code_rl([sample])
    assert batch["tasks"][0].task_id == "inc"
