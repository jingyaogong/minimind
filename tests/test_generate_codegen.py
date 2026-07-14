from types import SimpleNamespace

import pytest

from scripts.generate_codegen import build_prompt, generate_predictions


def test_generate_predictions_groups_samples_by_task():
    task = SimpleNamespace(task_id="add", entry_point="add", prompt="Implement add(a, b).")
    prompts = []

    def generate(prompt):
        prompts.append(prompt)
        return f"candidate-{len(prompts)}"

    records = generate_predictions([task], generate, samples_per_task=3)

    assert records == [{"task_id": "add", "candidates": ["candidate-1", "candidate-2", "candidate-3"]}]
    assert all("Required entry point: add" in prompt for prompt in prompts)
    assert "exactly one ```python``` code block" in build_prompt(task)


def test_generate_predictions_rejects_empty_sample_count():
    with pytest.raises(ValueError, match="at least 1"):
        generate_predictions([], lambda prompt: prompt, samples_per_task=0)
