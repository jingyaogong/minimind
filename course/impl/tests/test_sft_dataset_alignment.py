"""Alignment test for lesson 16 CourseSFTDataset.

Run after implementing:

    python course/impl/tests/test_sft_dataset_alignment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from course.impl.core.datasets import CourseSFTDataset  # noqa: E402
from dataset.lm_dataset import SFTDataset as SourceSFTDataset  # noqa: E402


def test_sft_dataset_alignment() -> None:
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model")
    data_path = ROOT / "course/labs/tiny_sft.jsonl"
    max_length = 96

    source = SourceSFTDataset(str(data_path), tokenizer, max_length=max_length)
    target = CourseSFTDataset(str(data_path), tokenizer, max_length=max_length)

    assert len(target) == len(source)

    for index in range(len(source)):
        source_input_ids, source_labels = source[index]
        target_input_ids, target_labels = target[index]
        assert isinstance(target_input_ids, torch.Tensor)
        assert isinstance(target_labels, torch.Tensor)

        input_diff = (source_input_ids - target_input_ids).abs().max().item()
        label_diff = (source_labels - target_labels).abs().max().item()
        print(f"sft_sample_{index}_input_diff={input_diff}")
        print(f"sft_sample_{index}_label_diff={label_diff}")
        assert input_diff == 0
        assert label_diff == 0

        non_ignored = int((target_labels != -100).sum().item())
        print(f"sft_sample_{index}_non_ignored_labels={non_ignored}")
        assert non_ignored > 0


def main() -> None:
    try:
        test_sft_dataset_alignment()
    except NotImplementedError as exc:
        raise SystemExit(f"TODO not implemented yet: {exc}") from exc
    print("sft_dataset_alignment=passed")


if __name__ == "__main__":
    main()
