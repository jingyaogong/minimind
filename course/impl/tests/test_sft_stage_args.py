"""Smoke test for lesson 17 SFT stage-script defaults.

Run with:

    python course/impl/tests/test_sft_stage_args.py
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from course.impl.train_sft_impl import parse_args  # noqa: E402


def test_default_args() -> None:
    args = parse_args([])
    print(f"data_path={args.data_path}")
    print(f"from_weight={args.from_weight}")
    print(f"save_weight={args.save_weight}")
    print(f"learning_rate={args.learning_rate}")
    assert args.data_path == "course/labs/tiny_sft.jsonl"
    assert args.from_weight == "course_pretrain"
    assert args.save_weight == "course_sft"
    assert args.learning_rate == 1e-5
    assert args.accumulation_steps == 1


def test_override_args() -> None:
    args = parse_args(["--from_weight", "pretrain", "--save_weight", "full_sft", "--use_moe"])
    assert args.from_weight == "pretrain"
    assert args.save_weight == "full_sft"
    assert args.use_moe


def main() -> None:
    test_default_args()
    test_override_args()
    print("sft_stage_args=passed")


if __name__ == "__main__":
    main()
