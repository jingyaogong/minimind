"""Alignment tests for lesson 14 training-loop mechanics.

Run after implementing:

    python course/impl/tests/test_train_loop_mechanics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from course.impl.core.train_loop import (  # noqa: E402
    cosine_lr,
    scale_loss_for_accumulation,
    should_flush_tail,
    should_step_optimizer,
)
from trainer.trainer_utils import get_lr  # noqa: E402


def test_cosine_lr() -> None:
    base_lr = 5e-4
    total_steps = 20
    for current_step in [0, 1, 5, 10, 19, 20]:
        expected = get_lr(current_step, total_steps, base_lr)
        actual = cosine_lr(current_step, total_steps, base_lr)
        diff = abs(expected - actual)
        print(f"lr_step_{current_step}_diff={diff:.12f}")
        assert diff < 1e-12


def test_loss_scaling() -> None:
    loss = torch.tensor(8.0)
    scaled = scale_loss_for_accumulation(loss, accumulation_steps=4)
    print(f"scaled_loss={scaled.item():.6f}")
    assert torch.equal(scaled, torch.tensor(2.0))


def test_step_timing() -> None:
    stepped = [step for step in range(1, 11) if should_step_optimizer(step, accumulation_steps=4)]
    print(f"optimizer_step_micro_steps={stepped}")
    assert stepped == [4, 8]

    assert should_flush_tail(last_step=10, start_step=0, accumulation_steps=4)
    assert not should_flush_tail(last_step=8, start_step=0, accumulation_steps=4)
    assert not should_flush_tail(last_step=0, start_step=0, accumulation_steps=4)


def main() -> None:
    try:
        test_cosine_lr()
        test_loss_scaling()
        test_step_timing()
    except NotImplementedError as exc:
        raise SystemExit(f"TODO not implemented yet: {exc}") from exc
    print("train_loop_mechanics=passed")


if __name__ == "__main__":
    main()
