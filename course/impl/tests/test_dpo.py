"""Alignment tests for lesson 22 DPO loss helpers.

Run after implementing:

    python course/impl/tests/test_dpo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from course.impl.core.losses import dpo_loss as course_dpo_loss  # noqa: E402
from course.impl.core.losses import sequence_log_probs  # noqa: E402
from trainer.train_dpo import dpo_loss as source_dpo_loss  # noqa: E402
from trainer.train_dpo import logits_to_log_probs  # noqa: E402


def test_sequence_log_probs_alignment() -> None:
    torch.manual_seed(42)
    logits = torch.randn(3, 5, 7)
    labels = torch.tensor(
        [
            [0, 1, 2, 3, 4],
            [4, 3, 2, 1, 0],
            [1, 1, 5, 6, 2],
        ],
        dtype=torch.long,
    )
    mask = torch.tensor(
        [
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [1, 1, 0, 1, 0],
        ],
        dtype=torch.float32,
    )

    expected = (logits_to_log_probs(logits, labels) * mask).sum(dim=1)
    actual = sequence_log_probs(logits, labels, mask)
    diff = (expected - actual).abs().max().item()
    print(f"sequence_log_probs_max_abs_diff={diff:.12f}")
    assert diff < 1e-7


def test_dpo_loss_alignment() -> None:
    policy_chosen = torch.tensor([-4.0, -2.5])
    policy_rejected = torch.tensor([-5.0, -4.0])
    ref_chosen = torch.tensor([-4.5, -3.0])
    ref_rejected = torch.tensor([-4.8, -3.8])
    beta = 0.15

    actual = course_dpo_loss(
        policy_chosen,
        policy_rejected,
        ref_chosen,
        ref_rejected,
        beta,
    )

    # Build source-shaped token logprob inputs whose masked sums equal the
    # sequence logprobs above. The source function splits batch front/back.
    policy_log_probs = torch.tensor(
        [
            [-1.0, -3.0, 0.0],
            [-0.5, -2.0, 0.0],
            [-2.0, -3.0, 0.0],
            [-1.0, -3.0, 0.0],
        ]
    )
    ref_log_probs = torch.tensor(
        [
            [-1.5, -3.0, 0.0],
            [-1.0, -2.0, 0.0],
            [-2.0, -2.8, 0.0],
            [-1.0, -2.8, 0.0],
        ]
    )
    mask = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    expected = source_dpo_loss(ref_log_probs, policy_log_probs, mask, beta)
    diff = abs(expected.item() - actual.item())
    print(f"dpo_loss_abs_diff={diff:.12f}")
    assert diff < 1e-7


def main() -> None:
    try:
        test_sequence_log_probs_alignment()
        test_dpo_loss_alignment()
    except NotImplementedError as exc:
        raise SystemExit(f"TODO not implemented yet: {exc}") from exc
    print("dpo_core=passed")


if __name__ == "__main__":
    main()
