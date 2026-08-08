"""Alignment tests for lesson 24 GRPO/CISPO helpers.

Run after implementing:

    python course/impl/tests/test_grpo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from course.impl.core.grpo import cispo_policy_loss  # noqa: E402
from course.impl.core.grpo import grpo_policy_loss  # noqa: E402
from course.impl.core.grpo import group_relative_advantages  # noqa: E402
from course.impl.core.grpo import token_kl_penalty  # noqa: E402


def assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, tol: float = 1e-7) -> None:
    diff = (actual - expected).abs().max().item()
    print(f"{name}_max_abs_diff={diff:.12f}")
    assert diff < tol, f"{name} diff {diff} >= {tol}"


def expected_group_relative_advantages(
    rewards: torch.Tensor,
    num_generations: int,
    eps: float = 1e-4,
) -> torch.Tensor:
    grouped_rewards = rewards.view(-1, num_generations)
    mean_r = grouped_rewards.mean(dim=1).repeat_interleave(num_generations)
    std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(num_generations)
    return (rewards - mean_r) / (std_r + eps)


def expected_token_kl_penalty(policy_logps: torch.Tensor, ref_logps: torch.Tensor) -> torch.Tensor:
    kl_div = ref_logps - policy_logps
    return torch.exp(kl_div) - kl_div - 1.0


def reduce_like_source(per_token_loss: torch.Tensor, completion_mask: torch.Tensor) -> torch.Tensor:
    per_response = (per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)
    return per_response.mean()


def expected_grpo_policy_loss(
    policy_logps: torch.Tensor,
    old_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    epsilon: float,
    beta: float,
) -> torch.Tensor:
    per_token_kl = expected_token_kl_penalty(policy_logps, ref_logps)
    ratio = torch.exp(policy_logps - old_logps)
    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
    per_token_loss1 = ratio * advantages.unsqueeze(1)
    per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
    per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - beta * per_token_kl)
    return reduce_like_source(per_token_loss, completion_mask)


def expected_cispo_policy_loss(
    policy_logps: torch.Tensor,
    old_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    epsilon_high: float,
    beta: float,
) -> torch.Tensor:
    per_token_kl = expected_token_kl_penalty(policy_logps, ref_logps)
    ratio = torch.exp(policy_logps - old_logps)
    clamped_ratio = torch.clamp(ratio, max=epsilon_high).detach()
    per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * policy_logps - beta * per_token_kl)
    return reduce_like_source(per_token_loss, completion_mask)


def test_group_relative_advantages_alignment() -> None:
    rewards = torch.tensor([1.0, 3.0, 2.0, 2.0, -1.0, 1.0])
    expected = expected_group_relative_advantages(rewards, num_generations=2)
    actual = group_relative_advantages(rewards, num_generations=2)
    assert_close("group_relative_advantages", actual, expected)
    print("group_relative_advantages=passed")


def test_token_kl_penalty_alignment() -> None:
    policy_logps = torch.tensor(
        [
            [-1.0, -0.7, -2.2],
            [-0.5, -1.3, -0.9],
        ]
    )
    ref_logps = torch.tensor(
        [
            [-1.1, -0.9, -2.0],
            [-0.4, -1.0, -0.8],
        ]
    )
    expected = expected_token_kl_penalty(policy_logps, ref_logps)
    actual = token_kl_penalty(policy_logps, ref_logps)
    assert_close("token_kl_penalty", actual, expected)
    print("token_kl_penalty=passed")


def test_grpo_policy_loss_alignment() -> None:
    policy_logps = torch.tensor(
        [
            [-1.00, -0.70, -2.20, -0.30],
            [-0.50, -1.30, -0.90, -2.00],
            [-1.50, -0.20, -1.10, -0.80],
            [-0.90, -1.80, -0.40, -1.20],
        ]
    )
    old_logps = torch.tensor(
        [
            [-1.20, -0.80, -1.90, -0.40],
            [-0.70, -1.00, -0.90, -1.70],
            [-1.60, -0.50, -1.00, -1.10],
            [-1.20, -1.60, -0.60, -1.00],
        ]
    )
    ref_logps = torch.tensor(
        [
            [-1.10, -0.90, -2.00, -0.30],
            [-0.40, -1.00, -0.80, -2.10],
            [-1.30, -0.40, -1.30, -0.70],
            [-1.00, -1.70, -0.70, -1.40],
        ]
    )
    advantages = torch.tensor([1.2, -0.8, 0.5, -1.1])
    completion_mask = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
        ]
    )
    epsilon = 0.2
    beta = 0.03

    expected = expected_grpo_policy_loss(
        policy_logps,
        old_logps,
        ref_logps,
        advantages,
        completion_mask,
        epsilon,
        beta,
    )
    actual = grpo_policy_loss(
        policy_logps,
        old_logps,
        ref_logps,
        advantages,
        completion_mask,
        epsilon,
        beta,
    )
    assert_close("grpo_policy_loss", actual, expected)


def test_cispo_policy_loss_alignment() -> None:
    policy_logps = torch.tensor(
        [
            [-1.00, -0.70, -2.20, -0.30],
            [-0.50, -1.30, -0.90, -2.00],
            [-1.50, -0.20, -1.10, -0.80],
            [-0.90, -1.80, -0.40, -1.20],
        ]
    )
    old_logps = torch.tensor(
        [
            [-1.20, -0.80, -1.90, -0.40],
            [-0.70, -1.00, -0.90, -1.70],
            [-1.60, -0.50, -1.00, -1.10],
            [-1.20, -1.60, -0.60, -1.00],
        ]
    )
    ref_logps = torch.tensor(
        [
            [-1.10, -0.90, -2.00, -0.30],
            [-0.40, -1.00, -0.80, -2.10],
            [-1.30, -0.40, -1.30, -0.70],
            [-1.00, -1.70, -0.70, -1.40],
        ]
    )
    advantages = torch.tensor([1.2, -0.8, 0.5, -1.1])
    completion_mask = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
        ]
    )
    epsilon_high = 1.3
    beta = 0.03

    expected = expected_cispo_policy_loss(
        policy_logps,
        old_logps,
        ref_logps,
        advantages,
        completion_mask,
        epsilon_high,
        beta,
    )
    actual = cispo_policy_loss(
        policy_logps,
        old_logps,
        ref_logps,
        advantages,
        completion_mask,
        epsilon_high,
        beta,
    )
    assert_close("cispo_policy_loss", actual, expected)


def test_cispo_clamped_ratio_detached_gradient() -> None:
    policy_logps = torch.tensor(
        [
            [-1.00, -0.70],
            [-0.50, -1.30],
        ],
        requires_grad=True,
    )
    expected_policy_logps = policy_logps.detach().clone().requires_grad_(True)
    old_logps = torch.tensor(
        [
            [-1.20, -0.80],
            [-0.70, -1.00],
        ]
    )
    ref_logps = torch.tensor(
        [
            [-1.10, -0.90],
            [-0.40, -1.00],
        ]
    )
    advantages = torch.tensor([1.2, -0.8])
    completion_mask = torch.ones_like(old_logps)
    epsilon_high = 1.3
    beta = 0.0

    actual = cispo_policy_loss(
        policy_logps,
        old_logps,
        ref_logps,
        advantages,
        completion_mask,
        epsilon_high,
        beta,
    )
    actual.backward()

    expected = expected_cispo_policy_loss(
        expected_policy_logps,
        old_logps,
        ref_logps,
        advantages,
        completion_mask,
        epsilon_high,
        beta,
    )
    expected.backward()

    assert_close("cispo_detached_gradient", policy_logps.grad, expected_policy_logps.grad)


def main() -> None:
    tests = [
        test_group_relative_advantages_alignment,
        test_token_kl_penalty_alignment,
        test_grpo_policy_loss_alignment,
        test_cispo_policy_loss_alignment,
        test_cispo_clamped_ratio_detached_gradient,
    ]
    failures = []
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - report all lesson test failures.
            failures.append((test.__name__, exc))
            print(f"{test.__name__}=failed: {type(exc).__name__}: {exc}")

    if failures:
        names = ", ".join(name for name, _ in failures)
        raise SystemExit(f"grpo_core=failed: {names}")
    print("grpo_core=passed")


if __name__ == "__main__":
    main()
