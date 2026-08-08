"""Alignment tests for lesson 23 PPO helpers.

Run after implementing:

    python course/impl/tests/test_ppo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from course.impl.core.ppo import clipped_value_loss  # noqa: E402
from course.impl.core.ppo import masked_gae  # noqa: E402
from course.impl.core.ppo import ppo_policy_loss  # noqa: E402
from course.impl.core.ppo import reference_kl_penalty  # noqa: E402
from course.impl.core.ppo import response_logp_positions  # noqa: E402


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp(min=1)


def assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, tol: float = 1e-7) -> None:
    diff = (actual - expected).abs().max().item()
    print(f"{name}_max_abs_diff={diff:.12f}")
    assert diff < tol, f"{name} diff {diff} >= {tol}"


def expected_masked_gae(
    token_rewards: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    gen_len = values.size(1)
    lastgaelam = torch.zeros(values.size(0), device=values.device, dtype=values.dtype)
    advs_rev = []
    for t in reversed(range(gen_len)):
        next_value = values[:, t + 1] if t < gen_len - 1 else 0.0
        delta = token_rewards[:, t] + gamma * next_value - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        advs_rev.append(lastgaelam)

    advantages = torch.stack(advs_rev[::-1], dim=1)
    returns = advantages + values
    adv_mean = masked_mean(advantages, mask)
    adv_var = masked_mean((advantages - adv_mean) ** 2, mask)
    advantages = (advantages - adv_mean) * torch.rsqrt(adv_var + 1e-8) * mask
    return advantages, returns


def test_response_logp_positions() -> None:
    prompt_lens = torch.tensor([3, 5], dtype=torch.long)
    expected = torch.tensor(
        [
            [2, 3, 4, 5],
            [4, 5, 6, 7],
        ],
        dtype=torch.long,
    )
    actual = response_logp_positions(prompt_lens, response_len=4)
    assert torch.equal(actual, expected), f"actual={actual.tolist()}, expected={expected.tolist()}"
    print("response_logp_positions=passed")


def test_masked_gae_alignment() -> None:
    token_rewards = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, -0.5, 0.0, 0.0],
        ]
    )
    values = torch.tensor(
        [
            [0.10, 0.20, 0.30, 0.00],
            [0.50, 0.10, 0.00, 0.00],
        ]
    )
    mask = torch.tensor(
        [
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ]
    )
    gamma = 1.0
    lam = 0.95

    expected_adv, expected_returns = expected_masked_gae(token_rewards, values, mask, gamma, lam)
    actual_adv, actual_returns = masked_gae(token_rewards, values, mask, gamma, lam)

    assert_close("masked_gae_advantages", actual_adv, expected_adv)
    assert_close("masked_gae_returns", actual_returns, expected_returns)


def test_ppo_policy_loss_alignment() -> None:
    new_logps = torch.tensor(
        [
            [-1.0, -0.7, -2.2],
            [-0.5, -1.3, -0.9],
        ]
    )
    old_logps = torch.tensor(
        [
            [-1.2, -0.8, -1.9],
            [-0.7, -1.0, -0.9],
        ]
    )
    advantages = torch.tensor(
        [
            [1.0, -0.5, 0.3],
            [-1.2, 0.7, 0.0],
        ]
    )
    mask = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    clip_epsilon = 0.2

    log_ratio = new_logps - old_logps
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    expected = masked_mean(torch.max(-advantages * ratio, -advantages * clipped_ratio), mask)
    actual = ppo_policy_loss(new_logps, old_logps, advantages, mask, clip_epsilon)
    assert_close("ppo_policy_loss", actual, expected)


def test_reference_kl_penalty_alignment() -> None:
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
    mask = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )

    delta = ref_logps - policy_logps
    expected = masked_mean(torch.exp(delta) - delta - 1.0, mask)
    actual = reference_kl_penalty(policy_logps, ref_logps, mask)
    assert_close("reference_kl_penalty", actual, expected)


def test_clipped_value_loss_alignment() -> None:
    values = torch.tensor(
        [
            [0.0, 1.0, 2.5],
            [1.5, -0.5, 0.0],
        ]
    )
    old_values = torch.tensor(
        [
            [0.0, 0.5, 1.0],
            [1.0, -0.2, 0.0],
        ]
    )
    returns = torch.tensor(
        [
            [0.5, 1.5, 3.0],
            [2.0, -1.0, 0.0],
        ]
    )
    mask = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    cliprange_value = 0.2

    raw_error = (values - returns) ** 2
    clipped_values = torch.clamp(values, old_values - cliprange_value, old_values + cliprange_value)
    clipped_error = (clipped_values - returns) ** 2
    expected = 0.5 * masked_mean(torch.max(raw_error, clipped_error), mask)
    actual = clipped_value_loss(values, old_values, returns, mask, cliprange_value)
    assert_close("clipped_value_loss", actual, expected)


def main() -> None:
    tests = [
        test_response_logp_positions,
        test_masked_gae_alignment,
        test_ppo_policy_loss_alignment,
        test_reference_kl_penalty_alignment,
        test_clipped_value_loss_alignment,
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
        raise SystemExit(f"ppo_core=failed: {names}")
    print("ppo_core=passed")


if __name__ == "__main__":
    main()
