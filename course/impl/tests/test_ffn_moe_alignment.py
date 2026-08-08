"""Alignment test for course FeedForward and MOEFeedForward.

Run after implementing:

    python course/impl/tests/test_ffn_moe_alignment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from course.impl.core.model_parts import FeedForward as CourseFeedForward
from course.impl.core.model_parts import MOEFeedForward as CourseMOEFeedForward
from model.model_minimind import FeedForward as SourceFeedForward
from model.model_minimind import MiniMindConfig, MOEFeedForward as SourceMOEFeedForward


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def copy_dense_weights(source: SourceFeedForward, target: CourseFeedForward) -> None:
    target.gate_proj.weight.data.copy_(source.gate_proj.weight.data)
    target.up_proj.weight.data.copy_(source.up_proj.weight.data)
    target.down_proj.weight.data.copy_(source.down_proj.weight.data)


def copy_moe_weights(source: SourceMOEFeedForward, target: CourseMOEFeedForward) -> None:
    target.gate.weight.data.copy_(source.gate.weight.data)
    for source_expert, target_expert in zip(source.experts, target.experts):
        copy_dense_weights(source_expert, target_expert)


def test_feedforward() -> None:
    torch.manual_seed(42)
    config = MiniMindConfig(
        hidden_size=32,
        intermediate_size=64,
        hidden_act="silu",
        dropout=0.0,
    )
    source = SourceFeedForward(config).eval()
    target = CourseFeedForward(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        hidden_act=config.hidden_act,
    ).eval()
    copy_dense_weights(source, target)

    x = torch.randn(2, 5, config.hidden_size)
    with torch.no_grad():
        source_y = source(x)
        target_y = target(x)

    diff = max_abs_diff(source_y, target_y)
    print(f"feedforward_max_abs_diff={diff:.12f}")
    assert diff < 1e-6


def test_moe() -> None:
    torch.manual_seed(42)
    config = MiniMindConfig(
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=64,
        hidden_act="silu",
        dropout=0.0,
        use_moe=True,
        num_experts=4,
        num_experts_per_tok=1,
        norm_topk_prob=True,
        router_aux_loss_coef=5e-4,
    )
    source = SourceMOEFeedForward(config)
    target = CourseMOEFeedForward(
        hidden_size=config.hidden_size,
        intermediate_size=config.moe_intermediate_size,
        num_experts=config.num_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        hidden_act=config.hidden_act,
        norm_topk_prob=config.norm_topk_prob,
        router_aux_loss_coef=config.router_aux_loss_coef,
    )
    copy_moe_weights(source, target)

    x = torch.randn(2, 5, config.hidden_size)

    source.eval()
    target.eval()
    with torch.no_grad():
        source_eval_y = source(x)
        target_eval_y = target(x)

    eval_diff = max_abs_diff(source_eval_y, target_eval_y)
    print(f"moe_eval_max_abs_diff={eval_diff:.12f}")
    assert eval_diff < 1e-6

    source.train()
    target.train()
    source_train_y = source(x)
    target_train_y = target(x)
    train_diff = max_abs_diff(source_train_y, target_train_y)
    aux_diff = abs(source.aux_loss.item() - target.aux_loss.item())
    print(f"moe_train_max_abs_diff={train_diff:.12f}")
    print(f"moe_aux_loss_abs_diff={aux_diff:.12f}")
    assert train_diff < 1e-6
    assert aux_diff < 1e-8


def main() -> None:
    try:
        test_feedforward()
        test_moe()
    except NotImplementedError as exc:
        raise SystemExit(f"TODO not implemented yet: {exc}") from exc
    print("ffn_moe_alignment=passed")


if __name__ == "__main__":
    main()
