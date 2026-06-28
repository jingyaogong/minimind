"""Alignment tests for lesson 19 LoRA injection flow.

Run after implementing:

    python course/impl/tests/test_lora_injection.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from course.impl.core.lora import (  # noqa: E402
    LoRALinear,
    apply_lora_to_linear_layers,
    lora_state_dict,
    mark_only_lora_as_trainable,
)


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def test_injection_preserves_initial_output() -> None:
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(4, 4, bias=False),
        nn.Linear(4, 3, bias=False),
        nn.Sequential(nn.Linear(3, 3, bias=False)),
    )
    x = torch.randn(2, 4)

    with torch.no_grad():
        before = model(x)

    injected = apply_lora_to_linear_layers(model, rank=2, alpha=2.0, dropout=0.0, square_only=True)
    print(f"injected_lora_layers={injected}")
    assert injected == 2
    assert isinstance(model[0], LoRALinear)
    assert isinstance(model[1], nn.Linear)
    assert isinstance(model[2][0], LoRALinear)

    with torch.no_grad():
        after = model(x)

    diff = max_abs_diff(before, after)
    print(f"initial_output_max_abs_diff={diff:.12f}")
    assert diff < 1e-6


def test_lora_trainable_and_state_dict_after_injection() -> None:
    model = nn.Sequential(
        nn.Linear(4, 4, bias=False),
        nn.Linear(4, 3, bias=False),
        nn.Sequential(nn.Linear(3, 3, bias=False)),
    )
    apply_lora_to_linear_layers(model, rank=2, alpha=2.0, dropout=0.0, square_only=True)

    lora_params = mark_only_lora_as_trainable(model)
    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    print(f"trainable_names={trainable_names}")
    assert trainable_names == ["0.A.weight", "0.B.weight", "2.0.A.weight", "2.0.B.weight"]
    assert list(lora_params) == [
        model[0].A.weight,
        model[0].B.weight,
        model[2][0].A.weight,
        model[2][0].B.weight,
    ]

    state = lora_state_dict(model)
    print(f"lora_state_keys={sorted(state.keys())}")
    assert sorted(state.keys()) == ["0.A.weight", "0.B.weight", "2.0.A.weight", "2.0.B.weight"]


def main() -> None:
    try:
        test_injection_preserves_initial_output()
        test_lora_trainable_and_state_dict_after_injection()
    except NotImplementedError as exc:
        raise SystemExit(f"TODO not implemented yet: {exc}") from exc
    print("lora_injection=passed")


if __name__ == "__main__":
    main()
