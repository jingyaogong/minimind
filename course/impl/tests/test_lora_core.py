"""Alignment tests for lesson 18 LoRA core modules.

Run after implementing:

    python course/impl/tests/test_lora_core.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from course.impl.core.lora import LoRALinear, lora_state_dict, mark_only_lora_as_trainable  # noqa: E402


def test_lora_linear_forward() -> None:
    torch.manual_seed(42)
    base = nn.Linear(4, 4, bias=False)
    wrapped = LoRALinear(base, rank=2, alpha=2.0, dropout=0.0)

    assert hasattr(wrapped, "A")
    assert hasattr(wrapped, "B")

    with torch.no_grad():
        base.weight.copy_(torch.arange(16, dtype=torch.float32).view(4, 4) / 100)
        wrapped.A.weight.copy_(torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]))
        wrapped.B.weight.copy_(torch.tensor([[0.2, -0.1], [0.0, 0.3], [-0.4, 0.1], [0.5, 0.2]]))

    x = torch.randn(3, 4)
    actual = wrapped(x)
    expected = base(x) + wrapped.B(wrapped.A(x))
    diff = (actual - expected).abs().max().item()
    print(f"lora_linear_forward_diff={diff:.12f}")
    assert diff < 1e-6


def test_freeze_and_state_dict() -> None:
    model = nn.Sequential(LoRALinear(nn.Linear(4, 4, bias=False), rank=2, alpha=2.0))
    lora_params = mark_only_lora_as_trainable(model)
    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    print(f"trainable_names={trainable_names}")
    assert trainable_names == ["0.A.weight", "0.B.weight"]
    assert list(lora_params) == [model[0].A.weight, model[0].B.weight]

    state = lora_state_dict(model)
    print(f"lora_state_keys={sorted(state.keys())}")
    assert sorted(state.keys()) == ["0.A.weight", "0.B.weight"]


def main() -> None:
    try:
        test_lora_linear_forward()
        test_freeze_and_state_dict()
    except NotImplementedError as exc:
        raise SystemExit(f"TODO not implemented yet: {exc}") from exc
    print("lora_core=passed")


if __name__ == "__main__":
    main()
