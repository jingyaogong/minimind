"""LoRA modules for the course implementation."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.scaling = alpha / rank

        self.A = nn.Linear(base.in_features, rank, bias=False)
        self.B = nn.Linear(rank, base.out_features, bias=False)
        self.A.to(device=base.weight.device, dtype=base.weight.dtype)
        self.B.to(device=base.weight.device, dtype=base.weight.dtype)

        with torch.no_grad():
            self.A.weight.normal_(mean=0.0, std=0.02)
            self.B.weight.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lora_x = F.dropout(x, p=self.dropout, training=self.training)
        return self.base(x) + self.scaling * self.B(self.A(lora_x))


def apply_lora_to_linear_layers(
    model: nn.Module,
    rank: int = 16,
    alpha: float | None = None,
    dropout: float = 0.0,
    square_only: bool = True,
) -> int:
    """TODO: inject LoRA adapters into selected Linear layers.

    Align with: model/model_lora.py::apply_lora
    """
    injected = 0
    for name,child in model.named_children():
        if isinstance(child,nn.Linear) and (child.in_features==child.out_features or not square_only):
            setattr(model, name,LoRALinear(child,rank,alpha,dropout))
            injected += 1
        else:
            injected +=apply_lora_to_linear_layers(child,rank,alpha,dropout,square_only)
    return injected
    raise NotImplementedError("Implement in the LoRA lesson.")


def mark_only_lora_as_trainable(model: nn.Module) -> list[nn.Parameter]:
    """TODO: freeze base parameters and return LoRA parameters.

    Align with: trainer/train_lora.py:139-147
    """
    for param in model.parameters():
        param.requires_grad = False

    lora_params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.A.weight.requires_grad = True
            module.B.weight.requires_grad = True
            lora_params.extend([module.A.weight, module.B.weight])

    return lora_params


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """TODO: return only LoRA parameters from a model.

    Align with: model/model_lora.py::save_lora
    """
    raw_model = getattr(model, "_orig_mod", model)
    state_dict = {}

    for name, module in raw_model.named_modules():
        if isinstance(module, LoRALinear):
            clean_name = name[7:] if name.startswith("module.") else name
            prefix = f"{clean_name}." if clean_name else ""
            state_dict[f"{prefix}A.weight"] = module.A.weight.detach().cpu().half()
            state_dict[f"{prefix}B.weight"] = module.B.weight.detach().cpu().half()

    return state_dict
