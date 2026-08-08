"""Minimal CausalLM skeleton for the course implementation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class CourseMiniMindConfig:
    vocab_size: int = 6400
    hidden_size: int = 768
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 96
    intermediate_size: int = 2432
    max_position_embeddings: int = 32768
    rope_theta: float = 1e6
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0


class CourseMiniMindForCausalLM(nn.Module):
    """TODO: assemble embedding, blocks, final norm, lm_head, and loss.

    Align with:
    - model/model_minimind.py::MiniMindModel
    - model/model_minimind.py::MiniMindForCausalLM
    """

    def __init__(self, config: CourseMiniMindConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
        use_cache: bool = False,
        labels: torch.Tensor | None = None,
    ):
        raise NotImplementedError("Implement during model assembly and pretrain lessons.")
