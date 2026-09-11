"""Generation helpers for the course implementation."""

from __future__ import annotations

import torch


@torch.inference_mode()
def generate(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
    use_cache: bool = True,
):
    """TODO: implement minimal autoregressive generation with optional KV cache.

    Align with: model/model_minimind.py::MiniMindForCausalLM.generate
    """
    raise NotImplementedError("Implement in the generation/KV cache lesson.")
