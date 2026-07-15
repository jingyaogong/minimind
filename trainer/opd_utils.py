"""Core tensor utilities for Generalized Knowledge Distillation (GKD)."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def build_completion_mask(
    completion_ids: Tensor,
    base_mask: Optional[Tensor],
    eos_token_id: Optional[int],
    pad_token_id: Optional[int] = None,
) -> Tensor:
    """Keep valid completion tokens up to and including the first EOS.

    MiniMind's native ``generate`` pads finished rows with repeated EOS tokens
    until every row is complete. Those repeated tokens are valid model inputs,
    but they must not contribute to the distillation objective.
    """
    if completion_ids.ndim != 2:
        raise ValueError(f"completion_ids must be 2D, got {tuple(completion_ids.shape)}")
    if base_mask is None:
        mask = torch.ones_like(completion_ids, dtype=torch.bool)
    else:
        if base_mask.shape != completion_ids.shape:
            raise ValueError(
                f"base_mask shape {tuple(base_mask.shape)} does not match "
                f"completion_ids shape {tuple(completion_ids.shape)}"
            )
        mask = base_mask.bool().clone()

    # Some tokenizers intentionally use EOS as PAD. In that case the first EOS
    # still belongs to the completion and is handled by the cumulative mask.
    if pad_token_id is not None and pad_token_id != eos_token_id:
        mask &= completion_ids.ne(pad_token_id)

    if eos_token_id is not None:
        eos_hits = completion_ids.eq(eos_token_id) & mask
        eos_seen_before = eos_hits.long().cumsum(dim=1) - eos_hits.long()
        mask &= eos_seen_before.eq(0)
    return mask


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    mask = mask.to(device=values.device, dtype=values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def generalized_jsd_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    completion_mask: Tensor,
    beta: float = 0.5,
    temperature: float = 1.0,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Compute the exact full-vocabulary generalized JSD from GKD.

    With teacher distribution ``p`` and student distribution ``q``, the
    mixture is ``m = beta * p + (1 - beta) * q`` and the objective is
    ``beta * KL(p || m) + (1 - beta) * KL(q || m)``. Its limiting cases are
    forward KL at ``beta=0`` and reverse KL at ``beta=1``.

    The teacher is always treated as a fixed target. Only completion tokens
    selected by ``completion_mask`` contribute to the token-mean loss.
    """
    if student_logits.ndim != 3 or teacher_logits.ndim != 3:
        raise ValueError("student_logits and teacher_logits must be [batch, time, vocab]")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "student and teacher logits must have identical shapes, got "
            f"{tuple(student_logits.shape)} and {tuple(teacher_logits.shape)}"
        )
    if completion_mask.shape != student_logits.shape[:2]:
        raise ValueError(
            f"completion_mask shape {tuple(completion_mask.shape)} does not match "
            f"logits shape {tuple(student_logits.shape[:2])}"
        )
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be in [0, 1], got {beta}")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    # Float32 log-softmax keeps the divergence stable under fp16/bf16 model
    # autocast, while preserving gradients back to the student parameters.
    student_logps = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_logps = F.log_softmax(
        teacher_logits.detach().float() / temperature, dim=-1
    )
    student_probs = student_logps.exp()
    teacher_probs = teacher_logps.exp()

    forward_kl = (teacher_probs * (teacher_logps - student_logps)).sum(dim=-1)
    reverse_kl = (student_probs * (student_logps - teacher_logps)).sum(dim=-1)

    if beta == 0.0:
        per_token_loss = forward_kl
    elif beta == 1.0:
        per_token_loss = reverse_kl
    else:
        beta_tensor = student_logps.new_tensor(beta)
        mixture_logps = torch.logsumexp(
            torch.stack(
                (
                    student_logps + torch.log1p(-beta_tensor),
                    teacher_logps + torch.log(beta_tensor),
                )
            ),
            dim=0,
        )
        teacher_to_mixture = (
            teacher_probs * (teacher_logps - mixture_logps)
        ).sum(dim=-1)
        student_to_mixture = (
            student_probs * (student_logps - mixture_logps)
        ).sum(dim=-1)
        per_token_loss = (
            beta_tensor * teacher_to_mixture
            + (1.0 - beta_tensor) * student_to_mixture
        )

    mask = completion_mask.bool()
    loss = _masked_mean(per_token_loss, mask)
    top1_agreement = _masked_mean(
        student_logps.detach()
        .argmax(dim=-1)
        .eq(teacher_logps.argmax(dim=-1))
        .float(),
        mask,
    )
    metrics = {
        "divergence": loss.detach(),
        "forward_kl": _masked_mean(forward_kl.detach(), mask),
        "reverse_kl": _masked_mean(reverse_kl.detach(), mask),
        "top1_agreement": top1_agreement.detach(),
        "valid_tokens": mask.sum().detach(),
    }
    return loss, metrics


def use_on_policy_batch(lmbda: float, random_value: float) -> bool:
    """Return the Algorithm 1 branch for a pre-sampled value in ``[0, 1)``."""
    if not 0.0 <= lmbda <= 1.0:
        raise ValueError(f"lmbda must be in [0, 1], got {lmbda}")
    if not 0.0 <= random_value < 1.0:
        raise ValueError(f"random_value must be in [0, 1), got {random_value}")
    return random_value < lmbda
