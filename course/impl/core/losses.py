"""Loss functions to implement and align with MiniMind source."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """TODO: implement shifted next-token cross entropy.

    Align with: model/model_minimind.py::MiniMindForCausalLM.forward
    """
    raise NotImplementedError("Implement in the causal LM loss lesson.")


def sequence_log_probs(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Gather per-token logprobs and reduce by sequence.

    Align with: trainer/train_dpo.py logprob handling.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return (log_probs_per_token * mask).sum(dim=1)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Compute DPO preference loss from split sequence logprobs.

    Align with: trainer/train_dpo.py::dpo_loss
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios
    return -F.logsigmoid(beta * logits).mean()
