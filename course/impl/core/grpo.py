"""GRPO/CISPO core functions to implement and align with MiniMind source."""

from __future__ import annotations

import torch


def group_relative_advantages(
    rewards: torch.Tensor,
    num_generations: int,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Compute group-relative advantages from flat rewards.

    Align with: trainer/train_grpo.py:121-124

    Args:
        rewards: Flat tensor with shape [batch * num_generations].
        num_generations: Number of responses sampled per prompt.
        eps: Stabilizer added to group std.

    Returns:
        Tensor with shape [batch * num_generations].

    Formula:
        A_i = (R_i - mean(group)) / (std(group) + eps)
    """
    groups=rewards.view(-1,num_generations)
    group_mean=groups.mean(dim=1,keepdim=True)
    group_var=((groups-group_mean)**2).sum(dim=1)/num_generations
    group_var=torch.sqrt(group_var)
    group_mean=group_mean.repeat_interleave(num_generations)
    group_var=group_var.repeat_interleave(num_generations)
    return (rewards-group_mean)/(group_var+eps)
    raise NotImplementedError("Implement in lesson 24: GRPO and CISPO.")


def token_kl_penalty(policy_logps: torch.Tensor, ref_logps: torch.Tensor) -> torch.Tensor:
    """Compute MiniMind's token-level reference KL penalty tensor.

    Align with: trainer/train_grpo.py:132-133

    Args:
        policy_logps: Current policy logprobs, shape [batch * num_generations, response_len].
        ref_logps: Frozen reference logprobs, same shape.

    Returns:
        Per-token KL penalty tensor with the same shape.

    Formula:
        delta = ref_logps - policy_logps
        penalty = exp(delta) - delta - 1
    """
    delta = ref_logps - policy_logps
    penalty = torch.exp(delta) - delta - 1
    return penalty
    raise NotImplementedError("Implement in lesson 24: GRPO and CISPO.")


def grpo_policy_loss(
    policy_logps: torch.Tensor,
    old_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    epsilon: float,
    beta: float,
) -> torch.Tensor:
    """Compute MiniMind's GRPO token loss reduced to a scalar.

    Align with: trainer/train_grpo.py:132-143

    Args:
        policy_logps: Current policy logprobs, shape [batch * num_generations, response_len].
        old_logps: Rollout policy logprobs, same shape.
        ref_logps: Frozen reference model logprobs, same shape.
        advantages: Sequence-level advantages, shape [batch * num_generations].
        completion_mask: Valid response-token mask, shape [batch * num_generations, response_len].
        epsilon: PPO-style clip epsilon.
        beta: KL penalty coefficient.

    Returns:
        Scalar policy loss.
    """
    ratio=policy_logps-old_logps
    ratio=torch.exp(ratio)
    loss1=torch.clamp(ratio,1-epsilon,1+epsilon)*advantages.unsqueeze(1)
    loss2=ratio*advantages.unsqueeze(1)
    kl_pl=token_kl_penalty(policy_logps,ref_logps)
    per_token_loss=-(torch.min(loss1,loss2)-beta*kl_pl)
    loss = (
      (per_token_loss * completion_mask).sum(dim=1)
      / completion_mask.sum(dim=1).clamp(min=1)
  ).mean()
    return loss
    raise NotImplementedError("Implement in lesson 24: GRPO and CISPO.")


def cispo_policy_loss(
    policy_logps: torch.Tensor,
    old_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    epsilon_high: float,
    beta: float,
) -> torch.Tensor:
    """Compute MiniMind's CISPO token loss reduced to a scalar.

    Align with: trainer/train_grpo.py:132-143

    Args:
        policy_logps: Current policy logprobs, shape [batch * num_generations, response_len].
        old_logps: Rollout policy logprobs, same shape.
        ref_logps: Frozen reference model logprobs, same shape.
        advantages: Sequence-level advantages, shape [batch * num_generations].
        completion_mask: Valid response-token mask, shape [batch * num_generations, response_len].
        epsilon_high: Upper bound used to clamp ratio as a detached weight.
        beta: KL penalty coefficient.

    Returns:
        Scalar policy loss.
    """
    ratio=policy_logps-old_logps
    ratio=torch.exp(ratio)
    climp_ratio=torch.clamp(ratio,max=epsilon_high).detach()
    kl_pl=token_kl_penalty(policy_logps,ref_logps)
    per_token_loss=-climp_ratio*advantages.unsqueeze(1)*policy_logps+beta*kl_pl
    loss = (
      (per_token_loss * completion_mask).sum(dim=1)
      / completion_mask.sum(dim=1).clamp(min=1)
  ).mean()
    return loss
    raise NotImplementedError("Implement in lesson 24: GRPO and CISPO.")
