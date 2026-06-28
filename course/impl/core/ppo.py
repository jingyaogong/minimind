"""PPO core functions to implement and align with MiniMind source."""

from __future__ import annotations

import torch


def response_logp_positions(prompt_lens: torch.Tensor, response_len: int) -> torch.Tensor:
    """Return logits positions for response-token logprobs.

    Align with: trainer/train_ppo.py:121-122

    Args:
        prompt_lens: Tensor with shape [batch].
        response_len: Number of response tokens R.

    Returns:
        Tensor with shape [batch, R].

    Rule:
        The j-th response token is predicted by logits at prompt_len - 1 + j.
    """
    response_idx=torch.arange(response_len).unsqueeze(0)
    return prompt_lens.unsqueeze(1)+response_idx-1
    raise NotImplementedError("Implement in lesson 23: PPO training flow.")


def masked_gae(
    token_rewards: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute masked GAE advantages and returns.

    Align with: trainer/train_ppo.py:140-151

    Args:
        token_rewards: Tensor with shape [batch, response_len].
        values: Tensor with shape [batch, response_len].
        mask: Tensor with shape [batch, response_len].
        gamma: Discount factor.
        lam: GAE lambda.

    Returns:
        advantages: Tensor with shape [batch, response_len].
        returns: Tensor with shape [batch, response_len].

    Requirements:
        - Recurse from the last response token to the first.
        - Normalize advantages only over mask == 1 positions.
        - Set padding / invalid positions to zero in the final advantages.
    """
    advs_rev=[]
    advantage=torch.zeros(token_rewards.size(0),device=token_rewards.device)
    response_len=token_rewards.size(1)
    for i in reversed(range(response_len)):
        nv=values[:,i+1] if i<response_len-1 else 0
        delta=token_rewards[:,i]+gamma*nv-values[:,i]
        advantage=delta+gamma*lam*advantage
        advs_rev.append(advantage)
    advantages=torch.stack(advs_rev[::-1], dim=1)
    returns=advantages+values
    advs_mean=(advantages*mask).sum()/mask.sum().clamp(min=1)
    advs_var=((advantages-advs_mean)**2*mask).sum()/mask.sum().clamp(min=1)
    advantages=(advantages-advs_mean)*torch.rsqrt(advs_var+1e-8)*mask
    return (advantages,returns)
    raise NotImplementedError("Implement in lesson 23: PPO training flow.")


def ppo_policy_loss(
    new_logps: torch.Tensor,
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_epsilon: float,
) -> torch.Tensor:
    """Compute the clipped PPO policy loss.

    Align with: trainer/train_ppo.py:180-199

    Args:
        new_logps: Current actor logprobs, shape [batch, response_len].
        old_logps: Rollout actor logprobs, shape [batch, response_len].
        advantages: GAE advantages, shape [batch, response_len].
        mask: Valid response-token mask, shape [batch, response_len].
        clip_epsilon: PPO ratio clipping range.

    Returns:
        A scalar loss.

    Requirements:
        - log_ratio = new_logps - old_logps
        - ratio = exp(log_ratio)
        - clipped_ratio = clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
        - loss = masked mean(max(-advantages * ratio, -advantages * clipped_ratio))

    This function intentionally does not include reference KL. Add
    kl_coef * reference_kl_penalty(...) at the stage assembly level.
    """
    log_ratio = new_logps - old_logps
    ratio=torch.exp(log_ratio)
    clipped_ratio=torch.clamp(ratio,1-clip_epsilon,1+clip_epsilon)
    loss=torch.max(-advantages * ratio,-advantages*clipped_ratio)
    return (loss*mask).sum()/mask.sum().clamp(min=1)
    raise NotImplementedError("Implement in lesson 23: PPO training flow.")


def reference_kl_penalty(
    policy_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute MiniMind's token-level reference KL penalty.

    Align with: trainer/train_ppo.py:194-195

    Args:
        policy_logps: Current actor logprobs, shape [batch, response_len].
        ref_logps: Frozen reference model logprobs, shape [batch, response_len].
        mask: Valid response-token mask, shape [batch, response_len].

    Returns:
        A scalar penalty.

    Formula:
        delta = ref_logps - policy_logps
        penalty = exp(delta) - delta - 1
        loss = masked mean(penalty)
    """
    delta = ref_logps - policy_logps
    penalty = torch.exp(delta) - delta - 1
    return (penalty*mask).sum()/mask.sum().clamp(min=1)
    raise NotImplementedError("Implement in lesson 23: PPO training flow.")


def clipped_value_loss(
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    cliprange_value: float,
) -> torch.Tensor:
    """Compute MiniMind's clipped critic value loss.

    Align with: trainer/train_ppo.py:200-203

    Args:
        values: Current critic values, shape [batch, response_len].
        old_values: Rollout critic values, shape [batch, response_len].
        returns: Value targets, shape [batch, response_len].
        mask: Valid response-token mask, shape [batch, response_len].
        cliprange_value: Value clipping range.

    Returns:
        A scalar loss.

    Requirements:
        - raw_error = (values - returns) ** 2
        - clipped_values = clamp(values, old_values - cliprange_value, old_values + cliprange_value)
        - clipped_error = (clipped_values - returns) ** 2
        - loss = 0.5 * masked mean(max(raw_error, clipped_error))
    """
    raw_error = (values - returns) ** 2
    clipped_values = torch.clamp(values, old_values - cliprange_value, old_values + cliprange_value)
    clipped_error = (clipped_values - returns) ** 2
    loss=0.5*((torch.max(raw_error,clipped_error)*mask).sum())/mask.sum().clamp(min=1)
    return loss
    raise NotImplementedError("Implement in lesson 23: PPO training flow.")
