import argparse
import random

import torch
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp(min=1)


def response_logp_positions(prompt_lens: torch.Tensor, response_len: int) -> torch.Tensor:
    response_idx = torch.arange(response_len, device=prompt_lens.device).unsqueeze(0)
    return prompt_lens.unsqueeze(1) - 1 + response_idx


def response_token_logps(logits: torch.Tensor, output_ids: torch.Tensor, logp_pos: torch.Tensor) -> torch.Tensor:
    labels = output_ids[:, 1:]
    all_token_logps = F.log_softmax(logits[:, :-1, :], dim=-1).gather(
        2,
        labels.unsqueeze(-1),
    ).squeeze(-1)
    return all_token_logps.gather(1, logp_pos)


def put_sequence_reward_on_last_token(rewards: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    token_rewards = torch.zeros_like(response_mask)
    lengths = response_mask.sum(dim=1).long().clamp(min=1)
    last_idx = lengths - 1
    token_rewards[torch.arange(rewards.size(0)), last_idx] = rewards
    return token_rewards


def group_relative_advantages(rewards: torch.Tensor, num_generations: int, eps: float = 1e-4) -> torch.Tensor:
    grouped_rewards = rewards.view(-1, num_generations)
    mean_r = grouped_rewards.mean(dim=1).repeat_interleave(num_generations)
    std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(num_generations)
    return (rewards - mean_r) / (std_r + eps)


def reference_kl_penalty(policy_logps: torch.Tensor, ref_logps: torch.Tensor) -> torch.Tensor:
    delta = ref_logps - policy_logps
    return torch.exp(delta) - delta - 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace MiniMind reward/logprob/KL tensor relationships.")
    parser.add_argument("--vocab_size", type=int, default=64)
    parser.add_argument("--prompt_len", type=int, default=3)
    parser.add_argument("--response_len", type=int, default=3)
    parser.add_argument("--num_generations", type=int, default=3)
    args = parser.parse_args()

    random.seed(7)
    torch.manual_seed(7)

    output_ids = torch.tensor(
        [
            [11, 12, 13, 21, 22, 2],
            [14, 15, 16, 31, 2, 0],
        ],
        dtype=torch.long,
    )
    prompt_lens = torch.full((output_ids.size(0),), args.prompt_len, dtype=torch.long)
    completion_ids = output_ids[:, args.prompt_len:]
    response_mask = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    logp_pos = response_logp_positions(prompt_lens, args.response_len)

    logits_shape = (output_ids.size(0), output_ids.size(1), args.vocab_size)
    old_logits = torch.randn(logits_shape)
    policy_logits = old_logits + 0.15 * torch.randn(logits_shape)
    ref_logits = old_logits + 0.10 * torch.randn(logits_shape)

    old_logps = response_token_logps(old_logits, output_ids, logp_pos)
    policy_logps = response_token_logps(policy_logits, output_ids, logp_pos)
    ref_logps = response_token_logps(ref_logits, output_ids, logp_pos)

    log_ratio = policy_logps - old_logps
    ratio = torch.exp(log_ratio)
    approx_kl = masked_mean(0.5 * log_ratio.pow(2), response_mask)
    ref_kl = reference_kl_penalty(policy_logps, ref_logps)
    ref_kl_mean = masked_mean(ref_kl, response_mask)

    ppo_rewards = torch.tensor([2.0, -0.5])
    token_rewards = put_sequence_reward_on_last_token(ppo_rewards, response_mask)

    grpo_rewards = torch.tensor([2.0, 0.0, -1.0, 0.5, 0.5, 0.5])
    grpo_advantages = group_relative_advantages(grpo_rewards, args.num_generations)

    chosen_policy_seq = torch.tensor([-4.0, -2.5])
    rejected_policy_seq = torch.tensor([-5.0, -4.0])
    chosen_ref_seq = torch.tensor([-4.5, -3.0])
    rejected_ref_seq = torch.tensor([-4.8, -3.8])
    pi_logratios = chosen_policy_seq - rejected_policy_seq
    ref_logratios = chosen_ref_seq - rejected_ref_seq
    dpo_improvement = pi_logratios - ref_logratios

    print("[Token logprobs]")
    print(f"output_ids.shape={tuple(output_ids.shape)}")
    print(f"completion_ids={completion_ids.tolist()}")
    print(f"logp_pos={logp_pos.tolist()}")
    print(f"old_logps.shape={tuple(old_logps.shape)}")
    print(f"policy_logps.shape={tuple(policy_logps.shape)}")
    print(f"ref_logps.shape={tuple(ref_logps.shape)}")
    print(f"response_mask={response_mask.tolist()}")
    print()

    print("[Reward to advantage]")
    print(f"ppo_sequence_rewards={ppo_rewards.tolist()}")
    print(f"ppo_token_rewards={token_rewards.tolist()}")
    print(f"grpo_grouped_rewards={grpo_rewards.view(-1, args.num_generations).tolist()}")
    print(f"grpo_group_advantages={grpo_advantages.view(-1, args.num_generations).tolist()}")
    print()

    print("[KL and ratio]")
    print(f"ratio.shape={tuple(ratio.shape)}")
    print(f"ratio_valid_min={ratio[response_mask.bool()].min().item():.8f}")
    print(f"ratio_valid_max={ratio[response_mask.bool()].max().item():.8f}")
    print(f"approx_kl_policy_vs_old={approx_kl.item():.8f}")
    print(f"reference_kl_penalty_mean={ref_kl_mean.item():.8f}")
    print()

    print("[DPO contrast]")
    print(f"pi_logratios={pi_logratios.tolist()}")
    print(f"ref_logratios={ref_logratios.tolist()}")
    print(f"dpo_improvement={dpo_improvement.tolist()}")
    print("dpo_reference_usage=sequence_logratio_not_token_kl_penalty")


if __name__ == "__main__":
    main()
