import argparse
import random

import torch


def masked_sequence_mean(per_token_loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def group_relative_advantages(rewards: torch.Tensor, num_generations: int, eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grouped_rewards = rewards.view(-1, num_generations)
    mean_r = grouped_rewards.mean(dim=1).repeat_interleave(num_generations)
    std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(num_generations)
    advantages = (rewards - mean_r) / (std_r + eps)
    return advantages, mean_r, std_r


def token_kl(policy_logps: torch.Tensor, ref_logps: torch.Tensor) -> torch.Tensor:
    kl_div = ref_logps - policy_logps
    return torch.exp(kl_div) - kl_div - 1.0


def grpo_loss(
    policy_logps: torch.Tensor,
    old_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ratio = torch.exp(policy_logps - old_logps)
    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
    per_token_kl = token_kl(policy_logps, ref_logps)
    loss1 = ratio * advantages.unsqueeze(1)
    loss2 = clipped_ratio * advantages.unsqueeze(1)
    per_token_loss = -(torch.min(loss1, loss2) - beta * per_token_kl)
    return masked_sequence_mean(per_token_loss, mask).mean(), ratio, per_token_kl


def cispo_loss(
    policy_logps: torch.Tensor,
    old_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    epsilon_high: float,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ratio = torch.exp(policy_logps - old_logps)
    clamped_ratio = torch.clamp(ratio, max=epsilon_high).detach()
    per_token_kl = token_kl(policy_logps, ref_logps)
    per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * policy_logps - beta * per_token_kl)
    return masked_sequence_mean(per_token_loss, mask).mean(), ratio, per_token_kl


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace MiniMind GRPO/CISPO group advantages and losses.")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_generations", type=int, default=3)
    parser.add_argument("--response_len", type=int, default=4)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--epsilon_high", type=float, default=5.0)
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)

    total = args.batch_size * args.num_generations
    rewards = torch.tensor([2.0, 1.0, -1.0, 0.5, 0.5, 0.5], dtype=torch.float32)[:total]
    if rewards.numel() < total:
        rewards = torch.linspace(-1.0, 1.0, total)

    old_logps = torch.randn(total, args.response_len) * 0.2 - 1.5
    policy_logps = old_logps + torch.randn(total, args.response_len) * 0.15
    ref_logps = old_logps + torch.randn(total, args.response_len) * 0.10
    completion_mask = torch.ones(total, args.response_len)
    completion_mask[1, -1] = 0
    completion_mask[-1, -2:] = 0

    advantages, mean_r, std_r = group_relative_advantages(rewards, args.num_generations, eps=1e-4)
    grpo, grpo_ratio, grpo_kl = grpo_loss(
        policy_logps,
        old_logps,
        ref_logps,
        advantages,
        completion_mask,
        args.epsilon,
        args.beta,
    )
    cispo, cispo_ratio, cispo_kl = cispo_loss(
        policy_logps,
        old_logps,
        ref_logps,
        advantages,
        completion_mask,
        args.epsilon_high,
        args.beta,
    )

    grouped_adv = advantages.view(args.batch_size, args.num_generations)
    print("[Group rewards]")
    print(f"rewards.shape={tuple(rewards.shape)}")
    print(f"grouped_rewards={rewards.view(args.batch_size, args.num_generations).tolist()}")
    print(f"group_mean={mean_r.view(args.batch_size, args.num_generations)[:, 0].tolist()}")
    print(f"group_std={std_r.view(args.batch_size, args.num_generations)[:, 0].tolist()}")
    print(f"advantages={grouped_adv.tolist()}")
    print(f"group_adv_mean={grouped_adv.mean(dim=1).tolist()}")
    print()
    print("[Token tensors]")
    print(f"policy_logps.shape={tuple(policy_logps.shape)}")
    print(f"old_logps.shape={tuple(old_logps.shape)}")
    print(f"ref_logps.shape={tuple(ref_logps.shape)}")
    print(f"completion_mask={completion_mask.tolist()}")
    print()
    print("[Losses]")
    print(f"ratio_min={grpo_ratio[completion_mask.bool()].min().item():.8f}")
    print(f"ratio_max={grpo_ratio[completion_mask.bool()].max().item():.8f}")
    print(f"kl_mean={(grpo_kl * completion_mask).sum().item() / completion_mask.sum().item():.8f}")
    print(f"grpo_loss={grpo.item():.8f}")
    print(f"cispo_loss={cispo.item():.8f}")
    print(f"ratio_tensors_match={torch.allclose(grpo_ratio, cispo_ratio)}")
    print(f"kl_tensors_match={torch.allclose(grpo_kl, cispo_kl)}")


if __name__ == "__main__":
    main()
