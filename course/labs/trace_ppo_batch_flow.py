import argparse
import os
import random
import sys
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", message="urllib3 .* doesn't match a supported version.*")

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402
from trainer.rollout_engine import compute_per_token_logps  # noqa: E402


class TinyCritic(MiniMindForCausalLM):
    def __init__(self, params):
        super().__init__(params)
        self.value_head = nn.Linear(params.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = self.model.norm(outputs[0])
        return self.value_head(hidden_states).squeeze(-1)


def build_synthetic_rollout(batch_size, prompt_len, response_len, vocab_size, pad_id, eos_id, device):
    prompt_ids = torch.randint(3, vocab_size, (batch_size, prompt_len), device=device)
    completion_ids = torch.randint(3, vocab_size, (batch_size, response_len), device=device)
    completion_mask = torch.ones(batch_size, response_len, dtype=torch.long, device=device)

    for row in range(batch_size):
        valid_len = max(1, response_len - (row % 2))
        completion_ids[row, valid_len - 1] = eos_id
        if valid_len < response_len:
            completion_ids[row, valid_len:] = pad_id
            completion_mask[row, valid_len:] = 0

    output_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    prompt_lens = torch.full((batch_size,), prompt_len, dtype=torch.long, device=device)
    full_mask = (output_ids != pad_id).long()
    return output_ids, completion_ids, prompt_lens, full_mask, completion_mask


def masked_mean(x, mask):
    return (x * mask).sum() / mask.sum().clamp(min=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace the tensor flow of one MiniMind PPO update batch.")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--prompt_len", type=int, default=5)
    parser.add_argument("--response_len", type=int, default=4)
    parser.add_argument("--vocab_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--cliprange_value", type=float, default=0.2)
    parser.add_argument("--kl_coef", type=float, default=0.02)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(42)
    torch.manual_seed(42)

    pad_id = 0
    eos_id = 2
    device = torch.device(args.device)

    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        vocab_size=args.vocab_size,
        num_attention_heads=4,
        num_key_value_heads=2,
        flash_attn=False,
    )
    actor = MiniMindForCausalLM(config).to(device)
    ref = MiniMindForCausalLM(config).to(device)
    ref.load_state_dict(actor.state_dict())
    ref.eval().requires_grad_(False)
    critic = TinyCritic(config).to(device)

    gen_out, completion_ids, prompt_lens, full_mask, completion_mask = build_synthetic_rollout(
        args.batch_size,
        args.prompt_len,
        args.response_len,
        args.vocab_size,
        pad_id,
        eos_id,
        device,
    )

    actor.eval()
    with torch.no_grad():
        old_resp_logp = compute_per_token_logps(actor, gen_out, completion_ids.size(1), attention_mask=full_mask)

        labels = gen_out[:, 1:].clone()
        resp_idx = torch.arange(completion_ids.size(1), device=device).unsqueeze(0)
        logp_pos = prompt_lens.unsqueeze(1) - 1 + resp_idx
        resp_pad_mask = completion_mask.bool()
        resp_lengths = resp_pad_mask.sum(dim=1).long().clamp(min=1)
        eos_mask = completion_ids.eq(eos_id) & resp_pad_mask
        has_eos = eos_mask.any(dim=1)
        eos_pos = torch.argmax(eos_mask.int(), dim=1)
        resp_lengths = torch.where(has_eos, eos_pos + 1, resp_lengths).long().clamp(min=1)
        resp_policy_mask = ((resp_idx < resp_lengths.unsqueeze(1)) & resp_pad_mask).float()
        resp_value_mask = resp_policy_mask.clone()

        values_seq = critic(input_ids=gen_out, attention_mask=full_mask)
        old_resp_values = values_seq.gather(1, logp_pos) * resp_value_mask

        ref_logits = ref(input_ids=gen_out, attention_mask=full_mask).logits[:, :-1]
        ref_resp_logp = F.log_softmax(ref_logits, dim=-1).gather(
            2, labels.unsqueeze(-1)
        ).squeeze(-1).gather(1, logp_pos)

        rewards = torch.linspace(1.0, -0.5, args.batch_size, device=device)
        token_rewards = torch.zeros_like(old_resp_logp)
        last_idx = resp_lengths - 1
        token_rewards[torch.arange(args.batch_size, device=device), last_idx] += rewards

        lastgaelam = torch.zeros(args.batch_size, device=device)
        advs_rev = []
        for t in reversed(range(old_resp_values.size(1))):
            next_value = old_resp_values[:, t + 1] if t < old_resp_values.size(1) - 1 else 0.0
            delta = token_rewards[:, t] + args.gamma * next_value - old_resp_values[:, t]
            lastgaelam = delta + args.gamma * args.lam * lastgaelam
            advs_rev.append(lastgaelam)
        advantages = torch.stack(advs_rev[::-1], dim=1)
        returns = advantages + old_resp_values

        adv_mean = masked_mean(advantages, resp_policy_mask)
        adv_var = masked_mean((advantages - adv_mean) ** 2, resp_policy_mask)
        advantages = (advantages - adv_mean) * torch.rsqrt(adv_var + 1e-8) * resp_policy_mask

    actor.train()
    critic.train()
    res = actor(input_ids=gen_out, attention_mask=full_mask)
    current_logits = res.logits[:, :-1]
    new_resp_logp = F.log_softmax(current_logits, dim=-1).gather(
        2, labels.unsqueeze(-1)
    ).squeeze(-1).gather(1, logp_pos)

    current_values_seq = critic(input_ids=gen_out, attention_mask=full_mask)
    current_resp_values = current_values_seq.gather(1, logp_pos)

    log_ratio = new_resp_logp - old_resp_logp
    ratio = torch.exp(log_ratio)
    approx_kl = masked_mean(0.5 * log_ratio.pow(2), resp_policy_mask)
    clipped_ratio = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon)
    clipfrac = masked_mean(((ratio - 1.0).abs() > args.clip_epsilon).float(), resp_policy_mask)
    ref_delta = ref_resp_logp - new_resp_logp
    kl_ref_penalty = masked_mean(torch.exp(ref_delta) - ref_delta - 1.0, resp_policy_mask)
    policy_loss = (
        masked_mean(torch.max(-advantages * ratio, -advantages * clipped_ratio), resp_policy_mask)
        + args.kl_coef * kl_ref_penalty
    )

    value_unclipped = (current_resp_values - returns).pow(2)
    value_clipped = torch.clamp(
        current_resp_values,
        old_resp_values - args.cliprange_value,
        old_resp_values + args.cliprange_value,
    )
    value_loss = 0.5 * masked_mean(torch.max(value_unclipped, (value_clipped - returns).pow(2)), resp_value_mask)

    total_loss = policy_loss + args.vf_coef * value_loss
    total_loss.backward()

    print("[Rollout tensors]")
    print(f"gen_out.shape={tuple(gen_out.shape)}")
    print(f"completion_ids.shape={tuple(completion_ids.shape)}")
    print(f"full_mask.shape={tuple(full_mask.shape)}")
    print(f"old_resp_logp.shape={tuple(old_resp_logp.shape)}")
    print(f"completion_mask={completion_mask.tolist()}")
    print()
    print("[Response indexing]")
    print(f"prompt_lens={prompt_lens.tolist()}")
    print(f"logp_pos={logp_pos.tolist()}")
    print(f"resp_lengths={resp_lengths.tolist()}")
    print(f"resp_policy_mask={resp_policy_mask.tolist()}")
    print()
    print("[Reward and GAE]")
    print(f"rewards={rewards.tolist()}")
    print(f"token_rewards={token_rewards.tolist()}")
    print(f"advantages.shape={tuple(advantages.shape)}")
    print(f"returns.shape={tuple(returns.shape)}")
    print(f"masked_adv_mean={masked_mean(advantages, resp_policy_mask).item():.8f}")
    print()
    print("[PPO loss]")
    print(f"ratio_min={ratio[resp_policy_mask.bool()].min().item():.8f}")
    print(f"ratio_max={ratio[resp_policy_mask.bool()].max().item():.8f}")
    print(f"approx_kl={approx_kl.item():.8f}")
    print(f"clipfrac={clipfrac.item():.8f}")
    print(f"kl_ref_penalty={kl_ref_penalty.item():.8f}")
    print(f"policy_loss={policy_loss.item():.8f}")
    print(f"value_loss={value_loss.item():.8f}")
    print(f"total_loss={total_loss.item():.8f}")


if __name__ == "__main__":
    main()
