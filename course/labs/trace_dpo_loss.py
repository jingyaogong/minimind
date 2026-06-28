import argparse
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset.lm_dataset import DPODataset  # noqa: E402
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402
from trainer.train_dpo import dpo_loss as source_dpo_loss  # noqa: E402
from trainer.train_dpo import logits_to_log_probs  # noqa: E402


def sequence_log_probs(token_log_probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (token_log_probs * mask).sum(dim=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace MiniMind DPO loss from token logprobs to -logsigmoid.")
    parser.add_argument("--tokenizer_path", default=str(ROOT / "model"))
    parser.add_argument("--data_path", default=str(ROOT / "course/labs/tiny_dpo.jsonl"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=96)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--beta", type=float, default=0.15)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(42)
    torch.manual_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    dataset = DPODataset(args.data_path, tokenizer, max_length=args.max_length)
    sample = dataset[args.index]

    x_chosen = sample["x_chosen"].unsqueeze(0).to(args.device)
    y_chosen = sample["y_chosen"].unsqueeze(0).to(args.device)
    mask_chosen = sample["mask_chosen"].unsqueeze(0).to(args.device)
    x_rejected = sample["x_rejected"].unsqueeze(0).to(args.device)
    y_rejected = sample["y_rejected"].unsqueeze(0).to(args.device)
    mask_rejected = sample["mask_rejected"].unsqueeze(0).to(args.device)

    x = torch.cat([x_chosen, x_rejected], dim=0)
    y = torch.cat([y_chosen, y_rejected], dim=0)
    mask = torch.cat([mask_chosen, mask_rejected], dim=0)

    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        vocab_size=len(tokenizer),
    )
    policy = MiniMindForCausalLM(config).to(args.device).eval()
    ref = MiniMindForCausalLM(config).to(args.device).eval()
    ref.load_state_dict(policy.state_dict())
    ref.requires_grad_(False)

    with torch.no_grad():
        ref_logits = ref(x).logits
    policy_logits = policy(x).logits

    ref_token_logps = logits_to_log_probs(ref_logits, y)
    policy_token_logps = logits_to_log_probs(policy_logits, y)

    ref_seq_logps = sequence_log_probs(ref_token_logps, mask)
    policy_seq_logps = sequence_log_probs(policy_token_logps, mask)

    half = x.shape[0] // 2
    chosen_ref_logps = ref_seq_logps[:half]
    rejected_ref_logps = ref_seq_logps[half:]
    chosen_policy_logps = policy_seq_logps[:half]
    rejected_policy_logps = policy_seq_logps[half:]

    pi_logratios = chosen_policy_logps - rejected_policy_logps
    ref_logratios = chosen_ref_logps - rejected_ref_logps
    dpo_logits = pi_logratios - ref_logratios
    manual_loss_per_pair = -F.logsigmoid(args.beta * dpo_logits)
    manual_loss = manual_loss_per_pair.mean()
    source_loss = source_dpo_loss(ref_token_logps, policy_token_logps, mask, beta=args.beta)

    print("[Shapes]")
    print(f"x.shape={tuple(x.shape)}")
    print(f"y.shape={tuple(y.shape)}")
    print(f"mask.shape={tuple(mask.shape)}")
    print(f"policy_token_logps.shape={tuple(policy_token_logps.shape)}")
    print(f"ref_token_logps.shape={tuple(ref_token_logps.shape)}")
    print(f"policy_seq_logps.shape={tuple(policy_seq_logps.shape)}")
    print()
    print("[Sequence Logprobs]")
    print(f"chosen_policy_logp={chosen_policy_logps[0].item():.8f}")
    print(f"rejected_policy_logp={rejected_policy_logps[0].item():.8f}")
    print(f"chosen_ref_logp={chosen_ref_logps[0].item():.8f}")
    print(f"rejected_ref_logp={rejected_ref_logps[0].item():.8f}")
    print()
    print("[DPO]")
    print(f"beta={args.beta}")
    print(f"pi_logratio={pi_logratios[0].item():.8f}")
    print(f"ref_logratio={ref_logratios[0].item():.8f}")
    print(f"dpo_logits={dpo_logits[0].item():.8f}")
    print(f"manual_loss_per_pair={manual_loss_per_pair[0].item():.8f}")
    print(f"manual_dpo_loss={manual_loss.item():.8f}")
    print(f"source_dpo_loss={source_loss.item():.8f}")
    print(f"abs_diff={abs(manual_loss.item() - source_loss.item()):.12f}")


if __name__ == "__main__":
    main()
