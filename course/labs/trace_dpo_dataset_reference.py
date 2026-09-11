import argparse
import os
import random
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset.lm_dataset import DPODataset  # noqa: E402
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402
from trainer.train_dpo import logits_to_log_probs  # noqa: E402


def shape(x: torch.Tensor) -> tuple[int, ...]:
    return tuple(x.shape)


def show_active_tokens(tokenizer, y: torch.Tensor, mask: torch.Tensor, label: str, limit: int = 16) -> None:
    active_positions = torch.nonzero(mask, as_tuple=False).flatten().tolist()
    print(f"{label}_active_positions_first={active_positions[:limit]}")
    token_pieces = []
    for pos in active_positions[:limit]:
        token_id = int(y[pos].item())
        token_pieces.append(tokenizer.decode([token_id]))
    print(f"{label}_active_label_tokens_first={token_pieces}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace MiniMind DPO dataset and reference-model logprob flow.")
    parser.add_argument("--tokenizer_path", default=str(ROOT / "model"))
    parser.add_argument("--data_path", default=str(ROOT / "course/labs/tiny_dpo.jsonl"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=96)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
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

    ref_log_probs = logits_to_log_probs(ref_logits, y)
    policy_log_probs = logits_to_log_probs(policy_logits, y)
    diff = (policy_log_probs - ref_log_probs).abs().max().item()

    print("[Dataset]")
    print(f"num_samples={len(dataset)}")
    print(f"x_chosen.shape={shape(x_chosen)}")
    print(f"y_chosen.shape={shape(y_chosen)}")
    print(f"mask_chosen.shape={shape(mask_chosen)}")
    print(f"x_rejected.shape={shape(x_rejected)}")
    print(f"y_rejected.shape={shape(y_rejected)}")
    print(f"mask_rejected.shape={shape(mask_rejected)}")
    print(f"mask_chosen_active={int(mask_chosen.sum().item())}")
    print(f"mask_rejected_active={int(mask_rejected.sum().item())}")
    show_active_tokens(tokenizer, y_chosen[0].cpu(), mask_chosen[0].cpu(), "chosen")
    show_active_tokens(tokenizer, y_rejected[0].cpu(), mask_rejected[0].cpu(), "rejected")
    print()
    print("[Concat]")
    print(f"x_cat.shape={shape(x)}")
    print(f"y_cat.shape={shape(y)}")
    print(f"mask_cat.shape={shape(mask)}")
    print()
    print("[Policy / Reference]")
    print(f"policy_logits.shape={shape(policy_logits)}")
    print(f"ref_logits.shape={shape(ref_logits)}")
    print(f"policy_log_probs.shape={shape(policy_log_probs)}")
    print(f"ref_log_probs.shape={shape(ref_log_probs)}")
    print(f"initial_policy_ref_logprob_max_abs_diff={diff:.12f}")
    print(f"ref_requires_grad={any(p.requires_grad for p in ref.parameters())}")
    print(f"policy_requires_grad={any(p.requires_grad for p in policy.parameters())}")


if __name__ == "__main__":
    main()
