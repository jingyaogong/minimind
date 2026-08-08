import argparse
import os
import sys
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset.lm_dataset import PretrainDataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import get_lr


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser(description="Trace one MiniMind pretrain step on CPU.")
    parser.add_argument("--tokenizer_path", default=str(ROOT / "model"))
    parser.add_argument("--data_path", default=str(ROOT / "course/labs/tiny_pretrain.jsonl"))
    parser.add_argument("--max_length", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    args = parser.parse_args()

    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        vocab_size=len(tokenizer),
        dropout=0.0,
    )
    model = MiniMindForCausalLM(config).train()
    dataset = PretrainDataset(args.data_path, tokenizer, max_length=args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    input_ids, labels = next(iter(loader))
    lr = get_lr(current_step=1, total_steps=10, lr=args.learning_rate)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    first_param = next(model.parameters()).detach().clone()
    outputs = model(input_ids, labels=labels)
    loss = outputs.loss + outputs.aux_loss
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    param_delta = (next(model.parameters()).detach() - first_param).abs().max().item()

    print("[Config]")
    print(f"hidden_size={config.hidden_size}")
    print(f"num_hidden_layers={config.num_hidden_layers}")
    print(f"vocab_size={config.vocab_size}")
    print(f"params={count_params(model):,}")
    print()
    print("[Batch]")
    print(f"input_ids.shape={tuple(input_ids.shape)}")
    print(f"labels.shape={tuple(labels.shape)}")
    print(f"non_ignored_labels={(labels != -100).sum().item()}")
    print()
    print("[Forward]")
    print(f"logits.shape={tuple(outputs.logits.shape)}")
    print(f"loss={outputs.loss.item():.6f}")
    print(f"aux_loss={outputs.aux_loss.item():.6f}")
    print()
    print("[Backward / Optimizer]")
    print(f"lr={lr:.8f}")
    print(f"grad_norm_before_clip={float(grad_norm):.6f}")
    print(f"max_first_param_delta_after_step={param_delta:.10f}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_printoptions(linewidth=160)
    main()
