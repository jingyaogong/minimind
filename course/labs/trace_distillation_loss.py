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

from dataset.lm_dataset import SFTDataset  # noqa: E402
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402
from trainer.train_distillation import distillation_loss  # noqa: E402


def shape(x: torch.Tensor) -> tuple[int, ...]:
    return tuple(x.shape)


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace MiniMind distillation CE + KL loss on a tiny SFT sample.")
    parser.add_argument("--tokenizer_path", default=str(ROOT / "model"))
    parser.add_argument("--data_path", default=str(ROOT / "course/labs/tiny_sft.jsonl"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=96)
    parser.add_argument("--student_hidden_size", type=int, default=64)
    parser.add_argument("--student_num_layers", type=int, default=2)
    parser.add_argument("--teacher_hidden_size", type=int, default=96)
    parser.add_argument("--teacher_num_layers", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(42)
    torch.manual_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    dataset = SFTDataset(args.data_path, tokenizer, max_length=args.max_length)
    input_ids, labels = dataset[args.index]
    input_ids = input_ids.unsqueeze(0).to(args.device)
    labels = labels.unsqueeze(0).to(args.device)

    student_config = MiniMindConfig(
        hidden_size=args.student_hidden_size,
        num_hidden_layers=args.student_num_layers,
        vocab_size=len(tokenizer),
    )
    teacher_config = MiniMindConfig(
        hidden_size=args.teacher_hidden_size,
        num_hidden_layers=args.teacher_num_layers,
        vocab_size=len(tokenizer),
    )
    student = MiniMindForCausalLM(student_config).to(args.device).eval()
    teacher = MiniMindForCausalLM(teacher_config).to(args.device).eval()
    teacher.requires_grad_(False)

    with torch.no_grad():
        student_outputs = student(input_ids)
        teacher_outputs = teacher(input_ids)

    student_logits = student_outputs.logits[..., :-1, :].contiguous()
    teacher_logits = teacher_outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_mask = (shift_labels != -100)
    loss_mask_flat = loss_mask.view(-1)

    flat_student_logits = student_logits.view(-1, student_logits.size(-1))
    flat_teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))
    flat_labels = shift_labels.view(-1)

    per_token_ce = F.cross_entropy(
        flat_student_logits,
        flat_labels,
        ignore_index=-100,
        reduction="none",
    )
    ce_loss_raw = torch.sum(per_token_ce * loss_mask_flat.float()) / (loss_mask_flat.float().sum() + 1e-8)

    active_student_logits = flat_student_logits[loss_mask_flat]
    active_teacher_logits = flat_teacher_logits[loss_mask_flat]
    distill = distillation_loss(
        active_student_logits,
        active_teacher_logits,
        temperature=args.temperature,
    )
    total = args.alpha * ce_loss_raw + (1 - args.alpha) * distill

    print("[Config]")
    print(f"student_hidden_size={args.student_hidden_size}")
    print(f"teacher_hidden_size={args.teacher_hidden_size}")
    print(f"alpha={args.alpha}")
    print(f"temperature={args.temperature}")
    print()
    print("[Shapes]")
    print(f"input_ids.shape={shape(input_ids)}")
    print(f"labels.shape={shape(labels)}")
    print(f"student_logits.shape={shape(student_logits)}")
    print(f"teacher_logits.shape={shape(teacher_logits)}")
    print(f"shift_labels.shape={shape(shift_labels)}")
    print(f"active_student_logits.shape={shape(active_student_logits)}")
    print(f"active_teacher_logits.shape={shape(active_teacher_logits)}")
    print()
    print("[Loss]")
    print(f"active_tokens={int(loss_mask_flat.sum().item())}")
    print(f"ce_loss_raw={ce_loss_raw.item():.8f}")
    print(f"distill_loss={distill.item():.8f}")
    print(f"total_loss={total.item():.8f}")
    print(f"manual_total_formula={args.alpha:.2f}*CE + {1 - args.alpha:.2f}*Distill")


if __name__ == "__main__":
    main()
