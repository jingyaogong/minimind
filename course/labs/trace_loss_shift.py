import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset.lm_dataset import PretrainDataset, SFTDataset, post_processing_chat, pre_processing_chat
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def decode_token(tokenizer, token_id):
    if int(token_id) == -100:
        return "<ignored>"
    return tokenizer.decode([int(token_id)], skip_special_tokens=False).replace("\n", "\\n")


def make_pretrain_sample(tokenizer, data_path, index, max_length):
    dataset = PretrainDataset(data_path, tokenizer, max_length=max_length)
    input_ids, labels = dataset[index]
    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.loads(f.readlines()[index])
    title = "Pretrain"
    text = raw["text"]
    return title, text, input_ids, labels


def make_sft_sample(tokenizer, data_path, index, max_length):
    dataset = SFTDataset(data_path, tokenizer, max_length=max_length)
    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.loads(f.readlines()[index])

    conversations = pre_processing_chat(raw["conversations"], add_system_ratio=0.0)
    prompt = dataset.create_chat_prompt(conversations)
    prompt = post_processing_chat(prompt, empty_think_ratio=1.0)
    input_ids = tokenizer(prompt).input_ids[:max_length]
    input_ids += [tokenizer.pad_token_id] * (max_length - len(input_ids))
    labels = dataset.generate_labels(input_ids)

    title = "SFT"
    return title, prompt, torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def print_alignment(tokenizer, input_ids, labels, show_rows):
    print("[Shift Alignment]")
    print("idx | logits position input token | target label position | participates | label token")
    print("-" * 104)
    rows = min(show_rows, len(input_ids) - 1)
    for i in range(rows):
        label = int(labels[i + 1])
        participates = label != -100
        print(
            f"{i:03d} | "
            f"logits[{i:03d}] from input_ids[{i:03d}]={int(input_ids[i]):5d} {decode_token(tokenizer, input_ids[i])!r:18s} | "
            f"labels[{i + 1:03d}]={label:6d} | "
            f"{str(participates):11s} | "
            f"{decode_token(tokenizer, label)!r}"
        )
    print()
    print("Note:")
    print("- labels[0] is never used because the loss reads labels[..., 1:].")
    print("- logits at the final sequence position is never used because the loss reads logits[..., :-1, :].")
    print()


def main():
    parser = argparse.ArgumentParser(description="Trace MiniMind causal LM loss shift and ignore_index=-100.")
    parser.add_argument("--mode", choices=["sft", "pretrain"], default="sft")
    parser.add_argument("--tokenizer_path", default=str(ROOT / "model"))
    parser.add_argument("--pretrain_path", default=str(ROOT / "course/labs/tiny_pretrain.jsonl"))
    parser.add_argument("--sft_path", default=str(ROOT / "course/labs/tiny_sft.jsonl"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=80)
    parser.add_argument("--show_rows", type=int, default=60)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    args = parser.parse_args()

    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    if args.mode == "pretrain":
        title, source_text, input_ids, labels = make_pretrain_sample(
            tokenizer, args.pretrain_path, args.index, args.max_length
        )
    else:
        title, source_text, input_ids, labels = make_sft_sample(
            tokenizer, args.sft_path, args.index, args.max_length
        )

    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=len(tokenizer),
        max_position_embeddings=128,
        dropout=0.0,
    )
    model = MiniMindForCausalLM(config).eval()

    batch_input_ids = input_ids.unsqueeze(0)
    batch_labels = labels.unsqueeze(0)

    with torch.no_grad():
        outputs = model(batch_input_ids, labels=batch_labels)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch_labels[..., 1:].contiguous()
        manual_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        flat_labels = shift_labels.view(-1)
        active_mask = flat_labels != -100
        active_labels = flat_labels[active_mask]
        active_log_probs = F.log_softmax(shift_logits.view(-1, shift_logits.size(-1))[active_mask], dim=-1)
        per_token_loss = -active_log_probs[torch.arange(active_labels.numel()), active_labels]

    print(f"[{title} Source]")
    print(source_text)
    print()

    print("[Shapes]")
    print(f"input_ids.shape={tuple(batch_input_ids.shape)}")
    print(f"labels.shape={tuple(batch_labels.shape)}")
    print(f"logits.shape={tuple(logits.shape)}")
    print(f"shift_logits.shape={tuple(shift_logits.shape)}")
    print(f"shift_labels.shape={tuple(shift_labels.shape)}")
    print()

    print("[Loss]")
    print(f"model_forward_loss={outputs.loss.item():.8f}")
    print(f"manual_shift_loss={manual_loss.item():.8f}")
    print(f"abs_diff={abs(outputs.loss.item() - manual_loss.item()):.12f}")
    print(f"active_loss_positions={int(active_mask.sum().item())}")
    print(f"ignored_loss_positions={int((~active_mask).sum().item())}")
    if per_token_loss.numel() > 0:
        print(f"mean_active_per_token_loss={per_token_loss.mean().item():.8f}")
    print()

    print_alignment(tokenizer, input_ids, labels, args.show_rows)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
