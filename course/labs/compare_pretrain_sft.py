import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset.lm_dataset import PretrainDataset, SFTDataset, post_processing_chat, pre_processing_chat


def decode_label(tokenizer, token_id):
    if int(token_id) == -100:
        return "<ignored>"
    return tokenizer.decode([int(token_id)], skip_special_tokens=False).replace("\n", "\\n")


def print_table(title, tokenizer, input_ids, labels, rows):
    print(f"[{title}]")
    print(f"input_ids.shape={tuple(input_ids.shape)}")
    print(f"labels.shape={tuple(labels.shape)}")
    print(f"non_ignored_labels={(labels != -100).sum().item()}")
    print("idx | input_id | token | label | label_token")
    print("-" * 72)
    for i in range(min(rows, len(input_ids))):
        tid = int(input_ids[i])
        label = int(labels[i])
        token = tokenizer.decode([tid], skip_special_tokens=False).replace("\n", "\\n")
        print(f"{i:03d} | {tid:8d} | {token!r:16s} | {label:6d} | {decode_label(tokenizer, label)!r}")
    print()


def make_deterministic_sft_sample(tokenizer, data_path, index, max_length):
    ds = SFTDataset(data_path, tokenizer, max_length=max_length)
    with open(data_path, "r", encoding="utf-8") as f:
        raw_sample = json.loads(f.readlines()[index])

    conversations = pre_processing_chat(raw_sample["conversations"], add_system_ratio=0.0)
    prompt = ds.create_chat_prompt(conversations)
    prompt = post_processing_chat(prompt, empty_think_ratio=1.0)
    input_ids = tokenizer(prompt).input_ids[:max_length]
    input_ids += [tokenizer.pad_token_id] * (max_length - len(input_ids))
    labels = ds.generate_labels(input_ids)
    return prompt, torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser(description="Compare MiniMind pretrain and SFT dataset targets.")
    parser.add_argument("--tokenizer_path", default=str(ROOT / "model"))
    parser.add_argument("--pretrain_path", default=str(ROOT / "course/labs/tiny_pretrain.jsonl"))
    parser.add_argument("--sft_path", default=str(ROOT / "course/labs/tiny_sft.jsonl"))
    parser.add_argument("--pretrain_index", type=int, default=0)
    parser.add_argument("--sft_index", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=80)
    parser.add_argument("--show_rows", type=int, default=70)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    pretrain_ds = PretrainDataset(args.pretrain_path, tokenizer, max_length=args.max_length)
    pretrain_input_ids, pretrain_labels = pretrain_ds[args.pretrain_index]

    sft_prompt, sft_input_ids, sft_labels = make_deterministic_sft_sample(
        tokenizer, args.sft_path, args.sft_index, args.max_length
    )

    with open(args.pretrain_path, "r", encoding="utf-8") as f:
        pretrain_raw = json.loads(f.readlines()[args.pretrain_index])

    print("[Pretrain raw text]")
    print(pretrain_raw["text"])
    print()
    print_table("Pretrain input_ids / labels", tokenizer, pretrain_input_ids, pretrain_labels, args.show_rows)

    print("[SFT prompt]")
    print(sft_prompt)
    print()
    print_table("SFT input_ids / labels", tokenizer, sft_input_ids, sft_labels, args.show_rows)

    print("[Comparison]")
    print("Pretrain: labels mostly copy input_ids, except padding is -100.")
    print("SFT: labels are -100 for user/system/padding, and token ids only for assistant response spans.")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_printoptions(linewidth=160)
    main()
