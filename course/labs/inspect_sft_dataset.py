import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset.lm_dataset import SFTDataset, post_processing_chat, pre_processing_chat


def decode_one(tokenizer, token_id):
    if token_id == -100:
        return "<ignored>"
    return tokenizer.decode([int(token_id)], skip_special_tokens=False).replace("\n", "\\n")


def main():
    parser = argparse.ArgumentParser(description="Inspect MiniMind SFTDataset prompt and labels.")
    parser.add_argument("--data_path", default=str(ROOT / "course/labs/tiny_sft.jsonl"))
    parser.add_argument("--tokenizer_path", default=str(ROOT / "model"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=96)
    parser.add_argument("--show_rows", type=int, default=80)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_length)

    with open(args.data_path, "r", encoding="utf-8") as f:
        raw_sample = json.loads(f.readlines()[args.index])

    conversations = pre_processing_chat(raw_sample["conversations"], add_system_ratio=0.0)
    prompt = ds.create_chat_prompt(conversations)
    prompt = post_processing_chat(prompt, empty_think_ratio=1.0)
    input_ids = tokenizer(prompt).input_ids[: args.max_length]
    input_ids += [tokenizer.pad_token_id] * (args.max_length - len(input_ids))
    labels = ds.generate_labels(input_ids)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    print("[Raw conversations]")
    print(json.dumps(raw_sample["conversations"], ensure_ascii=False, indent=2))
    print()

    print("[Chat prompt]")
    print(prompt)
    print()

    print("[Shapes]")
    print(f"input_ids.shape={tuple(input_ids.shape)}")
    print(f"labels.shape={tuple(labels.shape)}")
    print(f"non_ignored_labels={(labels != -100).sum().item()}")
    print()

    print("[Assistant span markers]")
    print(f"bos_id={ds.bos_id} -> {tokenizer.convert_ids_to_tokens(ds.bos_id)}")
    print(f"eos_id={ds.eos_id} -> {tokenizer.convert_ids_to_tokens(ds.eos_id)}")
    print()

    print("[Token / label table]")
    print("idx | input_id | token | label | label_token")
    print("-" * 72)
    rows = min(args.show_rows, args.max_length)
    for i in range(rows):
        tid = int(input_ids[i])
        label = int(labels[i])
        token = tokenizer.decode([tid], skip_special_tokens=False).replace("\n", "\\n")
        label_token = decode_one(tokenizer, label)
        print(f"{i:03d} | {tid:8d} | {token!r:16s} | {label:6d} | {label_token!r}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_printoptions(linewidth=160)
    main()
