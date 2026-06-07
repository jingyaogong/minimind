import argparse
import json
import math
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trainer.trainer_utils import (  # noqa: E402
    add_model_profile_args,
    apply_model_profile,
    build_lm_config,
    init_model,
    resolve_attention_type,
)


class JsonlPretrainEvalDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=1024, max_samples=2000, start_index=0):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        max_samples = int(max_samples)
        start_index = int(start_index)

        with open(data_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if line_idx < start_index:
                    continue
                if max_samples > 0 and len(self.samples) >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                text = str(item.get("text", ""))
                if text:
                    self.samples.append(text)

        if not self.samples:
            raise ValueError(f"No eval samples loaded from {data_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        tokens = self.tokenizer(
            self.samples[index],
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True,
        ).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate pretrain checkpoint loss/PPL on JSONL text samples")
    parser.add_argument("--data_path", required=True, help="JSONL file with a text field")
    parser.add_argument("--save_dir", default="out", help="Directory containing native .pth weights")
    parser.add_argument("--weight", required=True, help="Weight prefix, e.g. pretrain_searchlm_300m_v1")
    parser.add_argument("--tokenizer_path", default="model")
    parser.add_argument("--output", default="", help="Optional JSON output path")
    parser.add_argument("--max_samples", type=int, default=2000, help="Number of eval samples; <=0 means all after start_index")
    parser.add_argument("--start_index", type=int, default=0, help="Skip this many JSONL rows before evaluation")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--num_attention_heads", default=8, type=int)
    parser.add_argument("--num_key_value_heads", default=4, type=int)
    parser.add_argument("--intermediate_size", default=None, type=int)
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1])
    parser.add_argument("--attention_type", default="gqa", choices=["gqa", "mha", "mqa", "mla"])
    parser.add_argument("--use_mla", default=0, type=int, choices=[0, 1])
    parser.add_argument("--kv_lora_rank", default=128, type=int)
    parser.add_argument("--q_lora_rank", default=256, type=int)
    parser.add_argument("--rope_dim", default=None, type=int)
    add_model_profile_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    apply_model_profile(args)
    attention_type = resolve_attention_type(args)
    lm_config = build_lm_config(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        attention_type=attention_type,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        intermediate_size=args.intermediate_size,
        kv_lora_rank=args.kv_lora_rank,
        q_lora_rank=args.q_lora_rank,
        rope_dim=args.rope_dim,
    )

    model, tokenizer = init_model(
        lm_config,
        from_weight=args.weight,
        tokenizer_path=args.tokenizer_path,
        save_dir=args.save_dir,
        device=args.device,
    )
    if args.device.startswith("cuda") and args.dtype != "float32":
        dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
        model = model.to(dtype=dtype)
    model.eval()

    dataset = JsonlPretrainEvalDataset(
        args.data_path,
        tokenizer,
        max_length=args.max_seq_len,
        max_samples=args.max_samples,
        start_index=args.start_index,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    total_nll = 0.0
    total_tokens = 0
    start_time = time.time()
    with torch.inference_mode():
        for step, (input_ids, labels) in enumerate(loader, start=1):
            input_ids = input_ids.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            outputs = model(input_ids, labels=labels)
            valid_tokens = (labels[..., 1:] != -100).sum().item()
            total_nll += float(outputs.loss.item()) * valid_tokens
            total_tokens += valid_tokens

            if step % args.log_interval == 0 or step == len(loader):
                loss = total_nll / max(total_tokens, 1)
                ppl = math.exp(loss) if loss < 100 else float("inf")
                print(
                    f"Eval step [{step}/{len(loader)}], "
                    f"loss: {loss:.4f}, ppl: {ppl:.2f}, tokens: {total_tokens}"
                )

    loss = total_nll / max(total_tokens, 1)
    ppl = math.exp(loss) if loss < 100 else float("inf")
    result = {
        "weight": args.weight,
        "model_profile": args.model_profile,
        "data_path": args.data_path,
        "start_index": args.start_index,
        "num_samples": len(dataset),
        "max_seq_len": args.max_seq_len,
        "batch_size": args.batch_size,
        "loss": loss,
        "ppl": ppl,
        "tokens": total_tokens,
        "elapsed_sec": time.time() - start_time,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
