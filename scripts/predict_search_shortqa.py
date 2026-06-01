import argparse
import json
import os
import sys
import time

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.search_shortqa_dataset import build_search_shortqa_messages
from model.model_minimind import MiniMindForCausalLM
from model.model_minimind_mla import MiniMindMLAConfig, MiniMindMLAForCausalLM
from trainer.trainer_utils import (
    add_model_profile_args,
    apply_model_profile,
    build_lm_config,
    get_model_params,
    get_model_suffix,
    resolve_attention_type,
)


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_model(args):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("predict_search_shortqa.py requires transformers. Install project requirements before inference.") from exc
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if args.load_from == "model" or os.path.basename(args.load_from.rstrip("/")) == "model":
        config = build_lm_config(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            attention_type=resolve_attention_type(args),
            kv_lora_rank=args.kv_lora_rank,
            q_lora_rank=args.q_lora_rank,
            rope_dim=args.rope_dim,
            num_attention_heads=args.num_attention_heads,
            num_key_value_heads=args.num_key_value_heads,
            intermediate_size=args.intermediate_size,
        )
        model = MiniMindMLAForCausalLM(config) if isinstance(config, MiniMindMLAConfig) else MiniMindForCausalLM(config)
        model_suffix = get_model_suffix(config)
        ckp = os.path.join(args.save_dir, f"{args.weight}_{args.hidden_size}{model_suffix}.pth")
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    get_model_params(model, model.config)
    model = model.eval().to(args.device)
    if args.dtype == "float16" and "cuda" in args.device:
        model = model.half()
    elif args.dtype == "bfloat16" and "cuda" in args.device:
        model = model.to(torch.bfloat16)
    return model, tokenizer


@torch.inference_mode()
def generate_one(model, tokenizer, sample, args):
    messages = build_search_shortqa_messages(
        sample,
        max_contexts=args.max_contexts,
        max_chars_per_context=args.max_chars_per_context,
    )
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        open_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_seq_len).to(args.device)
    start = time.time()
    output_ids = model.generate(
        inputs=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        max_new_tokens=args.max_new_tokens,
        do_sample=bool(args.do_sample),
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
    )
    elapsed = max(time.time() - start, 1e-6)
    new_tokens = output_ids.size(1) - inputs["input_ids"].size(1)
    response = tokenizer.decode(output_ids[0][inputs["input_ids"].size(1):], skip_special_tokens=True).strip()
    return response, new_tokens / elapsed, new_tokens


def main():
    parser = argparse.ArgumentParser(description="Generate SearchShortQA prediction JSONL")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--load_from", default="model")
    parser.add_argument("--save_dir", default="out")
    parser.add_argument("--weight", default="search_grpo")
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
    parser.add_argument("--max_seq_len", default=1536, type=int)
    parser.add_argument("--max_new_tokens", default=128, type=int)
    parser.add_argument("--max_contexts", default=6, type=int)
    parser.add_argument("--max_chars_per_context", default=700, type=int)
    parser.add_argument("--temperature", default=0.2, type=float)
    parser.add_argument("--top_p", default=0.9, type=float)
    parser.add_argument("--top_k", default=50, type=int)
    parser.add_argument("--do_sample", default=0, type=int)
    parser.add_argument("--repetition_penalty", default=1.05, type=float)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", default=0, type=int)
    add_model_profile_args(parser)
    args = parser.parse_args()
    apply_model_profile(args)

    model, tokenizer = load_model(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    total_tokens, total_tps, count = 0, 0.0, 0
    with open(args.output, "w", encoding="utf-8") as out:
        for sample in iter_jsonl(args.data):
            if args.limit and count >= args.limit:
                break
            response, tps, new_tokens = generate_one(model, tokenizer, sample, args)
            row = {
                "id": sample.get("id"),
                "question": sample.get("question"),
                "prediction": response,
                "tokens": new_tokens,
                "tokens_per_second": tps,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            total_tokens += new_tokens
            total_tps += tps
            count += 1
            if count % 20 == 0:
                print(f"generated={count}, avg_tokens/s={total_tps / count:.2f}")
    if count:
        print(f"Wrote {count} predictions to {args.output}")
        print(f"avg_tokens/s={total_tps / count:.2f}, total_new_tokens={total_tokens}")


if __name__ == "__main__":
    main()
