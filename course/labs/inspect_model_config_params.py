import argparse
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


PRESETS = {
    "tiny": {
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 6400,
        "max_position_embeddings": 128,
    },
    "minimind3": {
        "hidden_size": 768,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "vocab_size": 6400,
        "max_position_embeddings": 32768,
    },
}


def fmt_params(value):
    return f"{value:,} ({value / 1_000_000:.3f}M)"


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def build_config(args, overrides=None):
    values = dict(PRESETS[args.preset])
    values.update(
        {
            "hidden_size": args.hidden_size or values["hidden_size"],
            "num_hidden_layers": args.num_hidden_layers or values["num_hidden_layers"],
            "num_attention_heads": args.num_attention_heads or values["num_attention_heads"],
            "num_key_value_heads": args.num_key_value_heads or values["num_key_value_heads"],
            "vocab_size": args.vocab_size or values["vocab_size"],
            "max_position_embeddings": args.max_position_embeddings or values["max_position_embeddings"],
            "use_moe": bool(args.use_moe),
            "tie_word_embeddings": bool(args.tie_word_embeddings),
            "dropout": 0.0,
        }
    )
    if overrides:
        values.update(overrides)
    return MiniMindConfig(**values)


def make_model(config, device):
    if device == "meta":
        with torch.device("meta"):
            return MiniMindForCausalLM(config)
    return MiniMindForCausalLM(config)


def collect_row(name, config, device):
    model = make_model(config, device)
    block0 = model.model.layers[0]
    token_matrix = model.lm_head.weight.numel()
    all_blocks = count_params(model.model.layers)
    return {
        "name": name,
        "hidden": config.hidden_size,
        "layers": config.num_hidden_layers,
        "vocab": config.vocab_size,
        "heads": config.num_attention_heads,
        "kv_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "tied": model.model.embed_tokens.weight is model.lm_head.weight,
        "total": count_params(model),
        "token": token_matrix,
        "block0": count_params(block0),
        "attn0": count_params(block0.self_attn),
        "mlp0": count_params(block0.mlp),
        "blocks": all_blocks,
    }


def print_single_report(config, device):
    model = make_model(config, device)
    block0 = model.model.layers[0]
    tied = model.model.embed_tokens.weight is model.lm_head.weight
    token_matrix_params = model.lm_head.weight.numel()

    print("[Config]")
    print(f"hidden_size={config.hidden_size}")
    print(f"num_hidden_layers={config.num_hidden_layers}")
    print(f"vocab_size={config.vocab_size}")
    print(f"num_attention_heads={config.num_attention_heads}")
    print(f"num_key_value_heads={config.num_key_value_heads}")
    print(f"head_dim={config.head_dim}")
    print(f"intermediate_size={config.intermediate_size}")
    print(f"use_moe={config.use_moe}")
    print(f"tie_word_embeddings={config.tie_word_embeddings}")
    print()

    print("[Important Shapes]")
    print(f"embed_tokens.weight={tuple(model.model.embed_tokens.weight.shape)}")
    print(f"lm_head.weight={tuple(model.lm_head.weight.shape)}")
    print(f"embedding_and_lm_head_are_same_parameter={tied}")
    print(f"hidden_states=[batch_size, seq_len, {config.hidden_size}]")
    print(f"attention q heads={config.num_attention_heads}, kv heads={config.num_key_value_heads}, head_dim={config.head_dim}")
    print()

    print("[Parameter Breakdown]")
    print(f"total_unique_params={fmt_params(count_params(model))}")
    print(f"token_matrix_params_once={fmt_params(token_matrix_params)}")
    print(f"all_blocks={fmt_params(count_params(model.model.layers))}")
    print(f"block0_total={fmt_params(count_params(block0))}")
    print(f"block0.self_attn={fmt_params(count_params(block0.self_attn))}")
    print(f"block0.mlp={fmt_params(count_params(block0.mlp))}")
    print(f"final_norm={fmt_params(count_params(model.model.norm))}")
    print()

    print("[Simple Formulas]")
    print(f"vocab_size * hidden_size = {config.vocab_size} * {config.hidden_size} = {fmt_params(config.vocab_size * config.hidden_size)}")
    print(f"num_hidden_layers * block0_total = {config.num_hidden_layers} * {count_params(block0):,} = {fmt_params(config.num_hidden_layers * count_params(block0))}")
    print(f"q projection output dim = num_attention_heads * head_dim = {config.num_attention_heads} * {config.head_dim} = {config.num_attention_heads * config.head_dim}")


def print_compare(args):
    base = build_config(args)
    variants = [
        ("base", {}),
        ("wider_hidden", {"hidden_size": base.hidden_size + 32}),
        ("deeper_layers", {"num_hidden_layers": base.num_hidden_layers + 2}),
        ("larger_vocab", {"vocab_size": base.vocab_size + 1600}),
        ("untied_lm_head", {"tie_word_embeddings": False}),
    ]
    rows = [collect_row(name, build_config(args, overrides), args.device) for name, overrides in variants]

    print("name | hidden | layers | vocab | heads/kv/head_dim | tied | total | token | block0 | attn0 | mlp0")
    print("-" * 128)
    for row in rows:
        print(
            f"{row['name']:15s} | "
            f"{row['hidden']:6d} | "
            f"{row['layers']:6d} | "
            f"{row['vocab']:5d} | "
            f"{row['heads']}/{row['kv_heads']}/{row['head_dim']:<6d} | "
            f"{str(row['tied']):5s} | "
            f"{row['total']:10d} | "
            f"{row['token']:9d} | "
            f"{row['block0']:9d} | "
            f"{row['attn0']:9d} | "
            f"{row['mlp0']:9d}"
        )


def main():
    parser = argparse.ArgumentParser(description="Inspect MiniMind config, tensor shapes, and parameter counts.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="tiny")
    parser.add_argument("--device", choices=["cpu", "meta"], default="cpu")
    parser.add_argument("--hidden_size", type=int)
    parser.add_argument("--num_hidden_layers", type=int)
    parser.add_argument("--num_attention_heads", type=int)
    parser.add_argument("--num_key_value_heads", type=int)
    parser.add_argument("--vocab_size", type=int)
    parser.add_argument("--max_position_embeddings", type=int)
    parser.add_argument("--use_moe", type=int, choices=[0, 1], default=0)
    parser.add_argument("--tie_word_embeddings", type=int, choices=[0, 1], default=1)
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    if args.compare:
        print_compare(args)
    else:
        print_single_report(build_config(args), args.device)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
