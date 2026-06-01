import argparse
import json
import math
import os


def round_intermediate(hidden_size):
    return math.ceil(hidden_size * math.pi / 64) * 64


def dense_params(cfg):
    h = cfg["hidden_size"]
    layers = cfg["num_hidden_layers"]
    vocab = cfg.get("vocab_size", 6400)
    heads = cfg.get("num_attention_heads", 8)
    kv_heads = cfg.get("num_key_value_heads", heads // 2)
    head_dim = cfg.get("head_dim", h // heads)
    intermediate = cfg.get("intermediate_size", round_intermediate(h))
    attention_type = cfg.get("attention_type", "gqa")

    embed = vocab * h
    if attention_type == "mla":
        kv_rank = cfg.get("kv_lora_rank", 128)
        q_rank = cfg.get("q_lora_rank", 256)
        rope_dim = cfg.get("rope_dim", head_dim // 2)
        attn = (
            h * q_rank
            + q_rank * heads * head_dim
            + q_rank * heads * rope_dim
            + h * kv_rank
            + kv_rank * kv_heads * head_dim
            + kv_rank * kv_heads * head_dim
            + h * rope_dim
            + heads * head_dim * h
        )
        kv_cache_floats = kv_rank + rope_dim
    else:
        q = h * heads * head_dim
        k = h * kv_heads * head_dim
        v = h * kv_heads * head_dim
        o = heads * head_dim * h
        attn = q + k + v + o
        kv_cache_floats = 2 * kv_heads * head_dim

    ffn = 3 * h * intermediate
    norms = 4 * h
    total = embed + layers * (attn + ffn + norms) + h
    active = total
    return {
        "params": total,
        "active_params": active,
        "embed_params": embed,
        "attn_params_per_layer": attn,
        "ffn_params_per_layer": ffn,
        "kv_cache_floats_per_token_layer": kv_cache_floats,
    }


def fmt_params(n):
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    return f"{n / 1e6:.1f}M"


def print_table(profiles):
    header = f"{'profile':<20}{'params':>10}{'active':>10}{'attn':>12}{'ffn':>12}{'kv_cache':>12}  usage"
    print(header)
    print("-" * len(header))
    for name, cfg in profiles.items():
        if "hidden_size" not in cfg:
            print(f"{name:<20}{'-':>10}{'-':>10}{'-':>12}{'-':>12}{'-':>12}  {cfg.get('usage', '')}")
            continue
        est = dense_params(cfg)
        print(
            f"{name:<20}"
            f"{fmt_params(est['params']):>10}"
            f"{fmt_params(est['active_params']):>10}"
            f"{fmt_params(est['attn_params_per_layer']):>12}"
            f"{fmt_params(est['ffn_params_per_layer']):>12}"
            f"{est['kv_cache_floats_per_token_layer']:>12}  "
            f"{cfg.get('usage', '')}"
        )


def main():
    parser = argparse.ArgumentParser(description="Estimate SearchLM model scales without importing torch/transformers")
    parser.add_argument("--profiles", default="configs/searchlm_profiles.json")
    parser.add_argument("--json", action="store_true", help="Print detailed JSON")
    args = parser.parse_args()

    with open(args.profiles, "r", encoding="utf-8") as f:
        profiles = json.load(f)
    if args.json:
        result = {}
        for name, cfg in profiles.items():
            result[name] = dict(cfg)
            if "hidden_size" in cfg:
                result[name].update(dense_params(cfg))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_table(profiles)
        print()
        print("Note: 2B+ full-parameter DDP is not recommended on 24GB RTX3090 cards; use LoRA/QLoRA as a baseline instead.")


if __name__ == "__main__":
    main()
