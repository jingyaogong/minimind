import argparse
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM, apply_rotary_pos_emb, precompute_freqs_cis


def shape(x):
    return tuple(x.shape)


def max_abs_diff(a, b):
    return (a - b).abs().max().item()


def main():
    parser = argparse.ArgumentParser(description="Trace MiniMind RoPE position slices and KV cache growth.")
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_attention_heads", type=int, default=4)
    parser.add_argument("--num_key_value_heads", type=int, default=2)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--vocab_size", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=3)
    args = parser.parse_args()
    if args.seq_len < 2:
        raise ValueError("--seq_len must be at least 2 so the script can compare prefix + next-token cache.")

    torch.manual_seed(42)
    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        vocab_size=args.vocab_size,
        max_position_embeddings=128,
        dropout=0.0,
        flash_attn=False,
    )
    model = MiniMindForCausalLM(config).eval()
    attn = model.model.layers[0].self_attn

    print("[Config]")
    print(f"hidden_size={config.hidden_size}")
    print(f"num_attention_heads={config.num_attention_heads}")
    print(f"num_key_value_heads={config.num_key_value_heads}")
    print(f"head_dim={config.head_dim}")
    print(f"max_position_embeddings={config.max_position_embeddings}")
    print(f"rope_theta={config.rope_theta}")
    print()

    freqs_cos, freqs_sin = precompute_freqs_cis(
        dim=config.head_dim,
        end=config.max_position_embeddings,
        rope_base=config.rope_theta,
        rope_scaling=config.rope_scaling,
    )
    print("[RoPE Table]")
    print(f"freqs_cos.shape={shape(freqs_cos)}")
    print(f"freqs_sin.shape={shape(freqs_sin)}")
    print(f"model.model.freqs_cos.shape={shape(model.model.freqs_cos)}")
    print(f"model.model.freqs_sin.shape={shape(model.model.freqs_sin)}")
    print()

    q = torch.randn(1, args.seq_len, config.num_attention_heads, config.head_dim)
    k = torch.randn(1, args.seq_len, config.num_key_value_heads, config.head_dim)
    cos = model.model.freqs_cos[: args.seq_len]
    sin = model.model.freqs_sin[: args.seq_len]
    q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)

    print("[Apply RoPE]")
    print(f"q_before_rope.shape={shape(q)}")
    print(f"k_before_rope.shape={shape(k)}")
    print(f"cos_slice.shape={shape(cos)}")
    print(f"sin_slice.shape={shape(sin)}")
    print(f"q_after_rope.shape={shape(q_rope)}")
    print(f"k_after_rope.shape={shape(k_rope)}")
    print(f"q_changed_max_abs_diff={max_abs_diff(q, q_rope):.8f}")
    print(f"k_changed_max_abs_diff={max_abs_diff(k, k_rope):.8f}")
    print()

    input_ids = ((torch.arange(args.seq_len, dtype=torch.long) * 4 + 3) % args.vocab_size).unsqueeze(0)
    prefix_ids = input_ids[:, :-1]
    last_id = input_ids[:, -1:]

    with torch.no_grad():
        full_outputs = model(input_ids, use_cache=False)
        prefix_outputs = model(prefix_ids, use_cache=True)
        prefix_cache = prefix_outputs.past_key_values
        step_outputs = model(last_id, past_key_values=prefix_cache, use_cache=True)
        step_cache = step_outputs.past_key_values

    prefix_k, prefix_v = prefix_cache[0]
    step_k, step_v = step_cache[0]
    print("[KV Cache Growth]")
    print(f"input_ids={input_ids.tolist()}")
    print(f"prefix_ids={prefix_ids.tolist()}")
    print(f"last_id={last_id.tolist()}")
    print(f"prefix_layer0_k.shape={shape(prefix_k)}")
    print(f"prefix_layer0_v.shape={shape(prefix_v)}")
    print(f"step_layer0_k.shape={shape(step_k)}")
    print(f"step_layer0_v.shape={shape(step_v)}")
    print(f"expected_prefix_cache_len={prefix_ids.shape[1]}")
    print(f"expected_step_cache_len={input_ids.shape[1]}")
    print()

    full_last_logits = full_outputs.logits[:, -1, :]
    step_last_logits = step_outputs.logits[:, -1, :]
    print("[Full Forward vs Incremental Cache Forward]")
    print(f"full_logits.shape={shape(full_outputs.logits)}")
    print(f"step_logits.shape={shape(step_outputs.logits)}")
    print(f"full_last_logits.shape={shape(full_last_logits)}")
    print(f"step_last_logits.shape={shape(step_last_logits)}")
    print(f"full_vs_incremental_last_logits_max_abs_diff={max_abs_diff(full_last_logits, step_last_logits):.12f}")
    print()

    with torch.no_grad():
        generated_with_cache = model.generate(
            inputs=prefix_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            top_k=0,
            top_p=1.0,
            temperature=1.0,
            eos_token_id=None,
            use_cache=True,
        )
        generated_without_cache = model.generate(
            inputs=prefix_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            top_k=0,
            top_p=1.0,
            temperature=1.0,
            eos_token_id=None,
            use_cache=False,
        )

    print("[Generate With Cache vs Without Cache]")
    print(f"generated_with_cache={generated_with_cache.tolist()}")
    print(f"generated_without_cache={generated_without_cache.tolist()}")
    print(f"cache_and_no_cache_generate_match={torch.equal(generated_with_cache, generated_without_cache)}")
    print()

    print("[Reading Rule]")
    print("RoPE keeps Q/K shapes unchanged, but changes their values according to absolute position.")
    print("KV cache stores per-layer K/V before repeat_kv: [batch, cached_seq, num_key_value_heads, head_dim].")
    print("During cached decode, seq_len can be 1 while start_pos equals cached_seq.")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
