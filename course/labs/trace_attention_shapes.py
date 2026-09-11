import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM, apply_rotary_pos_emb, repeat_kv


def shape(x):
    return tuple(x.shape)


def max_abs_diff(a, b):
    return (a - b).abs().max().item()


def print_matrix(title, matrix):
    print(title)
    for row in matrix.tolist():
        print(" ".join(f"{value:7.4f}" for value in row))
    print()


def main():
    parser = argparse.ArgumentParser(description="Trace MiniMind attention Q/K/V shapes, causal mask, and output.")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_attention_heads", type=int, default=4)
    parser.add_argument("--num_key_value_heads", type=int, default=2)
    parser.add_argument("--num_hidden_layers", type=int, default=1)
    parser.add_argument("--vocab_size", type=int, default=6400)
    args = parser.parse_args()

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

    x = torch.randn(args.batch_size, args.seq_len, args.hidden_size)
    position_embeddings = (
        model.model.freqs_cos[: args.seq_len],
        model.model.freqs_sin[: args.seq_len],
    )

    print("[Config]")
    print(f"hidden_size={config.hidden_size}")
    print(f"num_attention_heads={config.num_attention_heads}")
    print(f"num_key_value_heads={config.num_key_value_heads}")
    print(f"n_rep={attn.n_rep}")
    print(f"head_dim={config.head_dim}")
    print(f"flash_attn={config.flash_attn}")
    print()

    with torch.no_grad():
        print("[Input]")
        print(f"x.shape={shape(x)}")
        print()

        xq_linear = attn.q_proj(x)
        xk_linear = attn.k_proj(x)
        xv_linear = attn.v_proj(x)
        print("[Linear Projections]")
        print(f"q_proj(x).shape={shape(xq_linear)}")
        print(f"k_proj(x).shape={shape(xk_linear)}")
        print(f"v_proj(x).shape={shape(xv_linear)}")
        print(f"o_proj.weight.shape={shape(attn.o_proj.weight)}")
        print()

        xq = xq_linear.view(args.batch_size, args.seq_len, attn.n_local_heads, attn.head_dim)
        xk = xk_linear.view(args.batch_size, args.seq_len, attn.n_local_kv_heads, attn.head_dim)
        xv = xv_linear.view(args.batch_size, args.seq_len, attn.n_local_kv_heads, attn.head_dim)
        print("[Head Reshape]")
        print(f"xq.shape={shape(xq)}")
        print(f"xk.shape={shape(xk)}")
        print(f"xv.shape={shape(xv)}")
        print()

        xq_normed = attn.q_norm(xq)
        xk_normed = attn.k_norm(xk)
        print("[Q/K Norm]")
        print(f"q_norm(xq).shape={shape(xq_normed)}")
        print(f"k_norm(xk).shape={shape(xk_normed)}")
        print()

        cos, sin = position_embeddings
        xq_rope, xk_rope = apply_rotary_pos_emb(xq_normed, xk_normed, cos, sin)
        print("[RoPE Shape Only]")
        print(f"cos.shape={shape(cos)}")
        print(f"sin.shape={shape(sin)}")
        print(f"xq after RoPE shape={shape(xq_rope)}")
        print(f"xk after RoPE shape={shape(xk_rope)}")
        print()

        xq_t = xq_rope.transpose(1, 2)
        xk_repeated = repeat_kv(xk_rope, attn.n_rep)
        xv_repeated = repeat_kv(xv, attn.n_rep)
        xk_t = xk_repeated.transpose(1, 2)
        xv_t = xv_repeated.transpose(1, 2)
        print("[Repeat KV and Transpose]")
        print(f"xq_t.shape={shape(xq_t)}")
        print(f"xk before repeat shape={shape(xk_rope)}")
        print(f"xk after repeat shape={shape(xk_repeated)}")
        print(f"xk_t.shape={shape(xk_t)}")
        print(f"xv_t.shape={shape(xv_t)}")
        print()

        scores = (xq_t @ xk_t.transpose(-2, -1)) / (attn.head_dim**0.5)
        causal_mask = torch.full((args.seq_len, args.seq_len), float("-inf")).triu(1)
        masked_scores = scores + causal_mask
        attn_weights = F.softmax(masked_scores.float(), dim=-1).type_as(xq_t)
        print("[Scores and Causal Mask]")
        print(f"scores.shape={shape(scores)}")
        print(f"causal_mask.shape={shape(causal_mask)}")
        print(f"attention_weights.shape={shape(attn_weights)}")
        print_matrix("attention weights for batch 0, head 0:", attn_weights[0, 0])

        future_mass = torch.triu(attn_weights[0, 0], diagonal=1).sum().item()
        print(f"future_attention_mass_for_batch0_head0={future_mass:.10f}")
        print()

        weighted_values = attn_weights @ xv_t
        merged_heads = weighted_values.transpose(1, 2).reshape(args.batch_size, args.seq_len, -1)
        manual_output = attn.resid_dropout(attn.o_proj(merged_heads))
        module_output, past_kv = attn(x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None)

        print("[Output]")
        print(f"weighted_values.shape={shape(weighted_values)}")
        print(f"merged_heads.shape={shape(merged_heads)}")
        print(f"manual_output.shape={shape(manual_output)}")
        print(f"module_output.shape={shape(module_output)}")
        print(f"past_kv={past_kv}")
        print(f"manual_vs_module_max_abs_diff={max_abs_diff(manual_output, module_output):.12f}")
        print()

    print("[Shape Rule]")
    print("input hidden_states:  [batch, seq, hidden_size]")
    print("Q after reshape:      [batch, seq, num_attention_heads, head_dim]")
    print("K/V after reshape:    [batch, seq, num_key_value_heads, head_dim]")
    print("scores:               [batch, num_attention_heads, seq, seq]")
    print("attention output:     [batch, seq, hidden_size]")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
