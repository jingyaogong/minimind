import argparse
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def shape(x):
    return tuple(x.shape)


def mean_rms(x):
    return torch.sqrt(x.float().pow(2).mean(dim=-1)).mean().item()


def print_step(name, tensor):
    print(f"{name:38s} shape={shape(tensor)} mean_rms={mean_rms(tensor):.6f}")


def main():
    parser = argparse.ArgumentParser(description="Trace MiniMind embedding, RMSNorm, residual, and block shapes.")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=6)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--vocab_size", type=int, default=6400)
    args = parser.parse_args()

    torch.manual_seed(42)
    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=args.vocab_size,
        max_position_embeddings=128,
        dropout=0.0,
    )
    model = MiniMindForCausalLM(config).eval()

    input_ids = torch.arange(args.batch_size * args.seq_len, dtype=torch.long).view(args.batch_size, args.seq_len)
    input_ids = input_ids % args.vocab_size

    print("[Config]")
    print(f"hidden_size={config.hidden_size}")
    print(f"num_hidden_layers={config.num_hidden_layers}")
    print(f"num_attention_heads={config.num_attention_heads}")
    print(f"num_key_value_heads={config.num_key_value_heads}")
    print(f"head_dim={config.head_dim}")
    print(f"vocab_size={config.vocab_size}")
    print()

    print("[Input]")
    print(f"input_ids.shape={shape(input_ids)}")
    print(f"input_ids[0]={input_ids[0].tolist()}")
    print(f"embed_tokens.weight.shape={shape(model.model.embed_tokens.weight)}")
    print()

    with torch.no_grad():
        hidden_states = model.model.dropout(model.model.embed_tokens(input_ids))
        print("[Embedding]")
        print_step("after embed_tokens + dropout", hidden_states)
        print()

        seq_length = input_ids.shape[1]
        position_embeddings = (model.model.freqs_cos[:seq_length], model.model.freqs_sin[:seq_length])
        print("[Position Embeddings]")
        print(f"freqs_cos slice shape={shape(position_embeddings[0])}")
        print(f"freqs_sin slice shape={shape(position_embeddings[1])}")
        print()

        past_key_values = [None] * len(model.model.layers)
        for layer_id, (layer, past_key_value) in enumerate(zip(model.model.layers, past_key_values)):
            print(f"[Block {layer_id}]")
            print_step("block input", hidden_states)

            residual = hidden_states
            normed_for_attn = layer.input_layernorm(hidden_states)
            print_step("after input_layernorm", normed_for_attn)

            attn_out, present = layer.self_attn(
                normed_for_attn,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=False,
                attention_mask=None,
            )
            print_step("self_attn output", attn_out)

            after_attn_residual = attn_out + residual
            print_step("after attention residual add", after_attn_residual)

            normed_for_mlp = layer.post_attention_layernorm(after_attn_residual)
            print_step("after post_attention_layernorm", normed_for_mlp)

            mlp_out = layer.mlp(normed_for_mlp)
            print_step("mlp output", mlp_out)

            hidden_states = after_attn_residual + mlp_out
            print_step("block output", hidden_states)
            print()

        hidden_states = model.model.norm(hidden_states)
        print("[Final Norm]")
        print_step("after final norm", hidden_states)
        print()

        logits = model.lm_head(hidden_states)
        print("[LM Head]")
        print(f"lm_head.weight.shape={shape(model.lm_head.weight)}")
        print(f"logits.shape={shape(logits)}")
        print()

    print("[Shape Rule]")
    print("input_ids:     [batch_size, seq_len]")
    print("hidden_states: [batch_size, seq_len, hidden_size]")
    print("logits:        [batch_size, seq_len, vocab_size]")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
