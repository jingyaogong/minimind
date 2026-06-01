"""
Lightweight sanity checks for MiniMind development.

This script intentionally uses tiny random models, so it can run on CPU before
launching expensive DDP jobs on the training server.

Examples:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --device cuda:0 --hidden_size 128 --seq_len 32
"""
import argparse
import os
import sys
import tempfile

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_minimind_mla import MiniMindMLAConfig, MiniMindMLAForCausalLM
from trainer.trainer_utils import lm_checkpoint


def assert_shape(name, actual, expected):
    if tuple(actual) != tuple(expected):
        raise AssertionError(f"{name} shape mismatch: got {tuple(actual)}, expected {tuple(expected)}")


def build_config(attention_type, args):
    common = dict(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        vocab_size=args.vocab_size,
        max_position_embeddings=max(args.seq_len + args.max_new_tokens + 8, 64),
        num_attention_heads=args.num_attention_heads,
        dropout=0.0,
        flash_attn=args.flash_attn,
    )
    if attention_type == "mla":
        return MiniMindMLAConfig(kv_lora_rank=args.kv_lora_rank, **common)
    return MiniMindConfig(attention_type=attention_type, **common)


def build_model(config):
    if isinstance(config, MiniMindMLAConfig):
        return MiniMindMLAForCausalLM(config)
    return MiniMindForCausalLM(config)


def check_forward_and_cache(attention_type, args):
    config = build_config(attention_type, args)
    model = build_model(config).to(args.device).eval()
    input_ids = torch.randint(3, args.vocab_size, (args.batch_size, args.seq_len), device=args.device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        out = model(input_ids, attention_mask=attention_mask, labels=input_ids, use_cache=True)

    assert out.loss is not None, f"{attention_type}: loss is None"
    assert_shape(f"{attention_type}: logits", out.logits.shape, (args.batch_size, args.seq_len, args.vocab_size))
    if len(out.past_key_values) != args.num_hidden_layers:
        raise AssertionError(f"{attention_type}: unexpected cache layers={len(out.past_key_values)}")

    first_cache = out.past_key_values[0]
    if attention_type == "mla":
        kv_latent, k_rope = first_cache
        assert_shape("mla: kv_latent", kv_latent.shape, (args.batch_size, args.seq_len, args.kv_lora_rank))
        assert_shape("mla: k_rope", k_rope.shape, (args.batch_size, args.seq_len, config.rope_dim))
        cache_floats = args.kv_lora_rank + config.rope_dim
    else:
        key, value = first_cache
        assert_shape(
            f"{attention_type}: key",
            key.shape,
            (args.batch_size, args.seq_len, config.num_key_value_heads, config.head_dim),
        )
        assert_shape(
            f"{attention_type}: value",
            value.shape,
            (args.batch_size, args.seq_len, config.num_key_value_heads, config.head_dim),
        )
        cache_floats = 2 * config.num_key_value_heads * config.head_dim

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            top_k=0,
            eos_token_id=None,
            use_cache=True,
        )
    assert_shape(
        f"{attention_type}: generated",
        generated.shape,
        (args.batch_size, args.seq_len + args.max_new_tokens),
    )

    params = sum(p.numel() for p in model.parameters())
    print(
        f"[OK] {attention_type:<3} params={params / 1e6:.3f}M "
        f"cache={cache_floats} floats/token/layer loss={out.loss.item():.4f}"
    )
    return model, config


def check_checkpoint_roundtrip(args):
    config = build_config("gqa", args)
    model = build_model(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    with tempfile.TemporaryDirectory() as tmpdir:
        lm_checkpoint(config, weight="smoke", model=model, optimizer=optimizer, epoch=1, step=3, save_dir=tmpdir)
        loaded = lm_checkpoint(config, weight="smoke", save_dir=tmpdir)
        if loaded is None:
            raise AssertionError("checkpoint roundtrip failed: no resume data")
        if loaded.get("epoch") != 1 or loaded.get("step") != 3:
            raise AssertionError(f"checkpoint metadata mismatch: {loaded.get('epoch')=}, {loaded.get('step')=}")
    print("[OK] checkpoint save/load roundtrip")


def main():
    parser = argparse.ArgumentParser(description="MiniMind smoke tests")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=4)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--vocab_size", type=int, default=128)
    parser.add_argument("--kv_lora_rank", type=int, default=16)
    parser.add_argument("--flash_attn", action="store_true")
    parser.add_argument("--skip_checkpoint", action="store_true")
    args = parser.parse_args()

    if args.hidden_size % args.num_attention_heads != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads")

    torch.manual_seed(42)
    for attention_type in ("mha", "gqa", "mqa", "mla"):
        check_forward_and_cache(attention_type, args)
    if not args.skip_checkpoint:
        check_checkpoint_roundtrip(args)
    print("All smoke tests passed.")


if __name__ == "__main__":
    main()
