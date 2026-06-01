# Attention Architecture Experiments

This project supports four attention variants for controlled small-LLM experiments:

- `mha`: Multi-Head Attention, one K/V head per query head.
- `gqa`: Grouped-Query Attention, the default MiniMind-style setting.
- `mqa`: Multi-Query Attention, one shared K/V head.
- `mla`: Multi-head Latent Attention with low-rank KV cache and decoupled RoPE.

## Training Commands

Run all commands from `trainer/`.

```bash
torchrun --nproc_per_node=4 train_pretrain.py --attention_type gqa
torchrun --nproc_per_node=4 train_pretrain.py --attention_type mqa --save_weight pretrain_mqa
torchrun --nproc_per_node=4 train_pretrain.py --attention_type mla --save_weight pretrain_mla
```

```bash
torchrun --nproc_per_node=4 train_full_sft.py --attention_type gqa --from_weight pretrain
torchrun --nproc_per_node=4 train_full_sft.py --attention_type mqa --from_weight pretrain_mqa --save_weight full_sft_mqa
torchrun --nproc_per_node=4 train_full_sft.py --attention_type mla --from_weight pretrain_mla --save_weight full_sft_mla
```

The legacy MLA flag still works:

```bash
torchrun --nproc_per_node=4 train_pretrain.py --use_mla 1
```

## RL Stages

The same `--attention_type` argument is available in DPO, PPO, GRPO, LoRA, distillation, Agent RL, and DeepSpeed pretraining.

```bash
torchrun --nproc_per_node=4 train_dpo.py --attention_type gqa --from_weight full_sft
torchrun --nproc_per_node=4 train_ppo.py --attention_type gqa --from_weight full_sft
torchrun --nproc_per_node=4 train_grpo.py --attention_type gqa --from_weight full_sft
```

## Benchmark

```bash
python scripts/benchmark_gqa_vs_mla.py --device cuda:0 --hidden_size 768
```

With checkpoints:

```bash
python scripts/benchmark_gqa_vs_mla.py \
  --device cuda:0 \
  --hidden_size 768 \
  --gqa_ckpt out/pretrain_768.pth \
  --mla_ckpt out/pretrain_mla_768_mla.pth \
  --data_path dataset/pretrain_t2t_mini.jsonl \
  --tokenizer_path model
```

Key metrics to record for a resume or report:

- Train setup: GPU type/count, batch size, sequence length, tokens trained.
- Architecture: total params, attention params/layer, KV-cache floats/token/layer.
- Efficiency: prefill latency, decode tokens/s, training tokens/s, peak GPU memory.
- Quality: pretrain loss, SFT PPL, DPO/PPO/GRPO reward and KL curves.

For `hidden_size=768`, the default GQA KV cache is `2 * 4 * 96 = 768` floats/token/layer. MLA with `kv_lora_rank=128` and `rope_dim=48` stores `128 + 48 = 176` floats/token/layer, a theoretical `4.4x` KV-cache compression.
