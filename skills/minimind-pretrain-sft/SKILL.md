---
name: minimind-pretrain-sft
description: Reproduce MiniMind Zero with pretraining then full SFT. Use when the user wants from-scratch pretrain, train_pretrain.py, train_full_sft.py, MiniMind Zero, or the default mini jsonl training path.
---

# MiniMind Zero — pretrain + SFT

Run all training commands from `trainer/` unless noted. Defaults assume MiniMind-3 scale: `--hidden_size 768`, `--num_hidden_layers 8`, `--use_moe 0`.

Prerequisites: `pretrain_t2t_mini.jsonl` and `sft_t2t_mini.jsonl` in `../dataset/` (see `minimind-setup`).

## 1. Pretrain

```bash
cd trainer
python train_pretrain.py
# multi-GPU:
torchrun --nproc_per_node N train_pretrain.py
```

Important defaults from `train_pretrain.py`:

| Flag | Default |
| --- | --- |
| `--save_dir` | `../out` |
| `--save_weight` | `pretrain` |
| `--epochs` | `2` |
| `--batch_size` | `32` |
| `--learning_rate` | `5e-4` |
| `--accumulation_steps` | `8` |
| `--max_seq_len` | `340` |
| `--data_path` | `../dataset/pretrain_t2t_mini.jsonl` |
| `--from_weight` | `none` (train from scratch) |
| `--from_resume` | `0` |
| `--dtype` | `bfloat16` |
| `--use_wandb` | off (pass flag to enable) |
| `--use_compile` | `0` |

Writes `../out/pretrain_<hidden_size>.pth` (e.g. `pretrain_768.pth`), overwritten each `--save_interval` (default `1000`).

Resume after interrupt:

```bash
python train_pretrain.py --from_resume 1
```

Checkpoints live under `./checkpoints/` as `<weight>_<dim>_resume.pth`.

Smoke-test pretrain (from repo root):

```bash
python eval_llm.py --weight pretrain
```

## 2. Full SFT

```bash
cd trainer
python train_full_sft.py
# or: torchrun --nproc_per_node N train_full_sft.py
```

Important defaults from `train_full_sft.py`:

| Flag | Default |
| --- | --- |
| `--save_weight` | `full_sft` |
| `--epochs` | `2` |
| `--batch_size` | `16` |
| `--learning_rate` | `1e-5` |
| `--max_seq_len` | `768` |
| `--data_path` | `../dataset/sft_t2t_mini.jsonl` |
| `--from_weight` | `pretrain` |
| `--save_interval` | `1000` |

Loads `../out/pretrain_*.pth` when `--from_weight pretrain`. Output: `../out/full_sft_<hidden_size>.pth`.

```bash
python train_full_sft.py --from_resume 1
```

## 3. Test the Zero model

From repo root:

```bash
python eval_llm.py --weight full_sft
```

## Full vs mini data

For the full main-branch corpora, point paths explicitly, for example:

```bash
python train_pretrain.py --data_path ../dataset/pretrain_t2t.jsonl
python train_full_sft.py --data_path ../dataset/sft_t2t.jsonl
```

Keep `--hidden_size` / `--num_hidden_layers` / `--use_moe` consistent across stages and with later eval/serve.

## Optional MoE / wandb / compile

```bash
python train_pretrain.py --use_moe 1 --use_wandb --wandb_project MiniMind-Pretrain
python train_full_sft.py --use_compile 1
```

Do not invent flags — run `python train_pretrain.py -h` / `python train_full_sft.py -h` if unsure.
