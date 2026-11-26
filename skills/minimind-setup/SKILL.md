---
name: minimind-setup
description: Set up the MiniMind environment and download training datasets or pretrained weights. Use when cloning MiniMind, installing dependencies, placing jsonl data under dataset/, or downloading minimind-3 / reward-model weights.
---

# MiniMind setup

Work from the MiniMind repository root. Prefer the project's README (`README.md` / `README_en.md`) and the real script `--help` over memory.

## Clone and dependencies

```bash
git clone --depth 1 https://github.com/jingyaogong/minimind
cd minimind
pip install -r requirements.txt
```

`requirements.txt` pins transformers, datasets, fastapi/uvicorn, streamlit, modelscope, swanlab/wandb, etc. PyTorch is commented out there — install a CUDA/CPU build that matches the machine from the official PyTorch index if `import torch` fails.

Confirm the backend before training:

```python
import torch
print(torch.cuda.is_available())
```

Training scripts default `--device` to `cuda:0` when CUDA is available, else `cpu`.

## Directory layout that matters

| Path | Role |
| --- | --- |
| `model/` | Tokenizer + MiniMind model code (`tokenizer.json`, `model_minimind.py`, `model_lora.py`) |
| `dataset/` | Place downloaded `*.jsonl` here |
| `trainer/` | All `train_*.py` entrypoints (run from this directory) |
| `out/` | Saved `*.pth` weights (created by training) |
| `scripts/` | Serving, conversion, WebUI, tool-call eval |
| `eval_llm.py` | Root CLI chat / smoke test |

## Download training data

Dataset hubs: [ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files) | [HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main)

Put files under `./dataset/`. Do not clone everything; download only what the task needs.

**MiniMind Zero (fastest from-scratch dialogue model)** — required pair:

- `pretrain_t2t_mini.jsonl`
- `sft_t2t_mini.jsonl`

These match the trainer defaults:

- `train_pretrain.py` → `--data_path ../dataset/pretrain_t2t_mini.jsonl`
- `train_full_sft.py` → `--data_path ../dataset/sft_t2t_mini.jsonl`

**Optional / stage-specific files** (names as documented in README):

- `pretrain_t2t.jsonl`, `sft_t2t.jsonl` — full main-branch pretrain/SFT
- `dpo.jsonl` — DPO (`train_dpo.py` default)
- `rlaif.jsonl` — PPO / GRPO / CISPO (`train_ppo.py` / `train_grpo.py` default)
- `agent_rl.jsonl`, `agent_rl_math.jsonl` — Agentic RL (`train_agent.py` default `agent_rl.jsonl`)
- `lora_medical.jsonl` (and similar) — LoRA (`train_lora.py` default `lora_medical.jsonl`)

## Download inference weights (optional)

Transformers-format model for chat without local training:

```bash
# ModelScope
modelscope download --model gongjy/minimind-3 --local_dir ./minimind-3
# or HuggingFace
git clone https://huggingface.co/jingyaogong/minimind-3
```

Native PyTorch `*.pth` weights (for `--load_from ./model --weight ...`) can come from training under `./out/`, or from the published pytorch collection (see README ModelScope/HuggingFace links for `minimind-3-pytorch`).

## RLAIF reward model (only for PPO/GRPO/Agent RL)

Default `--reward_model_path` is `../../internlm2-1_8b-reward` (sibling of the MiniMind checkout). Layout:

```text
root/
├── minimind/
└── internlm2-1_8b-reward/
```

Sources: ModelScope `Shanghai_AI_Laboratory/internlm2-1_8b-reward` or HuggingFace `internlm/internlm2-1_8b-reward`.

## After setup

- Zero reproduction → skill `minimind-pretrain-sft`
- LoRA / DPO / RLAIF / Agentic RL → `minimind-lora` or `minimind-rl`
- Chat a checkpoint → `minimind-eval`
- API / convert → `minimind-serve`
