---
name: minimind-rl
description: Run MiniMind RL stages — RLHF-DPO (train_dpo.py), RLAIF PPO/GRPO/CISPO (train_ppo.py, train_grpo.py), and Agentic RL (train_agent.py). Use when aligning preferences, on-policy RL, or multi-turn tool-use RL.
---

# MiniMind RL (DPO / PPO / GRPO / CISPO / Agentic)

All commands below run from `trainer/`. Start from a stable SFT weight (`--from_weight full_sft` by default). Place data under `../dataset/` and, for PPO/GRPO/Agent, the InternLM2 reward model at the path in `--reward_model_path` (see `minimind-setup`).

## 1. RLHF — DPO

```bash
python train_dpo.py
# or: torchrun --nproc_per_node N train_dpo.py
```

Defaults from `train_dpo.py`:

| Flag | Default |
| --- | --- |
| `--save_weight` | `dpo` |
| `--epochs` | `1` |
| `--batch_size` | `4` |
| `--learning_rate` | `4e-8` |
| `--max_seq_len` | `1024` |
| `--data_path` | `../dataset/dpo.jsonl` |
| `--from_weight` | `full_sft` |
| `--beta` | `0.15` |
| `--from_resume` | `0` |

Output: `../out/dpo_<hidden_size>.pth`.

## 2. RLAIF — PPO

Needs `rlaif.jsonl` and a reward model.

```bash
python train_ppo.py
# or: torchrun --nproc_per_node N train_ppo.py
```

Notable defaults from `train_ppo.py`:

| Flag | Default |
| --- | --- |
| `--save_weight` | `ppo_actor` |
| `--epochs` | `1` |
| `--batch_size` | `2` |
| `--learning_rate` | `3e-7` |
| `--critic_learning_rate` | `5e-7` |
| `--max_seq_len` | `768` |
| `--max_gen_len` | `1024` |
| `--data_path` | `../dataset/rlaif.jsonl` |
| `--from_weight` | `full_sft` |
| `--reward_model_path` | `../../internlm2-1_8b-reward` |
| `--clip_epsilon` | `0.2` |
| `--kl_coef` | `0.02` |
| `--ppo_update_iters` | `2` |
| `--mini_batch_size` | `2` |
| `--thinking_ratio` | `0.9` |
| `--rollout_engine` | `torch` (`sglang` also allowed) |

Output: `../out/ppo_actor_<hidden_size>.pth`.

SGLang rollout extras: `--sglang_base_url` (default `http://localhost:8998`), `--sglang_model_path` (`../model`), `--sglang_shared_path` (`./sglang_ckpt_ppo`).

## 3. RLAIF — GRPO / CISPO

Same data/reward setup as PPO. CISPO is a `--loss_type` on the GRPO script (no separate trainer).

```bash
python train_grpo.py
python train_grpo.py --loss_type cispo
# or: torchrun --nproc_per_node N train_grpo.py
```

Notable defaults from `train_grpo.py`:

| Flag | Default |
| --- | --- |
| `--save_weight` | `grpo` |
| `--epochs` | `1` |
| `--batch_size` | `2` |
| `--learning_rate` | `3e-7` |
| `--max_seq_len` | `768` |
| `--max_gen_len` | `1024` |
| `--data_path` | `../dataset/rlaif.jsonl` |
| `--num_generations` | `6` |
| `--beta` | `0.1` |
| `--loss_type` | `cispo` (`grpo` \| `cispo`) |
| `--epsilon` | `0.2` |
| `--epsilon_high` | `5.0` |
| `--from_weight` | `full_sft` |
| `--reward_model_path` | `../../internlm2-1_8b-reward` |
| `--thinking_ratio` | `0.9` |
| `--rollout_engine` | `torch` |

Output: `../out/grpo_<hidden_size>.pth`.

## 4. Agentic RL

Multi-turn tool-use RL via `train_agent.py`. Default data `../dataset/agent_rl.jsonl` (math variant: `agent_rl_math.jsonl`).

```bash
python train_agent.py
# or: torchrun --nproc_per_node N train_agent.py
```

Notable defaults:

| Flag | Default |
| --- | --- |
| `--save_weight` | `agent` |
| `--epochs` | `1` |
| `--batch_size` | `2` |
| `--learning_rate` | `3e-7` |
| `--max_seq_len` | `1024` |
| `--max_gen_len` | `768` |
| `--max_total_len` | `2500` |
| `--data_path` | `../dataset/agent_rl.jsonl` |
| `--num_generations` | `4` |
| `--beta` | `0.1` |
| `--loss_type` | `cispo` (`grpo` \| `cispo`) |
| `--from_weight` | `full_sft` |
| `--thinking_ratio` | `0.1` |
| `--reward_model_path` | `../../internlm2-1_8b-reward` |
| `--rollout_engine` | `torch` |

SGLang example (start server first, then train):

```bash
python -m sglang.launch_server --model-path ./minimind-3 --attention-backend triton --host 0.0.0.0 --port 8998
python train_agent.py --rollout_engine sglang --sglang_base_url http://localhost:8998 --sglang_shared_path ./ckpt_mm --data_path ../dataset/agent_rl_math.jsonl --use_wandb
```

Output: `../out/agent_<hidden_size>.pth`.

Tool-call smoke test (from `scripts/`):

```bash
cd scripts && python eval_toolcall.py --weight agent
```

## Shared tips

- Use `--from_resume 1` to continue; `--use_wandb` / `--wandb_project` for logging; `--debug_mode` on PPO/GRPO/Agent for sample dumps.
- Do not invent flags — check each script's `argparse` / `-h`.
- Evaluate with skill `minimind-eval` (`--weight dpo|ppo_actor|grpo|agent|...`).
