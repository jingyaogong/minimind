# Run: mini_mix_seed42

## 1. 目的

- reward_mode: reference=rule -> this run=mix
- reward_mode: mix, model: MiniMind-64M
- 预注册假设: calibrated mix - standardised rule plus pure-RM beats RM-only on accuracy while keeping the RM reward shape

## 2. 环境

- command: `python -u train_grpo.py \` (full command in command.txt)
- host: autodl-container-154d408a77-cd6ad9dc
- platform: Linux-5.15.0-94-generic-x86_64-with-glibc2.35
- python: 3.12.3 (CPython)
- gpu: NVIDIA GeForce RTX 4090
- git_commit: n/a (no git repository in this checkout)

## 3. 训练动态

- steps logged: 200 (step 5 -> 1000)
- avg_reward: -9.934e-09 -> -9.934e-09 (min -6.308e-07, max 3.378e-07)
- format_rate: 1 -> 1 (min 1, max 1)
- correct_rate: 0.5 -> 0.5 (min 0, max 1)
- avg_completion_tokens: 88.58 -> 93.17 (min 36.5, max 204.6)
- kl_ref: -0.01341 -> -0.01739 (min -0.1006, max 0.005669)
- entropy: 0.04552 -> 0.04751 (min 0.02881, max 0.2152)
- dynamic sampling: filtered_fraction 1 -> 1, valid_groups last 0

## 4. Held-out

- model: out/mini_mix_seed42_768.pth
- seed: 42
- temperature: 0.8
- num_samples: 16
- max_new_tokens: 512
- greedy_accuracy: 0.315
- sampled_pass_at_1: 0.2795
- pass_at_4: 0.4074
- pass_at_8: 0.4586
- pass_at_16: 0.5
- format_rate: 1
- no_answer_rate: 0
- mean_completion_tokens: 110.1
- median_completion_tokens: 111
- data_matches_manifest: True
- by_difficulty:
  - easy: greedy=0.5, pass@1=0.4636, n=n/a
  - hard: greedy=0.3534, pass@1=0.312, n=n/a
  - medium: greedy=0.09023, pass@1=0.06156, n=n/a
- rows_available: 400

## 5. Error analysis

- raw completions: runs/mini_mix_seed42/samples.jsonl
- status counts (training samples): n/a
- TODO: 人工标注 20 条 held-out 采样（正确/格式错/数值错/无答案/超长），把类别统计与典型路径写在这里。

## 6. 结论边界

- TODO: 支持/不支持哪条预注册假设（对照 §4 的数值，不要引用训练期 reward）。
- TODO: 哪些结论不能从本 run 推出（单 seed / 单规模 / 单难度分布 / 未做多轮评估）。
- TODO: 下一步最小实验是什么。
