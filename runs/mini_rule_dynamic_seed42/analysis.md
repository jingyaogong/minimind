# Run: mini_rule_dynamic_seed42

## 1. 目的

- dynamic_sampling: reference=0 -> this run=1
- reward_mode: rule, model: MiniMind-64M
- 预注册假设: remove degenerate groups - filtering all-correct and all-wrong groups raises accuracy per optimizer step

## 2. 环境

- command: `python -u train_grpo.py \` (full command in command.txt)
- host: autodl-container-154d408a77-cd6ad9dc
- platform: Linux-5.15.0-94-generic-x86_64-with-glibc2.35
- python: 3.12.3 (CPython)
- gpu: NVIDIA GeForce RTX 4090
- git_commit: n/a (no git repository in this checkout)

## 3. 训练动态

- steps logged: 200 (step 5 -> 1000)
- avg_reward: 0.5 -> 0.5 (min 0, max 1)
- format_rate: 1 -> 1 (min 1, max 1)
- correct_rate: 0.5 -> 0.5 (min 0, max 1)
- avg_completion_tokens: 99.08 -> 107.4 (min 36.5, max 203.6)
- kl_ref: -0.01638 -> -0.0009935 (min -0.02362, max 0.00292)
- entropy: 0.03408 -> 0.03183 (min 0.02414, max 0.2108)
- dynamic sampling: filtered_fraction 1 -> 1, valid_groups last 0

## 4. Held-out

- model: out/mini_rule_dynamic_seed42_768.pth
- seed: 42
- temperature: 0.8
- num_samples: 16
- max_new_tokens: 512
- greedy_accuracy: 0.39
- sampled_pass_at_1: 0.3325
- pass_at_4: 0.4658
- pass_at_8: 0.5127
- pass_at_16: 0.5475
- format_rate: 1
- no_answer_rate: 0
- mean_completion_tokens: 115.5
- median_completion_tokens: 124
- data_matches_manifest: True
- by_difficulty:
  - easy: greedy=0.5597, pass@1=0.4879, n=n/a
  - hard: greedy=0.4586, pass@1=0.4088, n=n/a
  - medium: greedy=0.1504, pass@1=0.09962, n=n/a
- rows_available: 400

## 5. Error analysis

- raw completions: runs/mini_rule_dynamic_seed42/samples.jsonl
- status counts (training samples): n/a
- TODO: 人工标注 20 条 held-out 采样（正确/格式错/数值错/无答案/超长），把类别统计与典型路径写在这里。

## 6. 结论边界

- TODO: 支持/不支持哪条预注册假设（对照 §4 的数值，不要引用训练期 reward）。
- TODO: 哪些结论不能从本 run 推出（单 seed / 单规模 / 单难度分布 / 未做多轮评估）。
- TODO: 下一步最小实验是什么。
