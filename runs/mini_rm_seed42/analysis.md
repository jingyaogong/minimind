# Run: mini_rm_seed42

## 1. 目的

- reward_mode: reference=rule -> this run=rm
- reward_mode: rm, model: MiniMind-64M
- 预注册假设: RM-only control - the original heuristic plus RM reward raises reward without raising held-out accuracy

## 2. 环境

- command: `python -u train_grpo.py \` (full command in command.txt)
- host: autodl-container-154d408a77-cd6ad9dc
- platform: Linux-5.15.0-94-generic-x86_64-with-glibc2.35
- python: 3.12.3 (CPython)
- gpu: NVIDIA GeForce RTX 4090
- git_commit: n/a (no git repository in this checkout)

## 3. 训练动态

- steps logged: 200 (step 5 -> 1000)
- avg_reward: 2.264 -> 2.359 (min 0.9981, max 2.606)
- format_rate: 1 -> 1 (min 1, max 1)
- correct_rate: 0.5 -> 0 (min 0, max 0.8333)
- avg_completion_tokens: 86 -> 96.08 (min 35.33, max 180.7)
- kl_ref: -0.02704 -> -0.1043 (min -0.2769, max -0.002073)
- entropy: 0.03375 -> 0.06627 (min 0.03375, max 0.2316)
- dynamic sampling: filtered_fraction 1 -> 1, valid_groups last 0

## 4. Held-out

- model: out/mini_rm_seed42_768.pth
- seed: 42
- temperature: 0.8
- num_samples: 16
- max_new_tokens: 512
- greedy_accuracy: 0.1325
- sampled_pass_at_1: 0.1167
- pass_at_4: 0.2005
- pass_at_8: 0.2433
- pass_at_16: 0.285
- format_rate: 1
- no_answer_rate: 0
- mean_completion_tokens: 105
- median_completion_tokens: 108
- data_matches_manifest: True
- by_difficulty:
  - easy: greedy=0.209, pass@1=0.1777, n=n/a
  - hard: greedy=0.1805, pass@1=0.1612, n=n/a
  - medium: greedy=0.007519, pass@1=0.01081, n=n/a
- rows_available: 400

## 5. Error analysis

- raw completions: runs/mini_rm_seed42/samples.jsonl
- status counts (training samples): n/a
- TODO: 人工标注 20 条 held-out 采样（正确/格式错/数值错/无答案/超长），把类别统计与典型路径写在这里。

## 6. 结论边界

- TODO: 支持/不支持哪条预注册假设（对照 §4 的数值，不要引用训练期 reward）。
- TODO: 哪些结论不能从本 run 推出（单 seed / 单规模 / 单难度分布 / 未做多轮评估）。
- TODO: 下一步最小实验是什么。
