# Run: mini_rule_seed7

## 1. 目的

- seed: reference=42 -> this run=7
- reward_mode: rule, model: MiniMind-64M
- 预注册假设: seed replication - the rule-baseline direction reproduces under a second seed

## 2. 环境

- command: `python -u train_grpo.py \` (full command in command.txt)
- host: autodl-container-154d408a77-cd6ad9dc
- platform: Linux-5.15.0-94-generic-x86_64-with-glibc2.35
- python: 3.12.3 (CPython)
- gpu: NVIDIA GeForce RTX 4090
- git_commit: n/a (no git repository in this checkout)

## 3. 训练动态

- steps logged: 200 (step 5 -> 1000)
- avg_reward: 0.3333 -> 0.25 (min 0, max 1)
- format_rate: 1 -> 1 (min 1, max 1)
- correct_rate: 0.3333 -> 0.25 (min 0, max 1)
- avg_completion_tokens: 128 -> 104.5 (min 35.92, max 190.7)
- kl_ref: -0.001792 -> -0.004803 (min -0.05323, max 0.004056)
- entropy: 0.03358 -> 0.07649 (min 0.02269, max 0.1896)
- dynamic sampling: filtered_fraction 0.5 -> 0.5, valid_groups last 1

## 4. Held-out

- model: out/mini_rule_seed7_768.pth
- seed: 42
- temperature: 0.8
- num_samples: 16
- max_new_tokens: 512
- greedy_accuracy: 0.3875
- sampled_pass_at_1: 0.3386
- pass_at_4: 0.4808
- pass_at_8: 0.534
- pass_at_16: 0.5775
- format_rate: 1
- no_answer_rate: 0
- mean_completion_tokens: 114.5
- median_completion_tokens: 123
- data_matches_manifest: True
- by_difficulty:
  - easy: greedy=0.5746, pass@1=0.5079, n=n/a
  - hard: greedy=0.4511, pass@1=0.4018, n=n/a
  - medium: greedy=0.1353, pass@1=0.1048, n=n/a
- rows_available: 400

## 5. Error analysis

- raw completions: runs/mini_rule_seed7/samples.jsonl
- status counts (training samples): n/a
- TODO: 人工标注 20 条 held-out 采样（正确/格式错/数值错/无答案/超长），把类别统计与典型路径写在这里。

## 6. 结论边界

- TODO: 支持/不支持哪条预注册假设（对照 §4 的数值，不要引用训练期 reward）。
- TODO: 哪些结论不能从本 run 推出（单 seed / 单规模 / 单难度分布 / 未做多轮评估）。
- TODO: 下一步最小实验是什么。
