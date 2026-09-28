# Run: math_cold_cot_seed42

## 1. 目的

- unknown: run_tag is missing from configs/experiment_registry.csv
- reward_mode: n/a, model: n/a
- 预注册假设: n/a (run is not in the registry)

## 2. 环境

- command: `n/a` (full command in command.txt)
- host: autodl-container-154d408a77-cd6ad9dc
- platform: Linux-5.15.0-94-generic-x86_64-with-glibc2.35
- python: 3.12.3 (CPython)
- gpu: NVIDIA GeForce RTX 4090
- git_commit: n/a (no git repository in this checkout)

## 3. 训练动态

- steps logged: 0 (step n/a -> n/a)

## 4. Held-out

- model: out/math_cold_cot_seed42_768.pth
- seed: 42
- temperature: 0.8
- num_samples: 16
- max_new_tokens: 512
- greedy_accuracy: 0.3725
- sampled_pass_at_1: 0.3113
- pass_at_4: 0.4631
- pass_at_8: 0.5158
- pass_at_16: 0.5525
- format_rate: 1
- no_answer_rate: 0
- mean_completion_tokens: 114.2
- median_completion_tokens: 121
- data_matches_manifest: True
- by_difficulty:
  - easy: greedy=0.5522, pass@1=0.4879, n=n/a
  - hard: greedy=0.4361, pass@1=0.3637, n=n/a
  - medium: greedy=0.1278, pass@1=0.08083, n=n/a
- rows_available: 400

## 5. Error analysis

- raw completions: n/a
- status counts (training samples): n/a
- TODO: 人工标注 20 条 held-out 采样（正确/格式错/数值错/无答案/超长），把类别统计与典型路径写在这里。

## 6. 结论边界

- TODO: 支持/不支持哪条预注册假设（对照 §4 的数值，不要引用训练期 reward）。
- TODO: 哪些结论不能从本 run 推出（单 seed / 单规模 / 单难度分布 / 未做多轮评估）。
- TODO: 下一步最小实验是什么。

## 7. 归档备注

- math_cold_cot_seed42 is not present in /root/autodl-tmp/minimind/configs/experiment_registry.csv; command.txt holds no command, only a warning
- metrics missing: logs/math_cold_cot_seed42_metrics.csv not found
- samples missing: logs/math_cold_cot_seed42_samples.jsonl not found
- config missing: logs/math_cold_cot_seed42_config.json not found
