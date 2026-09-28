# RLVR 审计汇总（自动生成，勿手改）

- 生成时间：2026-09-28T00:34:03+00:00
- 预注册表：`/Users/qi/实习/项目/minimind/configs/experiment_registry.csv`
- 训练日志：`runs`（归档）/ `logs`（未归档）
- 评估数据：`dataset/math_heldout.jsonl`（400 题，与训练集无重叠）
- 本文件与 `summary.csv` 只是**视图**；每个数字的原始来源是 `runs/<run_tag>/metrics.csv` 与 `runs/<run_tag>/eval.json`。
- 口径：训练侧为 log_interval 采样点的组内统计；held-out 侧为 greedy + pass@k（每个 run 的采样数与温度见 summary.csv 的 `eval_num_samples` / `eval_temperature`），长度=到首个 EOS 的 token 数（不含 prompt）。

## 1. 训练动态

| run_tag | evidence | steps | reward (首/末10点均值) | correct (首/末10点均值) | format last | tokens last | filtered last | ACR wrong last | 决策树提示 |
|---|---|---:|---|---|---:|---:|---:|---:|---|
| `mini_rule_seed42` | archived | 200 | 0.317 → 0.542 (+0.225) | 0.317 → 0.542 (+0.225) | 1.000 | 118.1 | 1.000 | 0.500 | 否：训练侧正确率上升但 held-out 未涨（reward-eval gap）：先查 held-out 难度/样本量/verifier 口径 |
| `mini_rule_dynamic_seed42` | archived | 200 | 0.383 → 0.433 (+0.050) | 0.383 → 0.433 (+0.050) | 1.000 | 116.7 | 1.000 | 0.500 | 否：held-out 下降（负结果）：先对照 paired_compare 的 p 值与本项目实测的噪声地板（report §4.7）再定性，别只看点估计方向 |
| `mini_rule_len_seed42` | archived | 200 | 0.317 → 0.417 (+0.100) | 0.317 → 0.417 (+0.100) | 1.000 | 117.2 | 1.000 | 0.500 | 否：held-out 下降（负结果）：先对照 paired_compare 的 p 值与本项目实测的噪声地板（report §4.7）再定性，别只看点估计方向 |
| `mini_rm_seed42` | archived | 200 | 1.684 → 2.011 (+0.327) | 0.233 → 0.025 (-0.208) | 1.000 | 110.3 | 1.000 | 1.000 | 否：held-out 下降 —— RM 奖励 hack：奖励与任务正确性背离（先拆 reward 组成 + 看 ACR；显著性见 paired_compare） |
| `mini_mix_seed42` | archived | 200 | n/a（mix 臂为组内标准分量，均值恒 0） | 0.208 → 0.350 (+0.142) | 1.000 | 113.1 | 1.000 | 0.500 | 否：held-out 下降 —— RM 分量污染了 rule 信号（负结果：先看 component 退化组；显著性见 paired_compare） |
| `mini_rule_seed7` | archived | 200 | 0.508 → 0.317 (-0.192) | 0.508 → 0.317 (-0.192) | 1.000 | 120.6 | 0.500 | 0.500 | 否：查 reward 范围、verifier、学习率、rollout、NaN |
| `math_cold_cot_seed42` | archived | 0 | — → — (—) | — → — (—) | — | — | — | — | 数据不足：需要至少两个 log_interval 的记录才能判断 |
| `math_cold_seed42` | missing | 0 | — → — (—) | — → — (—) | — | — | — | — | 数据不足：需要至少两个 log_interval 的记录才能判断 |

## 2. Held-out 评估矩阵（SOP §10.3）

`*` = 该行尚未归档（`runs/<run_tag>/manifest.json` 不存在），数字请先归档再引用。

| 权重 | 题数 | greedy | pass@1 | pass@16 | format | avg len | easy | medium | hard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 冷启动 SFT（CoT v2，RL 起点） (`math_cold_cot_seed42`) | 400.000 | 0.372 | 0.311 | 0.552 | 1.000 | 114.2 | 0.552 | 0.128 | 0.436 |
| rule GRPO (`mini_rule_seed42`) | 400.000 | 0.395 | 0.332 | 0.557 | 1.000 | 115.2 | 0.560 | 0.158 | 0.466 |
| dynamic sampling (`mini_rule_dynamic_seed42`) | 400.000 | 0.390 | 0.333 | 0.547 | 1.000 | 115.5 | 0.560 | 0.150 | 0.459 |
| length penalty (`mini_rule_len_seed42`) | 400.000 | 0.393 | 0.331 | 0.557 | 1.000 | 114.9 | 0.567 | 0.150 | 0.459 |
| RM-only (`mini_rm_seed42`) | 400.000 | 0.133 | 0.117 | 0.285 | 1.000 | 105.0 | 0.209 | 0.008 | 0.180 |
| calibrated mix (`mini_mix_seed42`) | 400.000 | 0.315 | 0.280 | 0.500 | 1.000 | 110.1 | 0.500 | 0.090 | 0.353 |
| 冷启动 SFT（CoT v1，pilot 对照） (`math_cold_seed42`) * | 400.000 | 0.000 | 0.010 | 0.098 | 1.000 | 49.5 | 0.000 | 0.000 | 0.000 |

### 2.1 reward–eval 对照（§11.2 决策树的定量版本）

`reward_gain` 是训练期组内 average reward 的首末差，`heldout_gain` 是相对参考臂的 greedy 差值；两者单位不同，`reward_eval_gap` 只用于排序，不能当作物理量。

| run_tag | reward_gain | train correct_gain | heldout_gain | reward_eval_gap | pass@k inversion | reward hacking 嫌疑 |
|---|---:|---:|---:|---:|---|---|---|
| `mini_rule_seed42` | +0.225 | +0.225 | +0.000 | +0.225 | false | false |
| `mini_rule_dynamic_seed42` | +0.050 | +0.050 | -0.005 | +0.055 | true | false |
| `mini_rule_len_seed42` | +0.100 | +0.100 | -0.003 | +0.103 | false | false |
| `mini_rm_seed42` | +0.327 | -0.208 | -0.263 | +0.589 | false | true |
| `mini_mix_seed42` | n/a（标准化） | +0.142 | -0.080 | +0.080 | false | false |
| `mini_rule_seed7` | -0.192 | -0.192 | -0.008 | -0.184 | false | false |
| `math_cold_cot_seed42` | — | — | -0.023 | — | false | false |
| `math_cold_seed42` | — | — | -0.395 | — | false | false |

## 3. 按难度的 held-out 正确率

| run_tag | easy greedy | medium greedy | hard greedy | easy pass@1 | medium pass@1 | hard pass@1 |
|---|---:|---:|---:|---:|---:|---:|
| `mini_rule_seed42` | 0.560 | 0.158 | 0.466 | 0.490 | 0.104 | 0.402 |
| `mini_rule_dynamic_seed42` | 0.560 | 0.150 | 0.459 | 0.488 | 0.100 | 0.409 |
| `mini_rule_len_seed42` | 0.567 | 0.150 | 0.459 | 0.492 | 0.094 | 0.406 |
| `mini_rm_seed42` | 0.209 | 0.008 | 0.180 | 0.178 | 0.011 | 0.161 |
| `mini_mix_seed42` | 0.500 | 0.090 | 0.353 | 0.464 | 0.062 | 0.312 |
| `mini_rule_seed7` | 0.575 | 0.135 | 0.451 | 0.508 | 0.105 | 0.402 |
| `math_cold_cot_seed42` | 0.552 | 0.128 | 0.436 | 0.488 | 0.081 | 0.364 |
| `math_cold_seed42` | 0.000 | 0.000 | 0.000 | 0.024 | 0.004 | 0.003 |

## 4. seed 汇总（SOP §10.4）

| 臂（除 seed 外全部相同） | seeds | greedy mean [min, max] | pass@1 mean [min, max] |
|---|---|---|---|
| MiniMind-64M|rule|6|512|0.1|0|0|0.5|0.5 | 42, 7 | 0.391 [0.388, 0.395] | 0.335 [0.332, 0.339] |
| |||||||| | ,  | 0.186 [0.000, 0.372] | 0.161 [0.010, 0.311] |

## 5. 图表清单（caption 全文）

### `reports/figures/fig1_training_dynamics.png`

models: MiniMind-64M | train data: dataset/math_rlvr_train.jsonl | eval data: dataset/math_heldout.jsonl (400 problems, no train overlap) | seeds: 42, 7 | held-out sampling: 16 completions per problem at temperature 0.8 | length convention: tokens up to the first EOS, prompt excluded | source: runs/<tag>/metrics.csv, one point per log_interval

### `reports/figures/fig2_acr.png`

models: MiniMind-64M | train data: dataset/math_rlvr_train.jsonl | eval data: dataset/math_heldout.jsonl (400 problems, no train overlap) | seeds: 42, 7 | held-out sampling: 16 completions per problem at temperature 0.8 | length convention: tokens up to the first EOS, prompt excluded | source: runs/<tag>/metrics.csv group counters; ACR = share of groups that step

### `reports/figures/fig3_heldout_arms.png`

models: MiniMind-64M | train data: dataset/math_rlvr_train.jsonl | eval data: dataset/math_heldout.jsonl (400 problems, no train overlap) | seeds: 42, 7 | held-out sampling: 16 completions per problem at temperature 0.8 | length convention: tokens up to the first EOS, prompt excluded | source: runs/<tag>/eval.json; a bar is absent when that run has no evaluation

### `reports/figures/fig4_difficulty.png`

models: MiniMind-64M | train data: dataset/math_rlvr_train.jsonl | eval data: dataset/math_heldout.jsonl (400 problems, no train overlap) | seeds: 42, 7 | held-out sampling: 16 completions per problem at temperature 0.8 | length convention: tokens up to the first EOS, prompt excluded | source: runs/<tag>/eval.json by_difficulty; the 400 held-out problems are split evenly

### `reports/figures/fig5_length_histogram.png`

models: MiniMind-64M | train data: dataset/math_rlvr_train.jsonl | eval data: dataset/math_heldout.jsonl (400 problems, no train overlap) | seeds: 42, 7 | held-out sampling: 16 completions per problem at temperature 0.8 | length convention: tokens up to the first EOS, prompt excluded | source: runs/<tag>/eval.jsonl per-completion token counts

### `reports/figures/fig6_reward_eval_gap.png`

models: MiniMind-64M | train data: dataset/math_rlvr_train.jsonl | eval data: dataset/math_heldout.jsonl (400 problems, no train overlap) | seeds: 42, 7 | held-out sampling: 16 completions per problem at temperature 0.8 | length convention: tokens up to the first EOS, prompt excluded | reference arm is the first baseline-signature row of the registry

## 6. 警告与缺口

- reference arm: mini_rule_seed42
- not plotted (missing train gain or eval): math_cold_cot_seed42, math_cold_seed42
- `math_cold_cot_seed42`: only 0 logged step(s); no trend can be read
- `math_cold_seed42`: not archived yet: run `python scripts/archive_run.py --run-tag math_cold_seed42`; only 0 logged step(s); no trend can be read
