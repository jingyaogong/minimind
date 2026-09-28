# 04 · 证据契约（每条 claim → 证据 → 复算命令）

C 级别：C0 了解 / C1 负责模块 / C2 设计优化 / C3 结果影响。C3 必须有指标定义 + 记录。

## 主表

| # | Claim | 级别 | 证据文件 | 复算命令（仓库根目录执行） | 面试风险 |
|---|---|---|---|---|---|
| 1 | 数据三件套确定性生成、去重、无泄漏 | C3 | `dataset/math_*.jsonl`、`dataset/manifests/` | `python scripts/clean_math_data.py --check-only` | 低 |
| 2 | 三态 verifier + 分档奖励 | C2 | `trainer/math_verifier.py`、`tests/test_math_verifier.py` | `python -m pytest -q tests/test_math_verifier.py` | 低 |
| 3 | 6 臂预注册 GRPO（rule / rm / mix / dynamic / len / seed7） | C2 | `configs/experiment_registry.csv`、`runs/*/manifest.json` | `python scripts/experiment_registry.py --print-command mini_rule_seed42` | 中 |
| 4 | held-out 400 题 × 16 采样，greedy + pass@k | C3 | `runs/*/eval.json`、`runs/*/eval.jsonl` | `python scripts/eval_heldout.py --model_path out/<tag>_768.pth --data_path dataset/math_heldout.jsonl --output runs/<tag>/eval.json --num_samples 16 --pass_at 1,4,8,16` | 中 |
| 5 | 逐题配对 McNemar + 分档 | C3 | `reports/paired_compare.json`、`reports/paired_compare.md` | `python scripts/paired_compare.py --baseline math_cold_cot_seed42 --arms mini_rule_seed42 mini_rule_seed7 mini_rule_dynamic_seed42 mini_rule_len_seed42 mini_rm_seed42 mini_mix_seed42` | 中 |
| 6 | RM 臂 reward hacking（核心负结果） | C3 | `logs/mini_rm_seed42_metrics.csv`、`runs/mini_rm_seed42/eval.json` | 上一条 + `python scripts/summarize_runs.py --require-eval` | 中（会被深挖机制） |
| 7 | 重跑噪声地板（同命令同种子） | C3 | `reports/paired_compare_rerun.json`、`logs/*_metrics.csv` | `python scripts/compare_runs.py --a mini_rule_seed42 --b mini_rule_rerun_seed42 --expect-identical` | 低（加分项） |
| 8 | 长度惩罚 no-op（分布外 cap） | C3 | `logs/*_samples.jsonl`、`reports/report.md` §4.6 | `python scripts/error_analysis.py --run-tag mini_rule_len_seed42 --mode sampled` | 低 |
| 9 | 错误类别 + 形态学（400 条全量） | C3 | `reports/error_analysis_*.md`、`reports/audit_*.md` | `python scripts/error_analysis.py --run-tag mini_rm_seed42 --mode greedy` | 中 |
| 10 | 工程规范（config / 日志 / 归档哈希） | C2 | `scripts/archive_run.py`、`runs/*/manifest.json` | `python scripts/archive_run.py --check --strict --run-tag mini_rule_seed42` | 低 |

## 关键数字卡（背下来；出处都在上面）

| 指标 | 值 |
|---|---|
| 冷启动基线 `math_cold_cot_seed42` | greedy 0.3725 / pass@1 0.3113 / pass@16 0.5525 |
| 模型规模 / 训练数据 / held-out | 63.91M 参数 / 2000 题 / 400 题 × 16 采样 |
| `mini_rule_seed42` | greedy 0.3950 / pass@1 0.3323 / pass@16 0.5575（greedy 净 +9，p=0.0636；能力净 +2，p=0.8679） |
| `mini_rule_seed7` | 0.3875 / 0.3386 / 0.5775（净 +6，p=0.1796；能力净 +10，p=0.1433） |
| `mini_rule_dynamic_seed42` | 0.3900 / 0.3325 / 0.5475（净 +7，p=0.1671；能力净 −2，p=0.8642） |
| `mini_rule_len_seed42` | 0.3925 / 0.3311 / 0.5575（净 +8，p=0.1153）；长度 114.9 vs 115.2 |
| `mini_rm_seed42` | greedy 0.1325 / pass@1 0.1167 / pass@16 0.2850（净 −96，p<0.0001）；训练 reward 1.684→2.011，同批 correct 0.233→0.025 |
| `mini_mix_seed42` | greedy 0.3150 / pass@1 0.2795 / pass@16 0.5000（净 −23，p=0.0011） |
| 重跑对照 `mini_rule_rerun_seed42` | 0.4000 / 0.3367 / 0.5425（对 `mini_rule_seed42` 净 +2，p=0.7539；能力净 −6，p=0.3075） |
| 训练成本 | 1.16–1.26 s/step，1000 步 ≈ 20 分钟，显存峰值 7.2 GB（单卡 4090） |
| 测试 | `python -m pytest -q` → 313 passed |

## 面试官要现场复算时的路径（按顺序给他看）

1. `runs/mini_rm_seed42/eval.json` → 0.1325 这个数从哪来；
2. `logs/mini_rm_seed42_metrics.csv` → reward 1.684→2.011、correct 0.233→0.025；
3. `reports/paired_compare.md` → 配对净变与 p 值（同一批 400 题，逐题配对，不是比较平均值）；
4. `reports/error_analysis_mini_rm_seed42.md` → 400 条输出 100% 格式完好、86.8% 算术错误；
5. `reports/report.md` §4.7 → 为什么单臂差异不能当效应量（35/200 复现）；
6. `runs/mini_rule_seed42/manifest.json` → 归档哈希与命令。

> 一句话原则：**每个数字都有 JSON/CSV 出处，每个结论都写了边界**。
