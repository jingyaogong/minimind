# 03 · 真话边界

图例：`可以写` / `谨慎写` / `补证据后写` / `不能写` / `无法判断`。

## 可以写（有归档证据）

| Claim | 证据 |
|---|---|
| 基于 MiniMind-64M 上游改造，搭建数学 RLVR（GRPO）训练—评估链路 | `trainer/`、`scripts/experiment_registry.py`、`runs/*/manifest.json` |
| 生成 2000 SFT / 2000 RLVR / 400 held-out 的确定性数据集，表达式 key 去重、train 与 held-out 重叠 0、sha256 清单、50 条人工审计 | `scripts/clean_math_data.py --check-only` → `PASS`；`dataset/manifests/` |
| 实现三态规则 verifier（correct / incorrect / no_answer）+ 分档奖励，配单测 | `trainer/math_verifier.py`、`tests/test_math_verifier.py`（全仓 313 tests passed） |
| 完成 6 个预注册臂 + 1 个重跑对照的训练与 400 题 × 16 采样评估 | `configs/experiment_registry.csv`、`runs/*/eval.json` |
| RM 奖励单独作为数学奖励时出现内容层 reward hacking：训练 reward 1.684→2.011，同批 rollout verifier 正确率 0.233→0.025，held-out greedy 0.373→0.133（配对 102 变错 / 6 变对，p<0.0001） | `reports/report.md` §4.8、`reports/paired_compare.json` |
| 纯 GT verifier 四臂方向一致但不显著（greedy 净 +6~+9，p 0.06~0.18），落在重跑噪声地板内 | `reports/paired_compare.json`、`reports/report.md` §4.7 |
| 长度惩罚臂为 no-op（cap 高于分布上界，240 条采样 0 触发） | `reports/report.md` §4.6 |
| 同命令同种子重跑不可复现（200 记录点仅 35 一致，step 180 分岔）；重跑模型的逐题净变为 greedy ±2 / 能力 −6 | `scripts/compare_runs.py` 输出、`reports/paired_compare_rerun.json`、`reports/report.md` §4.7 |
| 错误类别分析：RM 臂 400 条输出 100% 格式完好、0 循环 / 0 截断，86.8% 为纯算术错误 | `reports/error_analysis_mini_rm_seed42.md` |

## 谨慎写（真实，但必须配边界说明）

| Claim | 风险 | 安全措辞 |
|---|---|---|
| 「规则 verifier 带来提升」 | 不显著、且在噪声地板内 | 「四臂方向一致（净 +6~+9 / 400 题），但未越过同配置重跑的噪声地板，只能作为方向性证据」 |
| 「发现 reward hacking」 | 只在小模型 + 单一 RM 上验证 | 「在 64M 规模、InternLM2-1.8B-RM 的组合下观察到内容层 reward hacking；不外推到其他规模」 |
| 「两个种子」 | 只有 rule 臂有 seed 7 复现 | 「seed 42 与 seed 7 方向一致；但训练曲线在种子间反号，说明单次运行不足以支撑结论」 |
| 「冷启动 SFT」 | 只有 250 步、2000 条 | 「小规模冷启动 SFT（250 步 / 2000 条），用于给 RL 一个可读的 CoT 起点」 |

## 补证据后写

| Claim | 缺什么 |
|---|---|
| 多 seed（≥3）统计 | 每臂再加 2 个种子；当前只有 rule 臂 2 个种子 |
| pass@32 / 更大 held-out | 需要更多采样与题目；当前效应量落在噪声地板内，扩样不改结论（已评估后主动放弃） |
| 语音 / ASR 相关经验 | 需要真实动手（半天 demo 起步） |
| 可公开的代码仓库 | 需要整理脱敏 + 大文件排除 |

## 不能写

- Qwen2.5-1.5B / TRL / Open-R1 / verl 的任何实操（**没做**，SOP 工业近似层未执行）。
- 「RL 让模型学会了新推理能力」「提升了推理能力」。
- 任何多卡、大规模、生产/上线表述。
- C/C++/perl 精通（除非你能当面写出来）。
- 把上游 MiniMind 说成自己训练的模型。

## 无法判断

- 简历前两个项目的数字（94.12% / 84.16% / BLEU 35.62）：现有材料无法复核，**你需要自己准备口径自述**；面试前先假设面试官会问「测试集怎么划的、指标怎么算、baseline 是什么」。
