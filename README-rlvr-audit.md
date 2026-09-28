# RLVR 训练动态审计（MiniMind-64M / 单卡 4090）

**Auditing RLVR training dynamics: reward, length bias, and capability decoupling**

> 本文件是本副本（MiniMind 上游 + 本文实验）的**项目说明与复现入口**。
> 完整实验报告 → [`reports/report.md`](reports/report.md)（每个数字都能指回 `runs/` / `logs/` / `reports/` 里的原始文件）
> ｜实验协议 → [`docs/posttraining-rlvr-sop.md`](docs/posttraining-rlvr-sop.md)
> ｜服务器跑通手册 → [`docs/server_runbook.md`](docs/server_runbook.md)

---

## 0. 问题与结论

**问题**：在一个有确定答案的数学任务上，GRPO 训练里 reward 上升，是否代表模型能力上升？
还是模型只是学会了（a）固定格式、（b）把回答写长、（c）迎合 reward model、（d）在组内奖励退化时重复已有行为。

**结论**：在这个设置里，**奖励里有没有正确性项，比 optimizer 和超参更决定性**。

| # | 结论 | 强度 |
|---|---|---|
| 1 | **RM 奖励 = 内容层面的 reward hacking**：训练 reward +0.327、同一批 rollouts 上 verifier 正确率 −0.208、held-out greedy 0.373 → **0.133**（400 题里 102 题由对变错、6 题由错变对，p<0.0001） | 强 |
| 2 | 纯 GT verifier 的四个臂方向一致但**不显著**（greedy 净 +6~+9，p=0.06~0.18） | 中 |
| 3 | 这些收益全部落在**同命令、同种子重跑**的噪声地板之内（重跑 greedy 净 +2 / 能力净 −6）→ 单臂单次运行不能作为证据 | 强 |
| 4 | **长度惩罚臂是 no-op**：cap=300 落在观测分布之外（训练采样 max 228 / 评估采样 max 280），240 条采样 0 次触发 | 强 |
| 5 | CoT 模板是 RL **之前**的第一个瓶颈：只宣告结果的 CoT 让 held-out 变成 0/20，改成列竖式后升到 0.373 | 强（单变量，探索性） |
| 6 | held-out 的**难度标签倒挂**：medium（0.128）比 hard（0.436）更难 → 难度必须用模型侧标定 | 强（事实），原因未定 |
| 7 | **RM 臂崩的是算术、不是生成**：400 条 greedy 100% 格式完好、0 循环 / 0 截断 / 0 无答案，86.8% 是纯算术错误 | 强 |

## 1. 数据来源与完整性

数据**全部由本仓库脚本确定性生成**，不依赖任何外部数据集（`manifest.mode = synthetic-only`）：

```bash
python scripts/clean_math_data.py --synthetic-only --seed 42   # 生成（本仓库已带产物，不必重跑）
python scripts/clean_math_data.py --check-only                 # 校验，期望 PASS
# → sft=2000 rlvr=2000 heldout=400 unique_keys=2000/400 overlap=0 manifest=yes audit_rows=50
```

| 文件 | 行数 | sha256 | 用途 |
|---|---:|---|---|
| `dataset/math_sft_train.jsonl` | 2000 | `1740c0d038f4adbc8c339371ddb1ac9c198ad038a5e3b21898a83f93e7b79f6a` | 冷启动 SFT（含列竖式 CoT） |
| `dataset/math_rlvr_train.jsonl` | 2000 | `f55c0b6954ea5e57abd844e88309bf8a8b1e7d0e7f2b82bfb60fcae5b2286478` | GRPO 训练（user-only prompt + gt） |
| `dataset/math_heldout.jsonl` | 400 | `f829eb084a3e428dd157e7ec20c7ed4160de183990e2cc83d0688a4c558ce4fa` | held-out 评估（与训练 0 重叠） |
| `dataset/probe/calibration_probe.jsonl` | 120 | `2392857656fc5725c503104f168b232b1a88b9fb250edf03a24c2f9eeeda30d5` | 难度探针（6 档 × 20 题） |
| `dataset/manifests/math_manifest.json` | — | — | 来源/计数/丢弃原因（36 条重复被弃） |

**三条纪律**（违反任何一条，`reports/` 里的数字就失去意义）：

1. **不要重跑生成脚本**。数据是确定性生成的，重跑会改变 sha256；要换数据就换新 tag 并重跑整条链路。
2. **答案标记 `答案是` 只认一处**：数据模板、`trainer/math_verifier.py`、`--reward_marker`、`scripts/eval_heldout.py --marker` 必须同时一致。
3. **train 与 held-out 的 key 集合重叠必须为 0**（`--check-only` 会检查）。

权重与奖励模型（外部下载，不入库）：

| 用途 | 来源 | 落到哪 | sha256 |
|---|---|---|---|
| 起点权重 | ModelScope `gongjy/minimind-3-pytorch` → `full_sft_768.pth` | `out/` | `5b8c49f6c9d965092e651cafeeaeb8705558632b3fd4ac8ab319cbc5a3cbc4a0` |
| 冷启动权重 | **本实验产物**（`train_full_sft.py` 250 步） | `out/math_cold_cot_seed42_768.pth` | `a7beb45f94cb3fbf55810691b908455e859053c70aaf7fcedf9f5a00146c8308` |
| 奖励模型 | ModelScope `Shanghai_AI_Laboratory/internlm2-1_8b-reward` | 与仓库同级目录，或用 `--reward_model_path` 传绝对路径 | `python scripts/rm_load_check.py` 自检 |

> 冷启动权重是整条链路的起点：`configs/experiment_registry.csv` 的 `DEFAULT_FROM_WEIGHT` 指向它。

## 2. 环境

| 组件 | 版本 | 备注 |
|---|---|---|
| GPU | RTX 4090 24 GB（单卡，实测峰值 **7.2 GB**） | ≥16 GB 显存即可 |
| Python | 3.12.3 | |
| torch | 2.8.0+cu128 | AutoDL 镜像自带，**不要动** |
| transformers | 4.57.6 | |
| datasets | 3.6.0 | |
| numpy | 2.3.2 | ⚠️ 不要按 `requirements.txt` 降到 1.26.4 |
| matplotlib | 3.10.5 | `summarize_runs.py` 出图 |
| pyyaml | 6.0.2 | |
| sentencepiece | 0.2.0 | ⚠️ 奖励模型的 tokenizer 需要，**必须补装** |
| modelscope | 1.37.0 | 下载权重 |
| pytest | ≥8 | |

```bash
pip install "transformers==4.57.6" "datasets==3.6.0" pyyaml matplotlib sentencepiece==0.2.0 modelscope pytest
python -m pytest -q          # 期望 313 passed（纯 CPU 也能跑）
```

## 3. 最小复现路径（一张 4090，约 1 小时）

```bash
# 0) 权重（一次性 ~130 MB）
modelscope download --model gongjy/minimind-3-pytorch --local_dir ../models/minimind-3-pytorch
cp ../models/minimind-3-pytorch/full_sft_768.pth out/ && sha256sum out/full_sft_768.pth
#   期望 5b8c49f6c9d965092e651cafeeaeb8705558632b3fd4ac8ab319cbc5a3cbc4a0

# 1) 预检：环境 / 数据哈希 / 权重 / 磁盘（期望 0 blockers）
python scripts/server_preflight.py --expect-gpu --min-disk-gb 50

# 2) 数据校验（期望 PASS）
python scripts/clean_math_data.py --check-only

# 3) 冷启动 SFT：250 步（仓库已带同哈希权重时可跳过）
cd trainer && python -u train_full_sft.py \
  --data_path ../dataset/math_sft_train.jsonl --from_weight full_sft \
  --save_weight math_cold_cot_seed42 --epochs 2 --batch_size 16 --learning_rate 1e-5 \
  --max_seq_len 512 --num_workers 0 --save_interval 1000 --log_interval 20
cd ..

# 4) 训练一个 RL 臂：1000 步 ≈ 20 分钟；命令由预注册表渲染，勿手敲
bash -c "$(python scripts/experiment_registry.py --print-command mini_rule_seed42)"

# 5) 评估（400 题 × 16 采样 ≈ 15–25 分钟）→ 归档 → 汇总 → 配对检验 → 错误分类
python scripts/eval_heldout.py --model_path out/mini_rule_seed42_768.pth \
  --data_path dataset/math_heldout.jsonl --output runs/mini_rule_seed42/eval.json \
  --num_samples 16 --max_new_tokens 512 --temperature 0.8 --top_p 0.85 --top_k 50 \
  --seed 42 --device cuda --marker '答案是'
python scripts/archive_run.py --run-tag mini_rule_seed42 --force
python scripts/summarize_runs.py --require-eval
python scripts/paired_compare.py --baseline math_cold_cot_seed42 --arms mini_rule_seed42
python scripts/error_analysis.py --run-tag mini_rule_seed42 --mode greedy
```

## 4. 六个臂（`configs/experiment_registry.csv` 是唯一事实来源）

所有臂共用：`G=6`、`lr=3e-7`、`beta=0.1`、`temperature=0.8`、`batch_size=2`、`epochs=1`（1000 步）、
`max_seq_len=max_gen_len=512`、`thinking_ratio=1.0`、`loss_type=grpo`，起点都是 `math_cold_cot_seed42`。
命令一律用 `scripts/experiment_registry.py --print-command <tag>` 渲染，避免手敲漂移。

| run_tag | 单变量改动 | 预注册假设 | 结果（held-out greedy，400 题；净变=对变对/错变错 的配对 McNemar） |
|---|---|---|---|
| `math_cold_cot_seed42` | 冷启动（起点，不在登记表） | — | 0.3725（pass@16 0.5525） |
| `mini_rule_seed42` | 参照臂（纯 GT verifier） | verifier 即可抬升 | 0.3950；净 **+9**，p=0.064 |
| `mini_rule_seed7` | seed=7 | 方向可复现 | 0.3875；净 +6，p=0.180 |
| `mini_rule_dynamic_seed42` | 过滤退化组 | 提高每步有效梯度 | 0.3900；净 +7，p=0.167 |
| `mini_rule_len_seed42` | 长度惩罚 coef=0.05，cap=300 | 缩短且不掉准确率 | 0.3925；净 +8，p=0.115（**实为 no-op**，见结论 4） |
| `mini_rm_seed42` | 官方启发式 + 1.8B RM | RM 抬 reward 但不抬准确率 | **0.1325**；净 **−96**，p<0.0001 ✅ |
| `mini_mix_seed42` | rule 与 RM 标准化后加权 | 标准化混合优于 RM-only | **0.3150**；净 **−23**，p=0.0011（比 RM-only 轻，但仍退化） |
| `mini_rule_rerun_seed42` | 同命令同种子重跑（对照，非登记臂） | — | 0.4000；对 `mini_rule_seed42` 净 +2，p=0.754 → **噪声地板** |

## 5. 仓库导览（本实验新增/改造的部分）

```text
configs/experiment_registry.csv     # 预注册表：唯一事实来源（含 hypothesis 列）
dataset/                            # 三件套 + 探针 + manifests（见 §1）
trainer/math_verifier.py            # 三态 verifier（correct/incorrect/no_answer）+ 奖励分档
trainer/math_rl_dataset.py          # RLVR Dataset（prompt/gt/难度）
scripts/clean_math_data.py          # 确定性数据生成 + 校验
scripts/server_preflight.py         # 训练前预检（30 项）
scripts/experiment_registry.py      # 渲染训练命令（勿手敲）
scripts/eval_heldout.py             # held-out 评估（greedy + pass@k）
scripts/paired_compare.py           # 逐题配对 McNemar（含 JSON 输出）
scripts/compare_runs.py             # 两个 run 的逐 step 复现性对照
scripts/error_analysis.py           # SOP §10.4 错误类别统计 + 形态学
scripts/summarize_runs.py           # 跨臂汇总 + 6 张图 + 决策树诊断
scripts/archive_run.py              # 归档 + 哈希（manifest.json）
scripts/rm_load_check.py            # 奖励模型加载自检
runs/<tag>/                         # 每个 run 的证据：eval.json / eval.jsonl / manifest.json / train.log
logs/<tag>_metrics.csv|_samples.jsonl|_config.json
reports/report.md                   # ★ 完整报告（结论、方法、局限、面试表述）
reports/error_analysis_*.md, audit_*.md, summary.md, paired_compare*.md, figures/
tests/                              # 313 个测试（verifier / 数据 / 脚本 / 归档）
```

## 6. 失败结果与局限（诚实清单）

- **RM-alone 与 mix 都退化**（§0 结论 1）：这是本项目的核心负结果，不是"实验失败"。
  mix 的两种解释（RM 分量污染 vs 标准化改变了 rule 分量形态）**无法完全分离**。
- **长度惩罚的 cap 设错**（结论 4）：结论是"没有东西可惩罚"，不是"惩罚机制无效"。
- **难度标签不可信**（结论 6）:medium 档比 hard 档更难，原因未定位。
- **训练不可复现**：同命令同种子重跑，200 个记录点只有 35 个一致（step 180 起分岔）；
  评估侧噪声地板 = greedy 净变 ±2、能力净变 −6。**因此所有臂间差异都不能单独引用**。
- **规模边界**：63.91M 参数 + 2000 题 + 单卡，结论**不能外推**到 1.5B/7B 级模型。
- **未做（范围边界，已在报告 §9 记录理由）**：工业近似层（Qwen2.5-1.5B + TRL/verl）；
  held-out 扩到 500 题（实测效应量落在噪声地板内，扩样不会改变结论）。
- **工程缺陷两轮（已修并留档）**：决策树曾把 RM 臂误诊为"长度偏置"、把标准化奖励读成"没动"；
  第二轮又发现它把 held-out 的点估计下降写成"显著下降"。详见 `reports/report.md` §6。

## 7. 已知坑（复现时最可能卡住的地方）

| 症状 | 原因 | 处理 |
|---|---|---|
| `libgomp: Invalid value for environment variable OMP_NUM_THREADS` | 镜像把 `OMP_NUM_THREADS` 设成 0 | `export OMP_NUM_THREADS=8` |
| `ModuleNotFoundError: sentencepiece` | 奖励模型的 tokenizer 需要它 | `pip install sentencepiece==0.2.0` |
| numpy/torch 版本冲突 | `requirements.txt` 里的 `numpy==1.26.4` 是上游遗留 | 保持镜像自带的 numpy 2.x |
| 归档报 `[partial]`、缺 `metrics.csv`/`samples.jsonl` | `train_full_sft.py`（上游脚本）不写这些 | 冷启动臂的 `partial` 是**预期**，别手工造 CSV |
| 格式率统计把正常输出判成"没有 think 块" | prompt 末尾自带 `<think>`（`open_thinking=True`），补全里只有 `</think>` | 以 `think_closed` 字段为准，别找开标签 |
| 找不到权重 / 路径错 | 训练脚本从 `trainer/` 启动、评估脚本从仓库根启动 | 按 §3 的命令顺序执行；奖励模型路径用绝对路径 |
| 训练曲线对不上 | cuBLAS 归约 + backward 的 atomicAdd 非确定 | 已知问题；要严格复现需 `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8`（本轮未启用） |

## 8. 出处与许可

- 基于 [MiniMind](https://github.com/jingyaogong/minimind)（Apache-2.0，见 `LICENSE`）；本副本的改动同样以 Apache-2.0 发布。
- 奖励模型 [InternLM2-1.8B-Reward](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-1_8b-reward)（上海 AI Lab）。
- 实验中所有数字以 `reports/report.md` 为准；报告生成于 2026-09-28，全部可在本机复算：
  `python -m pytest -q`（313 passed）+ `scripts/summarize_runs.py` + `scripts/paired_compare.py` + `scripts/error_analysis.py`。
