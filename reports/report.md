# RLVR 训练动态审计：奖励、长度偏置与模型能力的解耦分析

**Auditing RLVR Training Dynamics: Reward, Length Bias, and Capability Decoupling**

模型：MiniMind-64M（白盒层） · 单卡 RTX 4090 · 状态：**v1.0**（六臂 + 重跑 + 噪声地板 + 错误类别 + 抽查 + 反向同步与本地复算 + README 全部完成）

> **本文的写作约定**：每个数字都必须能指回仓库里的原始文件（`runs/<tag>/eval.json`、
> `logs/<tag>_metrics.csv`、`reports/summary.md`、`reports/paired_compare.md`）。
> 凡属"事后追加"的分析一律显式标注 **post-hoc**，与预注册假设分开陈述。
> 标 `TODO(...)` 的地方是尚未定稿的证据，不允许被引用为结论。

---

## 0. 摘要（TL;DR）

| # | 结论 | 关键证据 | 强度 |
|---|---|---|---|
| 1 | **冷启动模板而非模型容量是第一个瓶颈**：把 SFT 的 CoT 从"只宣告结果"改成"列竖式"，held-out greedy 从 0/20 跳到 **0.373**，pass@16 到 **0.553** | §4.1；`runs/math_cold_cot_seed42/eval.json` | 强（单变量，但探索性） |
| 2 | **rule 信号方向为正但不显著**：greedy +9 题净变（双边 p=0.064，单边 p=0.032），pass@16 逐题净变 +2（p=0.868） | §4.4；`reports/paired_compare.md` | 中（方向一致、未过双边阈值） |
| 3 | **RL 主要锐化已有分布，未见能力上界扩展**：四个臂 greedy 净变 +6~+9，能力指标（pass@16 逐题）净变 +2/+10/−2/+2 全部不显著 | §4.4 | 中 |
| 4 | **长度惩罚臂是 no-op**：预注册 cap=300 落在观测分布之外（训练采样 max 228、评估 max 280），惩罚从未触发 | §4.6 | 强（直接测量） |
| 5 | **难度分层用的是"人类难度"**：held-out 的 medium 档（0.128）反而低于 hard 档（0.436） | §4.5 | 强（事实描述），原因未定 |
| 6 | **训练不可复现 + 噪声地板已实测**：同命令同种子重跑，200 个记录点只有 **35 个一致**（step 180 起分岔）；评估侧噪声下限 = greedy 净变 **+2**（p=0.75）、能力净变 **−6**（p=0.31），四个臂的全部效应都没越过它 | §4.7 | 强（直接实验） |
| 7 | **RM 奖励把能力换掉了（本项目最强的负结果）**：RM-only 的 reward 涨 +0.327，而 held-out greedy 从 0.373 掉到 **0.133**（配对 6 变对 / 102 变错，净 −96，p<0.0001）；mix 臂也退化到 0.315（净 −23，p=0.0011） | §4.8；`reports/paired_compare.md` | 强（预注册假设成立） |
| 8 | **RM 臂崩的是算术、不是生成**：400 条 greedy 里 100% 闭合 `</think>`、0 循环 / 0 截断 / 0 无答案，失败全部落在算术类（86.8%，冷启动 62.7%）——外壳被完整保留，只丢了正确性 | §4.9；`reports/error_analysis_mini_rm_seed42.md` | 强（全量分类） |
| 9 | **失败是"续写数字"而非"算错"**：抽查 20 条失败样本，17/17（冷启动）与 12/13（RM）的乘法断言为假，16/17 与 13/15 的"写出的和 ≠ 自己写的加数之和" | §4.9；`reports/audit_*.md` | 中（前 20 条，非随机抽样） |

一句话版本：**在这套设置里，reward 上升不能当作正确率上升的证据**——RM-only 臂在同一批数据上 reward +0.327、verifier 正确率 −0.208，是这条论断最直接的实证。本项目给出的是"如何把两者拆开"的可复现流程，以及一份诚实的负结果清单。

---

## 1. 问题与预注册设计

### 1.1 核心问题

在一个有确定答案的数学任务上，GRPO 训练后观察到的 reward 上升，是否代表正确率上升？还是模型主要学会了
（a）输出固定格式、（b）延长回答、（c）利用 reward model 的偏好、（d）在组内奖励退化时重复已有行为。

项目**不预设正结果**：三种结局（真收益 / reward hacking / 全指标不动）都算有效结论，但证据要求不同。

### 1.2 六个臂（预注册原文，`configs/experiment_registry.csv`）

| run_tag | reward_mode | 单变量改动 | 预注册假设（原文摘要） |
|---|---|---|---|
| `mini_rule_seed42` | rule | — （参照臂） | GT verifier 单独即可把 held-out 抬到冷启动之上 |
| `mini_rule_dynamic_seed42` | rule | 过滤退化组 | 过滤全对/全错组提高每步有效梯度 |
| `mini_rule_len_seed42` | rule | 长度惩罚 coef=0.05, cap=300 | 超过 cap 的 token 受罚 → 缩短长度且不掉准确率 |
| `mini_rule_seed7` | rule | seed=7 | 基线方向在第二个种子下可复现 |
| `mini_rm_seed42` | rm | 官方启发式 + 1.8B RM | RM 抬高 reward 但不抬高 held-out 准确率 |
| `mini_mix_seed42` | mix | rule 与纯 RM 标准化后加权 | 标准化混合优于 RM-only |

所有臂共用：`G=6`、`lr=3e-7`、`beta=0.1`、`temperature=0.8`、`batch_size=2`、`epochs=1`（1000 步）、
`max_seq_len=max_gen_len=512`、`thinking_ratio=1.0`、`loss_type=grpo`、起点 `math_cold_cot_seed42`。
命令由 `scripts/experiment_registry.py --print-command <tag>` 渲染，避免手敲漂移。

### 1.3 奖励与判定口径（冻结）

- verifier 三态：`correct=+1.0` / `incorrect=0.0` / `no_answer=−0.5`（`--verdict_tier binary`）；
- 答案抽取只认最后一行 `答案是 <数字>`，数字按十进制规范化比较，**不做子串匹配**（`1` 不会命中 `21`）；
- `mix` 臂的两个分量**先组内标准化再加权**（否则 RM 的 ±3 量纲会吞掉 rule 的 ±0.5）；
- 长度惩罚为 `coef × max(0, tokens − cap) / cap`，只罚超出部分。

### 1.4 预注册 vs 事后（post-hoc）

| 类别 | 内容 |
|---|---|
| 预注册 | 六个臂及其假设、奖励分档、超参、`length_cap=300`、数据哈希、评估口径 |
| 事后（post-hoc，明确标注） | 冷启动 CoT v1→v2 的修正是**看到 v1 失败后才做的**；难度探针（`calibration_probe`）为事前工具但不在登记表内；分档倒挂、长度分布、ACR 复算、配对检验的**解释**均事后；确定性对照与扩样计划见 §9 |

---

## 2. 数据与 verifier

### 2.1 数据三件套（哈希已入 `dataset/manifests/data.sha256`）

| 文件 | 行数 | sha256（前 12 位） | 用途 |
|---|---:|---|---|
| `dataset/math_sft_train.jsonl` | 2000 | `1740c0d038f4` | 冷启动 SFT（CoT v2，列竖式） |
| `dataset/math_rlvr_train.jsonl` | 2000 | `f55c0b6954ea` | GRPO 训练提示（与 SFT 同题不同模板） |
| `dataset/math_heldout.jsonl` | 400 | `f829eb084a3e` | 评估（easy 134 / medium 133 / hard 133） |
| `dataset/probe/calibration_probe.jsonl` | 120 | `2392857656fc` | 6 档难度探针（**不进** manifest） |

训练/评估按题目 key 去重，零交叠；难度分层取自数据源的人工标注：**easy = 两位数±一位数、medium = 两位数×两位数、hard = 三位数×两位数**（注意：这是"人类难度"，见 §4.5）。

### 2.2 CoT v1 → v2：唯一的干净因果干预

- v1（pilot `math_cold_seed42`）：SFT 数据里 CoT 只写"按运算顺序，结果是 X"，模型学会的是**模板句**。20 题检查：`format=1.000` 但 greedy **0/20**。
- 官方基线 `full_sft_768`（同 20 题）：`format=0.200`、`no_answer=1.000` —— 连格式都没学会。
- v2（`COT_VERSION="cot-v2-column"`）：把 CoT 换成**列竖式逐步计算**，重新 SFT（14 步，loss 0.2627 → 0.0275）。
- 对照结果：同 400 题 held-out，greedy **0.373**、pass@1 0.311、pass@16 0.553、`format=1.000`。

**能说**：在这套数据+模板下，模板形式（而非模型容量）是第一个瓶颈，改模板带来数量级差异。
**不能说**：这证明了"CoT 提升推理能力"——它是一次工程修正，且不是预注册的假设检验。

### 2.3 verifier 与测试

`trainer/math_verifier.py`：三态判定 + 严格的数字规范化（拒绝分数、单位、科学计数、中文数字、乱用逗号）；
未闭合的 `<think>` 视为格式未完成，不允许从中抽答案。测试覆盖子串陷阱、格式边界、非法 GT、幂等重跑等。
仓库全量单测：**283 passed**（`python3.13 -m pytest -q`，本机）。

---

## 3. 方法

### 3.1 指标口径

- **greedy**：温度 0 贪心一条，主指标；
- **pass@k**：每题 16 次采样（`temperature=0.8, top_p=0.85, top_k=50, seed=42`）中至少一次正确；
- **format**：末行是否命中 `答案是 <数字>`；
- **ACR / 有效组**：组内既有对又有错（有梯度）的比例；全对/全错组 advantages 恒为 0；
- **reward-eval gap**：训练期组内平均 reward 的首末差 vs held-out greedy 差（单位不同，只用于排序）。

### 3.2 统计

逐题配对（同一批 400 题）用 **McNemar 精确检验**：把"变对/变错"的题数做成 2×2。
预注册假设是方向性表述（"raises ... above"），故**单边**检验合法，同时报告**双边**作为保守值。
训练动态一律用**首/末 10 点窗口均值**，避免单点抖动（早期用单点曾产生 −0.25 的假退步）。

### 3.3 归档链

`experiment_registry.py`（渲染命令）→ `logs/*_config.json` + `*_metrics.csv` + `*_samples.jsonl`（训练时写）
→ `eval_heldout.py`（`runs/<tag>/eval.json` + `.jsonl`）→ `archive_run.py`（`manifest.json` + `analysis.md` + 哈希校验）
→ `summarize_runs.py` / `paired_compare.py`（跨臂汇总）。`archive_run.py --check --strict` 用于核验归档完整性。

---

## 4. 结果

### 4.1 冷启动（见 §2.2）

> 前两行是 20 题、单次采样的**格式检查**，只用于说明"模板没学会"，不要和下面的 400 题数字并列比较。

| 权重 | 题数 | greedy | pass@1 | pass@16 | format |
|---|---:|---:|---:|---:|---:|
| 官方 `full_sft_768` | 20 | 0.000 | 0.050 | — | 0.200 |
| pilot CoT v1 `math_cold_seed42` | 20 | 0.000 | —（单采样） | — | 1.000 |
| **CoT v2 `math_cold_cot_seed42`（RL 起点）** | 400 | **0.373** | 0.311 | 0.553 | 1.000 |

### 4.2 Held-out 矩阵（400 题 × 16 采样）

| 权重 | greedy | pass@1 | pass@16 | format | avg len | easy | medium | hard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 冷启动 CoT v2 | 0.373 | 0.311 | 0.553 | 1.000 | 114.2 | 0.552 | 0.128 | 0.436 |
| `mini_rule_seed42` | 0.395 | 0.332 | 0.558 | 1.000 | 115.2 | 0.560 | 0.158 | 0.466 |
| `mini_rule_seed7` | 0.388 | 0.339 | 0.578 | 1.000 | 114.5 | 0.575 | 0.135 | 0.451 |
| `mini_rule_dynamic_seed42` | 0.390 | 0.333 | 0.548 | 1.000 | 115.5 | 0.560 | 0.150 | 0.459 |
| `mini_rule_len_seed42` | 0.393 | 0.331 | 0.558 | 1.000 | 114.9 | 0.567 | 0.150 | 0.459 |
| `mini_rm_seed42`（RM-only） | **0.133** | 0.117 | 0.285 | 1.000 | 105.0 | 0.209 | 0.008 | 0.180 |
| `mini_mix_seed42`（calibrated mix） | **0.315** | 0.280 | 0.500 | 1.000 | 110.1 | 0.500 | 0.090 | 0.353 |

> 完整表格与图表由 `python scripts/summarize_runs.py` 生成到 `reports/summary.md`（含 6 张图）。

### 4.3 训练动态 vs held-out（reward-eval gap）

首/末 10 点窗口均值：`mini_rule_seed42` **0.317 → 0.542（+0.225）**、`mini_rule_dynamic_seed42` +0.050、
`mini_rule_len_seed42` +0.100、`mini_rule_seed7` **0.508 → 0.317（−0.192）**。

**训练侧正确率涨了 +0.225，held-out 只涨 +0.022** —— 这是本项目最需要解释的一组数字：它既可能是
"RL 学会了训练分布"，也可能是"评估集更难/更小、功效不足"，需要靠 §4.4 的配对检验和 §9 的扩样计划来区分。

### 4.4 配对检验：greedy 有方向，能力无扩展

| 臂 | greedy 变对/变错 | 净变 | 双边 p | 能力（pass@16 逐题）净变 | p |
|---|---|---:|---:|---:|---:|
| `mini_rule_seed42` | 14 / 5 | **+9** | 0.0636（单边 0.0318） | +2 | 0.868 |
| `mini_rule_seed7` | 10 / 4 | +6 | 0.180 | +10 | 0.143 |
| `mini_rule_dynamic_seed42` | 13 / 6 | +7 | 0.167 | −2 | 0.864 |
| `mini_rule_len_seed42` | 14 / 6 | +8 | 0.115 | +2 | 0.864 |

**读法**：四个臂 greedy 净变**同向为正**，但单个臂都过不了双边 0.05；同时"能力存在性"（该题 16 次里至少对过一次）
净变在 ±10 内不显著 —— 即 **RL 更像是在锐化已有分布，而不是扩展能力上界**（与 arXiv:2504.13837 的方向一致）。
注意：四个臂共享同一个基线、其中三个共享 seed=42，**不独立**，不能把 p 值合并成"联合显著"来宣传。

### 4.5 分档倒挂：medium 才是难点

greedy 分档（冷启动）：easy 0.552 > hard 0.436 > **medium 0.128**。四个臂训练后顺序不变。

- **事实**：人工标注的 medium（两位数×两位数）反而最难。
- **解释（post-hoc，未证实）**：乘法需要多步进位，而加法/减法可以凑整简化；训练集里 easy/hard 的比例与 held-out 不同，模型对乘法 2×2 的覆盖不足。
- **处置**：**不改数据、不重标难度**——预注册已冻结，重标等于事后调参。作为审计发现记录在案。

### 4.6 长度偏置：cap 落在分布之外（no-op）

| 证据 | 内容 |
|---|---|
| 评估侧 | 5 个 run × 6400 条采样，最大 token **279–280**，超 cap（300）率 **0.00%** |
| 训练侧 | `_samples.jsonl` 20 个采样点 × 12 条，最大 **217 / 228**，超 cap **0 条** |
| 奖励侧 | 用 `基准分(status) − 记录 reward` 反推惩罚，240 行**全为 0** |
| 结果侧 | greedy 0.393 vs 0.395（Δ−0.002）、平均长度 114.9 vs 115.2（Δ−0.3 token） |

**结论**：该臂对长度和准确率都没有可测影响，原因是**没有东西可惩罚**（`cap=300` 高于长度上界）。
假设失败的原因被定位到"阈值设置"而不是"惩罚机制无效"——这是可复用的教训：**先测分布，再设 cap**。

> **更正（2026-09-28）**：早期曾把"`len` 臂与基线从 step 335 起分岔"当作"惩罚至少触发过一次"的旁证。
> 同种子重跑实验（§4.7）证明**那个分岔是非确定性所致**，与惩罚无关——该旁证已撤回。
> 本节结论只依赖上面的分布测量（分布外 + 240 条采样 0 触发），不再依赖任何轨迹对比。

### 4.7 训练不可复现：同一条命令、同一个种子重跑也会分岔

**实验**：把 `mini_rule_seed42` 的命令原样重跑一遍（tag 换成 `mini_rule_rerun_seed42`，其余参数逐字相同，
由 `scripts/experiment_registry.py` 渲染后 `sed` 替换），再用 `scripts/compare_runs.py` 逐 step 对比
200 个记录点（`--log_interval 5`）。

| 对比 | 逐字段一致的行数 | 首处分岔 | 分岔幅度示例 |
|---|---:|---|---|
| **同命令、同种子重跑** | **35 / 200（17.5%）** | **step 180** | step 225：`median_completion_tokens` 66 vs 150；step 280：`all_correct_groups` 2 vs 0 |
| `seed42` vs `len` 臂（仅差惩罚系数） | 66 / 200（33%） | step 335 | `valid_groups` 2 vs 0、`adv_std` 1.04 vs 0、`avg_reward` 0.083 vs 0.667 |

**结论**：

1. **训练管线不是逐位可复现的。** `setup_seed()` 固定了 Python/numpy/torch 的种子和 cuDNN
   （`cudnn.deterministic=True, benchmark=False`），但 **cuBLAS 归约与 backward 中的 atomicAdd 仍是非确定性的**；
   在采样温度 0.8 下，一个 token 的翻转就会把轨迹推向完全不同的路径。严格复现需要
   `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8`（代价是吞吐下降），本轮未启用。
2. **"分岔位置"是随机的**：重跑在 step 180 分岔、`len` 臂在 step 335 分岔——这是同一现象的两次抽样，
   因此 §4.6 的轨迹级旁证不成立（已撤回）。
3. **单臂单种子的训练曲线不能作为结论依据**。held-out 数字仍是对**已训练模型**的有效测量，
   但臂间差异里必须扣掉非确定性噪声。
4. **种子效应与非确定性效应同量级**：seed 42 与 seed 7 的训练 reward 走向相反（+0.225 vs −0.192），
   而 held-out greedy 却几乎相同（0.395 vs 0.388）——即训练期曲线的差异远大于最终评估的差异。
5. 记录点之外的更早步骤可能已经存在**低于日志精度**的微小差异，因此"35/200 一致"是复现性的**上界**。

**噪声下限（已测，2026-09-28）**：重跑模型同样做 400×16 held-out 评估，与"同命令、同种子"的原运行配对：

| 口径 | 点估计（原运行 → 重跑） | 逐题变对 / 变错 | 净变 | McNemar p |
|---|---:|---:|---:|---:|
| greedy 正确率 | 0.3950 → 0.4000 | 6 / 4 | **+2** | 0.7539 |
| 能力（pass@16 逐题） | 0.5575 → 0.5425 | 9 / 15 | **−6** | 0.3075 |

**含义**：只换随机数（模型权重、评估采样都变，命令与种子不变）就能产生 ±6 的逐题翻转、
±2 的 greedy 净变、甚至 −6 的能力净变。对照 §4.4，四个臂的**全部**效应是：
greedy 净变 +6 ~ +9、能力净变 +2 / +10 / −2 / +2——**没有一个越过这个地板**。
因此 §4.4 的"净 +9 题"只能作为方向性证据，不能作为效应量；这也解释了为什么四个臂的
p 值全部停在 0.06–0.18：分母里同时有 400 题的抽样误差和重跑的非确定性，两者都比效应本身大。

### 4.8 RM 两臂：奖励与正确性的正面背离（本项目的核心负结果）

| run_tag | 奖励组成 | 训练 reward（首/末窗口） | 训练 correct | held-out greedy | 配对（变对/变错） | p |
|---|---|---:|---:|---:|---:|---:|
| `mini_rm_seed42` | 启发式 + 1.8B RM | 1.684 → 2.011（**+0.327**） | 0.233 → **0.025** | **0.133**（−0.263） | 6 / 102（净 **−96**） | <0.0001 |
| `mini_mix_seed42` | 0.5·z(rule) + 0.5·z(RM) | 组内标准化，均值恒 0（不可读趋势） | +0.142 | **0.315**（−0.058） | 12 / 35（净 **−23**） | 0.0011 |

**三件事同时发生**，构成一条完整的因果叙述：

1. **reward 上升**（+0.327）而**同一批 rollouts 上的 verifier 正确率从 0.233 掉到 0.025**——
   奖励和正确性在同一时刻、同一批数据上反向运动，这不是"评估集不同"能解释的。
2. **held-out 崩塌**：greedy 0.373 → 0.133，pass@16 0.553 → 0.285；400 题里 **102 题由对变错、只有 6 题由错变对**。
3. **分档全线下降**（easy −0.343 / medium −0.120 / hard −0.256），说明不是"难度迁移"，而是整体退化。

**机制**：`--reward_mode rm` 的奖励是 `0.5·长度项 + 1.0·think 项 + 0.25·单标签项 − 重复惩罚 + RM 分(±3)`，
**没有任何一项依赖答案是否正确**。RM 的自检已经预告了这一点：格式漂亮但算错的回答仍得 **+0.632**。
策略于是学会了"继续输出漂亮格式、放弃算术"，训练末期 `ACR wrong = 1.000`（所有组全错，advantage 归零）。

**两个值得写进报告的细节**：

- **它不是靠变长换分**：RM 臂的 held-out 平均长度 **105 < 基线 114**。所以"长度偏置/过长 hack"
  这个诊断对它不成立，正确的诊断是**内容层面的 reward hacking**（RM 偏好流畅与格式）。
  （`scripts/summarize_runs.py` 的决策树已按奖励来源修正，见 §6。）
- **mix 臂也退化，但轻得多**（−0.058 vs −0.263）：rule 分量提供了正确性锚点，但 RM 分量
  在 rule 无方差的组（全对/全错）上仍在提供梯度，把策略往正确性之外推。
  注意 mix 臂的 rule 分量是**组内 z-score**，与 rule 臂的原始 reward 形态不同，
  因此"RM 污染"与"标准化副作用"这两个解释无法完全分离（见 §6 局限）。

**与 SOP §0.2 的对应**：这条结果落在"reward/格式率上升、正确率不变或下降"那一行，
即**发现了 reward hacking**，而不是"RLVR 在数学上无效"。

### 4.9 错误类别分析：RM 臂崩的是算术，不是模型

`scripts/error_analysis.py` 把 `runs/<tag>/eval.jsonl` 的**全部 400 条 greedy 输出**逐条分类
（互斥主类别 + 非互斥标记；重复检测为"任意周期 ≤64 字符、连续 ≥3 次、跨度 ≥12 字符"）：

| 主类别 | `math_cold_cot_seed42`（RL 起点） | `mini_rm_seed42`（RM-only） |
|---|---:|---:|
| `correct` | 149（37.2%） | 53（13.2%） |
| `arithmetic_error` | 251（62.7%） | **347（86.8%）** |
| 其余 6 类：`repetition` / `unclosed_think` / `redundant_answer` / `length_burn` / `no_answer` / `other` | **全部 0** | **全部 0** |
| 标记 `marker_unparsed`（verifier 待复核） | 0 | 0 |

分档正确率（同一份 400 题）：

| 档位 | 冷启动 | RM-only | 比值 |
|---|---:|---:|---:|
| easy | 74/134 = 0.552 | 28/134 = 0.209 | 0.38 |
| medium | 17/133 = 0.128 | 1/133 = 0.008 | 0.06 |
| hard | 58/133 = 0.436 | 24/133 = 0.180 | 0.41 |

**读法**：RM 臂的退化**不在格式层**——400 条补全 100% 闭合了 `</think>`、100% 能被 verifier 抽出数字、
没有一条循环、没有一条被长度上限截断、没有一条"无答案"。它把 SFT 给的外壳（分步拆解 + `答案是 N`）
**完整保留**，只把内容层的算术正确率压掉了约三分之二——这是 §4.8"奖励对内容不敏感、策略只需保住外壳"
的行为层证据。

同一道题的 A/B（`66×23`，GT=1518，两条都错）：

```text
冷启动 : 66*23：23 拆成 20 和 3；66*20=1440；66*3=184；1440+184=1728。  → 1728
RM-only: 66*23：23 拆成 20 和 3；66*20=1280；66*3=184；1280+184=1334。  → 1334
```

**错误形态学**（`reports/audit_*.md`：每个臂按 held-out 文件顺序取**前 20 条失败样本**，非随机抽样）：

| 观测 | `math_cold_cot_seed42` | `mini_rm_seed42` |
|---|---:|---:|
| 含乘法断言的样本 | 17/20 | 13/20 |
| └ 其中至少有一条乘法断言为假 | **17/17** | **12/13** |
| 含加法断言的样本 | 17/20 | 15/20 |
| └ 其中"写出的和 ≠ 自己写出的两个加数之和" | **16/17** | **13/15** |
| 错答的相对误差 ≤100%（同量级） | 20/20 | 17/20 |

两条用了**同一个分解模板**，错的层次也一样：不只是乘法事实错（66×20 应为 1320、66×3 应为 198），
而且**写出来的和与自己写的加数也对不上**（`1440+184=1728`、`1280+184=1334`——实际分别是 1624 和 1464）。
减法档同理："逐位相减、忘记借位"（`56-49` → 个位写 0、十位写 1，得 10；`72-31` 借位写成 `12-1-1=9` 得 39）。
**模型是在"照着模板续写数字"，而不是在算**——真正的算术执行只在极少数样本里是对的。

这解释了一个反直觉的现象：RM 臂的**格式率仍是 1.0000**、`no_answer` 仍是 0——在"续写"这个层次上，
它输出的依旧是结构完美、可被 verifier 抽取的答案，崩掉的只有数值本身。

> **口径修正（2026-09-28）**：类别 `no_think_block` 已删除。本 pipeline 的 prompt 末尾自带
> `<|im_start|>assistant\n<think>\n`（`open_thinking=True`，见 `scripts/eval_heldout.py:build_prompt`），
> 补全里只可能出现 `</think>`，"没有开标签"是**正常**状态；旧版本把它误判成"模板没被遵循"，
> 一度在同一批数据上给出 400/400 的假阳性。现行判定与 `trainer/math_verifier.py:split_think`
> 的 `think_closed` 字段对齐（`tests/test_error_analysis.py`，18 passed）。

**人工抽查（DoD §10.4 要求 ≥20 条/权重）**：`reports/audit_mini_rm_seed42.md`、
`reports/audit_math_cold_cot_seed42.md`（每个失败类别 20 条原文；上面这张表的所有数字都能从这两份
文件逐条复算）。原始 400×16 条 rollout 在 `runs/<tag>/eval.jsonl` 与 `logs/<tag>_samples.jsonl`。

---

## 5. 结论与边界

### 5.1 支持的结论

1. **奖励源决定学到什么**：只有 GT verifier 参与奖励时方向为正；一旦把 RM 放进奖励
   （`rm` 或 `mix`），held-out 就显著下降（净 −96 / −23，p ≤ 0.0011）——奖励里有没有正确性，比 optimizer 和超参更决定性。
2. rule verifier 提供的是**真实但小**的信号（greedy 净 +9/400 题，方向四臂一致）；
   但按 §4.7 实测的噪声地板（同配置重跑即可净变 ±2、能力净变 −6），这个量级只能算**方向性证据**，
   不能当效应量引用。
3. 长度惩罚在本设置下**无效**，原因是 cap 设置错误（分布外）。
4. 难度分层必须用**模型侧**标定，不能直接信人工标签。
5. 该规模的 GRPO **单次运行不可复现**（同命令同种子重跑仅 35/200 记录点一致），叠加种子效应后，**单臂单次运行的曲线与效应量都不足以支撑结论**。

### 5.2 不能推出的结论

- **不能**说"RL 让模型学会了新推理能力"：pass@k 的能力指标不显著，且 64M 规模不支持泛化结论。
- **不能**说"RLVR 无效"：这是 64M + 2000 题 + 单卡的分层实验，结论边界限于此配置。
- **不能**把训练 reward 的涨幅当成任务收益：两者在 seed 维度上可以反号（seed 42 的 +0.225 与 seed 7 的 −0.192），
  在 RM 臂上更是**同批数据上直接背离**（reward +0.327 / 正确率 −0.208）。
- **不能**说"RM 完全没用"：本实验只说明**把 RM 单独作为数学奖励**会退化；RM 作为辅助信号（如过程奖励）
  的价值不在本实验的结论范围内。

### 5.3 与 SOP §0.2 三种结局的对应

本项目的落点是**第 2 种 + 第 3 种的混合**：reward/格式率上升、真实正确率几乎不动（对照臂之间），
且最关键的干预（CoT 模板）发生在 RL 之前而不是 RL 之中。

---

## 6. 工程观测与局限

| 观测 | 数值 | 说明 |
|---|---|---|
| 单步耗时 | **1.16 s/step**（rule）/ **1.26 s/step**（RM 臂前 170 步） | 1000 步 ≈ 19–21 分钟/臂 |
| GPU 利用率 | **~36%** | `--rollout_engine torch` 逐 token `generate`，**CPU 单核瓶颈**（CPU 102%） |
| 显存 | 7.2 GB / 23.5 GB | RM 3.17GB + policy/ref/优化器约 1.5GB + 激活与上下文 |
| 评估成本 | 400×16 约 15–25 分钟 | 单卡 |
| 已知加速路径（未采用） | `--rollout_engine sglang`、更大 batch | 预注册超参冻结、臂间可比性优先 |

其他局限：

- **单卡、单种子、且不可复现**：每个臂只有一个 seed（`mini_rule_seed7` 提供了一次种子复核），而同命令同种子的重跑只有 35/200 记录点一致（§4.7）——**单次运行的训练曲线不能作为证据**。
- **`mix` 奖励的绝对值不可读**：它是两个分量的组内 z-score 加权，均值恒为 0；
  因此它的"reward 涨了多少"没有信息量，只能看正确率与 held-out。
- **`mix` 的两个解释无法完全分离**：held-out 下降既可能来自 RM 分量的污染，也可能来自
  rule 分量被改成组内 z-score 后的形态变化（对比对象 `mini_rule_seed42` 用的是原始 reward）。
- **仪器缺陷（已修）**：`summarize_runs.py` 的决策树原先会把 `mini_rm_seed42` 误诊为"长度偏置/过长 hack"
  （该臂 held-out 长度 105 < 基线 114，实际是内容层面的 reward hacking），并把 `mix` 臂
  "均值为 0"的标准化奖励读成"reward 没动"。现已按奖励来源分派诊断、把标准化臂渲染为 `n/a`，并补了回归测试。
- **仪器缺陷第二轮（2026-09-28，已修）**：决策树把"held-out 点估计下降"直接写成"**显著**下降"，
  而该脚本并不做配对检验；`mini_rule_dynamic_seed42` 的 −0.005 恰好落在 §4.7 实测的噪声地板内，
  写成"显著"就是过度解读。现已去掉"显著"字样、统一指向 `paired_compare` 的 p 值与噪声地板，
  并补回归测试（`tests/test_summarize_runs.py`，58 passed）。**这条是本地复算时发现的**：
  同步回本机后重跑 `summarize_runs.py`，服务器上的 `summary.md` 仍是修复前的旧版，重算才暴露。
- **未持久化的指标**：`mix` 的 `component_degenerate_groups`（任一分量无方差的组数）在训练时被计算但没写进
  metrics CSV，事后无法复算——属于下一版要补的埋点。
- **困难档覆盖窄**：hard 只有三位数×两位数这一种形态；`pass@32` 未做。
- **held-out 400 题功效不足**：对本项目观测到的效应量（不一致率 4.8%），400 题的双边 p=0.064，
  估算需约 500 题才能压到 0.05（见 §9）。

---

## 7. 复现清单

```text
dataset/*.jsonl + manifests/data.sha256        # 数据与哈希
configs/experiment_registry.csv                # 预注册表（唯一事实来源）
scripts/server_preflight.py                    # 环境/数据/权重/磁盘预检
scripts/experiment_registry.py --print-command <tag>   # 渲染训练命令（勿手敲）
scripts/eval_heldout.py                        # held-out 评估
scripts/archive_run.py --run-tag <tag> --force # 归档 + 哈希
scripts/summarize_runs.py                      # 跨臂汇总 + 6 张图
scripts/paired_compare.py --baseline <tag> --arms ...  # 配对 McNemar
scripts/compare_runs.py --a <tag> --b <tag> --expect-identical  # 逐 step 复现性对照
scripts/error_analysis.py --run-tag <tag> --mode greedy         # SOP §10.4 错误类别统计
scripts/rm_load_check.py                       # RM 加载自检（RM 臂前置）
README-rlvr-audit.md                           # 项目说明与最小复现路径（SOP §18）
```

环境：Python 3.12.3 · torch 2.8.0+cu128 · transformers 4.57.6 · datasets 3.6.0 · numpy 2.3.2 ·
modelscope 1.37.0 · sentencepiece 0.2.0 · RTX 4090 24GB（单卡）· bf16。

**已完成（2026-09-28）：反向同步 + 本地复算验证。** `runs/`（61 MB）、`logs/`、`reports/` 已拉回本机
（`reports/report.md` 由本机维护，同步时显式排除，避免被服务器上的旧版覆盖；同步前后哈希不变可证）。

| 本地复算命令 | 结果 |
|---|---|
| `scripts/error_analysis.py --run-tag mini_rm_seed42 / math_cold_cot_seed42 --mode greedy` | 与服务器产物 **逐字节一致**（`diff` 无输出） |
| `scripts/paired_compare.py --baseline math_cold_cot_seed42 --arms <4 个 rule 臂>` | 四臂数值与服务器版一致；服务器版是**六臂**版（多 `rm`/`mix` 两行） |
| `scripts/summarize_runs.py` | 重算成功，并暴露出服务器 `summary.md` 是修复前的旧版（见 §6 第二轮仪器缺陷） |
| `python -m pytest -q` | **313 passed** |

---

## 8. 证据索引

| 结论 | 证据文件 |
|---|---|
| 冷启动修复 | `runs/math_cold_cot_seed42/eval.json`、`runs/math_cold_seed42/*`（pilot） |
| 六臂矩阵 | `reports/summary.md` §2、`runs/<tag>/eval.json` |
| 配对检验 | `reports/paired_compare.md`、`scripts/paired_compare.py` 的回归测试 |
| 训练动态 | `reports/summary.md` §1、`logs/<tag>_metrics.csv` |
| 长度 no-op | `runs/<tag>/eval.json` 的 `samples`、`logs/<tag>_samples.jsonl` |
| 难度探针 | `runs/probe/calibration.json`（v1）、`runs/probe/calibration_cot_v2.json`（v2） |
| ACR / 有效组 | §4 的组内直方图复算（`runs/<tag>/eval.jsonl`） |
| 归档完整性 | `python scripts/archive_run.py --check --strict --run-tag <tag>` |
| 错误类别（全量） | `reports/error_analysis_<tag>.md`、`scripts/error_analysis.py`、`tests/test_error_analysis.py` |
| 人工抽查（≥20 条/权重） | `reports/audit_<tag>.md`（原始文本，形态学表可逐条复算） |
| 噪声地板 | `runs/mini_rule_rerun_seed42/eval.json`、`reports/paired_compare_rerun.md` |
| 训练可复现性 | `scripts/compare_runs.py --a mini_rule_seed42 --b mini_rule_rerun_seed42 --expect-identical` |

---

## 9. 待办与缺口（按优先级）

1. ~~noise-floor~~ **已完成**（§4.7）：`mini_rule_rerun_seed42` 做了 400×16 评估与配对，
   greedy 净变 +2（p=0.75）、能力净变 −6（p=0.31）。
2. ~~error-analysis~~ **已完成**（§4.9）：全量 400 条 greedy 逐条分类，两类之外全为 0 条。
   ~~audit~~ **已完成**：`reports/audit_<tag>.md` 两个文件（每类 20 条原文），§4.9 的形态学表即由它们复算。
   **TODO(err-shape)**：目前只做了每臂**前 20 条**失败样本（非随机抽样）。全量 251/347 条的形态学
   需要把该统计并进 `scripts/error_analysis.py` 的输出（下一版），本轮不影响结论方向。
3. ~~power~~ **已评估、决定不做**（2026-09-28）：held-out 扩到 500 题只能把置信区间收窄
   （按实测效应量估算双边 p 从 0.064 → 约 0.023），但 §4.7 已实测出**同配置重跑就有 ±2 的 greedy 净变
   与 −6 的能力净变**，四个臂的效应量本身落在噪声地板内——增加样本不会改变"无法区分于噪声"这个结论。
   记录于此以避免重复讨论。
4. **TODO(instrumentation)**：把 `component_degenerate_groups` 等 mix 专属统计写进 metrics CSV（下一版埋点）。
5. **TODO(scope)**：SOP §0.3 的"工业近似层"（Qwen2.5-1.5B + TRL/verl）**未做**，属范围边界；
   若要做，按 SOP §9 先锁 commit + 10–20 step smoke。
6. ~~readme~~ **已完成**：`README-rlvr-audit.md`（项目说明 + 数据来源与哈希 + 环境 + 最小复现路径 +
   失败结果与局限 + 已知坑），并在上游 `README.md` 顶部加了 3 行指引（不改动上游正文）。
7. ~~sync~~ **已完成**（§7）：反向同步 + 本地复算（`error_analysis` 逐字节一致、`paired_compare` 一致、
   `summarize_runs` 重算成功）。反向同步还顺带查出两个仪器问题（§6），已在本地修复。
8. **已知缺口（刻意不补）**：`mini_rule_rerun_seed42`（重跑对照）与 `math_cold_seed42`（CoT v1 pilot）
   不在预注册表 `configs/experiment_registry.csv` 内，因此没有 `manifest.json`，也不会出现在
   `reports/summary.md` 的臂清单里。二者的原始证据（`train.log` / `eval.json` / `eval.jsonl` / `metrics.csv`）
   都在 `runs/<tag>/` 与 `logs/` 下，§4.7、§2.2 的数字可直接复算——**不为 post-hoc 运行补登记行**，
   以免污染"预注册"的语义。

---

## 10. 面试可用表述（数字定稿后再引用）

> MiniMind 的价值是让我能逐行理解 rollout、组内 advantage、KL 和 mask，但 64M 规模不支持泛化结论。
> 我把问题收窄成"reward 是否真的代表学习"：用带三态判定的 GT verifier 替换原来的格式分和 RM，
> 建 train/held-out 无泄漏数据，记录正确率、格式率、长度、KL、熵、退化组比例，并补上配对显著性检验与功效估算。
> 最终的重点不是一条漂亮的 reward 曲线，而是能解释 reward hacking、长度偏置、verifier 错误和小模型能力边界——
> 包括"长度惩罚的 cap 落在分布之外，所以这个消融其实是 no-op"这种负面结论。

（SOP §15.2 的 14 条必答追问需要逐条准备答案；简历数字**只填已经跑完并归档的**。）
