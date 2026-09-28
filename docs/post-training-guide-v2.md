# MiniMind 后训练升级指南 v2（2026-09 校验版）

> 本文档是 `docs/post-training-guide.md`（v1，2025-09 成文）的**增量升级版**：v1 的结论结构与主线定位（64M 数学 RLVR 诊断式复现）不变，本文记录 12 个月后的校验结果、新增技术取舍与实现级落地规格。
> 校验日 **2026-09-06**（arXiv 最新提交 2026-09-03）。由 main agent 调度 research / coding / reviewer 三个子 agent 协同产出：
> - research 校验与增量调研 → `planning/research_delta.md`
> - coding 实现级审计与规格 → `planning/implementation_spec.md`（所有行号锚点实测）
> - reviewer 交叉评审与裁决 → `planning/review_report.md`
> 配套产物均在仓库内，本文只保留裁决结论。

---

## 0. TL;DR（v2 增量）

- **定位不变**：基于官方 64M dense 权重，把 `train_grpo.py` 的"启发式格式分 + 1.8B LLM judge"奖励替换为**规则答案 verifier**，跑通 SFT 冷启动 → GRPO/RLVR → heldout 评估，用曲线 + 对照 + 消融 + 诚实负结果做"小模型 RL 能力门槛"诊断。2026 年这仍是正确选择（依据见 §1）。
- **技术坐标已过时约 12 个月，3 处必须修订**：① "0.5B 从零 RLVR 学不会推理"→ 改为"0.5B 短链可学（有论文先例）、长链涌现学不会"；② "PRM 小模型不推荐" → "生产未主流、研究活跃（0.5B 有 +9.98pp 声称）"；③ 冷启动叙事从"SFT 是地基"更新为"2026 大厂把冷启动演进为蒸馏（OPD），64M 无 teacher 不实现但面试必会讲"。
- **v2 低成本加料（推荐做，合计约 1-2 天）**：① ACR（advantage collapse rate）曲线，把"动态采样过滤"消融从有/无升级为**连续量化**（约 30 行）；② **pass@k 大 k 反转诊断**（评估矩阵加 pass@16/32，复现 NeurIPS 2025 Oral 方法学）；③ 数据**难度分层**（自造数据 2-3 档 + 难度×提升矩阵；≈5× 样本效率为论文声称，见 2604.18381）；④ verifier **低假阴纪律**（假阴比假阳更伤且有实验证据）。
- **实现面比 v1 预期更小**：`calculate_rewards` 实际只有 32 行（v1 估 150-300 行）；主实验改动 = train_grpo.py 约 10 处可开关补丁（约 +120 行）+ 4 个新文件。完整规格见 `planning/implementation_spec.md`。

---

## 1. v1 论断核对表（research 校验，2026-09）

| # | v1 论断 | 判定 | 2026-09 依据/修正 |
|---|---|---|---|
| 1 | GRPO/RLVR 是主流在线 RL 范式 | ✅ 维持 | 无范式级替代；GRPO 已"地基化"（TRL GRPOTrainer `loss_type` 原生含 `dr_grpo`；2026 论文里 GRPO 是 baseline 而非卖点） |
| 2 | GRPO 组内 advantage 有偏（Dr.GRPO） | ⚠️ 表述升级 | 双视角：有限组有偏 → 长度膨胀机制（Dr.GRPO v2，2025-10）；渐近理论证明 GRPO 梯度是 U-statistic、近 oracle 最优并给 group size 标度律（arXiv:2603.01162，2026-03）。面试只讲"有偏"显得停在 2025 |
| 3 | DAPO 四件套是复刻标配 | ⚠️ 表述升级 | 仍是稳定化基准，但 2026 语境扩展为"稳定化工具集"：clip-higher / 采样过滤 / token 级 loss / 长度 shaping / **advantage-collapse 监控（AVSPO，ICML'26）**；REFT 把熵修复延伸到首 token 多样化 |
| 4 | "0.5B 从零 RLVR 学不会推理" | ❌ **必须修订** | 0.5B SFT 冷启动 + GRPO 在 GSM8K 短链可学到 outcome 53.75% / process 63.73%（arXiv:2607.02869，未评审，引用带"声称"）。正确表述："0.5B 短链可学、长链涌现学不会；64M 两者皆难" |
| 5 | "64M 只能学格式、做诚实诊断" | ✅ **维持且证据增强** | 最贴量级的新负例：135M 单卡公开审计，直接 GRPO-RLVR 使 GSM8K EM 1.82%→1.59%（arXiv:2606.22189；**单次运行、192-token 完成度，作者声明不构成一般下界**）；NeurIPS 2025 Oral 总纲："RLVR 不外推 base 能力、小 k 涨大 k 反超、蒸馏才扩能"（arXiv:2504.13837） |
| 6 | "会跑 GRPO 不再是简历亮点" | ✅ 加速贬值 | 高志军课程在线、GRPOTrainer 库化、2026 论文 baseline 化；从零实现 + 动态量化理解（ACR/pass@k 反转）更硬 |
| 7 | PRM"成本高、未普及、小模型不推荐" | ⚠️ 需软化 | 生产仍 outcome 为主，但研究侧 0.5B process 超 outcome +9.98pp（2607.02869）且降低轨迹偏离；改"生产未主流、研究活跃；主线仍 outcome，过程奖励留作谈资/可选微对照" |
| 8 | 蒸馏起点可行（DeepScaleR 1.5B） | ✅ 维持 + 佐证 | Plasticity-vs-Rigidity：≤1.5B 高秩 LoRA（r=256）+ instruct 起点 → AIME24 P@1 40% 声称（2601.06677）；都强化"起点质量决定 RL 上限" |
| 9 | agentic RL 最热但本项目不做 | ✅ 维持 | K2 Thinking（2025-11）与 Qwen3.5（2026-02）证实风口；研究前沿同时被 OPD 分流；1-2 周项目仍不做 |
| 10 | 冷启动 SFT 是 RL 地基 | ⚠️ 更新叙事 | 64M 场景仍对；但 2026 行业结论="OPD 是比 SFT 更好的冷启动、distill→RL 两段优于 joint"（2609.04108，2026-09-03；ICML'26 OPD 研究 2604.13016）。64M 无 teacher 不实现，**面试口径必须能解释** |
| 11 | 简历红线总体成立 | ⚠️ 2 处更新 | ① R1 复现更不值钱（论文 2026-01 更新至 86 页全配方）；② "0.5B 数字禁写首创"（有公开论文先例，只可写"复现并诊断"并引用） |
| 12 | v1 参考链接 | ⚠️ 2 处需修 | DeepSeek-R1 → arXiv:2501.12948v2（86 页）；philschmid 教程 → HF open-r1 Mini-R1 博客 |

**仓库代码层面（coding 审计，相对 v1 §4.4 的三处修正）**：
1. "calculate_rewards 约 150-300 行" 不实——实际 **train_grpo.py L37-68 共 32 行**，改动面比 v1 预期更小；
2. "SFT 与 RLAIF eos 约定差异"属实但**无实义**（RLAIFDataset L202-203 的 eos 锚是死代码，GRPO mask 靠 `eos_token_id` 检测，L126-130）；
3. `reward_model_path="../../internlm2-1_8b-reward"`（L232）是 **cwd 相对路径**（cwd=trainer 时解析到仓库同级，与 README 布局一致），非"相对仓库根"。

---

## 2. 2026 新增技术取舍（对 64M 数学 RLVR 简历项目）

### 2.1 ✅ 做（低成本、高展示度，进 v2 主线）

| 技术 | 机制一句话 | 来源 | 落地方式 | 工作量 |
|---|---|---|---|---|
| **ACR / advantage-collapse 曲线** | 组内 reward 全同（全对/全错组）→ A=0，advantage 驱动的梯度为零（严格说 A=0 时 β·KL 项仍把策略拉回 ref，见 §6-Q3）；把两类退化组占比画成随步数曲线 | AVSPO，ICML 2026（arXiv:2605.21125） | 主循环已有分组 mean/std（L121-124），加两行统计（**全对组占比 / 全错组占比**）落 CSV；与"动态采样过滤"消融（开/关）配套。归因：全对主导→已收敛；全错主导→容量/难度（小模型早期全错组天然占多数，此时 ACR 高≠奖励分辨率问题）；对错混合仍高→奖励分辨率不足 | ~30 行 |
| **pass@k 大 k 反转诊断** | NeurIPS'25 Oral：小 k 下 RLVR 优于 base、大 k（pass@16/32）下 base 反超 → 推理被 base 封顶的证据 | arXiv:2504.13837 | heldout 评估矩阵加 pass@16/32（同引擎、固定采样数报方差）；若 RL 后大 k 反而不如 SFT 冷启动模型，直接复现顶会结论——2026 最有辨识度的诊断语言 | eval 脚本加列 |
| **数据难度分层** | 低数据小模型 RLVR：混合难度 ≈5× 样本效率；课程化声称可破 base 上界 | arXiv:2604.18381 / 2606.22317 | 自造数据按操作数/步数分 2-3 档并记录；评估矩阵加"难度×提升"；有余力再按先易后难进 RL（与 v1 能力门槛叙事互补：有效→亮点，无效→更硬证据） | 数据清洗时顺手 |
| **verifier 低假阴纪律** | verifier 假阴（漏判对）比假阳更伤训练，且加大算力无法弥补 | arXiv:2605.25252（ICML'26 WS） | v1"防作弊（低假阳）"之外补第二目标"保低假阴"：确定性匹配 + SFT 教格式 + 扫全文取最后锚点；负结果报告前自测并写明假阴率（单测 200 条含"格式对但答案对/格式对答案错"分桶） | 设计原则，0 成本 |

### 2.2 ◐ 可选（有余力才做）
- **REFT 首 token 多样化 rollout**：采样时首 token 在 top-N 均匀化并均分预算；0.5B-7B 四个 base 上优于 DAPO/GRPO（arXiv:2605.28295，声称）。torch 引擎改 ~10 行，可作第 3 项消融；先用组内多样性日志判断是否需要。
- **过程 vs 结果奖励微型对照**：0.5B 声称 process 63.73% vs outcome 53.75%、hybrid λ=0.1 反劣于纯 outcome（2607.02869，**未评审**）。64M 无过程标注预算 → 只进 B 支线或面试谈资（λ 非单调是反直觉谈资），不在主线引入 PRM。

### 2.3 ✗ 不做但要会讲（2026 面试弹药）
- **OPD（on-policy distillation）**：2026 推理后训练研究前沿最大主题（两周内 4+ 篇 follow-up）。机制：student 自采样 + teacher 逐 token 监督；结论："OPD 是比 SFT 更好的 RL 冷启动、distill→RL 两段优于 joint"（2609.04108 / 2604.13016，ICML'26）。64M/24GB 无 teacher 预算 → 不实现，但"冷启动从 SFT 演进为蒸馏"必须在面试口径里（可答："我项目里蒸馏角色的替代物是高质量冷启动 SFT + 拒绝采样自蒸馏"）。
- **R1 v2 全配方**：86 页公开四步管线（冷启动 SFT → 推理 RL + **语言一致性奖励** → 拒绝采样重 SFT → 对齐 RL），"aha" 用反思词频量化。教材级面试弹药，官方措辞可全文引用（v1 未覆盖"语言一致性奖励"，补入题库）。
- **GRPO U-statistic 理论**：策略梯度近 oracle 最优 + group size 标度律（2603.01162）→ 支撑"为什么 group=6/8"追问。
- **Nanbeige4-3B 工业小模型配方**：高质量 SFT（deliberative refinement）→ 蒸馏（Dual Preference Distillation）→ 多阶段 verifiable RL + 偏好 RL（2512.06266）——小模型配方模板，3B 级。

---

## 3. 实现规格汇总（详见 planning/implementation_spec.md）

### 3.1 改动面（比 v1 预期更小）
- **train_grpo.py ≈10 处可开关补丁（约 +120/-20 行）**：argparse 加 `--reward_mode {rule,rm,mix}` 等 10 个参数（d1）；RM 加载条件化（d2，L279）；数据集切换（d3，L292）；`calculate_rewards` 拆三纯函数（d4，L37-68）；调用点传 `batch['gt']`（d5，L95）；动态采样过滤（d6，消融 A）；长度惩罚（d7，消融 B）；日志统计与 CSV 落盘（d8，L154-178，口径见 §4.1）；run_tag 隔离（d9）；debug 样本存档（d10）。rule 模式下启发式格式分整体弃用（格式只进统计曲线），RM 臂原样保留保证对照公平。
- **新文件 4 个**：`trainer/math_verifier.py`（纯 Python 三态判定 + 分档奖励 + 单测）、`scripts/clean_math_data.py`（数据三件套 + 防泄漏 + 难度桶 + report）、`dataset/math_rl_dataset.py`（单轮 gt Dataset ~55 行，复用 `apply_chat_template`）、`eval_heldout.py`（heldout 矩阵：greedy/pass@1/pass@16/32、格式率、按难度）。
- **不改**：rollout_engine.py、train_agent.py、trainer_utils.py、model/*。回归护栏：`--reward_mode rm` 在官方 rlaif 数据跑 5 步与改造前 diff≈0。
- **答案锚点决策（v2 采纳 coding 规格）**：默认 `答案是`（v1 的 `#### 答案` 为备选）——必须在 数据清洗 / SFT gold / verifier / eval 四处同步，参数 `--reward_marker` 可配。SFT gold 形态：`<think>\n<简短推理>\n</think>\n\n答案是 <num>`。
- **正式 run 参数纪律（v1 §4.4 收编）**：所有正式 run 显式 `--loss_type grpo`（脚本默认 cispo 无公开出处，被深挖讲不清）；统计曲线自 rule 臂第一步起按 §4.1 口径记录。

### 3.2 关键新事实（coding 审计，v1 未覆盖，动手前必读）
1. **格式锚点是仓库内建的**：chat_template 对一切 assistant 消息强制 `<think>\n{reasoning}\n</think>\n\n{content}` 包装（tokenizer_config.json L333，已实测）；`<think>`/`</think>` = id 25/26 的**普通 token**（special:false），decode 后保留 → verifier 在文本层解析即可，SFT gold 不用手拼格式。
2. **占位 assistant 仅在"沿用官方 schema + 新 Dataset 不切片"组合下有污染风险**：官方路径里 RLAIFDataset 的 `conversations[:-1]` 切片（lm_dataset L212）会丢弃末条占位 assistant，RM 正则（train_grpo L50-52）根本看不到它、无污染；但自造 RLVR 数据若沿用官方 schema（user+占位 assistant）而新 `MathRLDataset` 不切片，占位会被 RM 正则当历史消息 → 自造数据一律**不放占位 assistant**。
3. **`save_weight` 同名互覆盖 + `_resume.pth` 跨 run 复用**（trainer_utils L63-116）→ 每 run 唯一前缀、`--from_resume` 只在明确续跑时开。
4. **MPS 只能冒烟**：autocast 仅 cuda 生效（L258-260）→ MPS 走 fp32；RM fp16 在 MPS 不可用 → **冒烟只跑 `--reward_mode rule`**；`--device mps` 必须显式传。正式实验在 24GB CUDA 卡。
5. 主循环每 step 三次全序列前向（rollout old-logp / policy / ref），64M 无压力；reward 计算是 per-sample 串行循环（B×6 次），RM 臂下是 step 热点 → verifier-only 省 ≈3.5GB 显存 + 每 step 12 次 1.8B 前向。

### 3.3 实现里程碑（M1-M7；与 §7 天计划 D1-D14 的映射见表内"≈天"列，勿与天数混淆）
| 里程碑 | 内容 | 验收 |
|---|---|---|
| M1 骨架（≈天 D1-2） | math_verifier.py + 单测；清洗脚本（--synthetic-only） | 200 条合成用例抽取命中 ≥99%、三态无歧义 |
| M2 数据（≈天 D3-4） | 下载 agent_rl_math 并清洗出 sft/rlvr/heldout 三件套 + report；math_rl_dataset.py | 50 条人工抽查 gt 无误；train/heldout 无 key 重叠；渲染与官方路径逐字一致 |
| M3 冒烟（≈天 D5 前段） | MPS 全链路（30 条合成数据、rule 模式） | ≥2 step 跑通、CSV/样本 jsonl 落盘、无 RM 加载；记录 step 耗时 |
| M4 冷启动 SFT（≈天 D5 后段） | 24GB 卡：full_sft_768 起步 1-2 epoch | heldout greedy 人工 20 条格式+锚全过；抽取命中 ≥99% |
| M5 主实验三臂（≈天 D6-8） | rule 先通（50 步小跑）→ rm/mix 臂；全部 `--loss_type grpo` | 曲线（含 ACR 双线）形态可解释；RM 加载断言 + 显存记录 |
| M6 全量+消融+评估（≈天 D9-12） | 三臂全量 + 动态采样/长度惩罚/（可选 REFT）+ pass@k 矩阵 | 对照表 + 消融表（无效也是结果）+ ACR 归因 |
| M7 归档（≈天 D13-14） | 曲线组图、抽查存档、runbook 回放 | 空 out/ 到全流程可复现；风险核销 |

---

## 4. 修订后的实验矩阵与结局叙事

### 4.1 指标集（v2）
- 训练侧曲线：reward / **格式率** / **正确率** / 长度 + **ACR 双线**（全对组占比、全错组占比）（+KL、Adv-std 既有）；CSV 落盘 `logs/{run_tag}_metrics.csv`。
- heldout 矩阵（每个权重：math_cold / rlvr_rule / rlvr_rm / rlvr_mix / 官方 grpo_768）：greedy、pass@1、**pass@16/32**（temp 0.8，固定采样数报方差；**pass@32 在 heldout 100-200 条子集上执行以控采样成本**，口径写入报告）、格式率、**难度×提升**、平均长度。
- 抽查：debug 存档 20 条，标注借用过程奖励论文的失败模式二分（outcome 轨迹短但推导易错 / process 结构稳但算术错）——把"抽查"升级为结构化审计。
- 报告前置：**verifier 假阴率**（合成单测分桶统计），负结果前先排除 verifier 漏洞。

### 4.2 预期管理（新增外部对照）
- 先验不是"涨点"而是：reward/格式率上升 + 正确率停滞或微跌。对照数字：**L20-Edu-135M GSM8K EM 1.82%→1.59%**（arXiv:2606.22189；**单次运行、192-token 完成度，作者声明不构成一般下界**）——64M 若同构，不是失败而是该量级的**同构参照/先例**（非同数据集直接比较），负结果有了外部基准。
- 若做 0.5B 冒烟，必须与 64M 用**同一套数据与 verifier**，才能归因差异是容量还是冷启动（否则冒烟无诊断价值）。
- pass@k 反转三种形态的解读：① RL 后小 k 涨大 k 跌 → 复现 NeurIPS'25 Oral 上界结论；② 全档位涨 → 值得写（64M 也分档位，报难度）；③ 全档位跌/格式涨正确率跌 → 信号被格式吸收 + 容量不足，v1 已备叙事。

### 4.3 三臂对照与消融（v1 不变 + v2 升级）
- 三臂：verifier-only / RM-only / 混合臂。**混合臂定义（reviewer 裁决）**：模型分量用**纯 RM score**（去掉启发式项，只取 train_grpo L62 的 RM 分），两分量先各自**组内标准化**再按 α/β 加权（α/β 作为显式超参报告；默认 1.0 裸相加不可用——rule∈[-0.5,1] 与 RM∈[-3,3] 尺度差数倍，rule 信号会被淹没）；RM-only 臂保持官方原样（启发式+RM，作公平对照）。整体在 README RLAIF 定义域内、有仓库背书。曲线对比只比格式率/正确率/长度等无量纲量。
- 消融 A：动态采样过滤 开/关 + **ACR 连续曲线**（升级）；消融 B：长度惩罚；消融 C（可选）：REFT 首 token 多样化。

---

## 5. 简历红线与可写版本 v2

**红线（v1 保留；v2 新增条目逐条标注）**：复现 R1/零复现路线；把 0.5B 或公开 GSM8K 类数字写成首创（**新增**：0.5B 短链已有公开论文先例，只可写"复现并诊断"且引用 arXiv:2607.02869）；任何未经自跑的 0.5B 效果数字；照抄 README 曲线/anecdote；把仓库既有 DPO/PPO/agent 写成"我实现"；cispo 等讲不清出处的公式（**新增口径**：与 cispo 相关的稳定化技巧统一用可引用术语描述，如 advantage collapse / clip-higher / token 级 loss）；PRM"未普及"式绝对断言（**新增**：会被 0.5B 正例论文挑战）；**新增**：ACR、pass@k 反转、难度×提升等诊断指标均为**方法学引用**（NeurIPS'25 Oral / AVSPO ICML'26 / 低数据 RLVR 2604.18381），简历与口头一律表述为"复现并应用于 64M 诊断"，不写"提出/首创"。

**可写（≤3 条，v2 文案）**：
- CN 主条：基于开源 MiniMind（64M，官方权重）二次开发，将启发式+RM 奖励替换为**规则答案 verifier（GT 三态判定）**，自建数据转换/难度分层与评估脚本，跑通 SFT 冷启动 → GRPO/RLVR → heldout pass@1/16 全链；verifier 低假阴设计有单测与人工审计背书。
- CN 次条：记录 reward/格式率/正确率/长度 + **ACR 双线**与同数据三奖励源对照、动态采样/长度惩罚消融；用 **pass@k 大 k 反转**与难度×提升矩阵诊断小模型 RL 的能力门槛（格式先于正确率收敛、长度膨胀、负结果如实报告，含 135M 级外部参照——同构诊断语境，非同数据集比较）。
- EN 主条：Built an RLVR (GRPO) pipeline on MiniMind-64M: replaced heuristic+RM rewards with a rule-based answer verifier (three-state GT checking, false-negative-audited), added difficulty-stratified data & eval harness (greedy/pass@1/pass@16, format-length-accuracy decomposition, advantage-collapse rate). Reported honest findings incl. a negative result on emergent reasoning at 64M, contextualized with published 135M-scale RLVR findings (same diagnostic framing, not a same-benchmark comparison).

**面试必会推导（v1 五件事不变）+ v2 新增口径**：GRPO 组内 advantage 推导后补一句"有限组有偏（Dr.GRPO）vs 渐近 U-statistic 近最优"的统一视角；DPO/SimPO 不变；RLVR 分档抗 hack 后补"低假阴优先"设计论据。

---

## 6. 面试题库增量（v2 新增/修订 6 题）

1. 【修订】GRPO 组内 advantage 到底有没有偏？（→ 双视角：Dr.GRPO 有限组偏差与长度膨胀机制；U-statistic 渐近理论显示策略梯度类近最优 + group size 标度律；工程修正已库化。group=6 怎么来的？）
2. 【新】为什么 2026 年大厂把 RL 冷启动从 SFT 换成蒸馏（OPD）？"distill→RL 两段优于 joint"指什么？你的 64M 项目为什么不跟？（→ 无 teacher 预算；替代物=高质量冷启动 + 自蒸馏/拒绝采样）
3. 【新】你日志里的 advantage collapse rate 是什么？全对/全错组为什么无梯度？怎么缓解？（→ ACR 拆两条：全对组占比 / 全错组占比；严格说 A=0 只让 advantage 项归零，**β·KL 项仍把策略拉回 ref**——"整组无梯度"需动态采样过滤整组掩码才能做到；缓解：动态采样过滤 / AVSPO 虚拟样本 / REFT 多样化；归因：全对主导→已收敛，全错主导→容量/难度（小模型早期全错组天然占多数，ACR 高未必是奖励分辨率问题），对错混合仍高→奖励分辨率不足）
4. 【新】pass@k 反转诊断是什么？为什么大 k 下 base 反超 RL 模型？（→ NeurIPS'25 Oral：RL 不产生 base 之外的新推理模式；小 k 是概率集中收益，大 k 依赖覆盖度 → 蒸馏才扩能。你 64M 结果落在哪种形态？）
5. 【新】verifier 假阴和假阳哪个更伤？为什么"加大算力无法弥补"？（→ 假阴把正确解答判错更伤：直接压制正确行为且无补救；verifier 质量是数据问题不是算力问题 → 确定性匹配 + SFT 教格式 + 假阴率审计）
6. 【新】过程奖励在小模型有了 +9.98pp 的声称，为什么生产还在用 outcome verifier？（→ 标注成本/覆盖度/λ 非单调反直觉；主线 outcome、过程奖励是研究活跃区——体现你对"研究 vs 工程"的边界感）

---

## 7. 执行地图 v2（两周，并入 v2 加料）

| 天 | 任务 | 交付/退出标准 |
|---|---|---|
| D1-2 | 环境 + M1 骨架（verifier + 单测 + 清洗脚本 synthetic 模式）；下载 full_sft_768、RM、rlaif.jsonl（回归护栏用） | 单测 ≥99%；权重/RM 路径断言通过 |
| D3-4 | M2 数据：真实清洗三件套 + 难度桶 + report + math_rl_dataset.py（ModelScope 不可达时退 `--synthetic-only` 全合成完成管线，报告如实标注数据来源） | 50 条人工抽查无误；train/heldout 零重叠 |
| D5 | M3 MPS 冒烟 + M4 冷启动 SFT（GPU） | 冒烟 ≥2 step；heldout greedy 20 条格式全过 |
| D6-8 | M5 三臂全量（全部 `--loss_type grpo`；lr 3e-7/1e-6 取 1 档） | 曲线 + 对照表；RM 臂显存/耗时记录 |
| D9-10 | M6 前半：heldout pass@1/16（+pass@32 子集）矩阵 + 难度×提升 + 抽查 20 条结构化标注 + 假阴率报告 | 一张矩阵表 + 结论段草稿 |
| D11-12 | M6 后半：消融 A（动态采样±ACR 双线）/B（长度惩罚）/（可选 C REFT） | 消融表（无效也是结果） |
| D13-14 | M7 归档：核对重跑、5 公式手推、repo diff + 诚实 README、演示脚本 | 面试可讲 30 分钟、命令可复现 |

**兜底降级（v1 不变，按里程碑表述）**：M5 主实验不通 → 砍到两臂（verifier vs RM）；M6 评估未完成 → 保单臂全曲线 + 抽查；环境失败 → 官方 grpo 权重冒烟 + 代码走读。**永不降级**：诚实口径、可复现命令、图表与数据归档。

---

## 8. 参考来源（2026-09-06 校验，A/B/C 分级；完整清单见 planning/research_delta.md §5）

**A 一手/arXiv**：DeepSeek-R1 [arXiv:2501.12948v2](https://arxiv.org/abs/2501.12948)（86 页全配方，2026-01）；NeurIPS 2025 Oral [arXiv:2504.13837](https://arxiv.org/abs/2504.13837)（RL 不外推 base、pass@k 反转）；Dr.GRPO [arXiv:2503.20783v2](https://arxiv.org/abs/2503.20783)；GRPO U-statistic [arXiv:2603.01162](https://arxiv.org/abs/2603.01162)；135M 掉点审计 [arXiv:2606.22189](https://arxiv.org/abs/2606.22189)；0.5B 过程/结果奖励 [arXiv:2607.02869](https://arxiv.org/abs/2607.02869)（未评审）；AVSPO/ACR [arXiv:2605.21125](https://arxiv.org/abs/2605.21125)（ICML'26）；REFT [arXiv:2605.28295](https://arxiv.org/abs/2605.28295)；verifier 噪声 [arXiv:2605.25252](https://arxiv.org/abs/2605.25252)；低数据小模型 RLVR [arXiv:2604.18381](https://arxiv.org/abs/2604.18381)；课程化 [arXiv:2606.22317](https://arxiv.org/abs/2606.22317)；OPD 研究 [arXiv:2604.13016](https://arxiv.org/abs/2604.13016)（ICML'26）、[arXiv:2609.04108](https://arxiv.org/abs/2609.04108)；Nanbeige4-3B [arXiv:2512.06266](https://arxiv.org/abs/2512.06266)；Plasticity-vs-Rigidity [arXiv:2601.06677](https://arxiv.org/abs/2601.06677)。
**B 官方/二级**：llm-stats 2026 post-training 综述（[llm-stats.com](https://llm-stats.com/blog/research/post-training-techniques-2026)，2026-03）；Kimi K2 Thinking（[HF](https://huggingface.co/moonshotai/Kimi-K2-Thinking)，2025-11）；Qwen3.5（2026-02）；高志军《LLM 后训练实践》（[posttrain.gaozhijun.me](https://posttrain.gaozhijun.me/)）；TRL GRPOTrainer（[grpo_config.py](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_config.py)）。
**C 教程/线索**：HF open-r1 Mini-R1（[blog](https://huggingface.co/blog/open-r1/mini-r1-contdown-game)，替代 v1 的 philschmid 链接）；TinyZero 生态（[GitHub](https://github.com/Jiayi-Pan/TinyZero)）。

> ⚠️ 引用纪律：2026 年多篇小模型正例为未大规模复现的预印本，口头/简历引用须带"声称（claimed）"字样；大厂配方多为发布会级披露，不进技术判定。

---

*本文档由 main agent 综合 research（增量调研）、coding（实现审计）、reviewer（裁决）三子 agent 产出；v2 校验日 2026-09-06。工作底稿：planning/research_delta.md、planning/implementation_spec.md、planning/review_report.md。*
