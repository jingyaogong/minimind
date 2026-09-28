# MiniMind 后训练指南 · 增量调研（research_delta）

> 性质：research 子 agent 增量核实与升级报告，校准基线为 [docs/post-training-guide.md](../docs/post-training-guide.md)（2025-09 成文）。
> 检索日：**2026-09-06**（全部来源均在该日检索/读取，链接为检索时有效；条目各自标注发布/更新日期）。
> 用途：供 main agent 与 reviewer 决定是否修订原指南；本报告只收有依据、有判断的内容。

---

## §0 日期校准与检索范围说明

### 0.1 当前日期判定：≈ **2026-09-06**（±1 天）

原文档成文于 2025-09，本次检索发现时间已推进约 **12 个月**。判定依据（按时间戳从硬到软）：

1. **arXiv API**（export.arxiv.org，最权威）：全量 query 响应头 `<updated>2026-09-06T07:59:54Z</updated>`；cs.CL 最新条目提交于 **2026-09-03**（如 arXiv:2609.04199、2609.04172、2609.04108）。
2. Simon Willison 博客最新帖 2026-09-04（[2026/Sep/4](https://simonwillison.net/2026/Sep/4/)）；量子位侧栏最新报道 2026-09-02（李飞飞多模态世界模型、阿里 Qwen3.8-Max 等）。
3. 模型发布锚点（见 §5 链接）：
   - **Qwen3.5**（"Towards Native Multimodal Agents"，除夕开源）2026-02-16/17；
   - **DeepSeek-R1 论文 v2**（22→86 页补全）2026-01-04（arXiv 提交时间）；
   - **Kimi K2 Thinking** 2025-11-06；
   - **GLM-5.3** 2026-08-14；**GPT-6 Astra** 约 2026-08-31/09-03 发布（灰测报道 2026-08-31，llm-stats 博客列 "Released Sep 2026"）。

### 0.2 检索范围与方法

- 手段：arXiv API（export.arxiv.org，按提交时间倒序 + 指定 id_list）+ web 检索 + 直接抓取 arXiv 摘要页与少量博客原文。**凡本报告引用的 arXiv 摘要均实际读取过**；媒体/综述类仅作二级佐证。
- 检索主题面：2025H2–2026-09 的 post-training 综述与论文、GRPO 系/DPO 系/奖励范式、大厂后训练披露（DeepSeek/Qwen/Kimi/GLM）、小模型 RLVR 正反证据、训练基础设施（TRL/veRL）、原文档 §8 链接抽查。
- 未做：付费墙报告、私有训练日志核对、数字复现。前沿命名混乱（GPT-6 Astra、Claude "Fable" 系、Qwen3.8 等）仅作时间锚，不作为技术依据。
- 检索期内**无范式级官宣替代 GRPO/RLVR**：DeepSeek-R2 未发布（kr-asia 曾报道团队否认 R2，检索期无新进展）；DeepSeek 最近重大后训练披露 = R1 v2 论文（2026-01）。

### 0.3 关键时间轴（2025-09 成文 → 2026-09 检索，按时间排序）

| 日期 | 事件 | 对指南的意义 |
|---|---|---|
| 2025-09 末 | DeepSeek V3.2（doc 时代末尾） | 文档基线内，无更新 |
| 2025-10-06 | Dr.GRPO 论文 v2（understand-r1-zero 更新） | 文档引用的"有偏"论断有了终版出处（43.3% AIME24 @7B 极简配方） |
| 2025-11-06 | Kimi K2 Thinking 发布（200–300 步工具调用 + INT4 QAT 原生） | agentic RL 风口的时间戳后移 |
| 2025-11-24 | NeurIPS 2025 Oral 论文（2504.13837）v5 | "RLVR 不外推 base 能力、蒸馏才扩能"定稿 |
| 2025-12 初 | NeurIPS 2025（圣地亚哥，论文获最佳论文亚军/清华）；Nanbeige4-3B 报告（12-06） | 能力门槛叙事获顶会背书；3B 小模型配方样例 |
| 2026-01-04 | **DeepSeek-R1 论文 v2：22→86 页全配方** | 复现 R1 彻底"教科书化"；拒绝采样/语言一致性奖励等官方措辞可用 |
| 2026-01-10 | Plasticity vs Rigidity（≤1.5B LoRA-r256，AIME24 40% 声称） | 1.5B 微预算 RL 最强声称（未复现） |
| 2026-02-16/17 | **Qwen3.5 发布**（除夕开源；397B-A17B 旗舰，含 ≤2B 小 dense；"原生多模态 agent"） | 2026 开源坐标；GRPO 系配方未变 |
| 2026-02 | GLM-5 技术报告（arXiv:2602.15763） | 国产模型后训练披露样例 |
| 2026-03-01/22 | GRPO U-statistic 理论（2603.01162） | 面试口径需双视角化 |
| 2026-03-11 | llm-stats《Post-Training in 2026》综述 | "GRPO/DAPO/RLVR 取代 RLHF"的 2026 坐标 |
| 2026-04 | OPD 系统研究（arXiv:2604.13016，ICML 2026） | 蒸馏×RL 成为研究前沿主词 |
| 2026-05 | AVSPO/ACR（ICML 2026）、REFT、verifier 噪声论文集中出现 | 小模型 RLVR 稳定化工具包扩充 |
| 2026-06-20 | L20-Edu-135M 审计（135M 直接 RLVR 掉点） | 64M 量级最直接负例 |
| 2026-07-03 | 0.5B 过程 vs 结果奖励论文（GSM8K 53.75/63.73%） | "0.5B 学不会"表述必须修订 |
| 2026-08-14 | GLM-5.3 发布（开源，"后训练 scaling"话术） | 时间锚 |
| 2026-08-31/09-03 | OPSA、OPD II、OPD-vs-RLVR 等 arXiv 密集提交；GPT-6 Astra 发布前后 | "当前"= 2026-09 上旬 |

### 0.4 未核实/刻意搁置项（避免污染报告）

- Qwen3.5 主技术报告全文（arXiv 未命中主条目，官方 blog/HF 未逐字核对）；K2 Thinking 的训练配方细节（无完整技术论文，只有发布会级材料）。
- DeepSeek-V4 / R2 的一切传闻（2026-01 起"爆料"级消息，无官方训练细节，不进技术判断）。
- 闭源系（OpenAI GPT-6 Astra、Anthropic Claude/Fable、Google Gemini 3.x）后训练细节——无公开论文级材料。

---

## §1 结论速览：现有指南论断核对表（≥8 条）

| # | 原指南论断（出处章节） | 是否仍成立 | 依据 / 需要更新的地方（截至 2026-09） |
|---|---|---|---|
| 1 | GRPO/RLVR 已成为在线 RL 主流范式（§3.1/3.2） | **成立（坐标不变）** | 2026-03 llm-stats 综述仍以 "GRPO/DAPO/RLVR + synthetic self-play 取代 RLHF" 为生产配方叙事；2026-09-03 论文仍以 "RLVR 与 on-policy distillation 是后训练推理两大主流方法" 开头（arXiv:2609.04108）；TRL `GRPOTrainer` 提供 `loss_type=dr_grpo` 等原生选项。GRPO 不是"过时"，而是"地基化"。 |
| 2 | GRPO 组内 advantage 有偏是学界共识（§3.1/7-Q5） | **基本成立，但表述需双视角化** | Dr.GRPO（arXiv:2503.20783v2，2025-10-06 更新）仍是权威引用：组内归一在"长度与正确率相关"时放大长答案优势、人为拉长（尤其错误样本）。**新增**：arXiv:2603.01162（2026-03）证明 GRPO 策略梯度是 U-statistic，渐近等价于"有理想 value function 的 oracle 算法"并给出最优 group size 标度律。面试口径建议升级为："有限组有偏（Dr.GRPO）→ 长度膨胀机制；渐近理论（U-statistic）显示 GRPO 在策略梯度类中近最优；工程修正（clip-higher/token 级 loss/length-adaptive）已成库默认"。 |
| 3 | DAPO 四件套是复刻 SOTA 的标配（§3.2/7-Q10、题库⑤） | **成立，但"四件套"叙述需要扩展** | DAPO 仍是稳定化基准：2026-03 综述仍以四技巧为正统；2026 新论文（REFT、AVSPO）直接以 "DAPO/GRPO baselines" 起跑；社区出现 "Qwen3.5-DAPO-RLVR" 命名的训练仓库。**新增补丁**：AVSPO/ACR（ICML 2026）把"全对/全错组无梯度"量化成 advantage collapse rate；REFT 把熵塌缩修法延伸到"首 token 采样多样化"。面试/消融口径建议从"DAPO 四项"扩展为"稳定化工具集（clip/采样过滤/token 级 loss/长度 shaping/advantage-collapse 监控）"。 |
| 4 | "0.5B 从零 RLVR 学不会推理"（§3.4.1，参考 TinyZero） | **需精化，不能原样表述** | 见 §3 详细证据。2026 年新证据显示：**0.5B SFT 冷启动 + GRPO 在 GSM8K 短链数学上可明显提升**（outcome-only 53.75%，arXiv:2607.02869）；rollout 首 token 多样化在 0.5B–7B 上优于 DAPO/GRPO（arXiv:2605.28295）。**但**"涌现级/长链推理"仍无 0.5B 成功证据；NeurIPS 2025 Oral（arXiv:2504.13837）给出总纲式解释——RLVR 不外推 base 模型能力（大 k pass@k 下 base 反而更高）。建议改写为"0.5B 能优化短链/可表达任务，学不会长链涌现；64M 两者都不能（见 135M 掉点证据）"。 |
| 5 | "64M 只能学格式，做诚实诊断"叙事（§4.1 等） | **成立且证据增强** | 最直接新证据：**L20-Edu-135M**（arXiv:2606.22189，2026-06）134.5M 单卡全流程公开审计，直接 GRPO-style RLVR 使 GSM8K EM 从 1.82% 掉到 1.59%——"~130M 直接 RLVR 掉点"的公开负例，比 TinyZero 0.5B anecdote 更贴 64M 量级。叠加 NeurIPS 2025 Oral 的"能力上界=base"框架，64M 诚实负结果叙事仍然安全，且应把 **pass@k 大 k 反转**写成标准诊断指标。 |
| 6 | "会跑 GRPO 不再是简历亮点"（§3.3） | **更加强成立（贬值加速）** | 2026 年 GRPO 已全面库化：TRL `GRPOTrainer`（`loss_type` 含 `dr_grpo` 等，[grpo_config.py](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_config.py)）；高志军《LLM 后训练实践》站点 2026-09 仍在线上；2026 论文里 0.5B–7B 的 GRPO 实验是常规手段。原指南"从零实现 + 训练动态量化理解 + 诚实负结果"作为差异点的判断**完全成立**，且差异化素材又多了 ACR、pass@k 反转、verifier 噪声三件可量化工具。 |
| 7 | PRM/过程奖励"成本高、未普及、小模型不推荐"（§3.2） | **表述过时，需软化** | 主流生产仍用 outcome verifier（未见大厂过程奖励规模化披露），但研究侧出现小模型正面证据：**Qwen2.5-0.5B + GRPO 上，process-only 63.73% vs outcome-only 53.75%（+9.98pp，GSM8K）**，且过程监督降低轨迹偏离（arXiv:2607.02869，2026-07）。"PRM 在小模型上无用"不再成立；建议改为"研究活跃、生产未主流；可选做 process-vs-outcome 微型对照（成本可控，见 §4）"。 |
| 8 | DeepScaleR "1.5B + 蒸馏起点可行（AIME24 43.1%）"（§3.2） | **成立且 2026 佐证更强** | 2026-01 "Plasticity vs Rigidity"（arXiv:2601.06677）报告 ≤1.5B、单 A40/24h、**高秩 LoRA（r=256）** + instruct-tuned 起点 → AIME24 Pass@1 40.0%（+11.1pp），Pass@16 70%；2026-08 OPSA（arXiv:2608.31046）在 Qwen3-1.7B base 上无 teacher 改进 AIME24 Avg@32（相对 +263%）。均需标注"单卡、未大规模复现"后再引用。**注意**：这些是 instruct 起点或带技巧，不是"从零"，反而强化了原指南"起点质量决定 RL 上限"的判断。 |
| 9 | 多轮 agentic RL 是最强风口、排第三优先级、不做 train_agent 主线（§3.2/§4.3） | **风口热度仍在，但研究前沿出现分化** | K2 Thinking（2025-11-06，200–300 次工具调用稳定性、INT4 QAT 原生）与 Qwen3.5（2026-02-16，原生多模态 agent）证明 agentic RL 是生产差异主战场；llm-stats 2026-03 把"environment-native training"列为 next-year 方向。**新增分化**：推理 RL 的研究前沿同时被 on-policy distillation（OPD）占去一半话语权（见 §2）。对 1–2 周简历项目，"不做 agentic、只讲概念"的裁决**维持不变**。 |
| 10 | 冷启动 SFT 是 RL 地基（§4.2） | **成立；行业做法已演进，需一句更新** | 原判断在 64M 场景仍对。但 2026 行业结论是 **"OPD 是比 SFT 更好的 RL 冷启动；distill→RL 两段优于 joint"**（arXiv:2609.04108，2026-09-03；ICML 2026 接收的 OPD 系统研究 arXiv:2604.13016）。64M/24GB 无 teacher 预算，不能实现 OPD，但面试语境必须能解释"为什么大厂把冷启动从 SFT 换成蒸馏"；Nanbeige4-3B（arXiv:2512.06266）也给出 3B 级"蒸馏 + 多阶段 RL"工业配方佐证。 |
| 11 | 简历红线与可写版本（§6） | **总体仍成立；2 处需更新** | ① "复现 R1"更不值钱：R1 v2 论文（2026-01）把 86 页全配方公开（含四步管线、语言一致性奖励、安全 RL、aha-moment 量化），复现 R1 已是"照着教科书跑"；但 v2 本身是极佳面试弹药（见 §4-10）。② "0.5B 效果数字禁写"的红线要小心：0.5B GSM8K 53–64% 已有公开论文（arXiv:2607.02869），若 0.5B 冒烟复现到类似数字，不可宣称首创，只能写"复现并诊断"，且必须引用该文。③ "PRM 未普及"若作为面试断言，会被 0.5B 正例论文挑战，措辞改为"生产未主流、研究活跃"。 |
| 12 | 原文档 §8 参考链接时效 | **大部分在线；2 处需修** | 抽查结果：Jiayi-Pan/TinyZero（GitHub 在线，多 fork 续作）、dapo-sia.github.io（2026-03 llm-stats 仍引用）、sail-sg/understand-r1-zero（arXiv v2 更新至 2025-10-06，非死链）、posttrain.gaozhijun.me（2026-09 在线）、DeepScaleR（模型仍在被第三方托管，featherless 等）。**需修**：① DeepSeek-R1 引用应升级为 **arXiv:2501.12948v2（2026-01-04，86 页）**；② philschmid.de/mini-deepseek-r1 未在检索中直接命中，同名教程现以 HF open-r1 博客 [Mini-R1](https://huggingface.co/blog/open-r1/mini-r1-contdown-game) 形态存在，建议换链。 |

---

## §2 新增主流/前沿技术清单（只收有可靠来源者）

判断列口径：**是**＝值得花时间做进 64M 主实验/消融；**可选**＝有余力/特定条件才做；**否**＝不实现（但可能值得讲/读）。64M 数学 RLVR 简历项目语境。

| 技术 | 一句话机制 | 来源链接 | 是否值得引入 | 理由 |
|---|---|---|---|---|
| **DeepSeek-R1 v2 全配方（22→86 页）** | R1 四步管线（冷启动 SFT → 推理 RL + 语言一致性奖励 → 拒绝采样重 SFT → 对齐 RL）首次完整公开；R1-Zero "aha" 用反思词频（wait/mistake…）量化（×5–7 倍，"wait" 峰值出现在 ~8k 步后）；安全 RM 与 GPT-4o LLM-as-judge 评测细节 | [arXiv:2501.12948v2](https://arxiv.org/abs/2501.12948)（2026-01-04）；中文报道 [qbitai](https://www.qbitai.com/2026/01/367678.html)（2026-01-08） | **是（读，不是做）** | 教材级一手材料：冷启动/拒绝采样/两阶段 RL 的官方措辞可全文引用于指南讲解与面试推导；"语言一致性奖励"是文档未覆盖的 R1 细节，1 句话可补进 §3.2/题库。 |
| **GRPO U-statistic 理论** | 证明 GRPO 策略梯度是 U-statistic：MSE/有限样本界/渐近分布；渐近等价于有理想 value function 的 oracle；给出最优 group size 的通用标度律 | [arXiv:2603.01162](https://arxiv.org/abs/2603.01162)（2026-03-01，v3 03-22） | **否（不实现），面试口径必读** | 与 Dr.GRPO"有偏"叙事构成一对张力，2026 面试若只讲"有偏"会显得知识停在 2025；group size 标度律可支撑"为什么 group=6/8"的追问。理论实现成本为零。 |
| **ACR + AVSPO（advantage collapse）** | 把"全对/全错组→advantage≈0→无梯度"量化成诊断指标 Advantage Collapse Rate；AVSPO 注入虚拟奖励样本缓解，0.5B–14B 声称 +4–6pp | [arXiv:2605.21125](https://arxiv.org/abs/2605.21125)（ICML 2026，2026-05 提交） | **是（轻量，约 20–30 行统计代码）** | 文档第 4 项消融正是"动态采样过滤"；ACR 曲线（横轴训练步，纵轴无效梯度批占比）把该消融从"有/无"升级为"连续量化"，且天然适配小模型"全对/全错组频繁出现"的情形——低风险高展示度。 |
| **REFT：首 token 多样化 rollout** | 采样时对推理起始 token 在其 top-N 里均匀化并均分 rollout 预算，扩组内覆盖而不动奖励信号；0.5B–7B 四个 base 上 Pass@1/8/64 优于 DAPO 与 GRPO | [arXiv:2605.28295](https://arxiv.org/abs/2605.28295)（2026-05-27） | **可选** | 改 rollout 采样约 10 行、与现有 torch 引擎兼容；若 D6–8 主 run 顺利且 group 内多样性不足（可先用 log 判断），可作为第 3 项消融候选。论文 0.5B 级有效是加分点。 |
| **过程奖励 × 小模型（process reward for SLM）** | 0.5B SFT + GRPO 上比较 process/outcome/hybrid 五条件：process-only 63.73% vs outcome-only 53.75%（GSM8K），hybrid λ=0.1 反而劣于纯 outcome | [arXiv:2607.02869](https://arxiv.org/abs/2607.02869)（2026-07-03，8 页短文，未评审，引用时标注） | **可选** | 文档现判"小模型不推荐 PRM"需修正；但 64M 无过程标注预算——务实做法是把"过程 vs 结果奖励"放进 B 支线或面试谈资（含 λ 非单调这一反直觉发现），不要在主线引入 PRM。 |
| **Verifier 噪声不对称结论** | 向 0.5B/1.5B GRPO 注入 verifier 假阳/假阴：假阴（漏判对）比假阳更伤；加大 compute 无法弥合噪声缺口 | [arXiv:2605.25252](https://arxiv.org/abs/2605.25252)（ICML 2026 Workshop，2026-05） | **是（采纳结论，不加实验）** | 直接背书文档 §4.2 verifier 工程取向："确定性精确匹配、宁缺勿错（低假阴）"；面试可讲"verifier 质量与算力不可互换"。 |
| **OPD（on-policy distillation）与 distill→RL 两段配方** | student 自采样 + teacher 逐 token 监督；系统研究（ICML 2026）+ "OPD 是比 SFT 更好的 RL 冷启动、distill-then-RL 两段优于 joint 融合" | [arXiv:2604.13016](https://arxiv.org/abs/2604.13016)（ICML 2026，2026-04）；[arXiv:2609.04108](https://arxiv.org/abs/2609.04108)（2026-09-03）；[Part II arXiv:2609.04172](https://arxiv.org/abs/2609.04172) | **否（64M/24GB 无 teacher 预算、仓库无蒸馏管线），但必须会讲** | 2026 推理后训练研究前沿的最大新主题（两周内 4+ 篇 follow-up）；简历项目不实现，但"冷启动从 SFT 演进为蒸馏"必须在面试口径里（见 §1-10）。若坚持加料，唯一低成本变体是把冷启动数据换成更强的离线蒸馏数据（Nanbeige4-3B 式），与文档"自造数据"章合并考虑。 |
| **课程化 / 难度分层 RLVR + 程序化数据** | 低数据小模型 RLVR 实证：混合难度数据 ≈ 5× 样本效率；Curriculum-RL（2026-06）声称课程化可使 RL 超越 base 上限——直接回应 NeurIPS 2025"RL 不外推"论文 | [arXiv:2604.18381](https://arxiv.org/abs/2604.18381)（2026-04-20）；[arXiv:2606.22317](https://arxiv.org/abs/2606.22317)（2026-06） | **可选（进数据章，成本≈给自造数据标难度再排序）** | 文档本来就要自造 ≥1000 条 + heldout；按"先易后难"分批进入 RL（或至少记录难度-提升矩阵）几乎零成本，且与"64M 能力门槛"叙事互斥又互补：如果课程化有效→正结果亮点；无效→更硬的门槛证据。 |
| **高秩 LoRA 做 RL（micro-budget 配方）** | ≤1.5B、单 A40 <24h：LoRA r=256 + instruct 起点 → AIME24 Pass@1 40.0%；r=8 失败；math 预训练过重的 base 反而塌缩 | [arXiv:2601.06677](https://arxiv.org/abs/2601.06677)（2026-01-10） | **可选（仅在 0.5B 冒烟解禁时）** | 仓库已有 LoRA 注入（教学级），但 RL 训练器是全参路径；若 main agent 同意 0.5B 冒烟，LoRA-RL 是显存可行解且"秩-塑性"故事可讲。r=8 失败这点与 64M 低秩容量担忧互相印证，可作 §4.1"为什么 64M 用全参"的论据。 |
| **OPSA：熵自适应负 advantage（无 teacher 自改进）** | 声称 OPD 的收益≈压制低概率尾 token（"does OPD really distill?"），提出用熵自适应负 advantage 复现该收益；Qwen3-1.7B AIME24 Avg@32 大幅提升 | [arXiv:2608.31046](https://arxiv.org/abs/2608.31046)（2026-08-31，太新） | **否** | 结论与机制争议大（刚上线一周），未沉淀；仅作为"2026 年熵塌缩修法仍在演进"的谈资一句带过，不做。 |
| **K2 Thinking 式"长程 agentic 生产化 + INT4 QAT 原生"** | 万亿 MoE 上做 agentic reasoning RL：200–300 步工具调用稳定 + 量化感知训练（qat int4）保证 256k 上下文可用 | [HF Kimi-K2-Thinking](https://huggingface.co/moonshotai/Kimi-K2-Thinking)；[Simon Willison 记录](https://simonwillison.net/2025/Nov/6/kimi-k2-thinking/)（2025-11-06） | **否** | 与 64M 项目无关；用途仅是更新文档"agentic RL 风口"的时间戳与例子（K2 已于 2025-11 落地，且 Qwen3.5 于 2026-02 走原生多模态 agent）。 |

**刻意不收**（检索到但判定噪音/不实/离题，逐条说明，供 reviewer 仲裁）：

1. **SPICE / SPIN 类 self-play 条目**（llm-stats 综述收录）：通用对话/文档 grounded 自博弈，与数学 RLVR 主任务、64M 诊断叙事无交集；引入会稀释实验主题。
2. **RISE 自验证**（arXiv:2505.13445，2025-05）：检索期内未见后续主流化/生产采用证据；"模型学会自我检查"在 64M 上无验证手段。
3. **"GRPO 是否过时"类自媒体问答**（36kr/知乎 等）：无可靠信息增量，仅折射"GRPO 地基化"的舆论信号；引用会拉低报告证据密度。
4. **ICT / Beyond Entropy**（arXiv:2606.19771）：token 分布熵调控属研究级，三篇基线里有 20-Entropy/STAPO 等未沉淀对象；64M 上先解决格式/正确率解耦，不解决熵精细调控。
5. **DeepSeek-V4 传闻**（2026-01 qbitai 爆料起）：无官方后训练细节，仅作时间锚。
6. **TGOPD / CA-OPD / reranker-OPD**（2026-09 一周内的 OPD 泛化浪潮）：方向验证 OPD 热度，但均为多 teacher/视觉/检索场景，与 64M 数学项目无关。
7. **frontier 命名与榜单数字**（GPT-6 Astra、Claude Fable 5.x、Qwen3.8-Max 等 2026-08/09 发布流）：时间锚用途，不做技术依据（llm-stats/zaobao/qbitai 口径混乱，无法交叉验证）。

---

## §3 小模型 RLVR 可行性证据更新（64M/0.5B/1.5B，正反）

### 3.1 正面（小模型可学会什么——用于校准"能力门槛"的边界）

1. **0.5B SFT + GRPO 在 GSM8K 短链数学可学到 50–64%**（outcome 53.75% / process 63.73%，arXiv:2607.02869，2026-07）。模型为 Qwen2.5-0.5B（SFT 后），任务是单步/短文答案数学。→ 证明"0.5B 从零（SFT 冷启动后）RLVR 学不会推理"是**错误表述**；正确的是"学不会涌现式长链推理，短链任务可被 RL 优化"。
2. **rollout 多样化技巧在 0.5B 有效**：REFT（arXiv:2605.28295）四个 base 含 0.5B，Pass@1/8/64 均优于 DAPO/GRPO。→ 0.5B 的问题部分在探索不够，不在纯容量。
3. **≤1.5B 微预算 RL 有激进声称**：LoRA r=256 + instruct 起点 → AIME24 Pass@1 40.0%、Pass@16 70%（arXiv:2601.06677，单 A40/24h，**未大规模复现，引用必须带 caveat**）；OPSA 在 Qwen3-1.7B base 无 teacher 提升 AIME24 Avg@32（arXiv:2608.31046）。
4. **3B 工业级小模型配方**：Nanbeige4-3B = 高质量 SFT（deliberative refinement + CoT 重建）→ 蒸馏（Dual Preference Distillation）→ 多阶段 verifiable RL + 偏好 RL（arXiv:2512.06266，2025-12）。→ 小模型"多阶段 + 蒸馏 + RL"仍是主流配方模板。
5. 已有基线仍有效：DeepScaleR-1.5B（AIME24 43.1%，2025-04，文档已引，检索期内仍被第三方托管使用）。

### 3.2 负面 / 门槛（支撑"诚实负结果"叙事）

1. **135M 直接 GRPO-RLVR 掉点（最贴 64M 的证据）**：L20-Edu-135M 单卡公开审计，GSM8K EM 1.82%→1.59%（192-token 完成度）；192/320 token 上限下 RLVR 无正收益（arXiv:2606.22189，2026-06）。→ 64M 主实验若全负/仅格式上涨，不是异常，是该量级的规律。
2. **NeurIPS 2025 Oral 总纲式结论**："当前 RLVR 训练设置不激发根本性新推理模式；小 k 下 RLVR 优于 base，大 k（pass@k）下 base 反而更高；推理源于并被 base 封顶；六种流行 RLVR 算法彼此接近且都远未榨干 base；**蒸馏才能引入 teacher 的新推理模式**"（arXiv:2504.13837，v5 2025-11-24；清华团队，NeurIPS 2025 最佳论文亚军/ICML'25 AI4MATH 最佳）。→ 文档"RL 只强化已有行为、不能教知识"从 anecdote 升级为顶会结论；**蒸馏 > RL 扩能**成为 2026 定论式表述。
3. **反例提醒：Curriculum-RL 声称可超越 base**（arXiv:2606.22317，2026-06）——"上界=base"不是物理定律；面试若讲死会被这类工作挑战，建议表述为"当前主流 RLVR 配方在绝大多数设置下不外推 base（NeurIPS'25 Oral），课程化/环境交互等新范式可能打破"。
4. **verifier 噪声不可被算力弥补**（0.5B/1.5B，假阴更毒，arXiv:2605.25252）→ 小模型实验结果对 verifier 质量更敏感，"负结果"须先排除 verifier 漏洞（文档 §4.2 的 verifier 单测纪律被背书）。
5. TinyZero "0.5B base fails to learn reasoning" 仍被 2026 社区当作起点引用（多 fork 续作），但只对 **base（无冷启动）** 成立——文档引用它时建议加此限定。

### 3.3 对 64M 叙事的综合校准

- "64M 只能学格式、做诚实诊断" **仍然站得住**，且有 135M 掉点 + NeurIPS'25 Oral 两个新支柱；但"0.5B 也学不会"必须撤下，改为"0.5B 学得会短链（GSM8K 50–64%），学不会长链涌现"。
- 若预算允许，最有解释力的单图是 **64M / 135M 级 / 0.5B 三点能力曲线（正确率-格式率-长度 vs 步数 + pass@k 反转点）**——把"能力门槛"做成横跨规模的定量诊断，正好回应 NeurIPS'25 Oral 的方法学（pass@k 大 k 视角）。0.5B 仍受"无官方权重需自 pretrain"约束，故只在文档已设的"冒烟"路径内做。
### 3.4 证据 → 实验动作映射（把 §3.1/3.2 变成可执行假设）

1. **预期管理**（L20-Edu-135M）：64M 主 run 的先行基线不是"涨点"，而是"reward/格式率上升 + 正确率停滞或微跌 + 135M 同构"。把 135M 的 1.82%→1.59% 当作对照数字写进分析段，负结果就有了外部基准，不再自说自话。
2. **pass@k 反转诊断**（NeurIPS'25 Oral）：heldout 评估矩阵加一列 pass@16/32（现文档只有 pass@1），若"RL 后小 k 涨、大 k 反而不如 SFT 冷启动模型"，直接复现顶会结论——这是 2026 年最有辨识度的诊断语言。
3. **冷启动归因**（0.5B 正例 2607.02869 + NeurIPS 蒸馏结论）：若 64M 全负而公开 0.5B 正例成立，差异变量是"模型容量"还是"冷启动数据质量"？文档已留 0.5B 冒烟路径，冒烟时建议与 64M 用同一套数据/verifier，才能归因。
4. **格式 vs 推理的解耦叙事**：process 奖励论文（2607.02869）给出"outcome 模型轨迹短但推导易错、process 模型结构不稳但算术扎实"的失败模式二分——64M 人工抽查 20 条的标注模板可借用这两类标签，升级"抽查"为结构化审计。
5. **verifier 假阴红线**（2605.25252）：负结果报告前必须先排除 verifier 系统性漏判（文档的合成 200 条单测 + 50 条人工抽查已覆盖，报告里写明"假阴率 < X%"）。
6. **课程化试点**（2604.18381/2606.22317）：自造数据若分易/中/难三档，则 RL 报告天然多一张"难度×提升"热力图；若 64M 只在易档有正收益，那本身就是"能力门槛"的最干净证据。
7. **熵/advantage 塌缩量化**（AVSPO-ACR）：四曲线之外加第 5 条 ACR 曲线（全对/全错组占比），用于回答"64M 的停滞是没信号还是没容量"——ACR 高说明信号问题（可归因奖励设计），ACR 低仍停滞说明容量问题。这条归因对面试叙事价值极高。

### 3.5 一句话裁决（给 main/reviewer）

- "64M 只能学格式、做诚实诊断" —— **维持**，理由换新（135M 掉点 + NeurIPS'25 Oral + pass@k 反转诊断法）。
- "0.5B 从零 RLVR 学不会推理" —— **必须修订**为"0.5B 短链可学（GSM8K 50–64% 有论文先例）、长链涌现学不会；64M 两者皆难"。
- "会跑 GRPO 已贬值" —— **加速贬值**，GRPOTrainer 与 2026 论文 baseline 化是证据；从零实现 + 诊断指标（ACR/pass@k 反转）是更硬通货。

---

## §4 对现有指南的具体修改建议（编号，指向原文档章节）

1. **全文时间戳与坐标**（§1–§3）：文档落款 2025-09 → 校验日 2026-09。§3.1/3.2 引用基线建议更新为：R1 v2（2026-01）、NeurIPS'25 Oral（2025-12 会议/2025-11 v5）、GRPO U-stat（2026-03）、llm-stats 2026-03 综述、K2 Thinking（2025-11）、Qwen3.5（2026-02）。
2. **§3.1 三件套表述**：把 "GRPO 免费去 critic 是简化叙事——组内归一 advantage 有偏（Dr.GRPO）" 升级为双视角（Dr.GRPO 有限组偏差 → 长度膨胀机制；U-statistic 渐近 oracle/最优 group size 标度律，arXiv:2603.01162）；注明 TRL `GRPOTrainer` 已把 `dr_grpo` 等做成 `loss_type` 选项（"库化"证据）。
3. **§3.2 新增一小节"蒸馏 × RL 的 2026 坐标"**：OPD 系统研究（ICML 2026）+ "distill→RL 两段优于 joint、OPD 是更好冷启动"（arXiv:2609.04108）；写明"64M/24GB 不实现 OPD（无 teacher），只更新面试口径与冷启动叙事"。同时补 R1 v2 四步管线里的"语言一致性奖励"与"拒绝采样重 SFT"官方措辞。
4. **§3.2 PRM 句**："成本高、未普及、小模型不推荐" → "生产未主流、研究活跃（含 0.5B +9.98pp 证据，arXiv:2607.02869）；主线仍用 outcome verifier，过程奖励留作可选微型对照"。
5. **§3.3 简历坐标系**：把 "认识 DAPO/Dr.GRPO 名词" 从差异点候选区彻底移入"人人都会"区（2026 论文把它们当 baseline）；新增候选差异点：ACR/advantage-collapse 连续量化、pass@k 大 k 反转诊断、verifier 噪声对照、与公开 0.5B/1.5B 结果的规模外推讨论。
6. **§3.4 小模型难点**：第 1 条改写为三点论（NeurIPS'25 Oral 上界 / 0.5B 短链正例 / 135M 掉点），并把"蒸馏才扩能"写进总纲；其余条目补充机制引用：熵塌缩→AVSPO/ACR（ICML'26）+ ICT（arXiv:2606.19771，可选一句）；奖励作弊→verifier 噪声不对称（假阴更毒）。
7. **§4.2 verifier 工程**：补一条来自 2605.25252 的设计论据——**假阴（把正确解答判错：漏提取/格式失败吃掉正确分）比假阳更伤，且加大 compute 无法弥补**。因此"确定性匹配 + SFT 教格式 + 扫全文"的目标不只是文档所写的"防作弊（低假阳）"，更是**保低假阴**；负结果报告前应自测并写明 verifier 假阴率（合成 200 条单测里加"格式正确但答案正确"用例即可）。
8. **§4.2 数据/冷启动**：自造数据章建议采纳"难度分层/课程化 + 程序化数据"（2604.18381 的 5× 样本效率、2606.22317 的 curriculum 反例）；heldout 之外按难度分批进入 RL（低风险，2–4 小时工作量）。
9. **§4.2 消融清单**：把"动态采样过滤"消融升级为 ACR 曲线 + 有/无过滤两臂（约 20–30 行，见 §2-ACR）；有余力再加 REFT 首 token 多样化（第 3 消融候选）。
10. **§6 红线/可写与题库**：红线区补 "0.5B GSM8K 类数字禁写首创（已有 2607.02869 先例，可写复现不可写首创）"；可写 CN/EN 文案建议把 "diagnosed dynamics" 具体化为 "advantage-collapse rate & pass@k inversion diagnostics"；题库补 2 题："为什么 2026 年大厂把冷启动从 SFT 换成蒸馏（OPD→RL）？64M 为何不跟？"、"过程 vs 结果奖励在小模型的新证据与 λ 非单调为什么？"；第 5 推导题组加入 R1 v2 的"语言一致性奖励"一句话。
11. **§5 执行地图**：D9–10 评估节加 pass@k（k≥16 与 greedy 同报）作为正式输出项（NeurIPS'25 Oral 方法学）；D7 兜底降级不变。
12. **§8 参考来源**：R1 引用升级 2501.12948v2（86 页，2026-01-04）；philschmid 链接换 [HF open-r1 Mini-R1](https://huggingface.co/blog/open-r1/mini-r1-contdown-game)；把本报告 §5 新增条目按 A/B 级并入。

---

## §5 参考来源（均为实际检索/读取，2026-09-06）

> 读取方式标注：**[全文]**＝抓取过正文/摘要；**[检索]**＝检索结果命中（含标题/摘要片段，未全文读取）。日期为条目自身发布/更新时间。

### A. 一手/arXiv（本报告判断的主要依据）

- **DeepSeek-R1 v2（22→86 页）**：[arXiv:2501.12948](https://arxiv.org/abs/2501.12948)（v2 提交 2026-01-04；期刊版 Nature 645:633-638, 2025）**[全文-元数据]**
- **NeurIPS 2025 Oral**：Does RL Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?：[arXiv:2504.13837](https://arxiv.org/abs/2504.13837)（v5: 2025-11-24；NeurIPS 2025 Oral + ICML 2025 AI4MATH best）**[全文]**；清华报道（2025-12-06）[链接](https://www.au.tsinghua.edu.cn/info/1305/4336.htm) **[检索]**
- **Dr.GRPO / understand-r1-zero**：[arXiv:2503.20783](https://arxiv.org/abs/2503.20783)（v2: 2025-10-06；43.3% AIME24 @7B minimal recipe）**[全文-元数据]**
- **GRPO 策略梯度是 U-statistic**：[arXiv:2603.01162](https://arxiv.org/abs/2603.01162)（2026-03-01，v3 03-22）**[全文]**
- **过程 vs 结果奖励 × 0.5B（GSM8K：process 63.73 / outcome 53.75）**：[arXiv:2607.02869](https://arxiv.org/abs/2607.02869)（2026-07-03）**[全文]**
- **L20-Edu-135M（135M 单卡审计，RLVR 掉点）**：[arXiv:2606.22189](https://arxiv.org/abs/2606.22189)（2026-06-20）**[全文-摘要]**
- **REFT（首 token 多样化，0.5B–7B）**：[arXiv:2605.28295](https://arxiv.org/abs/2605.28295)（2026-05-27）**[全文-摘要]**
- **ACR/AVSPO（ICML 2026）**：[arXiv:2605.21125](https://arxiv.org/abs/2605.21125)（2026-05-20，v2 05-30）**[全文-摘要]**
- **Verifier 噪声不对称（ICML 2026 Workshop）**：[arXiv:2605.25252](https://arxiv.org/abs/2605.25252)（2026-05-24，v2 05-27）**[全文-摘要]**
- **低数据小模型 RLVR**：[arXiv:2604.18381](https://arxiv.org/abs/2604.18381)（2026-04-20）**[全文-摘要]**
- **Curriculum RL 超越 base 的声称**：[arXiv:2606.22317](https://arxiv.org/abs/2606.22317)（2026-06）**[检索]**
- **Plasticity vs Rigidity（≤1.5B LoRA-RL, AIME24 40%）**：[arXiv:2601.06677](https://arxiv.org/abs/2601.06677)（2026-01-10）**[全文-摘要]**
- **OPSA（熵自适应负 advantage）**：[arXiv:2608.31046](https://arxiv.org/abs/2608.31046)（2026-08-31）**[全文-摘要]**
- **ICT/熵调控（0.5B/1.5B/7B）**：[arXiv:2606.19771](https://arxiv.org/abs/2606.19771)（2026-06-18）**[全文-摘要]**
- **OPD 系统研究 I（ICML 2026）**：[arXiv:2604.13016](https://arxiv.org/abs/2604.13016)（2026-04-14；会议页 [icml.cc/virtual/2026/70312](https://icml.cc/virtual/2026/70312)；代码 [github.com/thunlp/OPD](https://github.com/thunlp/OPD)）**[全文-摘要]**
- **OPD II：单样本极限**：[arXiv:2609.04172](https://arxiv.org/abs/2609.04172)（2026-09-03）**[全文-摘要]**
- **OPD vs RLVR："两段优于 joint、OPD 是更好冷启动"**：[arXiv:2609.04108](https://arxiv.org/abs/2609.04108)（2026-09-03）**[全文-摘要]**
- **TGOPD（teacher gating）**：[arXiv:2609.02998](https://arxiv.org/abs/2609.02998)（2026-09-02）**[全文-摘要]**
- **Nanbeige4-3B 技术报告（3B 小模型配方）**：[arXiv:2512.06266](https://arxiv.org/abs/2512.06266)（2025-12-06）**[全文-摘要]**
- **Qwen3.5-Omni 技术报告**：[arXiv:2604.15804](https://arxiv.org/abs/2604.15804)（2026-04-17）**[全文-摘要]**（佐证 Qwen3.5 家族时间线）
- **VAPO**：[arXiv:2504.05118](https://arxiv.org/abs/2504.05118)（2025-04）**[检索]**；技术解读（2026-06-09）[链接](https://www.zhongzhuzhou.org/blog/2026-06-09-vapo-technical-review-en/) **[检索]**
- **GLM-5 技术报告**：GLM-5: from Vibe Coding to Agentic Engineering：[arXiv:2602.15763](https://arxiv.org/abs/2602.15763)（2026-02）**[检索]**
- **Qwen3.5 主技术报告**：arXiv `ti:"Qwen3.5"` 检索未命中主条目；第三方论文引其标题为 "Qwen3.5: a new generation of hybrid-thinking open foundation models"（Qwen Team, 2026）**[检索-间接]**；官方发布见 B 类链接。

### B. 官方发布 / 媒体报道 / 综述（时间锚与二级佐证）

- **llm-stats《Post-Training in 2026: GRPO, DAPO, RLVR & Beyond》**：[链接](https://llm-stats.com/blog/research/post-training-techniques-2026)（2026-03-11，二级综述，DAPO/GRPO/RLVR/self-play/agentic 现状叙事）**[全文]**
- **Qwen3.5 发布**：科技日报 [链接](https://www.stdaily.com/web/gdxw/2026-02/16/content_474848.html)（2026-02-16 除夕开源）**[检索]**；gigazine [链接](https://gigazine.net/news/20260217-qwen3-5-397b-a17b/)（2026-02-17）**[检索]**；Simon Willison 笔记 [Qwen3.5: Towards Native Multimodal Agents](https://simonwillison.net/2026/Feb/17/qwen35/)（2026-02-17）**[检索]**
- **Kimi K2 Thinking（2025-11-06）**：[HF moonshotai/Kimi-K2-Thinking](https://huggingface.co/moonshotai/Kimi-K2-Thinking)；[Simon Willison 记录](https://simonwillison.net/2025/Nov/6/kimi-k2-thinking/) **[全文]**
- **DeepSeek R1 v2 报道**：[量子位](https://www.qbitai.com/2026/01/367678.html)（2026-01-08，64 页新增内容总结）**[全文]**；OFweek 同题 [链接](https://www.ofweek.com/ai/2026-01/ART-201721-8140-30679242.html)（2026-01）**[检索]**
- **GLM-5.3 发布**：[科技日报](https://www.stdaily.com/web/gdxw/2026-08/14/content_564134.html)（2026-08-14）**[检索]**；[联合早报](https://www.zaobao.com/news/china/story20260814-9519106)（2026-08-14）**[检索]**
- **GPT-6 Astra 发布（约 2026-09-03）**：[mashable](https://mashable.com/tech/openai-gpt-6-astra-launch-pricing-safety-benchmarks) **[检索]**；灰测报道 [量子位](https://www.qbitai.com/2026/08/481893.html)（2026-08-31）**[检索]**
- **arXiv API 全量时间戳（cs.CL 最新 2026-09-03；feed 2026-09-06）**：[export.arxiv.org](https://export.arxiv.org/api/query?search_query=cat:cs.CL&sortBy=submittedDate&sortOrder=descending&max_results=10) **[全文]**

### C. 社区 / 基础设施（时效与"课程化/库化"证据）

- **TRL GRPOTrainer `loss_type`（含 `dr_grpo`）**：[github.com/huggingface/trl grpo_config.py](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_config.py)（2026-09 检索有效）**[检索]**
- **高志军《LLM 后训练实践》**：[posttrain.gaozhijun.me](https://posttrain.gaozhijun.me/) 与 [课程总览](https://posttrain.gaozhijun.me/docs/)（2026-09 在线）**[检索]**
- **TinyZero 生态**：[github.com/Jiayi-Pan/TinyZero](https://github.com/Jiayi-Pan/TinyZero)（在线，多续作 fork，2026-09 检索）**[检索]**
- **Mini-R1 教程（原 philschmid 链接疑似迁移）**：[HF open-r1 Mini-R1: Reproduce DeepSeek R1 "aha moment"](https://huggingface.co/blog/open-r1/mini-r1-contdown-game) **[检索]**
- **DeepScaleR 仍被托管使用**：[featherless.ai/agentica-org/DeepScaleR-1.5B-Preview](https://featherless.ai/models/agentica-org/DeepScaleR-1.5B-Preview)（2026-09 检索）**[检索]**
- **DAPO**：[dapo-sia.github.io](https://dapo-sia.github.io/)（2025-03；被 2026-03 llm-stats 综述引用，判为在线）**[检索-间接]**
- **Qwen3-0.6B thinking 模式社区困惑**（C 级线索，仅用于说明"小模型 thinking 状态非官方承诺"）：[QwenLM/Qwen3 discussion #1329](https://github.com/QwenLM/Qwen3/discussions/1329) **[检索]**

---

## §6 检索局限声明

- arXiv 摘要为权威源，但 2026 年多篇小模型正例（2601.06677 / 2605.28295 / 2605.21125 / 2608.31046 等）均为**未大规模复现的预印本/新会议论文**，数字引用须带"声称"字样。
- 大厂 2026 后训练配方大多只有发布会/博客级披露（Qwen3.5 主报告未在 arXiv 命中主条目；K2 Thinking 无详细 RL 训练论文），未纳入 §2 的"机制"判定。
- 本报告未覆盖：OpenAI/Anthropic/Google 闭源后训练细节、付费分析报告、DeepSeek-V4 传闻细节。

*文档：research 子 agent，2026-09-06。*
