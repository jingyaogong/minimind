# MiniMind 后训练升级指南（简历 + 求职向）

> 背景：投递 LLM 后训练（post-training）方向；24GB 单卡；1~2 周预算；宁精勿滥。
> 本文由多 agent 协同产出：research（技术调研）+ coding（代码审计）+ reviewer（交叉评审与裁决）。
> 定位：**讲解 + 选型建议**，不做逐行实现手册。

---

## 1. 结论速览（TL;DR）

- **一句话定位**：基于开源 MiniMind（官方 64M dense 权重）二次开发，把仓库里"启发式格式分 + LLM judge"的通用 RLAIF 奖励，替换为**规则答案 verifier（GT 判定）**，跑通 SFT 冷启动 → GRPO/RLVR → heldout 评估的完整后训练链，产出**奖励/格式率/正确率/长度四曲线 + 对照 + 消融**，并把"小模型 RL 的能力门槛"做成**带数据的诚实诊断**。
- **核心洞察**：2025 年面试语境下"会跑 GRPO/RLVR"已经贬值（公开课都在教），真正稀缺的是：**从零实现（你能推导每个算子）+ 对训练动态的量化理解（熵塌缩/长度膨胀/奖励黑客）+ 严谨小消融与诚实负结果**。你仓库最大的存量资产是从零手写的 trainer，不要浪费在"再复现一个 R1"上。
- **三做**：① 数学 RLVR 诊断式复现（verifier 替换 judge，主实验）；② 三臂奖励对照 + 2 项消融；③ 四曲线 + heldout pass@1 + 人工抽查的评估纪律。
- **三不做**：① 自训 0.5B 从零 pretrain（预算杀手 + 实验混杂 + 无仓库背书）；② 198M-MoE 路线（路由/aux_loss 解释负担大）；③ 全套 DAPO/PRM/MCTS/sglang rollout（0.5B/64M 上测不出差异，时间不够）。

---

## 2. 仓库现状（coding 子 agent 审计 + reviewer 复核）

**训练栈**（trainer/ 目录，全部从零实现、无 TRL 依赖）：

| 文件 | 实现内容 | 缺口/注意 |
|---|---|---|
| train_pretrain.py / train_full_sft.py / train_lora.py | 预训练 / 全参 SFT / LoRA | LoRA 仅对方形 Linear 注入（教学级） |
| train_dpo.py | vanilla DPO（ref 冻结） | 无 SimPO/KTO 变体；离线偏好，跑起来便宜 |
| train_ppo.py | 真 PPO：critic + GAE(γ=1, λ=0.95) + value clip + KL 惩罚 | 双模型显存翻倍 |
| train_grpo.py | **GRPO/CISPO**：格式启发式奖励 + 1.8B LLM judge + token 级 KL + 组内归一 advantage(group=6) | **没有数学答案 verifier** → 这是你的主战场 |
| train_agent.py | 多轮工具调用 RL（模拟 env，GT 校验仅在 tool_call 分支生效） | 依赖 1.8B RM 无条件加载；时间成本高 |
| rollout_engine.py | torch 同步 / sglang HTTP 双后端 | sglang 权重滞后 save_interval 步（隐含 off-policy） |
| trainer_utils.py | init_model / 断点续训 / LMForRewardModel | 路径约定与 cwd 强相关 |

**关键事实**（动手前必读）：
- 仓库默认配置 minimind-3 = **64M**（h768/L8），另有 198M-A64M MoE；**没有 0.5B 配置与公开权重**。0.5B 需要自己从零 pretrain（≈8-10h/epoch × 2+，24GB 单卡）。
- 本地无训练好的权重、无数据；官方 ModelScope `gongjy/minimind-3-pytorch` 可下 pretrain_768 / full_sft_768 / full_sft_zero_768 / **grpo_768** / ppo_actor_768 / agent_768（及 _moe 版）。**无 dpo 权重**（DPO 只能自训）。
- RL 训练无条件加载 **InternLM2-1.8B-reward**（放仓库同级目录）；tokenizer 文件（tokenizer.json + tokenizer_config.json）已在 `model/` 内，但 transformers 加载需注册/复用兼容项（README L1712 附近有说明）。
- 数据来自 ModelScope `gongjy/minimind_dataset`：pretrain_t2t_mini(1.2GB)、sft_t2t_mini(1.6GB)、rlaif(24MB)、agent_rl(86MB)、**agent_rl_math(18MB，仓库里唯一带 gt 的数学数据)**、dpo(53MB)。无 GSM8K 中文数据。
- 算力锚点（README，单卡 3090 实测 64M）：pretrain≈1.2h/epoch、SFT≈1.1h、RLAIF≈1.1h；0.5B ≈ ×7~8。GRPO 瓶颈在 rollout（B×6 采样 × 长序列 + 每样本一次 1.8B RM 前向）。
- 评估：eval_llm.py 只是聊天 CLI；客观榜需自装 lm-eval-harness。64M 的 ceval/cmmlu 分数很低（20 出头），**别在客观榜上找简历数字**。

---

## 3. 2025 后训练全景与选型坐标系（讲解）

### 3.1 主流三件套 → 已是"必会项"
- **SFT**：真正的功夫在数据（配比/清洗/模板/多轮）；冷启动格式数据决定 RL 天花板。
- **离线偏好优化**：DPO 及其变体 SimPO（去 ref + 长度归一）/ KTO / ORPO / IPO（GSPO 统一框架）；Zephyr 式迭代 DPO。通用对齐/创意/安全域仍是主力。
- **在线 RL**：PPO（critic+GAE，显存翻倍）→ **GRPO 已成主流**（组内归一 advantage 去 critic；DeepSeekMath → R1 → Qwen3）；RLOO/ReMax 是低成本替代。注意：GRPO 免费去 critic 是简化叙事——组内归一的优势估计有偏（Dr.GRPO），面试能讲清这点是加分项。

### 3.2 前沿（差异点所在，但选做）
- **RLVR**（规则可验证奖励）：R1/Qwen3/开源复刻（TinyZero、DeepScaleR、oat-zero、DAPO）默认范式。DeepScaleR 证明 1.5B + 强蒸馏起点可行（AIME24 43.1%）。
- **2025 训练技巧**：DAPO 四项（Clip-Higher 治熵塌缩、Dynamic Sampling 过滤全对/全错组、token 级 PG loss、Overlong Reward Shaping）；Dr.GRPO（去偏 + length-adaptive advantage——"长度膨胀"的机制根因，长度与正确率相关时组内归一会放大长答案优势）。
- **迭代/多轮 RL**：R1 两阶段配方；R1-0528 用多轮工具调用 RL；Qwen3 hybrid thinking。
- **多轮 agentic RL**：2025 最强风口（Kimi K2：agentic 数据合成 + RLVR/self-critic）；进 JD，但 1~2 周内排第三优先级。
- **PRM/过程奖励**：成本高、未普及，小模型不推荐。

### 3.3 简历坐标系："主流" vs "差异点"
- 已主流到不算亮点：RLHF/DPO/GRPO 概念与公式、调 TRL/LLaMA-Factory/veRL 跑通 Countdown/GSM8K、认识 DAPO/Dr.GRPO 名词。（北大软微公开课已用 TRL 教 SFT→DPO→GRPO 复现迷你 R1-Zero，T4/16G 即可做。）
- 差异点（按面试杀伤力排序）：**① 从零实现并能推导每个算子**（你仓库最强资产）；② 训练动态的量化理解（熵塌缩/长度膨胀/reward hacking 的曲线证据与修法）；③ rigorous 小消融 + 诚实负结果；④ 在线-离线对照 / agentic 的独立视角；⑤ 评估素养（pass@1 vs greedy、self-consistency、把格式率/长度/正确率解耦）。

### 3.4 小模型做 RL 的真实难点（简历故事素材）
1. **能力门槛**：RL 只强化已有行为、不能教知识。TinyZero 作者明说 Qwen2.5-0.5B base "fails to learn reasoning"；DeepScaleR/oat-zero 的有效起点是 1.5B+ 或数学预训练 base。0.5B 能学会的是**格式/两步算术/指令遵循**——"学不会涌现推理"不是你的失败，是你的**研究结论**。
2. **奖励作弊**：二元稀疏奖励下最快作弊路径=先学格式拿格式分或凑长度。缓解：复合奖励分档（正确 1 / 格式 0.3 / 无答案 -0.3 这类）、verifier 强正则、答案提取只做确定性匹配。
3. **长度膨胀**：R1 系 12K→23K CoT 是能力；小模型里多半是"语无伦次凑长度"。缓解：cap max length、长度惩罚、用"每正确样本 token 数"做效率指标。
4. **熵塌缩**：小模型分布本来就窄，塌缩更快 → lr 压低（1e-6 量级）、Clip-Higher、组内全对/全错无梯度（动态采样过滤）。
5. **KL 与崩溃**：保留 ref 模型 + 合理 β 是底线；β=0 在小模型上容易塌缩成重复短句。冷启动用自己 SFT/Instruct 版而非 base。

---

## 4. 选型裁决（reviewer 最终意见）

### 4.1 为什么是 64M，不是 0.5B
- 0.5B 无官方权重，必须自己 pretrain（8-10h/epoch×2+），且 pretrain 质量会与 RLVR 效果**不可解耦地混杂**——不是预算问题，是实验有效性问题。
- 64M 能力门槛更低，格式过拟合/reward hacking 出现得更早更显著，"诊断负结果"的信号强度不输 0.5B。
- 面试上，"为什么选 64M"本身是可深挖的动机题；"0.5B 复现 R1"反而踩"把仓库作者未背书的 0.5B 效果包装成自己成果"的红线。
- 198M-MoE 直接否决（路由不稳定 + aux_loss 变量）。0.5B 仅在一切顺利且剩 ≥2 整天时做一次冒烟。

### 4.2 主实验：数学 RLVR 诊断式复现
- **改动核心**：重写 `train_grpo.py` 的 `calculate_rewards`（约 150-300 行）为规则 verifier 判定（判对/判错/无答案三态），保留单轮主循环（组内 advantage / KL / rollout 都现成）。不要碰 train_agent.py 的多轮 env 语义。
- **数据**：清洗 agent_rl_math（剥掉工具调用 schema → 纯单轮中文数学问答，保留 gt），不足自造补齐到 ≥1000 训练条；另留 200-400 条 heldout（换模板族防泄漏），全程不参与训练。否决 GSM8K 中文翻译版（转换成本高、64M 出不了公开榜数字）。
- **verifier 工程**：不照抄 veRL"只扫末尾 300 字符"（小模型输出不稳，截断丢答案）；改为"prompt 钉死 `#### 答案` 格式 + SFT 冷启动教会 + 扫全文取最后一个 `####`"。中文数字/单位/分数**在数据侧规避**（答案限阿拉伯数字、去歧义），解析器只做确定性精确匹配（Decimal 规约、去全半角/逗号/空格）。verifier 是确定性代码、与模型规模无关，成败只取决于格式是否被 SFT 教会 → **冷启动 SFT 是地基**。
- **对照与消融**：三臂（verifier-only / RM-only / 混合 α·r_rule+β·r_model——README 本身建议过这个配方，改造有仓库背书）；2 项消融（动态采样过滤、长度惩罚）；四曲线（reward/格式率/正确率/长度）+ 人工抽查 20 条 + heldout pass@1。
- **预期叙事**（必须是数据支撑后的结论，不是预设）：格式率快速收敛而正确率停滞 → 定位"信号被格式吸收 + 基础能力不足"；对照 DAPO 技巧在 64M 上的（可能微弱的）贡献 → 能力门槛证据；与 1.5B 公开日志对比 → 给出"冷启动 SFT → 短推理 RL → 蒸馏"的务实配方。**若全负结果，诚实报告本身就是亮点，但差异点要落在实验设计与呈现质量上。**

### 4.3 支线与点缀
- **B 支线（有余力）**：离线 DPO vs 在线 GRPO 对照——用 RL rollout 存档做错对配对（同数据），量化 DPO 长度偏置与 GRPO 长度膨胀。改动小（train_dpo 现成）。
- **不做**：sglang rollout（权重滞后坑直接绕开，固定 torch 引擎）、PRM/MCTS、分布式 rollout 重写、train_agent 主线、重训 tokenizer、ceval 客观榜。

### 4.4 代码疑点备忘（跑之前知道）
- 默认 loss 是 cispo（无公开可引用出处），**正式消融建议 loss_type=grpo（标准 clip）**，否则被深挖讲不清。
- cwd 约定混乱：训练脚本要 `cd trainer`（../out、../checkpoints 相对 trainer），而 reward_model_path 相对仓库根（../../internlm2-1_8b-reward 实际生效方式需实测）——统一固定 cwd + 绝对路径写进 runbook，冒烟第一件事验证权重真的被加载。
- RM 无条件加载；跑 RL 前确认 from_weight 指向的权重存在。
- RLAIFDataset 用 `conversations[:-1]` 切片构造 prompt、SFT 与 RLAIF 的 eos 约定有差异 → 自造数据建议**新写 ~60 行单轮 Dataset**，别硬复用。
- 两臂 reward 尺度不同（RM 打分 vs 0/0.3/1），曲线对比只看格式率/正确率/长度等无量纲量，不比 reward 绝对值。
- 动态采样过滤后有效组变少，小 batch 统计噪声大，如实记录。

---

## 5. 两周执行地图（含退出标准）

| 天 | 任务 | 退出标准 |
|---|---|---|
| D1-2 | 环境+冒烟：下载 full_sft_768 + InternLM2-1.8B-reward；eval_llm.py 实测对话；rlaif 抽 20-50 条跑官方 train_grpo 5-10 步 | 权重/RM 加载成功（打印参数量）、reward 曲线形态合理、日志落盘 |
| D3-4 | 数据：清洗 agent_rl_math → 单轮问答 + 自造补齐 + heldout 切分；verifier + 单测（合成 200 条） | 抽取命中 ≥99%；人工抽查 50 条无错 gt |
| D5 | 冷启动 SFT 1-2 epoch + greedy heldout 基线 | 模型稳定输出 `####` 格式（人工 20 条全过） |
| D6-8 | RLVR 主 run（SFT init，ref=SFT，lr 试 3e-7/1e-6）+ 三臂对照 | verifier 臂 reward/格式率/正确率上升且抽查内容相关 |
| D9-10 | heldout pass@1 + 解耦表 + 20 条抽查存档 + 长度统计；有余力做 DPO 错对配对 | 四曲线图 + 一张对照表 + 结论段 |
| D11-12 | 2 项消融 + 缓冲重跑 | 消融表（含无效结果） |
| D13-14 | 全量核对重跑、5 公式手推演练、repo diff + 诚实 README 记录、演示脚本 | 面试可讲 30 分钟、命令可复现 |

**兜底降级路径**：D7 不通 → 砍到两臂（verifier vs RM）；D10 未完成 → 保单臂 RLVR 全曲线 + SFT 基线 + 抽查；环境彻底失败 → 官方 grpo 权重冒烟 + 把 train_grpo 奖励机制讲透的代码走读。**永不降级**：诚实口径、可复现命令、图表与数据归档进 repo。地板件 = 四曲线 + 一张对照表 + 抽查 + 诚实结论，保住它面试即够用。

---

## 6. 简历红线与可写版本

**红线（禁写）**：复现 DeepSeek-R1 / 零复现路线；提升 GSM8K X→Y（无此类背书数字）；任何 0.5B 效果数字；照抄 README 曲线/anecdote；把仓库既有 DPO/PPO/agent 写成"我实现"（除非你重写过）；cispo 等讲不清出处的公式。
**可写（≤3 条，全部经得起深挖）**：

- CN：基于开源 MiniMind（64M，官方权重）二次开发：将启发式 + RM 打分替换为规则答案 verifier（GT 判定），自建数据转换与评估脚本，跑通 SFT 冷启动 → GRPO/RLVR → 评估全链。
- CN：设计同数据三奖励源对照与动态采样/长度惩罚消融，记录 reward/格式率/正确率/长度四曲线，诊断小模型 RL 中格式分先于正确率收敛、长度膨胀等动态（含人工抽查与诚实负结果）。
- EN：Built an RLVR (GRPO) pipeline on MiniMind-64M with a rule-based answer verifier replacing heuristic+RM rewards; released custom data/eval scripts in a fork. Devised heldout pass@1/16 with format–length–accuracy decomposition; findings (incl. a negative result on emergent reasoning at 64M) reported honestly.

**面试必须能现场推导的 5 件事**：① GRPO 组内 advantage (r−μ)/σ 与省 critic 的动机、与 PPO 的关系；② KL 惩罚近似形式与 β 的作用（β=0 为何塌缩）；③ DPO 损失与其隐式 reward、长度偏置机理；④ verifier 分档为何比 RM 抗 reward hacking；⑤ DAPO Clip-Higher 与全对/全错组过滤为何能缓解熵塌缩。

---

## 7. 面试高频深挖题库（18 题，要点自查）

1. 推导 DPO 损失；为什么免 RL？→ RLHF KL 约束闭式解：最优策略 ∝ π_ref·exp(r/β)，反解 r 代入 Bradley-Terry。
2. DPO 假设与局限？→ BT + 策略靠近 ref；隐式奖励只在数据覆盖域可信；长度偏置。
3. SimPO 为何去 ref？如何抑制长度偏置？→ 平均 log 概率当奖励；长度归一。
4. KTO/ORPO/IPO 区别？→ 单样本二值效用 / SFT loss 内加 odds-ratio（免参考免配对）/ 防过拟合正则（GSPO 特例）。
5. PPO vs GRPO 逐点差异？→ critic/GAE vs 组内均值归一；KL 注入位置；显存吞吐；GRPO 优势估计有偏（Dr.GRPO）。
6. GRPO 组内归一为何导致长度膨胀？→ 长度与正确率正相关时，长答案 advantage 系统性偏高；解法=length-adaptive advantage / 长度惩罚 / cap。
7. group size 影响？全对/全错组怎么办？→ 方差/显存/吞吐权衡；全对全错无梯度 → 动态采样过滤。
8. 熵塌缩是什么、怎么修？→ 后期分布变尖探索消失 → Clip-Higher / 降 lr / 提温度。
9. token 级 vs 序列级 PG loss？→ 长 CoT 下序列级归一稀释长序列梯度（DAPO 动机）。
10. RLVR 奖励怎么设计不易 hack？→ 复合分档、提取鲁棒（强正则/确定性）、无答案负分、verifier 先自测。
11. 两个真实 reward hacking 例子与防御？→ 只写格式不推理（复合奖励+看正确率曲线）；凑长度（长度惩罚+抽查）；防御=KL+奖励审计+人工看样本。
12. 过程奖励 vs 结果奖励？→ 稀疏但抗 hack；PRM 标成本高未主流。
13. 为什么在线 RL 常赢离线 DPO？→ 策略分布内采样无漂移（VPO 视角）；离线可用 SimPO/迭代 DPO 接近——用自己的对照实验说话。
14. 迭代 DPO / R1 两阶段为何有效？→ 每次用当前策略刷新数据缩小 gap；冷启动 SFT 教格式 → RL 探索 → 拒绝采样 SFT 巩固。
15. 在线 DPO（exPO 类）思路？→ 固定高质量正样本、在线生成负样本，近似策略内偏好。
16. KL 在 PPO/GRPO 的两种放法？→ 奖励项扣 vs loss 内正则；小模型 β=0 塌缩成重复。
17. 0.5B 上 RL 翻车的第一诊断顺序？→ 先看中间 checkpoint 的答案正确率（别看 reward）→ 格式率/长度/熵三曲线 → 抽 20 条人工判。
18. 评估陷阱？→ pass@1 固定采样数/温度并报方差；防泄漏（GSM8K 在预训练语料里）；self-consistency 与长度效率同报；别用 reward 当 eval。

---

## 8. 主要参考来源（分级）

**A 强证据**：
- DeepSeek-R1: arxiv.org/abs/2501.12948（RLVR/冷启动/蒸馏配方）
- GRPO/DeepSeekMath: arxiv.org/abs/2402.03300
- TinyZero: github.com/Jiayi-Pan/TinyZero（0.5B base 失败、单卡 ≤1.5B 可行性）
- DAPO: dapo-sia.github.io（Clip-Higher / Dynamic Sampling / token 级 loss / Overlong Shaping）
- sail-sg understand-r1-zero: github.com/sail-sg/understand-r1-zero；arxiv.org/abs/2503.20783（GRPO 有偏 / Dr.GRPO / 模板破坏 / "aha"存疑）
- DeepScaleR: agentica-project.com（1.5B 从 R1-Distill 再 RL，AIME24 43.1%）
- Kimi K2: arxiv.org/abs/2507.20534（agentic 数据合成、RLVR+self-critic、难度过滤）
- DPO: arxiv.org/abs/2305.18290；SimPO: arxiv.org/abs/2405.14734；RLOO: arxiv.org/abs/2402.14740（+ huggingface.co/blog/putting_rl_back_in_rlhf_with_rloo）；VPO: arxiv.org/abs/2405.19320；Zephyr: arxiv.org/abs/2310.16944

**B 中证据**：
- Mini-R1 / Philipp Schmid 教程: philschmid.de/mini-deepseek-r1
- 北大高志军《LLM 后训练实践》公开课: posttrain.gaozhijun.me（TRL 教学路线——"会跑 GRPO"课程化证据）
- DeepSeek-R1-0528 官方模型卡: huggingface.co/deepseek-ai/DeepSeek-R1-0528（经 SemiAnalysis / d.run 转述核对）

**C 弱证据（仅线索）**：个别 CSDN 个人日志（0.6B GRPO 62.5% GSM8K 等）细节完整但不可复现，勿当简历背书。

---

*文档由 main agent 综合三个子 agent（research/coding/reviewer）产出，2025-09。*
