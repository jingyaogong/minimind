# 后训练方向：1~2 个月旗舰项目选型指南（求职版）

> 画像：投递大厂/中厂算法岗（LLM 后训练/RL 方向）；已有 MiniMind 从零手写后训练全栈；自有 24GB 单卡 + 可少量租云卡；2025-09。
> 由多 agent 协同产出：research ①（生态盘点）+ research ②（出彩策略）→ reviewer（定版裁决）→ main 汇总。
> 配套文档：`docs/post-training-guide.md`（MiniMind 内 1-2 周方案，已封存为兜底，见 §3）。

---

## 1. TL;DR

- **旗舰项目定版**：「长度通道」——在 verl 上用 Qwen2.5-1.5B 全参对 GSM8K 做 RLVR，用受控消融**证明或证伪"GRPO 存在与正确率解耦的系统长度偏置"**，归因来源、验证缓解、诚实报负结果。
- **为什么是这个**：2025 年"跑通 GRPO/RLVR"已课程化贬值；面试官默认防备"第 N 个跑通版 R1"。唯一"别人照你 README 抄不走"的产出 = **带受控对照、可证伪、肯认负结果的机制量化**。长度偏置（Dr.GRPO 的核心命题）是可以证伪的锐利假说，且与 JD 高频词（reward hacking/长度膨胀/评估严谨性）完美对齐。
- **框架**：verl 主力（工业背书 + 单卡 ≤1.5B 有同栈先例），W2 末为环境决策点，不过即切 TRL。
- **MiniMind 方案**：封存为兜底，不并轨执行（题材重叠、0.5B 信号不可读）；简历呈现"从零实现（过去）→机制诊断（现在）"递进。

---

## 2. 为什么是这个项目（裁决逻辑）

### 2.1 岗位信号（真实 JD 证据）
大厂对齐岗 JD 的能力词高度收敛：PPO/GRPO/DPO、reward modeling、credit assignment、RLVR、熵坍缩/奖励饱和；verl/TRL/vLLM/SGLang、"训推不一致"定位；**判题器/去污染/失败机制配比/难度分层**；"reward 正确性锚定在模型之外"。面试官深挖三连问：数据哪来/超参为什么/什么评估与基线。**烂大街判据：换个人照你 README 能产出一模一样结果 = 烂大街。**

### 2.2 三个候选原型的取舍
| 原型 | 取向 | 裁决 |
|---|---|---|
| P1 长度偏置显微镜（research②主推） | 算法深度 | **定为科学主轴**（可证伪假说 + 缓解消融 + 诚实负结果） |
| RLVR 失败模式诊断 + DAPO trick 消融（research①主推） | 算法深度 | 不取为主轴：复现已知现象/审计已知配方 = 烂大街形态，TinyZero 已弃维护；其 verl/监控/云验证资产全部**吸收进旗舰** |
| P2 自研栈 × verl 数值对齐 + PR | 框架工程 | 不当旗舰（吃时间、价值依赖公司口味）；pr 默认不做，见 §5 |
| P3 可验证数据工厂 | 数据基建 | 不当旗舰；**判题器自测 + 去污染技术整体内嵌**为本项目数据纪律 |

### 2.3 MiniMind 1-2 周方案的去留
**封存，不并轨执行**。理由：题材重叠（verifier+诊断），先跑 0.5B 再跑 1.5B 是同一件事两遍；0.5B 学不会推理（TinyZero 实证），极易产出不可解读的全零结果。算法实现能力已是简历存量，无需新 run 背书。**唯一重启条件**：W2 末 verl 与 TRL 都无法产出首轮 run。

---

## 3. 旗舰项目实验设计（定版）

**名称**：长度通道——GRPO 在可验证 RLVR 中输出长度偏置的量化、归因与缓解

**配置**：verl（FSDP + vLLM colocate）· Qwen2.5-1.5B 全参 · GSM8K 规则判题（子集 1.5k-3k prompts）· G=8 · max_seq 768-1024 · 单臂预算 ≤1 GPU·日（先 500 prompts × 2 epoch 冒烟标定吞吐）

**数据纪律（地基，永不砍）**：
- 判题器启用门槛：与人工标注一致率 ≥95%；
- eval 用 GSM8K test + MATH-500，与 RL 集严格零交叠；
- 先测基座/SFT 零样本"记忆基线"隔离记忆效应；
- **预注册**全部假设与指标进 README 再开跑。

**必做 4 臂对照**（A/B 各 ≥2 seed，C/D ≥1-2 seed）：
- A = 标准 GRPO（带 ref-KL）
- B = 关 ref-KL（RLVR 常见设置）
- C = B + reward 层 overlong shaping（映射 DAPO Overlong Shaping）
- D = B + 显式长度惩罚（区分"无偏 shaping"与"惩罚"机制代价；天然负结果候选臂）

**每臂必采指标**：长度分层 pass@1、长度分布、熵、reward/advantage 对长度的回归斜率、错误样本分类。判定标准写死：长度涨而分层 pass@1 平 → 偏置实锤；斜率为零 → 证伪，转归因叙事。

**加分模块（按砍序收缩）**：① verifier 设计对照（格式约束有无 → 偏置是否随 verifier 变 = "reward 锚定在模型之外"的直接证据）；② 中文博客 + 可复现配方卡；③ 自研栈 × verl 数值对齐抽测或 advantage 无偏化（需动 verl 源码，时间富余才做）。

**规模外推**：3B 复核（本地试跑，不成就上云或砍）；7B 云上"基线 vs 最佳缓解"单对照一次——只回答规模敏感性，不做完整矩阵。

---

## 4. 框架与算力裁决

| 项 | 裁决 | 理由 |
|---|---|---|
| 框架 | **verl 主力** | JD 高频、工业背书、TinyZero 同栈"单卡 ≤1.5B"先例 |
| 降级链 | verl 不过 → TRL GRPOTrainer（设计/数据层不动，math-verify 内置省 verifier 工程）→ 自研栈 0.5B 收窄版 | 决策点 W2 末 |
| 1.5B@24GB | 现实但有约束：bf16 ≈3GB，显存大头在 rollout；OOM 按"降 G→4 → 限长 512 → offload"顺序降档 | TinyZero README 明示单卡 ≤1.5B |
| 3B | 单卡无官方实证（文献口径 2 卡）；LoRA 仅作本地失败次选 | EasyR1 估算表：3B LoRA = 1×24GB |
| 7B | **一律云** | — |
| ms-swift | 不作训练主框架 | 只作贡献入口/免费算力备选 |

**云卡边界**：单卡足够 = 1.5B 全矩阵与全部日常迭代；必须云上 = ① 7B 规模敏感性单对照（1×A100/H100 级，2 runs）② 本地失败的 3B 复核；最不值云上 = 7B 完整消融矩阵与"撞运气"调参。**云 gate 写死：本地 1.5B 结论成形前不下单。**

---

## 5. 8 周日历

| 周 | 主题 | 关键任务 | 退出标准 |
|---|---|---|---|
| W1 | 上膛+数据基建 | verl 环境+冒烟；判题器自测≥95%；数据切分+记忆基线；监控面板 v0 | verl 跑出 1.5B 首轮曲线（哪怕平坦） |
| W2 | 基线+锁定假说 | A/B 臂开跑；预注册 README；吞吐标定矩阵规模 | A/B 曲线+长度/熵/斜率图；**环境决策点（切 TRL 最后期限）** |
| W3 | 机制实验 | C/D 臂；3B 本地可行性试跑 | 4 臂数据齐，长度-正确率解耦结论初版 |
| W4 | 归因+补洞 | 错误样本分类；关键臂补 seed；verifier 设计对照 | 一页机理叙事+图集 draft |
| W5 | 规模决策+写作启动 | 云下单（gate：本地结论成形）或砍并记录理由；博客骨架 | 云 run 启动或明确放弃 |
| W6 | **显式缓冲** | 补洞/补 seed/云 run 执行与解读 | 云结论（支持/不支持外推） |
| W7 | 收尾+成稿 | 补最后对照；博客成稿（中文、含复现卡）；数字定稿 | 博客可发布；数字全部带口径 |
| W8 | 简历化+演练 | bullets 定稿；推导模拟面试 | 简历+模拟闭环 |

**每周超支砍序**：加分③ → ② → 3B → 7B → 误差深度分析。**永不砍**：判题器自测、受控对照、监控面板。

**ms-swift gfi PR 裁决**：默认不做。触发条件 = W6 末主线零延迟且富余 ≥3 天；只选文档/示例级 gfi；被拒或一轮大改即弃；**未 merge 不上简历**。

---

## 6. 简历红线与最终 bullets

**红线**：不出现"复现/超越 R1、DAPO、aha moment"叙事；未跑完/未 merge/未验证一律不写；每个数字必带口径（数据集/采样/seed/卡型）；至少保留一条诚实负结果；不引用未核验数字（如 gpt-oss 训练卡时）。每条 bullet 通过"面试官 3 分钟问到最深处不崩"自检。

**Bullets（CN ≤3，数字以诚实口径填）**：
1. 在 verl（FSDP+vLLM colocate）上以 Qwen2.5-1.5B 全参对 GSM8K 规则判题任务做 RLVR，用 [4 臂×seed] 受控消融量化 GRPO 长度偏置：关 ref-KL 后输出长度 +[X]% 而分层 pass@1 仅 [Y]%，证实长度是与正确率解耦的隐奖励通道。
2. 验证两类缓解：overlong shaping 保持/提升 pass@1（[Z]%）同时抑制膨胀；显式长度惩罚伤害探索（负结果如实报告），定位偏置主源为 [ref-KL 缺失/相对 advantage]，结论经 3B 复核及 7B 单点验证（如成行）。
3. 自建评测纪律：判题器-人工一致率 ≥95%、eval/RL 零交叠+记忆基线隔离、长度分层 pass@1 与熵/斜率监控；全量配方与失败案例开源为技术博客。
（EN 版对译 ≤3 条，保留 verl/GRPO/RLVR/rule-verifier/pass@1 检索词。）

---

## 7. 面试必推 5 件事

1. **GRPO 全链路推导**：REINFORCE→组内 baseline advantage、clip、ref-KL 的梯度角色，与 PPO 异同（MiniMind 推导直接复用；新增：关 ref-KL 后哪一项消失、为何只剩长度通道）。
2. **长度偏置机制链**：为何 verifiable 任务中"长度-正确率"正相关；组内相对 advantage 如何把相关性转成持续拉长梯度；从公式指出 reward shaping / advantage 层 / KL 项三个干预位及各自代价。
3. **reward hacking 与"reward 涨 eval 不涨"诊断法**：监控面板（长度/熵/斜率/错误分布）如何泛化为通用诊断；区分真学习、hack 与记忆。
4. **verifier/判题器设计**：规则 vs 学习式、答案归一化与自测协议、outcome vs process、"reward 锚定在模型之外"的实现口径；讲清 math-verify/distilabel 生态。
5. **verl 工程机制与训推不一致**：rollout/train 组织、vLLM 与 FSDP 交互、colocate 显存账、采样参数与推理时不一致的定位法；单卡受限下 G/序列长/批量的缩放取舍。

**学习路径（并行）**：手推 PPO/GRPO/DPO（numpy 级伪码）→ 论文清单：DeepSeekMath(GRPO) → R1/R1-Zero(RLVR+格式奖励) → DAPO(四技巧) → Dr.GRPO(长度偏置) → Kimi K2(agentic 数据) → Qwen3(hybrid thinking)；系统读物：The RLHF Book（rlhfbook.com，含配套代码与中译）、高志军《LLM 后训练实践》、HF "N+ Implementation Details of RLHF with PPO"。

---

## 8. 风险表与兜底

| 风险 | 触发点 | 降级路径 |
|---|---|---|
| verl 环境坑 | W1-W2 冒烟 run 出不来 | 切 TRL（设计不动）→ 自研栈 0.5B 收窄版；决策点 W2 末 |
| 云卡/预算 | 租不到或超支 | 砍 7B 并注明"未做规模外推"；3B 本地减配或 LoRA；不因云 run 推迟博客与简历 |
| 效果不显著/全负结果 | 缓解无效或基线无偏置 | 预注册使负结果成为结论本身；量化+归因+证伪即产出；图集与误差分析保底 |
| 时间超支 | 任一 gate 未过 | 按砍序收缩；缓冲周 W6 兜底；博客可降级为技术笔记但必须公开 |

**最不该做的一件事**：在标准 GRPO+ref-KL 受控基线与监控面板就位之前，直接跑任何"改进配方/更大模型"，最后拿一条漂亮 reward 曲线当成果——无对照无数值即死。次不该：与新旗舰并轨执行 MiniMind 方案。

---

## 9. 参考来源（选型证据）

**框架/复现项目（A=官方一手）**：verl（verl-project/verl，README/文档；旧版含 24GB L4 PPO notebook）；TinyZero（github.com/Jiayi-Pan/TinyZero：单卡 ≤1.5B、0.5B 学不会、弃维护声明）；oat-zero（github.com/sail-sg/oat-zero）；DAPO（github.com/BytedTsinghua-SIA/DAPO）；Open-R1（github.com/huggingface/open-r1）；Open-Reasoner-Zero（github.com/Open-Reasoner-Zero/Open-Reasoner-Zero：0.5B 单卡脚本）；OpenRLHF（github.com/OpenRLHF/OpenRLHF）；EasyR1（github.com/hiyouga/EasyR1，显存估算表）；TRL（huggingface.co/docs/trl，math-verify 集成于内置奖励函数）；LLaMA-Factory / ms-swift（swift.readthedocs.io GRPO 章节）；Unsloth（官方口径 24GB GRPO，无第三方复测）；gpt-oss（github.com/openai/gpt-oss + arXiv:2508.10925）。
**数据/评估/奖励**：HF math-verify（github.com/huggingface/math-verify）；distilabel（Argilla）；lm-eval / lighteval；RewardBench（AllenAI）；UltraFeedback / Skywork-Reward / RLHFlow（开源 RM）。
**学习资源**：The RLHF Book（rlhfbook.com / arXiv:2504.12501 / 中译 jweihe/RLHF-book-Chinese）；高志军《LLM 后训练实践》（posttrain.gaozhijun.me）；HF 博客 "The N+ Implementation Details of RLHF with PPO"。
**岗位证据**：Shopee 对齐岗实习 JD（川大就业网）；MisterBooo/llm-interview-questions；卡码简历专栏。
**论文**：GRPO（arXiv:2402.03300）；DeepSeek-R1（arXiv:2501.12948）；DAPO（arXiv:2503.14476）；Dr.GRPO（arXiv:2503.20783）；Kimi K2（arXiv:2507.20534）。
**诚实标注的存疑项**：DAPO/gpt-oss 训练集群规模、PRIME 细节、verl 内部 logp 对齐源码级机制、Unsloth 官方数字——均无一手核验，勿写入简历/报告当事实。

---

*文档由 main agent 综合三个子 agent（research×2 / reviewer）产出，2025-09。*
