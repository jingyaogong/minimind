# MiniMind 后训练指南 v2 草稿 · 交叉评审报告（reviewer 最终裁决）

- 角色：reviewer 子 agent（交叉评审与最终裁决）
- 审校对象：`docs/post-training-guide-v2.md`（v2 草稿）
- 基线：`docs/post-training-guide.md`（v1）；上游交付：`planning/research_delta.md`、`planning/implementation_spec.md`
- 审校日：2026-09（与 v2 校验日一致）

---

## §0 裁决总览

**最终意见：有条件接受（需修 6 项 P1 后定稿；无 P0）。**

一句话理由：v2 的核心事实（代码行号锚点、技术论断、来源归属、caveat 保留）经 reviewer 亲自 read 与 4 次 arXiv 网页复核全部成立、与 v1/research/coding 三份底稿自洽，无必须打回的错误；但存在 1 处 v2 内部机制表述自相矛盾、1 处三臂"混合臂"实验定义缺口、若干引用限定与简历措辞需在定稿前修正，均为 P1，不影响主线架构采纳。

---

## §1 P0 清单

**无 P0。** 任务清单指定的 4 组代码锚点（train_grpo / lm_dataset / tokenizer / trainer_utils）全部实测符合 v2 与 coding 报告的描述；web 抽查的 4 篇最承重 arXiv 来源全部真实存在且数字、日期、主张与 v2 转述一致；未发现代码谎报、数字夸大、来源错归属或整段丢失 caveat 的情形。

---

## §2 P1 清单（建议修改，按重要性排序）

**P1-1｜v2 §3.2 第 2 条与 §1 代码修正 2 内部矛盾，占位 assistant 机制转述不实**
- 位置：`post-training-guide-v2.md` §3.2 关键新事实 2（L78）。
- 问题：v2 写"官方 rlaif 数据 assistant 是占位（'无'），RM 的 prompt 正则解析（train_grpo L50-52）会把它当历史"。但实测 `RLAIFDataset.create_chat_prompt`（lm_dataset.py L211-216）对 `conversations[:-1]`（L212）切片后才 `apply_chat_template`，末条占位 assistant 在官方路径**永远进不了 prompt**，L50-52 正则也就见不到"无"。§1 修正 2 与本条自相矛盾。真正风险（impl_spec §1.6g⑧ 的原始表述更准）是"自造数据沿用官方 schema 的占位尾条 + 新 `MathRLDataset` 不切片 → 占位进入 prompt 并被 RM 正则当历史"这一**假设组合**。
- 证据：lm_dataset.py L212 `conversations[:-1]`；train_grpo.py L50-52 正则只作用在已渲染 prompt 上。
- 建议改法：按 impl_spec 口径改写为："官方路径的占位尾条会被 RLAIFDataset 的 `[:-1]` 切片丢弃，无污染；但自造 RLVR 数据若沿用官方 schema（user+占位 assistant）而新 Dataset 不切片，占位会被 RM 正则（L50-52）当历史——因此自造数据一律不放占位 assistant。"（结论不变，机制说对。）

**P1-2｜三臂"混合臂"（α·r_rule+β·r_rm）成分与尺度未定义，实验语义不清**
- 位置：v2 §4.3（L109）、§3.1；源头 impl_spec §2.d d4。
- 问题：① coding 规格把 `rewards_rm` 定义为"原 L44-66 原样搬入（含启发式格式分）"，若 mix 臂直接用该分量，混合臂=verifier+**启发式**+RM 三成分，违背 v1/v2"verifier 替换启发式+RM、RM-only 作对照"的设计意图——对照臂含启发式是刻意的官方 parity，但**混合臂含启发式不是**；② 尺度：r_rule∈{-0.5,0,1}（或 tiered），r_rm∈[-3,3]（trainer_utils L177 clip），原样相加时 RM 分量方差大 3-6 倍以上，混合臂会退化≈RM 臂，rule 信号被淹没，α/β=1.0 的默认值无意义。
- 证据：trainer_utils.py L177 `max(min(score,3.0),-3.0)`；train_grpo.py L54-66 启发式分；impl_spec §2.d d4 伪代码。
- 建议改法：在 v2 §4.3/§3.1 与实验报告口径中写明：mix 臂的模型分量用**纯 RM score**（只取 L62 项，去掉启发式），两分量先各自做组内/批内标准化（如 (r−μ)/σ）再按 α/β 加权，α/β 作为显式报告的超参；或退一步把 mix 臂明确定义为"verifier+官方(启发式+RM)"三成分并说明动机。二选一，不能留默认 1.0 裸加。

**P1-3｜135M 外部对照引用丢 caveat，"规律"表述偏强**
- 位置：v2 §1 表 #5（L29）、§4.2 预期管理（L105）。
- 问题：摘要页原文为 "1.82% to 1.59% **(192-token completions)**…These **single-run** results identify a concrete failure mode **rather than establishing a general lower bound**"。v2 引用"1.82%→1.59%"省略了 192-token 限定，并把"不是失败而是该量级**规律**"当作结论——超出作者自己声明的强度。
- 证据：arXiv:2606.22189 摘要页（reviewer 已 fetch 复核，2026-06-20 提交）。
- 建议改法：两处引用各加一个限定括注"（单次运行、192-token 完成度；作者声明不构成一般下界）"，并把"该量级规律"改为"该量级的同构参照/先例"。

**P1-4｜简历 EN 主条 "against a 135M external baseline" 可比性措辞有深挖风险**
- 位置：v2 §5 EN 主条（L121）。
- 问题：135M 数字是 GSM8K + 自建 harness 的公开审计；64M 主实验是自造中文合成题 + 自建 heldout——**数据集与评测口径不同**，简历写"against a 135M external baseline"暗示直接可比，面试深挖"你的数字凭什么跟它比"时会被动。
- 证据：2606.22189 摘要（GSM8K EM）；v2 §4.1 heldout 为自建 math_heldout。
- 建议改法：改为 "contextualized with published 135M-scale RLVR findings"（定性参照）或明确限定"同构诊断语境，非同数据集比较"；CN 次条相应加"外部参照"而非"对照"字样。

**P1-5｜红线缺口：ACR / pass@k 反转 / 难度分层等是方法学引用，须显式标"复现应用"而非"提出"**
- 位置：v2 §5 红线段（L116）与可写文案（L119-121）。
- 问题：审查维度 C4 提示的风险确实存在：v2 新增红线只封了"0.5B/GSM8K 数字写首创"，没有封"诊断指标写原创"。ACR 是 AVSPO（2605.21125）术语，pass@k 大 k 反转是 NeurIPS'25 Oral（2504.13837）方法学，难度×提升矩阵是低数据 RLVR（2604.18381）叙事——简历逐字使用这些名词而不带出处，被问"这个指标哪来的"即踩原创/复现边界。
- 建议改法：红线区补一句："ACR、pass@k 反转、难度×提升等诊断指标为**方法学引用**（NeurIPS'25 Oral / AVSPO / 低数据 RLVR），简历与口头一律写'复现并应用于 64M'，不写'提出/首创'。"CN/EN 文案可保留现有措辞（本就偏应用），但面试口径要能主动交代出处。

**P1-6｜"无梯度"与 ACR 归因表述需精确化（KL 项残留 + 全对/全错拆分）**
- 位置：v2 §2.1 ACR 行（L51）、§4.2、§6 Q3（L131）。
- 问题：① 代码 L137/L142 中 A=0 时 advantage 项为 0，但 `−β·per_token_kl` 项仍产生把策略拉向 ref 的梯度——"全对/全错组无梯度"严格说只指 advantage 驱动的 policy-gradient 项；动态采样过滤（d6 整组掩码）才会连 KL 项一并去掉。② v2 归因框架"ACR 高=信号问题、ACR 低仍停滞=容量问题"在小模型早期可能误诊：模型能力不足时**全错组**天然占多数（全错也 std=0），此时 ACR 高其实是容量/难度问题而非奖励分辨率问题。③ 数值面（审查维度 C2）确认：组内 reward 全同 → r−μ 精确为 0，分母 std+1e-4=1e-4 不爆，advantage 精确为 0，此点 v2 成立。
- 建议改法：ACR 拆两条曲线（全对组占比 / 全错组占比）各一行统计，归因句改为"全对主导→已收敛；全错主导→容量/难度；对错混合仍高→奖励分辨率不足"；Q3 答案补半句"A=0 组的 β·KL 项仍向 ref 回拉，动态采样过滤才是整组去除"。

**P1-7｜顶层文档缺执行关键参数收编，v1 的 loss_type 建议可能丢失**
- 位置：v2 §3.3 D5/D6 与 §3.1（整体）；对照 v1 §4.4。
- 问题：v1 §4.4 明确"正式消融建议 loss_type=grpo（标准 clip），否则 cispo 无公开出处被深挖讲不清"；v2 §3.1/§3.3 未转载这一行（只在 impl_spec runbook 的逐条命令里出现）。同时 §3.3 D5 验收写"四曲线形态可解释"，而 §4.1/简历口径已是"五曲线"（ACR 落地只要 ~30 行，D5 即应有）——只读 v2 的实现者会丢参数、口径不一。
- 证据：train_grpo.py L228 `default="cispo"`；v2 §3.3 D5 vs §4.1。
- 建议改法：§3.1 或 D5 处补一行："所有正式 run 显式 `--loss_type grpo`（cispo 无公开出处，v1 §4.4）；五曲线自 rule 臂第一步起即记录（ACR 行不计入 30 行成本），里程碑 D5 措辞统一为五曲线。"

---

## §3 P2 可选清单

1. **§3.3 里程碑 D1-D7 与 §7 天计划 D1-14 双编号并存**：内容经核对无冲突（§7 每行都显式映射到里程碑），但 §3.3 写"替代 v1 的 D1-D14 表述"而 §7 又保留 D1-D14 天计划，且两套都用 D 前缀、区间重叠（D5 在两表中含义不同），易误读。建议里程碑改名 M1-M7，或在 §3.3 表头加"与 §7 天计划的映射见下"一句。
2. **§8 人名笔误**："高志君"应为 v1/research_delta 一致的"高志军"（域名 posttrain.gaozhijun.me 拼音佐证）；请按作者主页核后统一。
3. **§5 "v1 保留 + 2 条新增"计数**：括号内实际标了"新增 0.5B 首创""新增口径 cispo"，另有未标的 PRM 断言（v1 红线本无此项）——"2 条"与条目数不严格对应，统一措辞（如把 PRM 条也标注"新增"）。
4. **回归护栏依赖缺失**：§3.1 回归护栏（`--reward_mode rm` 官方 rlaif 跑 5 步 diff≈0）需要 rlaif.jsonl（24MB），§7 D1-2 下载清单只列 full_sft_768 与 RM——补一行下载 rlaif.jsonl（RM 臂本身用 math 数据，不需要它）。
5. **数据下载无降级注**：真实数据依赖 ModelScope 可达性且 agent_rl_math schema 未知（impl_spec §4 开放问题 1）；建议在 §7 D3-4 加一句"网络/数据不可达时以 `--synthetic-only` 全合成数据完成管线，报告如实标注数据来源"，防 D2 被阻塞无出路。
6. **pass@32 采样成本**：400 条 heldout × pass@32 × 5 个权重 ≈ 6 万次生成，D9-10 两天内偏紧；建议 pass@32 用 heldout 子集（100-200 条）或与 pass@16 共享采样并写明口径。
7. **0.5B 冒烟归因**：research §3.4-3 要求"若做 0.5B 冒烟，须与 64M 同一套数据/verifier 才能归因容量 vs 冷启动"，v2 未带回；可在 §4.2 预期管理加一句（仅作冒烟路径的前置约束，不新增工作量）。
8. **"报告声称 5× 样本效率"行文晦涩**（§0 TL;DR）：改为"论文声称（2604.18381）的 ≈5× 样本效率"，避免读者误读为自报数字。

---

## §4 代码抽查记录（reviewer 亲自 read 核验，9 处全部通过）

| # | 锚点 | 核验结果 |
|---|---|---|
| 1 | `trainer/train_grpo.py` L37-68 `calculate_rewards` | **符合**：实为 32 行（37-68 含函数边界）；双层循环启发式分（L44-66，长度/think/rep_penalty）+ RM `get_score`（L62）相加（L66）不 clip；调用点 L95；返回 `torch.zeros(len(responses))` 于 `args.device`（L38） |
| 2 | 组内归一 L121-124 与 KL L132-133 | **符合**：`rewards.view(-1, G)`→`mean_r/std_r`（L122-123，`unbiased=False`）→`(r−μ)/(σ+1e-4)`（L124）；`kl_div` L132、`per_token_kl=exp(kl)−kl−1` L133；cispo 分支 L135-137（clip 上限 + detach），grpo 对称 clip 分支 L139-142（min(ratio,clip_ratio)·A − β·KL）存在 |
| 3 | L226/L228/L279 | **符合**：`num_generations=6`（L226）；`loss_type default="cispo"`、choices grpo/cispo（L228）；RM 无条件加载（L279 `LMForRewardModel(...)` 无任何条件） |
| 4 | `dataset/lm_dataset.py` RLAIFDataset | **符合**：L202-203 bos/eos 锚均**无 `\n`**（`f'{bos}assistant'`、`f'{eos}'`）；L212 `conversations[:-1]` 切片；L217-224 `__getitem__` 只返回 `{'prompt', 'answer': ""}`（answer 恒空、无 gt）；L202-203 两锚确为死代码（类内无消费方） |
| 5 | SFTDataset L65-66 | **符合**：bos 锚=`f'{bos}assistant\n'`、eos 锚=`f'{eos}\n'`，**含 `\n`**，与 RLAIF 常量差异属实（label 机制 L88-104 依赖该锚） |
| 6 | `model/tokenizer_config.json` chat_template（L333 一整行） | **符合**：assistant 分支一律输出 `<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{content}<|im_end|>\n`；content 含 `</think>` 时自动切分 reasoning/content；`add_generation_prompt` 末尾追加 `<|im_start|>assistant\n` +（`open_thinking is defined and true` ? `<think>\n` : `<think>\n\n</think>\n\n`） |
| 7 | `model/tokenizer.json` think token | **符合**：`<think>`=25、`</think>`=26，二者 `special:false`（added_tokens 仅 0-20 为 special）；decode 后文本保留 → 文本层解析成立 |
| 8 | `trainer/trainer_utils.py` L160-177 LMForRewardModel | **符合**：`AutoTokenizer/AutoModel.from_pretrained(path, trust_remote_code=True)`（本地目录）；`.to(device).eval()`；`get_score` 拼接"以上是对话历史…"后调远端 `.get_score`，返回 `max(min(score,3.0),-3.0)` clip 于 [-3,3]（L177） |
| 9 | `trainer/trainer_utils.py` L63-116 lm_checkpoint | **符合**：resume 文件 `{weight}_{hidden}[_moe]_resume.pth`（L67）；加载分支 L107-116 含 world_size 换算（L110-114）；保存分支 `model is not None`（L69）写 ckp + resume 双文件 |

结论：v2 与 impl_spec 中所有可静态核验的代码描述与仓库现状一致，未发现 v1/coding 报告向 v2 继承任何代码事实错误。

---

## §5 技术论断抽查记录（含 web 复核）

**已做 web 复核（fetch arXiv 摘要页，4 篇全部命中且与 v2 转述一致）**：

| 论断 | 复核结果 |
|---|---|
| ① 0.5B GSM8K outcome 53.75 / process 63.73（arXiv:2607.02869） | **一致**：论文真实存在（2026-07-03 提交，8 页）；摘要确认 process-only 63.73% vs outcome-only 53.75%（"nearly 10pp"）、hybrid λ∈{0.9,0.5,0.1}、λ=0.1 劣于纯 outcome、Qwen2.5-0.5B + GRPO；v2"未评审、引用带声称"的 caveat 保留得当 |
| ② 135M EM 1.82→1.59（arXiv:2606.22189） | **数字一致，caveat 有缺口**：论文真实存在（2026-06-20）；摘要原文含 "(192-token completions)"、"single-run"、"not establishing a general lower bound" 三个限定，v2 只引裸数字并把"该量级规律"当结论 → 见 P1-3 |
| ⑥ OPD 两段优于 joint（arXiv:2609.04108） | **一致**：论文真实存在（2026-09-03，cs.CL，标题"Sequential Beats Joint"）；摘要确认 OPD-then-RL 两段优于 pure OPD/RLVR/全部 joint baseline，"OPD is a better cold start for RL than SFT"；v2 归属（2604.13016=ICML'26 系统研究 + 2609.04108）与 research_delta 一致 |
| ③ NeurIPS 2025 Oral pass@k 反转（arXiv:2504.13837） | **一致**：v5=2025-11-24，Journal ref 确为 "NeurIPS 2025 Oral; ICML 2025 AI4MATH workshop best paper"；摘要确认"small k 优于 base、large k base 更高、能力源于并被 base 封顶、六算法接近且远未榨干 base、蒸馏才扩能"——v2"小 k 涨大 k 反超、蒸馏才扩能"转述准确 |

**未做 web 复核（依据 research_delta，其自称均实际读取/检索）**：④ ACR/AVSPO（2605.21125）、⑤ U-statistic（2603.01162）、⑦ verifier 噪声（2605.25252）、⑧ 难度混合 5×（2604.18381/2606.22317）、⑨ R1 v2 86 页与语言一致性奖励（2501.12948v2）、⑩ TRL `loss_type=dr_grpo`。
- 置信度意见：**高**。被复核的 4 篇（含两篇最易幻觉的 2026-09/2026-07 ID）全部真实且内容逐字吻合，research_delta 未见幻觉迹象；其 ID 格式、日期、摘要引文互相自洽。
- 若 main agent 定稿前有余力，建议对 ⑨（R1 v2 86 页 + "语言一致性奖励"措辞）与 ⑩（TRL grpo_config 的 loss_type 选项列表）各做 1 次 fetch 双确认——这两条是最常见的语义漂移点，且 v2 将"语言一致性奖励"写入题库口径。

**逐条转述核对结论**（①-⑩ 与 research_delta §1/§2 交叉比对）：① ✓（caveat 保留）② ⚠️（丢 192-token/single-run 限定，P1-3）③ ✓ ④ ✓ ⑤ ✓（双视角表述与来源归属正确）⑥ ✓ ⑦ ✓ ⑧ ✓ ⑨ ✓（86 页/语言一致性奖励/aha 词频量化均一致）⑩ ✓。未发现错归属或夸大转述。

---

## §6 对 v2 落地的一句话提醒（给 main agent 定稿）

代码事实与主干结论我全部复核通过、架构可采纳，定稿只需处理 7 项 P1（重点是：mix 臂用"纯 RM 分量+先归一再加权"并写进口径、135M 引用补"单次运行/192-token/非一般下界"、红线补"ACR/pass@k 反转等为方法学引用，只写复现应用"、§3.2 占位 assistant 机制按 impl_spec 1.6g⑧ 原文口径改写消除内部矛盾），P2 里务必顺手修 §8 人名与里程碑双编号，之后即可发布为 v2 终稿。

---

*reviewer 交叉评审报告，2026-09。*
