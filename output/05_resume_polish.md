# 05 · 简历诊断与润色

## 一、逐行诊断（现有简历）

| 现状 | 问题 | 改法 |
|---|---|---|
| 自我评价「统计学与数据科学双背景…全流程实战经验」 | 空词多、和项目不对应；「落地」暗示生产 | 改成能力指向：后训练与评测闭环 / 数据工具 / 统计口径 |
| 技能「熟练 TensorFlow」 | 无任何项目产出支撑 | 降级为「了解」，或删掉给后训练内容腾位置 |
| 核心算法「Transformer, BERT, RoBERTa, BiLSTM, CRF, Seq2Seq, Attention, Beam Search」 | 全是 NLP 传统组件，与「后训练 / RL / reward / 评测」岗位关键词错位 | 拆两行：①模型与微调 ②**后训练与评测**（RLHF/GRPO 概念、reward/verifier、pass@k、配对检验） |
| 项目 1「Accuracy 94.12%，Macro F1 84.16%」 | 无口径：测试集多大？几类？baseline 是什么？ | 补一句口径（如测试集规模、类别数、baseline），或至少能口述 |
| 项目 2「SacreBLEU 35.62」 | 同上，且缺 baseline 对比 | 补 baseline 或改成「相比 X 提升 / 持平」，无对比就写绝对值和测试集口径 |
| 两个项目都是「我实现了什么」 | 缺少约束与失败案例（面试官爱问） | 每个项目加一句「难点 / bad case / 怎么定位」 |
| 项目经历整体 2 个条 | 与 JD 的「大模型后训练」直接相关的项目缺失 | **新增 RLVR 审计项目置顶**（见下） |
| 「全国大学生数学建模竞赛山东赛区二等奖」等 | 保留，量化学术背景 | 保留 |
| 无 GitHub / 无代码链接 | 面试官无法验证 → 削弱可信度 | 面试前整理公开仓库（见 `09_upgrade_plan.md` 半天项） |

## 二、三档 Bullets

### A. 保守档（现有证据完全支撑，不需要额外解释）

```text
数学 RLVR 训练动态审计（MiniMind-64M / GRPO / 单卡 RTX 4090）
- 用确定性脚本生成 2000 SFT / 2000 RLVR / 400 held-out 数学数据，按表达式 key 去重（train 与 held-out 重叠为 0），
  附 sha256 清单与 50 条人工审计抽样。
- 实现三态规则 verifier（correct / incorrect / no_answer）与分档奖励，与训练、评估、数据模板共用同一答案标记，配单元测试（全仓 313 tests passed）。
- 搭建 held-out 评估链路：greedy + pass@1/4/8/16，16 采样/题；对 6 个预注册臂 × 400 题做逐题配对 McNemar 检验与分档统计。
- 完成 8 个 run 的归档（config / 命令 / 日志 / 权重哈希），支持任意数字回原始文件复算。
```

### B. 标准档（真实，但含判断与因果，需按答案卡准备追问）

```text
- 定位到 RM 奖励的内容层 reward hacking：训练 reward 1.684→2.011（+0.327）而同一批 rollout 的 verifier 正确率 0.233→0.025，
  held-out greedy 0.373→0.133（400 题配对 102 题由对变错 / 6 题由错变对，p<0.0001）；错误类别分析显示 100% 格式完好、86.8% 为纯算术错误。
- 纯 GT verifier 的四个对照臂方向一致（greedy 净 +6~+9 / 400 题，p=0.06~0.18），但同命令同种子重跑的噪声地板为 ±2~±6，
  据此把「RL 提升推理能力」降级为方向性证据，并记录单次运行不可复现（200 个记录点仅 35 个一致）。
- 发现长度惩罚消融为 no-op：cap=300 高于实测长度上界（采样 max 228 / 评估 max 280），240 条采样 0 次触发 —— 教训是先测分布再设阈值。
```

### C. 补证据后写（现在别写）

```text
- 将同一套 verifier / 评测协议迁移到 Qwen2.5-1.5B + TRL/verl（需要工业近似层实验，未做）。
- 多 seed（≥3）重复与 pass@32（当前效应量落在噪声地板内，扩样不改结论）。
- 语音 / ASR 相关实验（需要真实动手，见升级计划）。
```

## 三、英文版（投外企或英文岗用）

```text
RLVR Training-Dynamics Audit (MiniMind-64M, GRPO, single RTX 4090)
- Built a deterministic data pipeline (2,000 SFT / 2,000 RLVR / 400 held-out math problems) with expression-key
  deduplication, zero train/held-out overlap, SHA-256 manifests and a 50-row human audit sample.
- Implemented a three-state rule verifier (correct / incorrect / no_answer) with graded rewards, shared by the data,
  training and evaluation paths; covered by unit tests (313 tests passing repo-wide).
- Ran a held-out harness with greedy and pass@1/4/8/16 metrics (16 samples/problem) over 6 pre-registered GRPO arms
  and 1 rerun control, with per-problem paired McNemar tests.
- Found content-level reward hacking for the RM-only reward: training reward rose 1.684 → 2.011 while verifier accuracy on
  the same rollouts dropped 0.233 → 0.025 and held-out greedy fell 0.373 → 0.133 (102 broken / 6 fixed of 400, p<0.0001);
  error analysis showed 100% well-formed outputs and 86.8% pure arithmetic errors.
- Measured the reproducibility floor: an identical-command, identical-seed rerun matched only 35/200 logged steps,
  so the rule-verifier arms' gains (+6~+9 of 400, p=0.06~0.18) are reported as directional evidence, not effect sizes.
```

## 四、要删 / 要降级的表述

| 删掉或降级 | 原因 |
|---|---|
| 「落地」「上线」「主导」类词 | 无生产/主导证据 |
| 「全流程实战经验」 | 空泛；面试官会要求逐项举证 |
| 「熟练 TensorFlow」 | 无产出 |
| 任何 Qwen / 多卡 / 大规模暗示 | 未做 |
| 「提升推理能力」 | 数据不支持（不显著、在噪声地板内） |
