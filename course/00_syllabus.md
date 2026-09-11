# MiniMind LLM 系统学习课程总纲

本课程以 MiniMind 源码为主教材，目标不是记住一堆术语，而是把一条完整 LLM 链路真正跑通、看懂、改得动。

核心学习主线：

```text
jsonl 数据
-> tokenizer / chat template
-> input_ids / labels
-> MiniMindForCausalLM
-> logits / next-token loss
-> optimizer / checkpoint
-> SFT / LoRA / DPO / PPO / GRPO
-> eval / API / WebUI
```

建议节奏：28 节，每节 60-90 分钟。每节尽量配一个最小实验，实验目标是验证源码里的关键变量流转，不追求训练出好模型。

从第 13 课开始，课程增加一条手写复现主线：

```text
原理讲解
-> MiniMind 源码走读
-> 手写核心模块
-> 和原源码做对齐测试
-> 阶段结束时组装成可运行脚本
```

目标不是复制所有工程外围，而是在 `course/impl/` 下手写一个教学版 MiniMind 核心实现。tokenizer、原始数据、分布式训练、wandb/swanlab、断点续训、WebUI/API、模型转换等工程能力优先复用原项目。

## 当前课程进度

已完成：

| 课次 | 文件 | 当前状态 |
|---:|---|---|
| 1 | `01_project_overview.md` | 已写，早期格式，后续可统一成新格式 |
| 2 | `02_environment_and_resources.md` | 已写，早期格式，后续可统一成新格式 |
| 3 | `03_cli_inference.md` | 已按新格式重写 |
| 4 | `04_tokenizer_and_special_tokens.md` | 已按新格式重写 |
| 5 | `05_sft_dataset_and_labels.md` | 已按新格式完成 |
| 6 | `06_pretrain_vs_sft.md` | 已按新格式完成 |
| 7 | `07_pretrain_training_entry.md` | 已按新格式完成 |
| 8 | `08_minimind_config_and_params.md` | 已按新格式完成 |
| 9 | `09_embedding_norm_residual_block.md` | 已按新格式完成 |
| 10 | `10_causal_lm_forward_and_loss_shift.md` | 已按新格式完成 |
| 11 | `11_attention_principles_and_source.md` | 已按新格式完成 |
| 12 | `12_rope_and_kv_cache.md` | 已按新格式完成 |
| 13 | `13_ffn_and_moe.md` | 已按手写复现新格式完成 |
| 14 | `14_lr_grad_accum_amp.md` | 已按手写复现新格式完成 |
| 15 | `15_checkpoint_resume.md` | 已按手写复现新格式完成 |
| 16 | `16_full_sft_training_overview.md` | 已按手写复现新格式完成 |
| 17 | `17_pretrain_to_full_sft.md` | 已按手写复现新格式完成 |
| 18 | `18_lora_principles_and_implementation.md` | 已按手写复现新格式完成 |
| 19 | `19_lora_training_flow.md` | 已按手写复现新格式完成 |
| 20 | `20_distillation_training_flow.md` | 已按源码训练链路新格式完成 |
| 21 | `21_dpo_dataset_and_reference.md` | 已按源码训练链路新格式完成 |
| 22 | `22_dpo_loss_source.md` | 已按源码训练链路新格式完成 |
| 23 | `23_ppo_training_flow.md` | 已按源码训练链路新格式完成 |
| 24 | `24_grpo_cispo.md` | 已按源码训练链路新格式完成 |
| 25 | `25_reward_logprob_kl.md` | 已按源码训练链路新格式完成 |
| 26 | `26_tool_use_and_agentic_rl.md` | 已按源码训练链路新格式完成 |
| 27 | `27_model_conversion_api_webui.md` | 已按源码部署链路新格式完成 |
| 28 | `28_final_review_and_capstone.md` | 已按总复盘与小项目格式完成 |

28 节主线课程已完成；后续进入按方向扩展或补强实验。

## 课程结构

| 模块 | 课次 | 主题 | 主要源码 | 实验 |
|---|---:|---|---|---|
| 项目全局 | 1 | 项目地图与学习路线 | `README.md`, `eval_llm.py`, `trainer/*` | 画出主调用链 |
| 项目全局 | 2 | 环境、依赖、数据与权重 | `requirements.txt`, `trainer/trainer_utils.py` | 检查 import、本地数据和权重 |
| 推理入门 | 3 | CLI 推理流程 | `eval_llm.py`, `model/model_minimind.py` | 跑真实或 tiny 推理 |
| Tokenizer | 4 | Tokenizer 与特殊标记 | `model/tokenizer.json`, `model/tokenizer_config.json`, `trainer/train_tokenizer.py` | 编码/解码、观察 special tokens |
| 数据格式 | 5 | SFT 数据、Prompt 与 Labels | `dataset/lm_dataset.py`, `model/tokenizer_config.json` | 打印 SFT prompt 和 labels mask |
| 数据目标 | 6 | Pretrain 与 SFT 的数据目标对比 | `dataset/lm_dataset.py`, `model/model_minimind.py` | 对比 pretrain/SFT 的 `input_ids` 和 `labels` |
| 预训练 | 7 | Pretrain 训练入口如何串起来 | `trainer/train_pretrain.py`, `trainer/trainer_utils.py` | CPU 跑一个 tiny pretrain step |
| 模型结构 | 8 | MiniMindConfig 与参数规模 | `MiniMindConfig`, `MiniMindForCausalLM` | 对比 hidden/layer/head 改变后的参数量 |
| 模型结构 | 9 | Embedding、RMSNorm 与残差结构 | `MiniMindModel`, `RMSNorm`, `MiniMindBlock` | 跟踪 hidden states shape |
| 模型结构 | 10 | Causal LM forward 与 next-token loss | `MiniMindForCausalLM.forward` | 手动验证 logits/labels shift |
| 模型结构 | 11 | Attention 原理与源码 | `Attention.forward`, `repeat_kv` | 追踪 Q/K/V、scores、causal mask |
| 模型结构 | 12 | RoPE 与 KV cache | `precompute_freqs_cis`, `apply_rotary_pos_emb`, `generate` | 查看 RoPE shape，对比 cache 生成 |
| 模型结构 | 13 | FFN 与 MoE | `FeedForward`, `MOEFeedForward` | 对比 dense/MoE 参数量和 aux loss |
| 训练机制 | 14 | 学习率、梯度累积、混合精度 | `get_lr`, `train_epoch` | 打印 lr 曲线和 accumulation step |
| 训练机制 | 15 | checkpoint 与断点续训 | `lm_checkpoint`, `SkipBatchSampler` | 保存/恢复 tiny 权重 |
| SFT | 16 | Full SFT 训练脚本总览 | `trainer/train_full_sft.py`, `SFTDataset` | 跑一个 tiny SFT step |
| SFT | 17 | 从 pretrain 权重到 full_sft 权重 | `init_model`, `train_full_sft.py`, `eval_llm.py` | tiny pretrain -> tiny SFT -> 推理 |
| LoRA | 18 | LoRA 原理和手写实现 | `model/model_lora.py` | 给 Linear 注入 LoRA |
| LoRA | 19 | LoRA 训练流程 | `trainer/train_lora.py` | 验证只更新 LoRA 参数 |
| 蒸馏 | 20 | Distillation 训练链路 | `trainer/train_distillation.py` | 对比 teacher/student logits |
| 偏好学习 | 21 | DPO 数据和 reference model | `trainer/train_dpo.py` | 打印 chosen/rejected mask |
| 偏好学习 | 22 | DPO loss 源码 | `trainer/train_dpo.py` | 手算一个 batch 的偏好 loss |
| RL | 23 | PPO 训练链路 | `trainer/train_ppo.py`, `trainer/rollout_engine.py` | 拆解 rollout/reward/update |
| RL | 24 | GRPO 与 CISPO | `trainer/train_grpo.py` | 对比不同 loss_type |
| RL | 25 | reward、logprob 与 KL penalty | `trainer/trainer_utils.py`, `trainer/rollout_engine.py` | 查看 logprob/KL shape |
| Agent | 26 | Tool Use 与 Agentic RL | `trainer/train_agent.py`, `scripts/eval_toolcall.py` | 解析一次 tool_call |
| 部署 | 27 | 模型转换、API 与 WebUI | `scripts/convert_model.py`, `scripts/serve_openai_api.py`, `scripts/web_demo.py` | 启动 OpenAI 兼容接口 |
| 综合 | 28 | 总复盘与小项目 | 全项目 | 完成 tiny LLM 训练报告 |

## 每节课固定格式

第 3-12 课采用源码讲解 + 观察实验的新格式。第 13 课开始采用“原理 + 源码 + 手写 + 对齐 + 组装”的新格式。

```text
目录（使用可点击跳转链接）
0. 本节主线
1. 原理讲解
2. 源码阅读顺序图
3. MiniMind 源码走读
4. 本节必须会写 / 暂时不要求
5. 手写模块
6. 对齐测试
7. 阶段组装
本节检查
下一课
```

每个原理内部固定回答：

```text
这个原理到底是什么
为什么这节课要懂它
看哪段源码
这段源码证明了什么
理解到哪一步就够
哪些细节暂时不要看
```

源码证据不单独堆在最后，而是嵌入到对应原理里。每段源码都要说明“看它是为了理解什么”。

## 手写复现路线

每节课产出的代码都要能进入阶段实现，而不是停留在零散练习。阶段结束时，优先用 `course/impl/` 下的教学版实现跑通 tiny 或 mini 数据；工程外围需要成熟能力时，再复用 MiniMind 原源码。

| 阶段 | 手写产物 | 阶段验收 |
|---|---|---|
| 模型结构 | `core/model_parts.py`, `core/causal_lm.py`, `core/generation.py` | 手写 RMSNorm、Attention、RoPE、KV cache、FFN、CausalLM，并通过 shape/output 对齐测试 |
| Pretrain | `core/datasets.py`, `core/losses.py`, `core/train_loop.py`, `train_pretrain_impl.py` | 跑通 tiny pretrain，保存并加载 `course_pretrain` 权重 |
| SFT | `train_sft_impl.py` | 手写 SFT labels mask，从 pretrain 权重继续训练，保存 `course_sft` 权重 |
| LoRA | `core/lora.py`, `train_lora_impl.py` | 手写 LoRA Linear，验证只更新 LoRA 参数，跑通短 SFT |
| DPO | `train_dpo_impl.py` | 手写 chosen/rejected logprob 和 DPO loss，跑通短偏好训练 |
| PPO/GRPO | `core/ppo.py`, `core/grpo.py`, `train_grpo_impl.py` | 手写 rollout/logprob/reward/KL/advantage/policy loss 的最小闭环 |

阶段结束课额外包含：

```text
阶段能力检查表
阶段验收命令
源码差异解释
portfolio 记录
```

## 学习记录与产出

课程中的学习记录分三类：

| 位置 | 用途 |
|---|---|
| `course/notes/` | 个人笔记、错题和踩坑记录 |
| `course/impl/` | 手写教学版 MiniMind 核心实现 |
| `course/portfolio/` | 阶段进度、实验记录、实现说明、简历描述草稿 |

`course/notes/mistakes.md` 专门记录 shape、mask、labels、shift、cache、logprob、chosen/rejected 等错误。每次写错代码时，优先记录“错在哪里、为什么错、正确源码在哪里、以后怎么判断”。

## 学完后的能力目标

完成课程后，应该能独立说明并改动：

- tokenizer 如何把文本变成 token id。
- chat template 如何把多轮对话变成 prompt。
- pretrain 和 SFT 的数据、labels、loss 有何不同。
- `MiniMindForCausalLM` 如何从 `input_ids` 得到 logits 和 loss。
- Attention、RoPE、FFN、MoE 在源码里如何组织。
- 训练脚本如何组织 optimizer、lr、amp、checkpoint。
- LoRA 如何注入、训练和保存。
- DPO/PPO/GRPO 在训练目标上和 SFT 有什么差异。
- 如何把模型接到 CLI、OpenAI API 或 WebUI。
- 如何手写 MiniMind 核心训练链路，并说明教学版实现和原项目工程实现的差异。

## 推荐学习纪律

- 每节课先读“本节主线”，再看原理，不要一上来扎进整段源码。
- 每个源码片段只问一个问题：这段代码证明了哪个原理。
- 不跳过 tiny 实验。实验是为了确认变量形状、mask 和 loss 区域。
- 不追求第一次就训练出好模型，先追求能解释每个张量的来源和形状。
- 每个模块结束后写 5-10 行自己的总结，笔记可以放在 `course/notes/`。
