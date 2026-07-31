<div align="center">

# 🎯 学习路径与进度看板

*借鉴 learn-claude-code 的 Track 化设计：按目标选路径，checkbox 可勾选*

</div>

---

## 📝 学习承诺（请先填）

> 心理学技巧：**承诺一致性**。先写下你的目标日期，完成率会显著提升。

```
我承诺：将在 ____ 年 ____ 月 ____ 日 之前完成 Track ____
每日投入时间：____ 小时
我的硬件：________________
```

把上面这段复制到你的笔记里填好，再继续往下读。

---

## 🟢 Track 1：速通 Zero（~3 小时）

> **目标**：今天就让模型开口说话。最快路径，不做理论深挖。
> **适合**：好奇党、想快速验证 LLM 没那么神秘的人。

### T1.1 准备
- [ ] 装好依赖（[00-起步/01-环境准备](./00-起步/01-环境准备.md)）
- [ ] 下载 `pretrain_t2t_mini.jsonl` + `sft_t2t_mini.jsonl`（[00-起步/03-跑通现成模型](./00-起步/03-跑通现成模型.md) 下载数据小节）
- [ ] 下载现成 minimind-3 权重，跑通 CLI 推理（先体验"目标产物"长啥样）

### T1.2 预训练（~1.2h）
- [ ] 阅读 [02-实战/01-预训练Pretrain](./02-实战/01-预训练Pretrain.md)
- [ ] `cd trainer && python train_pretrain.py`
- [ ] 看到 `out/pretrain_768.pth` 生成
- [ ] `python eval_llm.py --weight pretrain`，问"为什么天空是蓝色的"看接龙

### T1.3 SFT（~1.1h）
- [ ] 阅读 [02-实战/02-指令微调SFT](./02-实战/02-指令微调SFT.md)
- [ ] `cd trainer && python train_full_sft.py`
- [ ] 看到 `out/full_sft_768.pth` 生成
- [ ] `python eval_llm.py --weight full_sft`，多轮对话

### T1.4 完成 🎉
- [ ] 给朋友 demo 你的对话模型
- [ ] commit 你的进度：`git commit -m "🎉 minimind Zero done"`
- [ ] 在 [minimind discussions](https://github.com/jingyaogong/minimind/discussions) 留个学习贴

> 🎁 **完成 T1 你将拥有**：一个由你亲手训练、能多轮对话的 64M LLM。可以选 T2 深入或 T4 部署。

---

## 🔵 Track 2：系统精读（1-2 周）

> **目标**：真正读懂 LLM 的每一行代码。
> **适合**：想从"调库侠"升级为"造轮子侠"的人。

### T2.1 起步
- [ ] [HOW_TO_LEARN.md](./HOW_TO_LEARN.md)
- [ ] [00-起步/01-环境准备](./00-起步/01-环境准备.md)
- [ ] [00-起步/02-代码地图](./00-起步/02-代码地图.md)
- [ ] [00-起步/03-跑通现成模型](./00-起步/03-跑通现成模型.md)

### T2.2 理论篇（结合代码逐行读）
- [ ] [01-理论/01-Tokenizer与词表](./01-理论/01-Tokenizer与词表.md) + 对照 `model/tokenizer.json`
- [ ] [01-理论/02-Transformer核心架构](./01-理论/02-Transformer核心架构.md) + 对照 `model/model_minimind.py` 全部 288 行
- [ ] [01-理论/03-训练数据格式](./01-理论/03-训练数据格式.md) + 对照 `dataset/lm_dataset.py`

### T2.3 实战篇（边读边跑）
- [ ] [02-实战/01-预训练Pretrain](./02-实战/01-预训练Pretrain.md) + 跑通 `train_pretrain.py`
- [ ] [02-实战/02-指令微调SFT](./02-实战/02-指令微调SFT.md) + 跑通 `train_full_sft.py`
- [ ] [02-实战/03-推理与采样](./02-实战/03-推理与采样.md) + 玩转 temp / top_p / top_k

### T2.4 自检
- [ ] 能默写出 Transformer 一个 Block 的 forward 流程
- [ ] 能解释 GQA 和 MHA 的区别
- [ ] 能说出 SFT 训练时为什么 labels 要把 prompt 部分置为 -100
- [ ] 能解释 RoPE 为什么能外推（YaRN 思路）

> 🎁 **完成 T2 你将拥有**：从 0 实现并训练一个对齐 Qwen3 结构的小型 LLM 的全部能力。

---

## 🟣 Track 3：研究者路线（2-3 周）

> **目标**：跑通 DPO → PPO → GRPO → Agentic RL 全链路。
> **适合**：想做 RL 研究、复现 DeepSeek-R1 / Qwen3 Reasoning 的人。

### T3.1 前置
- [ ] 完成 T1 或 T2（至少有 `full_sft_768.pth`）
- [ ] 下载 `rlaif.jsonl` + `dpo.jsonl` + `agent_rl.jsonl` + `agent_rl_math.jsonl`

### T3.2 偏好学习
- [ ] [03-进阶/02-DPO与强化学习](./03-进阶/02-DPO与强化学习.md) DPO 部分
- [ ] `cd trainer && python train_dpo.py`
- [ ] 对比 SFT vs DPO 输出差异

### T3.3 RLAIF
- [ ] 同上文档 PPO 部分 + `train_ppo.py`
- [ ] 同上文档 GRPO 部分 + `train_grpo.py`
- [ ] 同上文档 CISPO 部分

### T3.4 Agentic RL
- [ ] [03-进阶/03-工具调用与思考](./03-进阶/03-工具调用与思考.md)
- [ ] `cd trainer && python train_agent.py`（GRPO/CISPO 多轮 Tool-Use）

### T3.5 拓展
- [ ] 阅读 README "📌 实验 → 强化学习" 中"PO 算法统一视角"章节
- [ ] 尝试 YaRN 长文本外推：`--inference_rope_scaling`

> 🎁 **完成 T3 你将拥有**：完整 RL 训练链路经验，能复现 reasoning / tool use 能力。

---

## 🟠 Track 4：应用工程（1-2 天）

> **目标**：把模型部署成可用的 API 服务。
> **适合**：想集成到产品 / 自己玩的人。

### T4.1 前置
- [ ] 跑通过 T1 拿到权重，或下载 `minimind-3` 现成权重

### T4.2 部署
- [ ] [04-部署/01-推理服务与评测](./04-部署/01-推理服务与评测.md)
- [ ] OpenAI 兼容 API：`python scripts/serve_openai_api.py --weight full_sft`
- [ ] WebUI：`cd scripts && streamlit run web_demo.py`
- [ ] vllm 加速：`vllm serve /path/to/minimind-3 --served-model-name minimind`
- [ ] ollama：`ollama run jingyaogong/minimind-3`

### T4.3 评测
- [ ] C-Eval / C-MMLU 跑分对比

> 🎁 **完成 T4 你将拥有**：一个接入 OpenAI SDK / WebUI / 命令行的可用 LLM 服务。

---

## ⚪ Track 5：垂域微调（1 天）

> **目标**：让 minimind 变身医疗助手 / 角色扮演 / 客服。
> **适合**：有私有数据、想做小众场景助手的人。

### T5.1 前置
- [ ] 跑通 T1 拿到 `full_sft_768.pth`

### T5.2 LoRA 微调
- [ ] [03-进阶/01-LoRA微调](./03-进阶/01-LoRA微调.md)
- [ ] 准备垂域数据 `lora_xxx.jsonl`（医疗/自我认知样例）
- [ ] `cd trainer && python train_lora.py`
- [ ] `python eval_llm.py --weight full_sft --lora_weight lora_medical`

### T5.3 合并导出
- [ ] `cd scripts && python convert_model.py` 把 LoRA 合回基座

> 🎁 **完成 T5 你将拥有**：一个针对你自己场景的小型领域助手。

---

## 📊 全局进度看板

> 复制下面这行到你的笔记里，每完成一篇就改一个 checkbox。目标梯度效应会推着你走完。

```
T1 速通 Zero     [ ] [ ] [ ] [ ]      0/4
T2 系统精读      [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ]   0/14
T3 研究者路线    [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ]      0/8
T4 应用工程      [ ] [ ] [ ] [ ] [ ]      0/5
T5 垂域微调      [ ] [ ] [ ] [ ] [ ]      0/5
```

完成任意一条 Track，欢迎到 [minimind discussions](https://github.com/jingyaogong/minimind/discussions) 分享你的体验！

---

## 🔗 不在这五条 Track 里？

| 我想… | 看这里 |
|---|---|
| 训练自己的 tokenizer | `trainer/train_tokenizer.py` + 主 README 数据章节 |
| 蒸馏学生模型 | [03-进阶](./03-进阶/) + `train_distillation.py` |
| 看 MoE | `model_minimind.py:148` `MOEFeedForward` |
| 转出 HuggingFace 格式 | `scripts/convert_model.py` |
| 多卡训练 | `torchrun --nproc_per_node N train_xxx.py` |
| 视觉多模态 | 移步 [minimind-v](https://github.com/jingyaogong/minimind-v) |
| 全模态 Omni | 移步 [minimind-o](https://github.com/jingyaogong/minimind-o) |

---

<div align="center">

**选好 Track，写下日期，开始吧 →** [返回 README](./README.md)

</div>
