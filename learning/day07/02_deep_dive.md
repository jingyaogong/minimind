# Day 7 Deep Dive: 把 MiniMind 训练链路串起来

## 总链路

```text
Tokenizer
-> Pretrain
-> SFT
-> Evaluation
-> Optional: Distillation / DPO / PPO / GRPO / Agent RL
-> Convert / Serve / Web Demo
```

## 关键连接 1：模型结构和训练目标

模型只负责：

```text
input_ids -> logits
```

训练目标由 labels 决定：

- Pretrain labels：几乎所有文本 token。
- SFT labels：assistant token。
- DPO labels/masks：chosen/rejected assistant span。
- GRPO：生成 completion 后用 reward 调整 logprob。

真正改变学习行为的，不只是模型，而是目标函数和数据组织。

## 关键连接 2：SFT 为什么是主线核心

MiniMind 一周训练目标应以 `pretrain + full_sft` 为核心，因为：

- pretrain 让模型学会语言建模。
- SFT 让模型学会对话模板和助手行为。
- 这两步能在个人 GPU 上稳定复现。
- RL/蒸馏更适合理解和小规模实验，不适合作为第一周硬交付。

## 关键连接 3：RL 为什么难

RL 多了这些不稳定来源：

- policy 在线生成，数据分布不断变。
- reward model 不等于真实人类偏好。
- KL 太小会跑偏，太大会学不动。
- reward 方差太小没有学习信号。
- 小模型能力弱，容易产生无效 rollout。

所以 RL 训练要看：

- reward
- KL
- advantage std
- response length
- debug samples
- 人工评测

## 关键连接 4：工程上如何扩展

如果要继续提升：

1. 先扩大或清洗 SFT 数据。
2. 增加固定评测集。
3. 调整 `max_seq_len` 和 batch。
4. 尝试 MoE，但记录训练成本。
5. 尝试 LoRA 做领域适配。
6. 再做 GRPO/Agent RL，且必须设计可靠 reward。

## 最终复盘问题

你要能回答：

- 当前模型最强能力是什么？
- 最弱能力是什么？
- 训练瓶颈是数据、模型容量、训练时长还是评测方式？
- 下一周最值得投入的改进是什么？
