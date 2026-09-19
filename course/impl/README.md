# Course Implementation

这个目录用于手写一个教学版 MiniMind 核心实现。

目标不是复制原项目的所有工程能力，而是通过课程逐步实现核心链路：

```text
model parts
-> causal LM forward/loss
-> generation with KV cache
-> pretrain dataset and train loop
-> SFT labels and training
-> LoRA
-> DPO
-> PPO/GRPO minimal loop
```

## 边界

自己实现：

- RMSNorm / Attention / RoPE / KV cache / FFN / CausalLM
- causal LM loss / SFT labels mask / DPO loss / PPO/GRPO 核心 loss
- pretrain / SFT / DPO 的关键数据处理
- 最小训练循环、保存和加载教学版权重

直接复用原项目：

- tokenizer 文件
- 原始数据文件
- 分布式训练
- wandb/swanlab
- 完整断点续训
- rollout server
- WebUI/API
- 模型转换

## 使用方式

每节课会指定一个或多个 TODO。先读课程原理和 MiniMind 源码，再补本目录中的实现。补完后运行对应对齐测试。

不要提前把后续模块全部写完。这里的代码应该跟课程进度一起增长。
