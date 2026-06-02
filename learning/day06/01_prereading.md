# Day 6 Prereading: 蒸馏与 GRPO/CISPO 后训练

## 今日目标

今天选择一个后训练方向深入：白盒蒸馏或 GRPO/CISPO。推荐优先读 GRPO，因为你当前打开的是 `train_grpo.py`，但如果 reward model 没准备好，可以把实跑换成蒸馏代码理解。

## 必须掌握的基础概念

### 蒸馏

- Black-box distillation：学习 teacher 生成的硬标签答案。
- White-box distillation：学习 teacher logits/probability distribution。
- Temperature：软化概率分布。
- KL divergence：衡量 teacher/student 分布差异。

### GRPO/CISPO

- On-policy rollout：用当前模型实时采样。
- Reward model：给回答打分。
- Reference model：限制 policy 偏离原模型。
- Advantage：回答比同组其他回答好多少。
- KL penalty：防止 reward hacking 和分布漂移。
- CISPO：把 clipped ratio 作为权重，保留 logprob 梯度路径。

## 面试考点

- DPO、PPO、GRPO 的主要区别是什么？
- GRPO 为什么可以不训练 critic？
- 为什么每个 prompt 要采样多个 responses？
- KL penalty 防什么？
- 蒸馏中的 temperature 为什么要乘 `T^2`？
- Reward hacking 是什么？

## 工业要求

- RL 后训练必须监控 reward、KL、response length、advantage std。
- 只看 reward 上升很危险，可能是 reward hacking。
- 小模型 RL 容易奖励稀疏，要选择能力边界内任务。
- 蒸馏要确认 teacher/student vocab 对齐或正确裁剪。
- 后训练前必须保留 SFT baseline，便于回退和对比。

## 项目中要看的文件

- `trainer/train_grpo.py`
- `trainer/train_agent.py`
- `trainer/rollout_engine.py`
- `trainer/train_distillation.py`
- `dataset/lm_dataset.py::RLAIFDataset`
- `trainer/trainer_utils.py::LMForRewardModel`

## 今日建议

如果 reward model 已准备好：

```bash
cd trainer
python train_grpo.py --debug_mode --debug_interval 5
```

如果没有 reward model：

- 先只读代码。
- 或选择 `train_distillation.py` 做理解。
