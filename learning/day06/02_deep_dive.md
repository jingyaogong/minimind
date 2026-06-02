# Day 6 Deep Dive: GRPO/CISPO 与蒸馏

## 深挖问题 1：GRPO 的训练主线是什么？

代码位置：`trainer/train_grpo.py::grpo_train_epoch`

流程：

```text
batch prompts
-> rollout_engine.rollout 生成多个 completions
-> calculate_rewards 得到 reward
-> 当前 policy 重新算 per-token logp
-> ref_model 算 ref logp
-> group rewards -> advantages
-> 计算 KL
-> GRPO/CISPO policy loss
-> optimizer step
-> update rollout policy
```

这是 online RL，不是普通离线 SFT。

## 深挖问题 2：为什么 `num_generations` 是关键？

GRPO 对同一个 prompt 生成 N 个回答：

```python
grouped_rewards = rewards.view(-1, args.num_generations)
mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
advantages = (rewards - mean_r) / (std_r + 1e-4)
```

含义：

- 同题内部比较，降低不同题难度差异影响。
- 高于组均值的回答被鼓励。
- 低于组均值的回答被压制。
- 如果同组 reward 全一样，advantage 接近 0，学习信号消失。

## 深挖问题 3：reward 是怎么来的？

代码位置：`calculate_rewards`

包含：

- 长度奖励/惩罚。
- thinking 格式奖励。
- 重复惩罚 `rep_penalty`。
- reward model score。

工业理解：

- reward 不是客观真理，是训练目标。
- 设计不好会鼓励模型钻空子。
- 必须同时观察 reward 和真实输出。

## 深挖问题 4：GRPO 和 CISPO loss 差异

GRPO：

```python
ratio = exp(new_logp - old_logp)
clipped_ratio = clamp(ratio, 1-eps, 1+eps)
loss = -min(ratio*A, clipped_ratio*A) + beta*KL
```

CISPO：

```python
clamped_ratio = clamp(ratio, max=epsilon_high).detach()
loss = -(clamped_ratio * A * per_token_logps - beta*KL)
```

核心差异：

- GRPO/PPO-style clip 可能让梯度在 clip 后被截断。
- CISPO 把 ratio 当作 detached 权重，梯度仍来自 `log pi`。

## 深挖问题 5：蒸馏如何训练？

代码位置：`trainer/train_distillation.py`

总损失：

```python
loss = alpha * ce_loss + (1 - alpha) * distill_loss
```

其中：

- CE：学生拟合硬标签。
- KL：学生拟合 teacher token 分布。
- Temperature：让 teacher 分布更软，暴露 token 间偏好。

适合理解：

- teacher/student 双模型加载。
- teacher `eval()` 和 `requires_grad_(False)`。
- 只在 loss mask 位置做 KL。

## 今日输出

任选一个方向写 1 页报告：

### GRPO 报告

- rollout 输入输出是什么？
- old logp / new logp / ref logp 分别代表什么？
- advantage 怎么算？
- KL 为什么必要？
- 你认为当前 reward 设计有什么风险？

### 蒸馏报告

- CE 和 KL 各自训练什么？
- temperature 如何影响 teacher distribution？
- teacher/student 结构不同时要注意什么？
- 为什么 teacher 不需要反向传播？
