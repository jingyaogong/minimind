# Day 6 Exam

## A. GRPO 代码阅读题

1. `rollout_engine.rollout` 返回哪些字段？每个字段用于什么？
2. `old_per_token_logps` 和 `per_token_logps` 有什么区别？
3. `ref_per_token_logps` 为什么用 `torch.no_grad()`？
4. `completion_mask` 为什么要截到 EOS？
5. `advantages_std` 接近 0 说明什么？

## B. GRPO 推导题

给定同一个 prompt 的 4 个 reward：

```text
[1.0, 2.0, 2.0, 5.0]
```

计算：

- mean
- std
- 每个 response 的 advantage

回答：哪个 response 被最强鼓励？哪个被压制？

## C. CISPO 理解题

1. CISPO 为什么对 ratio 使用 `.detach()`？
2. `epsilon_high` 太大或太小分别有什么风险？
3. CISPO 和 GRPO 哪个更像 PPO clip？

## D. 蒸馏代码阅读题

1. `distillation_loss` 中 teacher logits 为什么要 `detach()`？
2. 为什么 KL 只在 `loss_mask == 1` 的 token 上算？
3. `alpha=0.5` 表示什么？
4. teacher vocab 大于 student vocab 时，代码如何处理？

## E. 工业题

1. RL 训练 reward 上升但人工评测下降，可能是什么问题？
2. KL 持续升高说明什么？
3. response length 越来越长，可能是 reward 哪部分导致？
4. 小模型做数学 RLVR 为什么容易奖励稀疏？

## F. 通过标准

你能讲清楚：

```text
SFT 是模仿标准答案；GRPO 是采样多个答案后按相对奖励调整概率；蒸馏是拟合 teacher 的硬答案或软分布。
```
