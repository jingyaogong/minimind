# CISPO 这段代码怎么理解

本文只解释第 24 课里 MiniMind 的 CISPO 分支：

```text
trainer/train_grpo.py:135-138
```

源码：

```python
if args.loss_type == "cispo":
    clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
    per_token_loss = -(
        clamped_ratio
        * advantages.unsqueeze(1)
        * per_token_logps
        - args.beta * per_token_kl
    )
```

## 0. 一句话

CISPO 的想法是：

```text
不要让 clipped ratio 变成一个切断梯度的目标；
把 clipped ratio 当成固定权重；
让真正的策略梯度从 log pi_theta(a|s) 走。
```

也就是：

```text
ratio 负责告诉模型“这次更新权重多大”
logprob 负责提供“往哪个方向更新”的梯度路径
```

## 1. 先看 GRPO 的问题

MiniMind 的 GRPO 分支是：

```python
clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
per_token_loss1 = ratio * advantages.unsqueeze(1)
per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)
```

对应公式：

$$
\mathcal{L}_{\mathrm{GRPO}}
=
-
\mathbb{E}
\left[
\min
\left(
r_t A,
\operatorname{clip}(r_t, 1-\epsilon, 1+\epsilon) A
\right)
-
\beta \operatorname{KL}_t
\right]
$$

其中：

$$
r_t
=
\frac{
\pi_{\theta}(a_t \mid s_t)
}{
\pi_{\mathrm{old}}(a_t \mid s_t)
}
$$

当 ratio 超过 clip 范围时，`clipped_ratio` 变成常数边界：

```text
1 + epsilon
```

或者：

```text
1 - epsilon
```

这能限制更新幅度，但也带来一个副作用：

```text
被 clip 的那一侧，梯度可能直接被截断。
```

README 里也说了这个问题：

```text
ratio 被 clip 之后，梯度流直接就被硬截断了。
```

源码证据：

文件：`README.md:1156-1170`

看它是为了理解：CISPO 为什么不是重新设计一套 RL 算法，而是改 GRPO 的 loss 形式。

代码/文本摘录：

```text
CISPO 的关注点并不是重新设计 group baseline，
而是用非常小的 loss 改动，更直接地修正这个问题。
```

这段说明：

- GRPO 的 group reward / advantage 仍然保留。
- CISPO 主要改策略项。
- 目标是避免 ratio clip 后梯度路径被截断。

### 1.1 clip 为什么会让梯度变成 0

先只看 `torch.clamp`：

$$
y
=
\operatorname{clamp}(x, l, u)
$$

它的导数大致是：

$$
\frac{\partial y}{\partial x}
=
\begin{cases}
1, & l < x < u \\
0, & x < l \text{ or } x > u
\end{cases}
$$

原因很简单：

```text
x 在区间内：clamp(x) = x，输出会跟着 x 变。
x 超过边界：clamp(x) = 边界常数，输出不再跟着 x 变。
```

所以当 ratio 超过 clip 边界时：

```python
clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
```

如果：

```text
ratio = 1.5
1 + epsilon = 1.2
```

那么：

```text
clipped_ratio = 1.2
```

这时把 `ratio` 从 `1.5` 改成 `1.6`，`clipped_ratio` 仍然是 `1.2`。也就是说：

```text
clipped_ratio 对 ratio 不敏感了；
梯度就不能从 clipped_ratio 继续传回 ratio。
```

再放回 GRPO/PPO 的 clipped surrogate：

$$
\min
\left(
r_t A_t,
\operatorname{clip}(r_t, 1-\epsilon, 1+\epsilon)A_t
\right)
$$

这里“梯度被截断”不是所有情况都会发生，而是发生在 `min` 选中了 clipped 分支、并且 `clamp` 已经落到边界时。

两个典型情况：

| advantage | ratio 状态 | 被限制的方向 | 结果 |
|---|---|---|---|
| $A_t > 0$ | $r_t > 1+\epsilon$ | 好 token 概率已经涨太多 | clipped 分支是常数边界，继续增大 ratio 没有梯度 |
| $A_t < 0$ | $r_t < 1-\epsilon$ | 坏 token 概率已经降太多 | clipped 分支是常数边界，继续减小 ratio 没有梯度 |

所以这句话的准确含义是：

```text
clip 限制了 policy 相对 old policy 变化太大；
但当 clipped 分支被选中时，策略项对 ratio 的梯度可能变成 0。
```

这就是 CISPO 想缓解的问题：不要把被 clip 的 ratio 本身当作主要梯度路径。

## 2. CISPO 的公式

README 写的 CISPO 损失是：

$$
\mathcal{L}_{\mathrm{CISPO}}
=
-
\mathbb{E}
\left[
\min(r_t, \epsilon_{\max})
\cdot
A_t
\cdot
\log \pi_{\theta}(a_t \mid s_t)
-
\beta \operatorname{KL}_t
\right]
$$

MiniMind 源码里的变量对应：

```text
min(r_t, epsilon_max)      -> clamped_ratio
A_t                       -> advantages.unsqueeze(1)
log pi_theta(a_t | s_t)   -> per_token_logps
beta * KL_t               -> args.beta * per_token_kl
```

所以源码：

```python
per_token_loss = -(
    clamped_ratio
    * advantages.unsqueeze(1)
    * per_token_logps
    - args.beta * per_token_kl
)
```

就是：

$$
\operatorname{loss}_{t}
=
-
\left[
\bar{r}_t
A
\ell^{\theta}_{t}
-
\beta \operatorname{KL}_t
\right]
$$

其中：

$$
\bar{r}_t
=
\min(r_t, \epsilon_{\max})
$$

$$
\ell^{\theta}_{t}
=
\log \pi_{\theta}(a_t \mid s_t)
$$

## 3. 为什么要乘 logprob

policy gradient 的基础形式是：

$$
\nabla_{\theta} J(\theta)
\approx
A_t
\nabla_{\theta}
\log \pi_{\theta}(a_t \mid s_t)
$$

所以如果要让模型：

```text
A_t > 0 时提高 token 概率
A_t < 0 时降低 token 概率
```

最直接的可微路径就是：

```text
log pi_theta(a_t | s_t)
```

CISPO 让损失里显式出现：

```python
advantages * per_token_logps
```

这样梯度就从 `per_token_logps` 走。

直观上：

```text
如果 advantage > 0:
  loss = -positive_weight * logprob
  最小化 loss 会提高 logprob。

如果 advantage < 0:
  loss = -negative_weight * logprob
  最小化 loss 会降低 logprob。
```

## 4. ratio 在 CISPO 里做什么

ratio 仍然有用。

源码：

```python
ratio = torch.exp(per_token_logps - old_per_token_logps)
clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
```

它对应：

$$
r_t
=
\exp
\left(
\log \pi_{\theta}(a_t \mid s_t)
-
\log \pi_{\mathrm{old}}(a_t \mid s_t)
\right)
$$

也就是：

$$
r_t
=
\frac{
\pi_{\theta}(a_t \mid s_t)
}{
\pi_{\mathrm{old}}(a_t \mid s_t)
}
$$

CISPO 不把 ratio 当成主要梯度路径，而是把它当作权重：

$$
\bar{r}_t
=
\min(r_t, \epsilon_{\max})
$$

如果某个 token 的当前概率相对 old policy 变大，ratio 大，那么这个 token 的策略项权重也变大。

但上限是：

```text
epsilon_high
```

避免权重爆炸。

## 5. 为什么要 detach

这行最关键：

```python
clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
```

`detach()` 的意思是：

```text
clamped_ratio 参与前向计算；
但不从 clamped_ratio 反传梯度。
```

也就是说，CISPO 的策略项：

```python
clamped_ratio * advantages * per_token_logps
```

反向传播时：

```text
clamped_ratio 被当成常数权重
advantages 被当成常数权重
per_token_logps 提供梯度
```

数学上近似看：

$$
\nabla_{\theta}
\left(
\bar{r}_t^{\mathrm{detach}}
A_t
\log \pi_{\theta}(a_t \mid s_t)
\right)
=
\bar{r}_t^{\mathrm{detach}}
A_t
\nabla_{\theta}
\log \pi_{\theta}(a_t \mid s_t)
$$

这就是 CISPO 的核心：

```text
用 ratio 控制权重；
用 logprob 保留梯度；
不要让梯度穿过 ratio 本身。
```

如果不 detach，梯度还会穿过：

```text
ratio = exp(new_logp - old_logp)
```

那策略项会同时从 ratio 和 logprob 两条路径反传，更新会更复杂、更容易偏离 CISPO 想表达的“裁剪权重 × logprob”。

## 6. 和 GRPO 的差别

GRPO 策略项：

$$
\min(r_t A_t, \bar{r}_t A_t)
$$

它的核心是：

```text
ratio 自己是优化目标的一部分。
```

CISPO 策略项：

$$
\bar{r}_t^{\mathrm{detach}}
A_t
\log \pi_{\theta}(a_t \mid s_t)
$$

它的核心是：

```text
ratio 只是权重；
logprob 才是梯度路径。
```

对比：

| 项目 | GRPO | CISPO |
|---|---|---|
| advantage 来源 | group reward | group reward |
| 是否需要 critic | 不需要 | 不需要 |
| 是否使用 old logprob | 使用 | 使用 |
| ratio 作用 | 进入 clipped surrogate | 作为 detached weight |
| 梯度主路径 | ratio / clipped surrogate | logprob |
| KL 约束 | token KL | token KL |

### 6.1 PPO/GRPO 更新 policy 到底靠谁

PPO/GRPO 更新 policy 不是“只靠 ratio”，也不是“只靠 advantage”。更准确的说法是：

```text
ratio/logprob 提供梯度路径；
advantage 提供方向和权重。
```

先看没有 clip 的策略目标：

$$
J_t
=
r_t A_t
$$

其中：

$$
r_t
=
\exp
\left(
\ell_t^\theta
-
\ell_t^{\mathrm{old}}
\right)
$$

$$
\ell_t^\theta
=
\log \pi_\theta(a_t \mid s_t)
$$

`old_per_token_logps` 是 rollout 时保存下来的旧 policy logprob，在训练当前 step 时当常数用。所以：

$$
\nabla_\theta J_t
=
A_t r_t
\nabla_\theta \ell_t^\theta
$$

也就是：

$$
\nabla_\theta J_t
=
A_t r_t
\nabla_\theta
\log \pi_\theta(a_t \mid s_t)
$$

这条公式说明三件事：

```text
1. 真正连到 policy net 参数的是 current logprob。
2. ratio 里面包含 current logprob，所以 ratio 是一条梯度路径。
3. advantage 自己通常不反传梯度，但它乘在梯度前面，决定方向和强度。
```

所以 advantage 不是“只和 reward model / critic 有关，然后和 policy 更新没关系”。它的来源确实可能来自 reward model、critic 或 group reward，但最后会直接进入 policy loss。

PPO 里：

```text
reward / reward_model -> reward
critic_model -> value
reward + value -> GAE advantage
policy_loss -> ratio * advantage 的 clipped 形式
```

GRPO 里：

```text
reward_model / rule -> reward
同一个 prompt 的多条 rewards -> group mean/std
group 标准化 -> advantage
policy_loss -> ratio * advantage 的 clipped 形式
```

两者的共同点：

```text
advantage 不一定接收梯度；
但 advantage 一定控制 policy gradient。
```

如果：

$$
A_t > 0
$$

优化会倾向于提高这个 token 的概率。

如果：

$$
A_t < 0
$$

优化会倾向于降低这个 token 的概率。

CISPO 的变化是：

```text
GRPO/PPO:
  ratio 既是权重，也是梯度路径的一部分。

CISPO:
  clamped_ratio.detach() 只当固定权重；
  per_token_logps 明确承担策略梯度路径。
```

## 7. 用正负 advantage 看方向

假设忽略 KL，只看 CISPO 策略项：

$$
\mathcal{L}
=
-
\bar{r}_t
A_t
\log \pi_{\theta}(a_t \mid s_t)
$$

### 7.1 advantage 大于 0

如果：

$$
A_t > 0
$$

那么：

$$
\mathcal{L}
=
-
\text{positive}
\cdot
\log \pi_{\theta}(a_t \mid s_t)
$$

最小化 loss 会让：

```text
log pi_theta(a_t | s_t) 变大
```

也就是提高这个 token 的概率。

### 7.2 advantage 小于 0

如果：

$$
A_t < 0
$$

那么：

$$
\mathcal{L}
=
+
\text{positive}
\cdot
\log \pi_{\theta}(a_t \mid s_t)
$$

最小化 loss 会让：

```text
log pi_theta(a_t | s_t) 变小
```

也就是降低这个 token 的概率。

这和 policy gradient 的直觉一致。

## 8. 源码证据：MiniMind 的 CISPO 分支

文件：`trainer/train_grpo.py:132-143`

看它是为了理解：GRPO 和 CISPO 共用 group advantage、ratio、reference KL，只切换 policy loss 形式。

代码摘录：

```python
kl_div = ref_per_token_logps - per_token_logps
per_token_kl = torch.exp(kl_div) - kl_div - 1
ratio = torch.exp(per_token_logps - old_per_token_logps)
if args.loss_type == "cispo":
    clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
    per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)
else:
    clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
    per_token_loss1 = ratio * advantages.unsqueeze(1)
    per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
    per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)
policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)).mean()
```

这段代码说明：

- `per_token_kl` 两种 loss 共用。
- `ratio` 两种 loss 都要计算。
- GRPO 使用 `torch.min(ratio*A, clipped_ratio*A)`。
- CISPO 使用 `clamped_ratio.detach() * A * per_token_logps`。
- `completion_mask` 只让有效 response token 参与 loss。

## 9. 记忆方式

```text
GRPO:
  clipped ratio objective
  ratio 是优化目标

CISPO:
  clipped importance weight * logprob
  ratio 是权重
  logprob 是梯度路径
```

一句话记：

```text
CISPO = GRPO 的 group advantage + reference KL + detached clipped ratio weighted logprob。
```

## 10. 检查问题

1. CISPO 为什么还需要 old logprob？
2. `clamped_ratio.detach()` 的作用是什么？
3. CISPO 的策略梯度主要从哪个变量走？
4. `advantages.unsqueeze(1)` 为什么可以乘到每个 token 上？
5. CISPO 和 GRPO 哪些部分是共用的？
6. 为什么 `torch.clamp` 到边界以后，对 ratio 的梯度可能是 0？
7. PPO/GRPO 里 advantage 不反传梯度，为什么仍然会影响 policy 更新？
