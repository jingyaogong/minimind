# PPO update loss block 源码解析

本文只解释 MiniMind PPO 训练中的这一段：

```text
trainer/train_ppo.py:171-214
```

它完成一次 PPO mini-batch update 的核心计算：

```text
当前 actor logprob
-> ratio
-> clipfrac
-> reference KL penalty
-> clipped policy loss
-> clipped value loss
-> 合成最终 loss
-> backward
```

## 0. 变量约定

这一节先按源码变量读。数学符号只用来把 `[batch, response_len]` 里的某个元素写清楚。

### 0.1 下标怎么读

```text
b = batch 里的第几条样本
t = response 里的第几个 token
```

所以：

```text
x[b, t]
```

就是：

```text
第 b 条样本，第 t 个 response token 位置上的值。
```

### 0.2 action 和 state

第 `b` 条样本第 `t` 个 response token 记作：

$$
a_{b,t}
$$

它对应源码里的：

```python
completion_ids[b, t]
```

生成这个 token 时，模型看到的上下文记作：

$$
s_{b,t}
$$

它表示：

```text
prompt + response 前 t-1 个 token
```

也就是说，PPO 里的：

```text
state  = s_{b,t}
action = a_{b,t}
```

### 0.3 logprob 符号

本文用小写希腊字母 ell：

$$
\ell
$$

表示 logprob。

当前 actor 的 token logprob：

$$
\ell^{\theta}_{b,t}
=
\log \pi_{\theta}(a_{b,t} \mid s_{b,t})
$$

对应源码：

```python
mb_resp_logp[b, t]
```

rollout 时 old actor 的 token logprob：

$$
\ell^{\mathrm{old}}_{b,t}
=
\log \pi_{\mathrm{old}}(a_{b,t} \mid s_{b,t})
$$

对应源码：

```python
old_resp_logp[b, t]
```

frozen reference model 的 token logprob：

$$
\ell^{\mathrm{ref}}_{b,t}
=
\log \pi_{\mathrm{ref}}(a_{b,t} \mid s_{b,t})
$$

对应源码：

```python
ref_resp_logp[b, t]
```

### 0.4 其它源码变量

有效 response token mask：

```python
resp_policy_mask[b, t]
```

公式里写成：

$$
m_{b,t}
$$

GAE 算出的 advantage：

```python
advantages[b, t]
```

公式里写成：

$$
A_{b,t}
$$

当前 critic value：

```python
mb_resp_values[b, t]
```

公式里写成：

$$
V_{\theta,b,t}
$$

rollout 阶段保存的 old critic value：

```python
old_resp_values[b, t]
```

公式里写成：

$$
V_{\mathrm{old},b,t}
$$

critic 的训练目标 return：

```python
returns[b, t]
```

公式里写成：

$$
G_{b,t}
$$

### 0.5 masked mean

后面很多 loss 都只在有效 response token 上求平均。统一写成：

$$
\operatorname{mean}_m(x)
=
\frac{
\sum_{b,t} m_{b,t} x_{b,t}
}{
\sum_{b,t} m_{b,t}
}
$$

其中 \(m_{b,t}\) 对应源码里的 `resp_policy_mask[inds]` 或 `resp_value_mask[inds]`。

## 1. 当前 actor 重新计算 response token logprob

源码：

```python
res = actor_unwrapped(input_ids=gen_out[inds], attention_mask=full_mask[inds])

mb_resp_logp = F.log_softmax(res.logits[:, :-1], dim=-1) \
    .gather(2, labels[inds].unsqueeze(-1)) \
    .squeeze(-1) \
    .gather(1, logp_pos[inds])
```

公式：

$$
\ell^{\theta}_{b,t}
=
\log \pi_{\theta}(a_{b,t} \mid s_{b,t})
$$

含义：

```text
rollout 阶段已经保存了 old actor 的 old_resp_logp。
现在 PPO update 阶段，要用当前 actor 对同一批 response token 重新算 logprob。
```

注意：

```text
这里没有重新生成 response。
它是在同一批 old rollout 上重新 forward，计算当前 actor 对这些 token 的概率。
```

## 2. log_ratio 和 ratio

源码：

```python
log_ratio = mb_resp_logp - old_resp_logp[inds]
ratio = torch.exp(log_ratio)
```

公式：

$$
\log r_{b,t}
=
\ell^{\theta}_{b,t}
-
\ell^{\mathrm{old}}_{b,t}
$$

$$
r_{b,t}
=
\exp(\log r_{b,t})
=
\frac{
\pi_{\theta}(a_{b,t} \mid s_{b,t})
}{
\pi_{\mathrm{old}}(a_{b,t} \mid s_{b,t})
}
$$

含义：

```text
ratio 表示当前 actor 相对 rollout 时 old actor，对同一个 token 的概率改了多少。
```

判断方式：

```text
ratio = 1:
  当前 actor 和 old actor 对这个 token 概率一样。

ratio > 1:
  当前 actor 更倾向这个 token。

ratio < 1:
  当前 actor 更不倾向这个 token。
```

## 3. approx_kl：监控 actor 是否离 old actor 太远

源码：

```python
approx_kl = (
    0.5 * (log_ratio ** 2) * resp_policy_mask[inds]
).sum() / resp_policy_mask[inds].sum().clamp(min=1)
```

公式：

$$
\operatorname{approx\_kl}
=
\operatorname{mean}_m
\left(
\frac{1}{2}
(\log r_{b,t})^2
\right)
$$

展开：

$$
\operatorname{approx\_kl}
=
\operatorname{mean}_m
\left(
\frac{1}{2}
\left[
\log \pi_{\theta}(a_{b,t} \mid s_{b,t})
-
\log \pi_{\mathrm{old}}(a_{b,t} \mid s_{b,t})
\right]^2
\right)
$$

用途：

```text
approx_kl 不直接作为 policy_loss 的主项。
它主要用于 early stop。
```

源码：

```python
if approx_kl_val > args.early_stop_kl:
    stop_ppo = True
```

含义：

```text
如果当前 actor 已经离 old actor 太远，就停止继续用这批 rollout 更新。
```

## 4. clipfrac：统计有多少 token 超出 PPO clip 范围

源码：

```python
clipfrac = (
    (((ratio - 1.0).abs() > args.clip_epsilon).float()
     * resp_policy_mask[inds]).sum()
    / resp_policy_mask[inds].sum().clamp(min=1)
)
```

公式：

$$
\operatorname{clipfrac}
=
\operatorname{mean}_m
\left(
\mathbf{1}
\left[
\left|r_{b,t} - 1\right| > \epsilon
\right]
\right)
$$

其中：

$$
\epsilon = \texttt{args.clip\_epsilon}
$$

含义：

```text
clipfrac 不是 loss。
它是一个监控指标，表示有多少有效 response token 的 ratio 已经超过 PPO 裁剪范围。
```

例子：

```text
clip_epsilon = 0.2
ratio 合理范围 = [0.8, 1.2]

ratio = 1.05 -> 不计入 clipfrac
ratio = 1.35 -> 计入 clipfrac
ratio = 0.70 -> 计入 clipfrac
```

如果 `clipfrac` 很大，说明：

```text
这批 update 里很多 token 的策略变化已经被 PPO clip 限制住了。
```

## 5. reference KL penalty：约束 actor 不要偏离 reference

源码：

```python
kl_ref_penalty = (
    (torch.exp(ref_resp_logp[inds] - mb_resp_logp)
     - (ref_resp_logp[inds] - mb_resp_logp)
     - 1.0)
    * resp_policy_mask[inds]
).sum() / resp_policy_mask[inds].sum().clamp(min=1)
```

先定义：

$$
\Delta_{b,t}
=
\ell^{\mathrm{ref}}_{b,t}
-
\ell^{\theta}_{b,t}
=
\log
\frac{
\pi_{\mathrm{ref}}(a_{b,t} \mid s_{b,t})
}{
\pi_{\theta}(a_{b,t} \mid s_{b,t})
}
$$

单个 token 的 penalty：

$$
k_{b,t}
=
\exp(\Delta_{b,t})
-
\Delta_{b,t}
-
1
$$

masked mean：

$$
\operatorname{kl\_ref\_penalty}
=
\operatorname{mean}_m
\left(
\exp(\Delta_{b,t}) - \Delta_{b,t} - 1
\right)
$$

含义：

```text
reference model 是 frozen SFT 模型。
这个项惩罚当前 actor 偏离 reference model。
```

注意它和 `approx_kl` 不同：

```text
approx_kl:
  当前 actor vs old actor
  用于 early stop

kl_ref_penalty:
  当前 actor vs reference model
  加进 policy_loss
```

## 6. policy_loss：PPO clipped surrogate + reference KL

源码：

```python
policy_loss = (
    (
        torch.max(
            -advantages[inds] * ratio,
            -advantages[inds] * torch.clamp(
                ratio,
                1.0 - args.clip_epsilon,
                1.0 + args.clip_epsilon,
            )
        )
        * resp_policy_mask[inds]
    ).sum() / resp_policy_mask[inds].sum().clamp(min=1)
    + args.kl_coef * kl_ref_penalty
)
```

先定义裁剪后的 ratio：

$$
\bar{r}_{b,t}
=
\operatorname{clip}
\left(
r_{b,t},
1-\epsilon,
1+\epsilon
\right)
$$

PPO 原始最大化目标通常写成：

$$
\mathcal{J}_{\mathrm{clip}}
=
\mathbb{E}
\left[
\min
\left(
r_{b,t} A_{b,t},
\bar{r}_{b,t} A_{b,t}
\right)
\right]
$$

源码是在最小化 loss，所以写成：

$$
\mathcal{L}_{\mathrm{policy\_clip}}
=
\operatorname{mean}_m
\left[
\max
\left(
-A_{b,t} r_{b,t},
-A_{b,t} \bar{r}_{b,t}
\right)
\right]
$$

MiniMind 最终 policy loss：

$$
\mathcal{L}_{\mathrm{policy}}
=
\mathcal{L}_{\mathrm{policy\_clip}}
+
\lambda_{\mathrm{KL}}
\operatorname{kl\_ref\_penalty}
$$

其中：

$$
\lambda_{\mathrm{KL}} = \texttt{args.kl\_coef}
$$

直观理解：

```text
advantage > 0:
  这个 token 比 critic 预期更好，希望提高它的概率。
  但 ratio 不能无限变大，最多按 1 + epsilon 裁剪。

advantage < 0:
  这个 token 比 critic 预期更差，希望降低它的概率。
  但 ratio 不能无限变小，最多按 1 - epsilon 裁剪。
```

所以 `policy_loss` 同时做两件事：

```text
1. 按 advantage 更新 actor。
2. 用 reference KL 约束 actor 不要偏离 SFT reference 太远。
```

## 7. value_loss：critic 的 clipped value loss

源码：

```python
value_loss = 0.5 * (
    torch.max(
        (mb_resp_values - returns[inds]) ** 2,
        (
            torch.clamp(
                mb_resp_values,
                old_resp_values[inds] - args.cliprange_value,
                old_resp_values[inds] + args.cliprange_value,
            )
            - returns[inds]
        ) ** 2
    )
    * resp_value_mask[inds]
).sum() / resp_value_mask[inds].sum().clamp(min=1)
```

当前 critic value：

$$
V_{\theta,b,t}
=
\texttt{mb\_resp\_values}_{b,t}
$$

旧 critic value：

$$
V_{\mathrm{old},b,t}
=
\texttt{old\_resp\_values}_{b,t}
$$

return target：

$$
G_{b,t}
=
\texttt{returns}_{b,t}
$$

裁剪后的 value：

$$
\bar{V}_{b,t}
=
\operatorname{clip}
\left(
V_{\theta,b,t},
V_{\mathrm{old},b,t} - c_v,
V_{\mathrm{old},b,t} + c_v
\right)
$$

其中：

$$
c_v = \texttt{args.cliprange\_value}
$$

两个 value error：

$$
e^{\mathrm{raw}}_{b,t}
=
\left(
V_{\theta,b,t} - G_{b,t}
\right)^2
$$

$$
e^{\mathrm{clip}}_{b,t}
=
\left(
\bar{V}_{b,t} - G_{b,t}
\right)^2
$$

最终 value loss：

$$
\mathcal{L}_{\mathrm{value}}
=
\frac{1}{2}
\operatorname{mean}_m
\left[
\max
\left(
e^{\mathrm{raw}}_{b,t},
e^{\mathrm{clip}}_{b,t}
\right)
\right]
$$

含义：

```text
critic 要学习预测 returns。
但源码也限制 critic value 不要一次离 old value 太远。
```

这里的 `0.5` 是 MSE 常见写法，方便梯度形式更简洁。

## 8. kl 和 kl_ref 只是日志变量

源码：

```python
kl = approx_kl_val
kl_ref = kl_ref_penalty.detach()
```

含义：

```text
kl:
  记录当前 actor vs old actor 的 approx_kl。

kl_ref:
  记录当前 actor vs reference model 的 KL penalty。
```

`.detach()` 表示：

```text
这里只用于记录，不让日志变量继续保留计算图。
```

## 9. stop_ppo 时为什么 loss 乘 0

源码：

```python
if stop_ppo:
    loss = (policy_loss + args.vf_coef * value_loss + aux_loss) * 0.0
else:
    loss = (
        policy_loss
        + args.vf_coef * value_loss
        + aux_loss
    ) / args.accumulation_steps

loss.backward()
```

正常训练时：

$$
\mathcal{L}
=
\frac{
\mathcal{L}_{\mathrm{policy}}
+
\lambda_v \mathcal{L}_{\mathrm{value}}
+
\mathcal{L}_{\mathrm{aux}}
}{
\texttt{accumulation\_steps}
}
$$

其中：

$$
\lambda_v = \texttt{args.vf\_coef}
$$

early stop 时：

$$
\mathcal{L}_{\mathrm{stop}}
=
0
\cdot
\left(
\mathcal{L}_{\mathrm{policy}}
+
\lambda_v \mathcal{L}_{\mathrm{value}}
+
\mathcal{L}_{\mathrm{aux}}
\right)
$$

为什么不直接 `continue` 或 `break`？

源码注释已经说明：

```text
早停时必须保证 forward-backward 闭环，故只截断 loss 不中断 DDP 通信
```

含义：

```text
分布式训练里，各卡最好执行相同的 forward/backward 结构。
如果某张卡提前 break，而其它卡还在 backward，可能导致 DDP 等通信卡死。
```

乘 `0.0` 的效果：

```text
仍然执行 backward；
但梯度为 0；
不会真正更新参数；
同时保持 DDP 通信路径完整。
```

## 10. 这段代码的完整执行顺序

```text
1. critic forward:
   得到当前 critic value: mb_resp_values

2. actor forward:
   得到当前 actor logits

3. gather:
   从 logits 中取 response token 的 mb_resp_logp

4. log_ratio:
   当前 actor logprob - old actor logprob

5. approx_kl:
   估计当前 actor 和 old actor 的距离，用于 early stop

6. ratio:
   exp(log_ratio)

7. clipfrac:
   统计多少 token 超出 PPO clip 范围

8. kl_ref_penalty:
   惩罚 actor 偏离 reference model

9. policy_loss:
   PPO clipped policy loss + reference KL

10. value_loss:
    clipped critic value loss

11. final loss:
    policy_loss + vf_coef * value_loss + aux_loss

12. backward:
    反向传播 actor 和 critic
```

## 11. 最容易混的 5 个点

1. `approx_kl` 不是 `kl_ref_penalty`。

```text
approx_kl: 当前 actor vs old actor
kl_ref_penalty: 当前 actor vs reference model
```

2. `clipfrac` 不是 loss。

```text
它只是监控有多少 token 的 ratio 超过 [1-eps, 1+eps]。
```

3. `policy_loss` 更新 actor。

```text
它使用 new/old logprob、advantage、reference KL。
```

4. `value_loss` 更新 critic。

```text
它使用 current value、old value、returns。
```

5. `stop_ppo` 时仍然 backward。

```text
loss 乘 0 是为了保留 DDP forward-backward 闭环。
```
