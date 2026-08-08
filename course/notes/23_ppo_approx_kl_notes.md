# PPO approx_kl 这段代码在做什么

对应源码：

```text
trainer/train_ppo.py:178-189
```

核心代码：

```python
mb_resp_logp = F.log_softmax(res.logits[:, :-1], dim=-1) \
    .gather(2, labels[inds].unsqueeze(-1)) \
    .squeeze(-1) \
    .gather(1, logp_pos[inds])

log_ratio = mb_resp_logp - old_resp_logp[inds]
approx_kl = (
    0.5 * (log_ratio ** 2) * resp_policy_mask[inds]
).sum() / resp_policy_mask[inds].sum().clamp(min=1)

if approx_kl_val > args.early_stop_kl:
    stop_ppo = True
```

## 1. 先看两个 logprob

`old_resp_logp`：

```text
rollout 时旧 actor 对 response token 的 logprob
```

数学上：

```text
old_resp_logp[b, t] = log pi_old(a_{b,t} | s_{b,t})
```

`mb_resp_logp`：

```text
PPO update 时当前 actor 对同一批 response token 重新算出来的 logprob
```

数学上：

```text
mb_resp_logp[b, t] = log pi_theta(a_{b,t} | s_{b,t})
```

这里的 `a_{b,t}` 是第 `b` 条样本第 `t` 个 response token。

这里的 `s_{b,t}` 是：

```text
prompt + response 前 t-1 个 token
```

## 2. log_ratio 是什么

源码：

```python
log_ratio = mb_resp_logp - old_resp_logp[inds]
```

公式：

```text
log_ratio[b, t]
= log pi_theta(a_{b,t} | s_{b,t})
  - log pi_old(a_{b,t} | s_{b,t})
```

也就是：

```text
log_ratio[b, t]
= log( pi_theta(a_{b,t} | s_{b,t}) / pi_old(a_{b,t} | s_{b,t}) )
```

所以后面：

```python
ratio = torch.exp(log_ratio)
```

就是：

```text
ratio[b, t]
= pi_theta(a_{b,t} | s_{b,t}) / pi_old(a_{b,t} | s_{b,t})
```

直观理解：

```text
ratio = 1:
  当前 actor 和 old actor 对这个 token 的概率一样。

ratio > 1:
  当前 actor 比 old actor 更倾向这个 token。

ratio < 1:
  当前 actor 比 old actor 更不倾向这个 token。
```

## 3. approx_kl 是什么

源码：

```python
approx_kl = (
    0.5 * (log_ratio ** 2) * resp_policy_mask[inds]
).sum() / resp_policy_mask[inds].sum().clamp(min=1)
```

公式：

```text
approx_kl
= sum_{b,t} m_{b,t} * 0.5 * log_ratio[b,t]^2
  / sum_{b,t} m_{b,t}
```

展开 `log_ratio`：

```text
approx_kl
= sum_{b,t} m_{b,t} * 0.5 *
   (log pi_theta(a_{b,t}|s_{b,t})
    - log pi_old(a_{b,t}|s_{b,t}))^2
  / sum_{b,t} m_{b,t}
```

其中：

```text
m_{b,t} = resp_policy_mask[b,t]
```

`resp_policy_mask` 的作用：

```text
只统计有效 response token；
不统计 prompt；
不统计 padding；
不统计 EOS 后面的 token。
```

## 4. 为什么叫 approx_kl

PPO 需要监控：

```text
当前 actor pi_theta 离 rollout 时的 old actor pi_old 有多远。
```

KL 本来是衡量两个分布距离的量。

当新旧策略很接近时，可以用二阶近似：

```text
KL(pi_old || pi_theta)
approx 0.5 * (log pi_theta - log pi_old)^2
```

所以源码用：

```python
0.5 * (log_ratio ** 2)
```

来近似新旧策略的 KL。

直观判断：

```text
log_ratio 接近 0:
  new actor 和 old actor 很接近，approx_kl 小。

log_ratio 绝对值变大:
  new actor 和 old actor 差得更远，approx_kl 变大。
```

## 5. 它是不是训练 loss

这段 `approx_kl` 不是 PPO 的主训练 loss。

PPO 的 actor loss 在后面：

```python
policy_loss = (
    clipped_policy_loss
    + args.kl_coef * kl_ref_penalty
)
```

`approx_kl` 的主要用途是 early stop：

```python
if approx_kl_val > args.early_stop_kl:
    stop_ppo = True
```

意思是：

```text
如果当前 actor 相对 old actor 改得太多，
就停止继续用这批 rollout 做 PPO 更新。
```

因为 PPO 是 on-policy 算法。

这批 response 是 old actor 生成的，如果当前 actor 已经离 old actor 太远，再继续拿这批旧 rollout 训练就不稳定。

## 6. 和 kl_ref_penalty 的区别

这两个很容易混：

```text
approx_kl:
  比较当前 actor 和 rollout 时的 old actor。
  用途是 early stop / 监控 PPO 更新幅度。
  源码位置: train_ppo.py:180-189

kl_ref_penalty:
  比较当前 actor 和 frozen reference model。
  用途是放进 actor loss，防止模型偏离 SFT reference。
  源码位置: train_ppo.py:194-199
```

一句话：

```text
approx_kl 控制“别离 old actor 太远”。
kl_ref_penalty 控制“别离 SFT reference 太远”。
```

## 7. 最小例子

假设某个 token：

```text
old_resp_logp = -2.0
mb_resp_logp  = -1.8
```

那么：

```text
log_ratio = -1.8 - (-2.0) = 0.2
ratio = exp(0.2) = 1.221
approx_kl_token = 0.5 * 0.2^2 = 0.02
```

含义：

```text
当前 actor 比 old actor 更倾向这个 token；
但变化不大，approx_kl_token 只有 0.02。
```

如果：

```text
old_resp_logp = -2.0
mb_resp_logp  = -0.5
```

那么：

```text
log_ratio = 1.5
ratio = exp(1.5) = 4.48
approx_kl_token = 0.5 * 1.5^2 = 1.125
```

含义：

```text
当前 actor 对这个 token 的概率提高太多；
如果 batch 平均也很大，就可能触发 early stop。
```

## 8. 记忆方式

```text
old_resp_logp:
  rollout 时候的概率，固定不变。

mb_resp_logp:
  当前 actor 重新 forward 的概率，会随着训练变化。

log_ratio:
  当前 actor 相对 old actor 的变化量。

approx_kl:
  用 log_ratio 的平方估计新旧 actor 的距离。

early_stop_kl:
  如果距离太大，就不要继续拿这批 old rollout 训练。
```
