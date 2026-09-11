# 第 25 课：Reward、Logprob 与 KL Penalty

这一课不是再讲一遍 PPO 或 GRPO，而是把第 21-24 课反复出现的三个核心量统一整理清楚：`reward`、`logprob`、`KL penalty`。看懂这一节，后面读 `train_agent.py` 时就不会把 reward model、old logprob、reference model、approx KL 混在一起。

## 目录

- [0. 本节主线](#l25-mainline)
- [1. 本节要懂的 7 个原理](#l25-principles)
- [2. 完整原理：三个量分别管什么](#l25-complete-principle)
- [3. 源码阅读顺序图](#l25-reading-order)
- [4. MiniMind 源码走读](#l25-source-walkthrough)
- [5. 本节必须会写 / 暂时不要求](#l25-must-write)
- [6. 手写模块复盘](#l25-handwrite)
- [7. 实验验证](#l25-experiment)
- [8. 阶段组装](#l25-stage-assembly)
- [9. 本节检查](#l25-check)
- [10. 下一课](#l25-next)

<a id="l25-mainline"></a>
## 0. 本节主线

MiniMind 的 RL 训练里，这条链路反复出现：

```text
rollout policy 生成 response
-> 保存 old_logp，也就是采样当时每个 response token 的 logprob
-> reward model / 规则函数给整条 response 一个 sequence reward
-> PPO 把 sequence reward 放到最后一个有效 token，再用 critic + GAE 得到 token advantage
-> GRPO 把同 prompt 多条 response 的 reward 做 group mean/std，得到 sequence advantage
-> 训练时用 current policy 重新计算 policy_logp
-> 用 frozen reference model 计算 ref_logp
-> ratio = exp(policy_logp - old_logp)
-> reference KL penalty = exp(ref_logp - policy_logp) - (ref_logp - policy_logp) - 1
-> mask 掉 padding / EOS 后无效 token
-> policy loss 反向更新 policy
```

一句话：

```text
reward 告诉模型哪条回答好；
logprob 连接 policy 参数并形成 ratio；
KL penalty 约束 policy 不要离 reference 太远。
```

这三个量的角色不能混：

```text
reward:
  来自外部评价或规则，通常不反传梯度。

policy_logp:
  来自当前正在训练的 policy，是策略梯度的主要路径。

old_logp:
  来自 rollout 时的旧 policy，用来算 ratio。

ref_logp:
  来自冻结 reference model，用来做 KL 约束或 DPO 偏好基准。
```

<a id="l25-principles"></a>
## 1. 本节要懂的 7 个原理

| 原理 | 要理解什么 | 源码位置 |
|---|---|---|
| reward 是 sequence 级信号 | MiniMind 先对整条 response 打分，不是每个 token 单独打分 | `trainer/train_ppo.py:52-76`, `trainer/train_grpo.py:37-68` |
| reward model wrapper 只负责打分 | `LMForRewardModel.get_score` 把对话历史和回答交给 reward model，并裁剪分数范围 | `trainer/trainer_utils.py:160-177` |
| old logprob 来自 rollout | 采样 response 时同时保存每个 response token 的 logprob | `trainer/rollout_engine.py:23-36`, `trainer/rollout_engine.py:71-92` |
| current/ref logprob 在 update 时重算 | 训练时对同一批 `output_ids` 重算 policy/ref logprob | `trainer/train_ppo.py:117-139`, `trainer/train_grpo.py:97-105` |
| reward 到 advantage 的算法不同 | PPO 用 critic + GAE；GRPO 用 group reward 标准化 | `trainer/train_ppo.py:140-155`, `trainer/train_grpo.py:121-124` |
| KL 有不同用途 | `approx_kl` 监控 policy-old 距离；reference KL penalty 约束 policy-ref 距离 | `trainer/train_ppo.py:186-205`, `trainer/train_grpo.py:132-143` |
| DPO 的 reference 不是 token KL | DPO 用 chosen/rejected 的 reference logratio 作为偏好基准 | `trainer/train_dpo.py:34-49` |

学完本节，你应该能独立说清楚这些 tensor：

```text
old_logps:        rollout policy 采样时的 response token logprob
policy_logps:     当前 policy 对同一 response token 的 logprob
ref_logps:        frozen reference 对同一 response token 的 logprob
rewards:          每条 response 或 trajectory 的外部分数
advantages:       reward 经过 PPO/GRPO 规则转换后的训练权重
ratio:            exp(policy_logps - old_logps)
ref_kl_penalty:   exp(ref_logps - policy_logps) - (ref_logps - policy_logps) - 1
mask:             哪些 response token 参与 loss
```

<a id="l25-complete-principle"></a>
## 2. 完整原理：三个量分别管什么

### 2.1 Reward：回答好不好

MiniMind 的 PPO/GRPO 都先生成完整 response，再计算一个标量 reward：

$$
R_i \in \mathbb{R}
$$

这个 reward 可以来自：

```text
规则项：
  长度是否合理、thinking 格式是否合理、重复惩罚等。

reward model：
  用外部模型给 prompt/response 打分。
```

它不是直接对 policy 参数求导的东西。更准确地说：

```text
reward 是训练信号来源；
policy_logp 才是把这个信号传回模型参数的路径。
```

PPO 会把 sequence reward 放到最后一个有效 response token 上：

$$
r_{i,t}
=
\begin{cases}
R_i, & t = T_i \\
0, & \text{otherwise}
\end{cases}
$$

然后用 critic 的 value estimate 做 GAE，得到 token-level advantage：

$$
A_{i,t}
$$

GRPO 不训练 critic，而是同一个 prompt 采样多条 response：

$$
R_{i,1}, R_{i,2}, \dots, R_{i,G}
$$

再做组内标准化：

$$
A_{i,j}
=
\frac{
R_{i,j} - \mu_i
}{
\sigma_i + \epsilon
}
$$

所以：

```text
PPO:
  sequence reward -> last token reward -> GAE -> token advantage

GRPO:
  group sequence rewards -> group normalization -> sequence advantage
```

### 2.2 Logprob：policy 更新的入口

对一个已经生成出来的 token：

$$
a_t
$$

current policy 的 logprob 是：

$$
\ell_t^\theta
=
\log \pi_\theta(a_t \mid s_t)
$$

rollout old policy 的 logprob 是：

$$
\ell_t^{old}
=
\log \pi_{old}(a_t \mid s_t)
$$

reference model 的 logprob 是：

$$
\ell_t^{ref}
=
\log \pi_{ref}(a_t \mid s_t)
$$

MiniMind 里这些 logprob 都是通过同一个基本操作拿到的：

```text
logits -> log_softmax -> gather 目标 token id
```

即：

$$
\log p(a_t)
=
\operatorname{log\_softmax}(\operatorname{logits}_t)_{a_t}
$$

注意 logprob 的对齐关系：

```text
logits[:, k] 预测 input_ids[:, k + 1]
```

所以 response 第 `j` 个 token 的 logprob，要从：

```text
prompt_len - 1 + j
```

这个 logits 位置取。

### 2.3 Ratio：current policy 相对 old policy 变了多少

PPO/GRPO 的 ratio 是：

$$
r_t
=
\frac{
\pi_\theta(a_t \mid s_t)
}{
\pi_{old}(a_t \mid s_t)
}
=
\exp
\left(
\ell_t^\theta - \ell_t^{old}
\right)
$$

它回答的问题是：

```text
当前 policy 对当时采样出来的这个 token，比 old policy 更想生成还是更不想生成？
```

如果：

```text
ratio > 1
```

说明 current policy 比 old policy 更偏向这个 token。

如果：

```text
ratio < 1
```

说明 current policy 比 old policy 更不偏向这个 token。

ratio 进入 PPO/GRPO/CISPO 的 policy loss；它不是 reward，也不是 KL。

### 2.4 Reference KL penalty：别离 SFT reference 太远

MiniMind 的 PPO/GRPO 使用同一个 token-level reference KL 近似：

$$
\Delta_t
=
\ell_t^{ref} - \ell_t^\theta
$$

$$
\operatorname{KLPenalty}_t
=
\exp(\Delta_t) - \Delta_t - 1
$$

源码里对应：

```python
kl_div = ref_logp - policy_logp
per_token_kl = torch.exp(kl_div) - kl_div - 1
```

它的作用是：

```text
policy 可以因为 reward 往某些方向移动；
但不要离 frozen reference 太远。
```

### 2.5 Approx KL：PPO 早停用，不是 reference KL

PPO 里还有一个：

$$
\operatorname{approx\_kl}
=
\frac{1}{2}
\left(
\ell_t^\theta - \ell_t^{old}
\right)^2
$$

MiniMind 用它判断当前 policy 相对 rollout old policy 是否更新过猛：

```text
如果 approx_kl 太大，就停止本轮 PPO 多 epoch 更新。
```

所以要区分：

| 名称 | 比较对象 | 主要用途 |
|---|---|---|
| `ratio` | current policy vs old policy | policy loss 的概率比 |
| `approx_kl` | current policy vs old policy | PPO update early stop |
| `reference KL penalty` | current policy vs reference model | 防止偏离 SFT reference |
| DPO `ref_logratio` | reference chosen vs rejected | 偏好基准，不是 token KL penalty |

<a id="l25-reading-order"></a>
## 3. 源码阅读顺序图

建议按这个顺序读：

```text
1. trainer/trainer_utils.py:160-177
   先看 reward model wrapper 返回什么。

2. trainer/train_ppo.py:52-76
   看 PPO 的规则 reward + reward model score 如何相加。

3. trainer/rollout_engine.py:23-36
   看 rollout 阶段如何保存 old per-token logprob。

4. trainer/train_ppo.py:117-139
   看 PPO 如何定位 response token，并计算 ref logprob。

5. trainer/train_ppo.py:140-155
   看 PPO 如何把 sequence reward 变成 token advantage。

6. trainer/train_ppo.py:186-205
   看 ratio、approx KL、reference KL 和 policy loss。

7. trainer/train_grpo.py:121-143
   对比 GRPO 如何用 group advantage 和同一套 KL/ratio。

8. trainer/train_dpo.py:34-49
   最后用 DPO 对照：reference logprob 也能服务偏好 logratio，而不是只服务 KL penalty。
```

<a id="l25-source-walkthrough"></a>
## 4. MiniMind 源码走读

### 4.1 reward model wrapper：外部模型只负责打分

#### 源码证据：`LMForRewardModel`

文件：`trainer/trainer_utils.py:160-177`

看它是为了理解：MiniMind 把 reward model 封装成 `get_score(messages, response)`，训练脚本只拿到一个标量分数。

代码摘录：

```python
class LMForRewardModel:
    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
        self.model = self.model.to(device).eval()

    @torch.no_grad()
    def get_score(self, messages, response):
        ...
        score = self.model.get_score(self.tokenizer, eval_messages)
        return max(min(score, 3.0), -3.0)
```

这段代码说明：

- reward model 是 `eval()`，不会被 PPO/GRPO 更新。
- `get_score` 返回一个 Python 标量分数。
- 分数被裁剪到 `[-3.0, 3.0]`，避免极端 reward。
- 这个分数后面会和规则 reward 相加。

### 4.2 PPO/GRPO 的 reward：规则项 + reward model score

#### 源码证据 A：PPO reward

文件：`trainer/train_ppo.py:52-76`

看它是为了理解：PPO 对每条 response 计算一个 sequence reward。

代码摘录：

```python
def calculate_rewards(prompts, responses, reward_model):
    rewards = torch.zeros(len(responses), device=args.device)
    ...
    rewards[i] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
    if '</think>' in response:
        rewards[i] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
        rewards[i] += 0.25 if response.count('</think>') == 1 else -0.25
    rewards[i] -= rep_penalty(answer)
    score = reward_model.get_score(messages, answer)
    reward_model_scores.append(score)
    ...
    rewards += reward_model_scores
```

这段代码说明：

- reward 不是纯 reward model 分数。
- MiniMind 还加了长度、thinking 格式、重复惩罚等规则项。
- 返回的 `rewards` shape 是 `[B]`。

#### 源码证据 B：GRPO reward

文件：`trainer/train_grpo.py:37-68`

看它是为了理解：GRPO 的 reward shape 是 `[B * num_generations]`，因为每个 prompt 会生成多条 response。

代码摘录：

```python
for i in range(batch_size):
    for j in range(args.num_generations):
        response_idx = i * args.num_generations + j
        response = responses[response_idx]
        prompt = prompts[i]
        ...
        rewards[response_idx] -= rep_penalty(answer)
        score = reward_model.get_score(messages, answer)
        reward_model_scores.append(score)
...
rewards += reward_model_scores
```

这段代码说明：

- flat 排列顺序是 `prompt_i` 的第 `j` 条生成。
- 这个顺序决定后面可以用 `rewards.view(-1, num_generations)`。
- reward 仍然是 sequence 级，不是 token 级。

### 4.3 old logprob：rollout 阶段保存采样策略概率

#### 源码证据 A：通用 token logprob helper

文件：`trainer/rollout_engine.py:23-36`

看它是为了理解：old logprob 本质也是 `log_softmax + gather`。

代码摘录：

```python
def compute_per_token_logps(model, input_ids, n_keep, attention_mask=None):
    logits = unwrapped(input_ids, attention_mask=attention_mask, logits_to_keep=n_keep + 1).logits[:, :-1, :]
    per_token_logps = []
    for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
        per_token_logps.append(
            torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1)
        )
    return torch.stack(per_token_logps)
```

这段代码说明：

- `n_keep` 是 response token 数量。
- `input_ids[:, -n_keep:]` 是要取 logprob 的 completion tokens。
- 返回 shape 是 `[B, R]`。
- 这个 helper 用在 rollout 阶段，所以结果就是 `old_logp`。

#### 源码证据 B：Torch rollout 保存 old logprob

文件：`trainer/rollout_engine.py:71-92`

看它是为了理解：生成 response 后立刻保存 `per_token_logps`。

代码摘录：

```python
output_ids = model.generate(...)
prompt_len = prompt_ids.size(1)
completion_ids = output_ids[:, prompt_len:]
full_mask = (output_ids != self.tokenizer.pad_token_id).long()
per_token_logps = compute_per_token_logps(
    self.policy_model,
    output_ids,
    completion_ids.size(1),
    attention_mask=full_mask
)
return RolloutResult(output_ids, completion_ids, per_token_logps, ...)
```

这段代码说明：

- `output_ids` 是 prompt + response。
- `completion_ids` 是 response 部分。
- `per_token_logps` 是 rollout policy 对 response tokens 的 logprob。
- PPO/GRPO 里后续会把它命名为 `old_resp_logp` 或 `old_per_token_logps`。

#### 源码证据 C：SGLang rollout 也返回 logprob

文件：`trainer/rollout_engine.py:115-173`

看它是为了理解：即使用服务端 rollout，也要拿回同样语义的 `per_token_logps` 和 `completion_mask`。

代码摘录：

```python
payload = {
    "input_ids": all_input_ids,
    "sampling_params": {...},
    "return_logprob": True,
}
...
raw_logprobs = meta.get("output_token_logprobs", [])
...
return RolloutResult(
    output_ids=...,
    completion_ids=...,
    per_token_logps=pad_to_tensor(all_logprobs, max_comp_len, pad_val=0.0),
    completion_mask=torch.tensor([[1] * len(ids) + [0] * ... for ids in all_completion_ids], device=device),
)
```

这段代码说明：

- SGLang 返回的 logprob 也被整理成 `[B * num_generations, R]`。
- SGLang 的 `completion_mask` 能标出 padding。
- Torch rollout 的 mask 可能全 1，SGLang rollout 的 mask 更接近真实变长 completion。

### 4.4 current/ref logprob：训练时对同一批 token 重算

#### 源码证据 A：PPO response 位置和 ref logprob

文件：`trainer/train_ppo.py:117-139`

看它是为了理解：response token 的 logprob 要从 `logits[:, :-1]` 的对应位置取。

代码摘录：

```python
labels = gen_out[:, 1:].clone()
resp_idx = torch.arange(resp_labels.size(1), device=gen_out.device).unsqueeze(0)
logp_pos = prompt_lens.unsqueeze(1) - 1 + resp_idx
...
ref_resp_logp = F.log_softmax(
    ref_model(input_ids=gen_out, attention_mask=full_mask).logits[:, :-1],
    dim=-1
).gather(2, labels.unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)
```

这段代码说明：

- `labels = gen_out[:, 1:]` 表示每个位置预测下一个 token。
- `logp_pos = prompt_len - 1 + response_idx` 是 response token 对应的 logits 位置。
- `ref_resp_logp` shape 是 `[B, R]`。
- reference model 是 frozen，只提供约束基准。

#### 源码证据 B：GRPO current/ref logprob

文件：`trainer/train_grpo.py:97-105`

看它是为了理解：GRPO 的 policy/ref logprob 和 PPO 是同一类 gather 操作。

代码摘录：

```python
res = model_unwrapped(outputs, attention_mask=full_mask)
per_token_logps = F.log_softmax(res.logits[:, :-1, :], dim=-1).gather(
    2, outputs[:, 1:].unsqueeze(-1)
).squeeze(-1).gather(1, logp_pos)

with torch.no_grad():
    ref_per_token_logps = F.log_softmax(
        ref_model(outputs, attention_mask=full_mask).logits[:, :-1, :],
        dim=-1
    ).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)
```

这段代码说明：

- `per_token_logps` 是 current policy，参与反传。
- `ref_per_token_logps` 在 `torch.no_grad()` 下计算，不参与训练。
- 两者都只取 response token 位置。

### 4.5 reward 怎么变成 advantage

#### 源码证据 A：PPO 把 sequence reward 放到最后有效 token

文件：`trainer/train_ppo.py:140-155`

看它是为了理解：PPO 的 reward 是 sequence 级，但 advantage 是 token 级。

代码摘录：

```python
token_rewards = torch.zeros_like(old_resp_logp)
last_idx = resp_lengths - 1
token_rewards[torch.arange(B, device=args.device)[valid_resp], last_idx[valid_resp]] += rewards[valid_resp]

for t in reversed(range(gen_len)):
    nv = old_resp_values[:, t + 1] if t < gen_len - 1 else 0.0
    delta = token_rewards[:, t] + args.gamma * nv - old_resp_values[:, t]
    lastgaelam = delta + args.gamma * args.lam * lastgaelam
    advs_rev.append(lastgaelam)
advantages = torch.stack(advs_rev[::-1], dim=1)
returns = advantages + old_resp_values
```

这段代码说明：

- 外部 reward 只加在最后一个有效 token。
- Critic 的 `old_resp_values` 参与 GAE。
- 最后得到的 `advantages` shape 是 `[B, R]`。
- padding / EOS 后无效位置后面还会用 mask 去掉。

#### 源码证据 B：GRPO 用组内 reward 标准化

文件：`trainer/train_grpo.py:121-124`

看它是为了理解：GRPO 不需要 critic，直接把 group reward 变成 sequence advantage。

代码摘录：

```python
grouped_rewards = rewards.view(-1, args.num_generations)
mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
advantages = (rewards - mean_r) / (std_r + 1e-4)
```

这段代码说明：

- `grouped_rewards` shape 是 `[B, num_generations]`。
- 每条 response 的 advantage 是同组相对分数。
- `advantages` shape 是 `[B * num_generations]`，后面通过 `unsqueeze(1)` 广播到 token 维。

### 4.6 KL 的两个名字：approx KL 和 reference KL penalty

#### 源码证据 A：PPO update 中的 ratio 与 KL

文件：`trainer/train_ppo.py:186-205`

看它是为了理解：PPO 同时计算 policy-old 的 approx KL 和 policy-reference 的 KL penalty。

代码摘录：

```python
log_ratio = mb_resp_logp - old_resp_logp[inds]
approx_kl = (0.5 * (log_ratio ** 2) * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
...
ratio = torch.exp(log_ratio)
kl_ref_penalty = ((torch.exp(ref_resp_logp[inds] - mb_resp_logp) - (ref_resp_logp[inds] - mb_resp_logp) - 1.0)
                  * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
policy_loss = (
    masked_ppo_loss
    + args.kl_coef * kl_ref_penalty
)
```

这段代码说明：

- `log_ratio` 比较 current policy 和 rollout old policy。
- `approx_kl` 用于 early stop，不是加到 loss 的 reference KL。
- `kl_ref_penalty` 比较 current policy 和 frozen reference。
- policy loss 里加的是 `kl_coef * kl_ref_penalty`。

#### 源码证据 B：GRPO/CISPO 中的 reference KL

文件：`trainer/train_grpo.py:132-143`

看它是为了理解：GRPO/CISPO 没有 PPO 的 critic/value loss，但仍然使用 reference KL penalty。

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
```

这段代码说明：

- `per_token_kl` 是 token-level reference KL penalty。
- GRPO 和 CISPO 共享 `ratio`、`advantages`、`per_token_kl`。
- 区别只在 policy loss 形式。

### 4.7 DPO 对照：reference logprob 不是只能做 KL

#### 源码证据：DPO logratio

文件：`trainer/train_dpo.py:34-49`

看它是为了理解：DPO 也有 policy/ref logprob，但它不是 token-level KL penalty，而是 chosen/rejected 偏好差。

代码摘录：

```python
ref_log_probs = (ref_log_probs * mask).sum(dim=1)
policy_log_probs = (policy_log_probs * mask).sum(dim=1)
...
pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
logits = pi_logratios - ref_logratios
loss = -F.logsigmoid(beta * logits)
```

这段代码说明：

- DPO 先把 token logprob 沿 sequence 求和。
- `ref_logratios` 是 reference 对 chosen/rejected 的偏好基准。
- DPO 的 reference 用法是“偏好相对提升”，不是 `exp(ref-policy)-...` 这种 KL penalty。

<a id="l25-must-write"></a>
## 5. 本节必须会写 / 暂时不要求

### 5.1 必须会写

这节课不新增一个大的训练模块，但你必须能独立写出这些小公式：

```python
log_probs = F.log_softmax(logits, dim=-1)
token_logps = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
response_logps = token_logps.gather(1, logp_pos)
```

```python
ratio = torch.exp(policy_logps - old_logps)
```

```python
delta = ref_logps - policy_logps
ref_kl_penalty = torch.exp(delta) - delta - 1.0
```

```python
approx_kl = 0.5 * (policy_logps - old_logps) ** 2
```

```python
masked_mean = (x * mask).sum() / mask.sum().clamp(min=1)
```

以及 GRPO 的 group advantage：

```python
grouped_rewards = rewards.view(-1, num_generations)
mean_r = grouped_rewards.mean(dim=1).repeat_interleave(num_generations)
std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(num_generations)
advantages = (rewards - mean_r) / (std_r + 1e-4)
```

### 5.2 暂时不要求

本节暂时不要求：

```text
真正加载 reward model 权重；
启动 SGLang server；
写完整 PPO/GRPO 训练脚本；
调 reward shaping 超参；
证明 KL 近似公式的理论来源。
```

现在只要求你能读懂源码中每个张量的身份。

<a id="l25-handwrite"></a>
## 6. 手写模块复盘

本节对应已有手写文件：

```text
course/impl/core/ppo.py
course/impl/core/grpo.py
```

你应该回头确认：

```text
course/impl/core/ppo.py:
  response_logp_positions
  reference_kl_penalty
  ppo_policy_loss

course/impl/core/grpo.py:
  group_relative_advantages
  token_kl_penalty
  grpo_policy_loss
  cispo_policy_loss
```

这些函数共同覆盖了第 25 课的核心：

```text
logprob 位置对齐；
ratio；
mask；
reference KL penalty；
reward -> advantage。
```

验收命令：

```bash
cd /home/sun/minimind
python course/impl/tests/test_ppo.py
python course/impl/tests/test_grpo.py
```

如果 `test_grpo.py` 的 policy loss 失败，优先检查 reduction：

```python
policy_loss = (
    (per_token_loss * completion_mask).sum(dim=1)
    / completion_mask.sum(dim=1).clamp(min=1)
).mean()
```

不要写成全 batch token 直接平均，二者在 response 长度不一致时不同。

<a id="l25-experiment"></a>
## 7. 实验验证

本节新增实验：

```text
course/labs/trace_reward_logprob_kl.py
```

它不加载真实模型，只用固定 logits/token/reward 张量复现 MiniMind 的关键公式。

运行：

```bash
cd /home/sun/minimind
python course/labs/trace_reward_logprob_kl.py
```

它会打印：

```text
[Token logprobs]
old_logps / policy_logps / ref_logps 的 shape

[Reward to advantage]
PPO sequence reward 如何落到最后有效 token
GRPO group rewards 如何变成 group-relative advantages

[KL and ratio]
ratio = exp(policy-old)
approx_kl = 0.5 * (policy-old)^2
reference_kl_penalty = exp(ref-policy) - (ref-policy) - 1

[DPO contrast]
DPO 的 policy/ref logratio 和 reference KL penalty 的区别
```

判断实验正确的重点不是某个固定数值，而是这些关系：

```text
old_logps.shape == policy_logps.shape == ref_logps.shape == response_mask.shape
ratio.shape == policy_logps.shape
reference_kl_penalty.shape == policy_logps.shape
PPO rewards.shape == [B]
GRPO rewards.shape == [B * num_generations]
```

<a id="l25-stage-assembly"></a>
## 8. 阶段组装

到这里，PPO/GRPO 阶段的核心张量已经完整：

```text
rollout_engine
-> output_ids / completion_ids / old_logps / completion_mask
-> reward model / rules
-> PPO: token_rewards + critic values -> GAE advantage
-> GRPO: group rewards -> group advantage
-> current policy logps
-> reference logps
-> ratio / KL penalty
-> policy loss
```

后面组装 `course/impl/train_grpo_impl.py` 时，不需要复刻 MiniMind 的所有工程外围。最小教学版可以只保留：

```text
synthetic prompts 或 tiny RLAIF prompt
-> rollout 生成 response
-> 简化 reward 函数
-> group_relative_advantages
-> token_kl_penalty
-> grpo_policy_loss / cispo_policy_loss
-> optimizer step
```

本节的约束是：

```text
所有 loss 函数必须明确：
  哪个 logprob 来自 old policy；
  哪个 logprob 来自 current policy；
  哪个 logprob 来自 reference model；
  哪个 mask 决定有效 response token；
  reward 是如何进入 advantage 的。
```

<a id="l25-check"></a>
## 9. 本节检查

1. `old_logp` 和 `policy_logp` 的区别是什么？
2. 为什么 PPO/GRPO 不能只保存 response 文本，还要保存 old per-token logprob？
3. `ref_logp - policy_logp` 在 PPO/GRPO 中用来算什么？
4. `approx_kl = 0.5 * log_ratio^2` 比较的是哪两个 policy？
5. PPO 为什么把 sequence reward 加到最后一个有效 response token？
6. GRPO 为什么要求同一个 prompt 的多条 response 在 flat tensor 中连续排列？
7. DPO 的 `ref_logratios` 和 PPO/GRPO 的 reference KL penalty 有什么区别？

<a id="l25-next"></a>
## 10. 下一课

第 26 课进入 Tool Use 与 Agentic RL。

下一课要解决：

```text
train_agent.py 如何组织 tool schema、tool_call、tool_response；
一次多轮 trajectory 如何 rollout；
环境反馈 reward 和普通 reward model score 有什么区别；
Agentic RL 为什么是延迟 reward；
MiniMind 如何把 tool calling 接回 chat template。
```
