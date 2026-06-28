# 第 24 课：GRPO 与 CISPO

这一课讲 MiniMind 的 `train_grpo.py`：它如何用同一个 prompt 的多条生成结果构造 group-relative advantage，并通过 `loss_type` 在 GRPO 和 CISPO 两种策略损失之间切换。

相关补充：如果只想看 CISPO 分支里 `clamped_ratio.detach() * A * logprob` 的解释，读 `course/notes/24_cispo_notes.md`。

## 目录

- [0. 本节主线](#l24-mainline)
- [1. 本节要懂的 7 个原理](#l24-principles)
- [2. GRPO/CISPO 完整原理](#l24-complete-principle)
- [3. 源码阅读顺序图](#l24-reading-order)
- [4. MiniMind 源码走读](#l24-source-walkthrough)
- [5. 本节必须会写 / 暂时不要求](#l24-must-write)
- [6. 手写模块](#l24-handwrite)
- [7. 实验验证](#l24-experiment)
- [8. 阶段组装](#l24-stage-assembly)
- [9. 本节检查](#l24-check)
- [10. 下一课](#l24-next)

<a id="l24-mainline"></a>
## 0. 本节主线

GRPO 的训练链路是：

```text
每个 prompt 采样 num_generations 条 response
-> reward model / rule 给每条 response 一个标量 reward
-> 同一个 prompt 的多条 reward 组成一个 group
-> group 内做 mean/std 归一化，得到每条 response 的 advantage
-> policy/ref 分别计算 response token logprob
-> ratio = exp(policy_logp - old_logp)
-> token KL = exp(ref_logp - policy_logp) - (ref_logp - policy_logp) - 1
-> loss_type == grpo 时，用 PPO-style clipped ratio loss
-> loss_type == cispo 时，用 clamped ratio 作为权重乘 logprob
-> mask 后按 response 求均值，再对 batch 求均值
-> 更新 policy，并同步 rollout engine
```

一句话：

```text
GRPO 用“同题多答的组内相对分数”替代 PPO 的 critic advantage；CISPO 沿用这套 group advantage，只改策略损失的梯度路径。
```

和 PPO 的差别：

```text
PPO:
  需要 critic_model，先估计 value，再用 GAE 得到 token advantage。

GRPO/CISPO:
  不需要 critic_model。
  每个 prompt 生成多条 response，用同组 reward 的相对高低得到 sequence advantage。
```

<a id="l24-principles"></a>
## 1. 本节要懂的 7 个原理

| 原理 | 要理解什么 | 源码位置 |
|---|---|---|
| GRPO 用多生成构造组 | 每个 prompt 生成 `num_generations` 条 response | `trainer/train_grpo.py:80-95`, `trainer/train_grpo.py:226` |
| group advantage 替代 critic | `A=(R-mean_group)/(std_group+eps)`，不需要 value model | `trainer/train_grpo.py:121-124`, `README.md:1128-1135` |
| GRPO 仍然需要 old logprob | `ratio=exp(policy-old)`，仍然是 on-policy update | `trainer/train_grpo.py:90-101`, `trainer/train_grpo.py:134` |
| response mask 截到 EOS | 只在有效 completion token 上计算 loss | `trainer/train_grpo.py:126-130` |
| reference KL 是 token 约束 | 使用 `exp(ref-policy)-(ref-policy)-1` 约束 policy | `trainer/train_grpo.py:132-133` |
| GRPO loss 是 PPO-style clipped loss | `min(ratio*A, clipped_ratio*A)` 加 KL penalty | `trainer/train_grpo.py:139-143`, `README.md:1124-1131` |
| CISPO 只改 loss 形式 | `clamped_ratio.detach() * A * logprob`，避免 clip 后梯度完全截断 | `trainer/train_grpo.py:135-138`, `README.md:1156-1170` |

学完本节，你应该能解释这些 shape：

```text
prompts:              length B
completion_ids:       [B * num_generations, R]
old_per_token_logps:  [B * num_generations, R]
rewards:              [B * num_generations]
grouped_rewards:      [B, num_generations]
advantages:           [B * num_generations]
completion_mask:      [B * num_generations, R]
per_token_loss:       [B * num_generations, R]
```

<a id="l24-complete-principle"></a>
## 2. GRPO/CISPO 完整原理

### 2.1 GRPO 为什么能不用 critic

PPO 用 critic 估计：

```text
这个 token 位置往后的 expected return 是多少
```

然后用 GAE 得到每个 token 的 advantage。

GRPO 换了一个角度：对同一个 prompt，一次采样多条 response：

```text
y_1, y_2, ..., y_G ~ pi_old(. | x)
```

再分别打 reward：

```text
R_1, R_2, ..., R_G
```

同组 reward 的均值和标准差：

$$
\mu_x
=
\frac{1}{G}
\sum_{i=1}^{G}
R_i
$$

$$
\sigma_x
=
\sqrt{
\frac{1}{G}
\sum_{i=1}^{G}
(R_i - \mu_x)^2
}
$$

每条 response 的 group-relative advantage：

$$
A_i
=
\frac{
R_i - \mu_x
}{
\sigma_x + \epsilon
}
$$

含义：

```text
同一个 prompt 下，比同组平均 reward 高的 response，A_i > 0；
比同组平均 reward 低的 response，A_i < 0。
```

所以 GRPO 不需要 critic：

```text
baseline 不是 V(s)，而是同组回答的平均 reward。
```

### 2.2 退化组是什么

如果同一个 prompt 生成的多条 response reward 几乎一样：

```text
R_1 ≈ R_2 ≈ ... ≈ R_G
```

那么：

```text
R_i - mean_group ≈ 0
```

最终 advantage 接近 0。

这意味着：

```text
这一组几乎没有学习信号。
```

这就是 README 里说的退化组。MiniMind 这种小模型如果生成差异很小，GRPO 的信号会明显变弱。

### 2.3 GRPO loss

每个 response token 仍然有新旧策略概率比：

$$
r_{i,t}
=
\frac{
\pi_{\theta}(a_{i,t} \mid s_{i,t})
}{
\pi_{\mathrm{old}}(a_{i,t} \mid s_{i,t})
}
=
\exp(
\ell^{\theta}_{i,t}
-
\ell^{\mathrm{old}}_{i,t}
)
$$

GRPO 的 clipped surrogate：

$$
\mathcal{L}_{\mathrm{GRPO}}
=
-
\mathbb{E}
\left[
\min
\left(
r_{i,t} A_i,
\operatorname{clip}(r_{i,t}, 1-\varepsilon, 1+\varepsilon) A_i
\right)
-
\beta \operatorname{KL}_{i,t}
\right]
$$

源码的写法等价于：

$$
\operatorname{per\_token\_loss}
=
-
\left[
\min(r_{i,t} A_i, \bar{r}_{i,t} A_i)
-
\beta \operatorname{KL}_{i,t}
\right]
$$

其中：

$$
\bar{r}_{i,t}
=
\operatorname{clip}(r_{i,t}, 1-\varepsilon, 1+\varepsilon)
$$

注意这里的 advantage 是 sequence-level：

```text
A_i 的 shape 是 [B * num_generations]
```

进入 token loss 时通过：

```python
advantages.unsqueeze(1)
```

广播到每个 response token。

### 2.4 reference KL

GRPO/CISPO 使用和 PPO 类似的 token-level reference KL：

$$
\Delta_{i,t}
=
\ell^{\mathrm{ref}}_{i,t}
-
\ell^{\theta}_{i,t}
$$

$$
\operatorname{KL}_{i,t}
=
\exp(\Delta_{i,t})
-
\Delta_{i,t}
-
1
$$

它的作用是：

```text
惩罚 policy 偏离 frozen SFT reference。
```

### 2.5 CISPO 改了什么

GRPO 的 clip 写法会出现一个问题：

```text
当 ratio 被 clip 到常数区间边界后，clip 分支对 ratio 的梯度可能被截断。
```

CISPO 不重做 group advantage，只改策略项：

$$
\mathcal{L}_{\mathrm{CISPO}}
=
-
\mathbb{E}
\left[
\min(r_{i,t}, \varepsilon_{\max})
A_i
\log \pi_{\theta}(a_{i,t} \mid s_{i,t})
-
\beta \operatorname{KL}_{i,t}
\right]
$$

MiniMind 源码里：

```python
clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)
```

关键是：

```text
clamped_ratio.detach()
```

这表示 ratio 只作为固定权重使用；梯度主要走：

```text
per_token_logps = log pi_theta(...)
```

所以即使 ratio 被上限截断，策略项仍然通过 logprob 保留梯度路径。

### 2.6 GRPO/CISPO 与 PPO 的关系

```text
PPO:
  old_logp -> ratio
  critic -> GAE advantage
  policy loss + value loss

GRPO:
  old_logp -> ratio
  group rewards -> group-relative advantage
  policy loss only, no value loss

CISPO:
  group rewards -> group-relative advantage
  clamped ratio as detached weight
  logprob carries policy gradient
```

<a id="l24-reading-order"></a>
## 3. 源码阅读顺序图

建议按这个顺序读：

```text
1. README.md:1120-1135
   先看 GRPO 公式和 group advantage。

2. README.md:1156-1170
   看 CISPO 为什么只是 GRPO 的 loss 变体。

3. trainer/train_grpo.py:271-292
   看 policy/reference/reward/rollout engine 初始化，没有 critic。

4. trainer/train_grpo.py:71-95
   看每个 prompt 如何 rollout 出 num_generations 条 response。

5. trainer/train_grpo.py:121-124
   看 group reward 如何变成 advantage。

6. trainer/train_grpo.py:126-143
   看 mask、reference KL、ratio、GRPO/CISPO loss。

7. trainer/train_grpo.py:144-193
   看 backward、optimizer、保存和 rollout engine 同步。
```

<a id="l24-source-walkthrough"></a>
## 4. MiniMind 源码走读

### 4.1 初始化：没有 critic

#### 源码证据：GRPO 初始化组件

文件：`trainer/train_grpo.py:271-292`

看它是为了理解：GRPO 初始化 policy、reference、reward model 和 rollout engine，但没有 critic。

代码摘录：

```python
base_weight = args.from_weight
model, tokenizer = init_model(lm_config, base_weight, device=args.device)
ref_model, _ = init_model(lm_config, base_weight, device=args.device)
ref_model = ref_model.eval().requires_grad_(False)
reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
rollout_engine = create_rollout_engine(
    engine_type=args.rollout_engine,
    policy_model=model,
    tokenizer=tokenizer,
    device=args.device,
    autocast_ctx=autocast_ctx,
)
train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len, thinking_ratio=args.thinking_ratio)
```

这段代码说明：

- `model` 是要训练的 policy。
- `ref_model` 从同一份 base weight 初始化，但冻结。
- `reward_model` 是外部打分器。
- `rollout_engine` 用 policy 生成 response。
- 这里没有 `CriticModel`、`value_head`、`value_loss`。

### 4.2 rollout：每个 prompt 生成多条 response

#### 源码证据：num_generations

文件：`trainer/train_grpo.py:71-95`

看它是为了理解：GRPO 的 group 来自同一个 prompt 的多次生成。

代码摘录：

```python
prompts = batch['prompt']
prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, ...)

rollout_result = rollout_engine.rollout(
    prompt_ids=prompt_inputs["input_ids"],
    attention_mask=prompt_inputs["attention_mask"],
    num_generations=args.num_generations,
    max_new_tokens=args.max_gen_len,
    temperature=0.8,
)
outputs = rollout_result.output_ids
completion_ids = rollout_result.completion_ids
old_per_token_logps = rollout_result.per_token_logps.to(args.device).detach()
prompt_lens = rollout_result.prompt_lens.to(args.device)
rewards = calculate_rewards(prompts, completions, reward_model).to(args.device)
```

这段代码说明：

- 输入 batch 有 `B` 个 prompt。
- rollout 后有 `B * num_generations` 条 response。
- `old_per_token_logps` 是采样时 policy 的 token logprob。
- `rewards` 是每条 response 一个标量。

### 4.3 reward：按 prompt 和 generation 编号写入

#### 源码证据：calculate_rewards

文件：`trainer/train_grpo.py:37-68`

看它是为了理解：`rewards` 的平铺顺序是 `prompt_i` 的第 `j` 条生成。

代码摘录：

```python
for i in range(batch_size):
    for j in range(args.num_generations):
        response_idx = i * args.num_generations + j
        response = responses[response_idx]
        prompt = prompts[i]
        ...
        rewards[response_idx] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
        if '</think>' in response:
            rewards[response_idx] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
            rewards[response_idx] += 0.25 if response.count('</think>') == 1 else -0.25
        rewards[response_idx] -= rep_penalty(answer)
        score = reward_model.get_score(messages, answer)
        reward_model_scores.append(score)
```

这段代码说明：

- 同一个 prompt 的多条 response 在 flat tensor 中连续排列。
- 这个顺序让后面可以直接 `view(-1, num_generations)`。
- reward 由长度规则、thinking 格式、重复惩罚和 reward model 分数组成。

### 4.4 group reward 变 advantage

#### 源码证据：组内标准化

文件：`trainer/train_grpo.py:121-124`

看它是为了理解：GRPO 用 group baseline 替代 critic。

代码摘录：

```python
grouped_rewards = rewards.view(-1, args.num_generations)
mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
advantages = (rewards - mean_r) / (std_r + 1e-4)
```

这段代码说明：

- `grouped_rewards` shape 是 `[B, num_generations]`。
- 同一行对应同一个 prompt 的多条 response reward。
- `advantages` shape 回到 `[B * num_generations]`。
- advantage 是 sequence-level，不是 token-level。

### 4.5 completion mask 截断 EOS 和 padding

#### 源码证据：completion_mask

文件：`trainer/train_grpo.py:126-130`

看它是为了理解：loss 只作用在有效 response token。

代码摘录：

```python
completion_pad_mask = rollout_result.completion_mask.to(args.device).bool()
is_eos = (completion_ids == tokenizer.eos_token_id) & completion_pad_mask
eos_idx = torch.full((is_eos.size(0),), is_eos.size(1) - 1, dtype=torch.long, device=args.device)
eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
completion_mask = ((torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)) & completion_pad_mask).int()
```

这段代码说明：

- Torch rollout 下 `completion_pad_mask` 可能全 1，真正截断靠 EOS。
- SGLang rollout 下 `completion_pad_mask` 会处理 padding。
- `completion_mask` 保留到第一个 EOS，包括 EOS 本身。

### 4.6 current/ref logprob

#### 源码证据：policy 和 reference token logprob

文件：`trainer/train_grpo.py:97-105`

看它是为了理解：GRPO/CISPO 都需要当前 policy logprob 和 reference logprob。

代码摘录：

```python
res = model_unwrapped(outputs, attention_mask=full_mask)
per_token_logps = F.log_softmax(res.logits[:, :-1, :], dim=-1).gather(
    2, outputs[:, 1:].unsqueeze(-1)
).squeeze(-1).gather(1, logp_pos)

with torch.no_grad():
    ref_per_token_logps = F.log_softmax(ref_model(outputs, attention_mask=full_mask).logits[:, :-1, :], dim=-1).gather(
        2, outputs[:, 1:].unsqueeze(-1)
    ).squeeze(-1).gather(1, logp_pos)
```

这段代码说明：

- `per_token_logps` 是当前正在训练的 policy。
- `ref_per_token_logps` 是 frozen reference。
- 两者都只取 response token 位置。

### 4.7 loss_type 分叉：GRPO 或 CISPO

#### 源码证据：KL、ratio、loss

文件：`trainer/train_grpo.py:132-143`

看它是为了理解：`loss_type` 只切换策略项，group advantage 和 KL 共享。

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

- `per_token_kl` 是 reference KL penalty。
- `ratio` 比较 current policy 和 rollout old policy。
- GRPO 用 `min(ratio*A, clipped_ratio*A)`。
- CISPO 用 `clamped_ratio.detach() * A * per_token_logps`。
- 最终先对每条 response 的有效 token 求均值，再对所有 response 求均值。

### 4.8 backward、保存、同步 rollout policy

#### 源码证据：训练闭环

文件：`trainer/train_grpo.py:144-193`

看它是为了理解：GRPO/CISPO 只更新一个 policy model，没有 critic optimizer。

代码摘录：

```python
loss = (policy_loss + aux_loss) / args.accumulation_steps
loss.backward()

if step % args.accumulation_steps == 0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

if step % args.save_interval == 0 or step == iters:
    rollout_engine.update_policy(model)
```

这段代码说明：

- loss 不包含 `value_loss`。
- optimizer 只更新 policy。
- 更新后要把最新 policy 同步到 rollout engine。

<a id="l24-must-write"></a>
## 5. 本节必须会写 / 暂时不要求

### 5.1 必须会写

本节教学版核心是下面 4 个函数：

```python
def group_relative_advantages(rewards: torch.Tensor, num_generations: int, eps: float = 1e-4) -> torch.Tensor:
    ...

def token_kl_penalty(policy_logps: torch.Tensor, ref_logps: torch.Tensor) -> torch.Tensor:
    ...

def grpo_policy_loss(...):
    ...

def cispo_policy_loss(...):
    ...
```

### 5.2 暂时不要求

本节先不要求手写：

```text
真实 reward model；
SGLang 服务；
真实 rollout 生成；
DDP、checkpoint、wandb/swanlab；
Agentic RL 多轮 tool-use 版本。
```

<a id="l24-handwrite"></a>
## 6. 手写模块

本节对应教学版文件：

```text
course/impl/core/grpo.py
```

### 6.1 补 `group_relative_advantages`

对齐源码：

```text
trainer/train_grpo.py:121-124
```

行为要求：

```text
输入:
  rewards: [batch * num_generations]

输出:
  advantages: [batch * num_generations]

步骤:
  grouped = rewards.view(-1, num_generations)
  mean = grouped.mean(dim=1).repeat_interleave(num_generations)
  std = grouped.std(dim=1, unbiased=False).repeat_interleave(num_generations)
  advantages = (rewards - mean) / (std + eps)
```

### 6.2 补 `token_kl_penalty`

对齐源码：

```text
trainer/train_grpo.py:132-133
```

行为要求：

```text
delta = ref_logps - policy_logps
return exp(delta) - delta - 1
```

### 6.3 补 `grpo_policy_loss`

对齐源码：

```text
trainer/train_grpo.py:134-143
```

行为要求：

```text
ratio = exp(policy_logps - old_logps)
clipped_ratio = clamp(ratio, 1 - epsilon, 1 + epsilon)
loss1 = ratio * advantages[:, None]
loss2 = clipped_ratio * advantages[:, None]
per_token_loss = -(min(loss1, loss2) - beta * token_kl)
loss = mean(sequence_masked_mean(per_token_loss))
```

### 6.4 补 `cispo_policy_loss`

对齐源码：

```text
trainer/train_grpo.py:135-138
```

行为要求：

```text
ratio = exp(policy_logps - old_logps)
clamped_ratio = clamp(ratio, max=epsilon_high).detach()
per_token_loss = -(clamped_ratio * advantages[:, None] * policy_logps - beta * token_kl)
loss = mean(sequence_masked_mean(per_token_loss))
```

<a id="l24-experiment"></a>
## 7. 实验验证

实验文件：

```text
course/labs/trace_grpo_cispo_loss.py
```

命令：

```bash
cd /home/sun/minimind
python course/labs/trace_grpo_cispo_loss.py
```

这个实验验证：

```text
1. flat rewards 如何 reshape 成 [B, num_generations]。
2. group mean/std 如何产生 advantages。
3. GRPO 和 CISPO 共享 ratio 与 token KL。
4. 两种 loss_type 的 policy loss 不同。
5. completion_mask 如何只统计有效 token。
```

记录输出：

```text
grouped_rewards =
group_mean =
group_std =
advantages =
ratio_min =
ratio_max =
kl_mean =
grpo_loss =
cispo_loss =
```

输出含义：

```text
同一组 advantages 的均值应接近 0；
如果某组 reward 完全一样，该组 advantages 接近 0；
GRPO/CISPO 的 ratio 和 KL 一样，但 loss 形式不同。
```

<a id="l24-stage-assembly"></a>
## 8. 阶段组装

PPO/GRPO 阶段现在有两个核心文件：

```text
course/impl/core/ppo.py
course/impl/core/grpo.py
```

最小教学版 GRPO/CISPO 可以这样组装：

```text
CoursePromptDataset
-> actor.generate 每个 prompt 生成 G 条 response
-> old_logps 保存 rollout policy logprob
-> reward_fn 给每条 response 打分
-> group_relative_advantages 得到 sequence advantage
-> policy/ref 计算 per-token logprob
-> grpo_policy_loss 或 cispo_policy_loss
-> optimizer 更新 policy
```

### 8.1 本节验收命令

```bash
cd /home/sun/minimind
python course/labs/trace_grpo_cispo_loss.py
```

如果你补完 `course/impl/core/grpo.py`，后续应新增：

```text
course/impl/tests/test_grpo.py
```

测试内容：

```text
group_relative_advantages 对齐 train_grpo.py:121-124；
token_kl_penalty 对齐 train_grpo.py:132-133；
grpo_policy_loss 对齐 train_grpo.py:139-143；
cispo_policy_loss 对齐 train_grpo.py:135-138。
```

### 8.2 源码差异

教学版先省略：

```text
真实 rollout engine；
真实 reward model；
SGLang；
大模型 checkpoint；
DDP 和日志系统。
```

但不省略：

```text
group rewards
group-relative advantage
token KL
GRPO/CISPO loss_type 分叉
mask reduction
```

### 8.3 Portfolio 记录

完成本节后，可以在 `course/portfolio/experiments.md` 记录：

```text
GRPO/CISPO trace:
- 构造每个 prompt 的多条 response reward。
- 用组内 mean/std 计算 relative advantage。
- 对比 GRPO clipped ratio loss 和 CISPO weighted logprob loss。
- 说明为什么 GRPO 不需要 critic。
```

<a id="l24-check"></a>
## 9. 本节检查

1. GRPO 为什么可以不训练 critic？
2. `rewards.view(-1, num_generations)` 依赖什么样的 response 排列顺序？
3. 如果同一组 reward 完全一样，advantages 会发生什么？
4. GRPO 的 `advantages.unsqueeze(1)` 为什么可以乘到每个 token 上？
5. `per_token_kl = exp(ref-policy) - (ref-policy) - 1` 约束的是什么？
6. GRPO loss 和 PPO policy loss 哪里相似？
7. CISPO 为什么要对 `clamped_ratio` 调用 `.detach()`？
8. MiniMind 的 `loss_type` 改成 `grpo` 和 `cispo` 时，哪些流程不变？

<a id="l24-next"></a>
## 10. 下一课

第 25 课进入 reward、logprob 与 KL penalty 的专项复盘。

下一课要解决：

```text
PPO/GRPO/DPO 中 logprob 的共同模式；
reference KL、approx KL、DPO reference logratio 的区别；
reward model wrapper 和规则 reward 如何影响 RL 训练；
为什么不同算法都绕不开 mask、EOS 和 token-level logprob。
```
