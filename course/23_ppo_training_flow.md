# 第 23 课：PPO 训练链路

这一课先讲 PPO 算法本体，再讲 MiniMind 的 `train_ppo.py` 如何把算法落到 LLM 训练里。

本节要解决的不只是“源码里有哪些变量”，而是这条完整主线：

```text
PPO 是一个 on-policy actor-critic 算法；
它用旧策略采样 trajectory；
用 reward 和 critic 算 advantage；
用新旧策略概率比 ratio 更新 actor；
用 clip 限制更新幅度；
在 RLHF 场景里再用 reference KL 约束模型不要偏离 SFT 模型太远。
```

## 目录

- [0. 本节主线](#l23-mainline)
- [1. 本节要懂的 7 个原理](#l23-principles)
- [2. PPO 完整原理](#l23-complete-principle)
- [3. 源码阅读顺序图](#l23-reading-order)
- [4. MiniMind 源码走读](#l23-source-walkthrough)
- [5. 本节必须会写 / 暂时不要求](#l23-must-write)
- [6. 手写模块](#l23-handwrite)
- [7. 实验验证](#l23-experiment)
- [8. 阶段组装](#l23-stage-assembly)
- [9. 本节检查](#l23-check)
- [10. 下一课](#l23-next)

<a id="l23-mainline"></a>
## 0. 本节主线

PPO 算法的主线是：

```text
初始化 actor policy pi_theta、critic value V_phi、reference policy pi_ref
-> 用当前 actor 采样 response trajectory
-> 保存采样时的 old_logprob
-> 用 reward model / rule 得到 response reward
-> 用 critic 估计 value
-> 用 reward 和 value 计算 advantage 与 return
-> 固定这批 rollout，多轮优化 actor/critic
-> actor 用 ratio = pi_theta / pi_old 做 clipped policy update
-> critic 用 return 做 value update
-> RLHF 额外用 reference KL 约束 actor
-> 更新后的 actor 再进入下一轮 rollout
```

MiniMind 源码里的落地链路是：

```text
RLAIFDataset 只提供 prompt
-> 当前 actor rollout 生成 response
-> rollout 时记录 old response token logprob
-> reward model 和规则给整条 response 一个标量 reward
-> 只在 response token 上建 mask 和 logp_pos
-> critic 给每个 response token 估计 value
-> 把整条 response reward 放到最后一个有效 token
-> GAE 从后往前算 advantages 和 returns
-> actor 重新 forward 得到 new response logprob
-> ratio = exp(new_logp - old_logp)
-> PPO clip policy loss + reference KL penalty 更新 actor
-> clipped value loss 更新 critic
-> 更新 rollout engine 里的 policy 权重
```

一句话：

```text
PPO 先用旧 actor 生成样本，再用 advantage 判断这些 token 应该被新 actor 提高概率还是降低概率，同时用 clip 防止新 actor 一步改得太猛。
```

这节和前两节的区别：

```text
DPO: 离线 chosen/rejected 偏好对，policy/ref 都在同一批静态数据上算 logprob。
PPO: 在线 rollout，数据来自当前 actor，old_logp 必须记录生成当时的策略概率。
```

<a id="l23-principles"></a>
## 1. 本节要懂的 7 个原理

| 原理 | 要理解什么 | 源码位置 |
|---|---|---|
| LLM PPO 的 RL 建模 | state 是 prompt+已生成前缀，action 是下一个 token，trajectory 是整段 response | `dataset/lm_dataset.py:195-224`, `trainer/rollout_engine.py:71-92` |
| PPO 是 on-policy 采样 | response 必须来自当前 actor，旧样本只能有限复用 | `README.md:1091-1100`, `trainer/train_ppo.py:89-101` |
| actor-critic 分工 | actor 生成和更新策略，critic 估计 value，reward model 给回报，reference 做约束 | `trainer/train_ppo.py:36-49`, `trainer/train_ppo.py:365-389`, `trainer/trainer_utils.py:160-177` |
| policy gradient 用 advantage 指方向 | advantage 大于 0 就提高 token 概率，小于 0 就降低 token 概率 | `trainer/train_ppo.py:136-151`, `trainer/train_ppo.py:196-199` |
| old/new logprob 形成 ratio | PPO 不直接最大化 logprob，而是比较新旧策略对同一 token 的概率比 | `trainer/rollout_engine.py:23-36`, `trainer/train_ppo.py:178-191` |
| clip surrogate 限制更新幅度 | ratio 被限制在 `1±epsilon` 附近，避免新策略离采样策略过远 | `README.md:1083-1089`, `trainer/train_ppo.py:191-199` |
| LLM 训练只作用 response token | prompt/padding 不进入 policy/value loss，`logp_pos` 对齐 next-token logits | `trainer/train_ppo.py:117-128`, `trainer/train_ppo.py:200-203` |

学完本节，你应该能画出：

```text
prompt -> rollout response -> old_logp/reward/value -> advantage/return -> new_logp -> ratio -> clipped actor loss + value loss
```

并且能解释下面这些张量的 shape：

```text
gen_out:          [batch, prompt_len + response_len]
completion_ids:  [batch, response_len]
old_resp_logp:   [batch, response_len]
logp_pos:        [batch, response_len]
resp_policy_mask:[batch, response_len]
advantages:      [batch, response_len]
returns:         [batch, response_len]
```

<a id="l23-complete-principle"></a>
## 2. PPO 完整原理

### 2.1 先把 LLM 训练写成 RL 问题

PPO 不是一种“特殊的 cross entropy loss”。它首先是强化学习算法。

一般 RL 里有三个核心对象：

```text
state s_t: 当前状态
action a_t: 当前动作
reward r_t: 动作之后得到的奖励
```

放到 LLM 里：

```text
state s_t  = prompt + response 前 t-1 个 token
action a_t = 第 t 个 response token
policy pi_theta(a_t | s_t) = 模型在当前位置输出这个 token 的概率
trajectory y = 一整段 response token
reward R(x, y) = reward model / rule 给 prompt-response 的分数
```

所以 LLM PPO 优化的不是某个固定标签，而是：

```text
让 actor 在 prompt x 下更容易生成高 reward 的 response y。
```

写成目标函数：

$$
J(\theta)
=
\mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)}
\left[
R(x, y)
\right]
$$

这里最关键的一点是：

```text
response y 是模型采样出来的离散 token 序列，不能像普通 CE loss 那样直接对 token 选择过程求导。
```

这就是为什么 PPO 要走 policy gradient 路线。

### 2.2 policy gradient：用 logprob 给采样动作分配责任

policy gradient 的核心结论是：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}\left[
\sum_t
A_t
\nabla_\theta \log \pi_\theta(a_t \mid s_t)
\right]
$$

先不用推导，先理解它的含义：

```text
如果某个 token 后面的结果比预期好，A_t > 0：
  提高 log pi_theta(a_t | s_t)

如果某个 token 后面的结果比预期差，A_t < 0：
  降低 log pi_theta(a_t | s_t)
```

`A_t` 叫 advantage。它不是原始 reward，而是：

```text
实际回报 - critic 预期回报
```

如果写成要最小化的 loss，最朴素的 policy gradient loss 是：

$$
\mathcal{L}_{\mathrm{PG}}
=
-\mathbb{E}\left[
\sum_t
A_t
\log \pi_\theta(a_t \mid s_t)
\right]
$$

这个式子能解释 PPO 的方向，但还不是 PPO。PPO 还要解决一个问题：策略每次不能改太猛。

### 2.3 为什么要 old policy 和 ratio

PPO 每轮先用当前 actor 采样一批 response。

采样那一刻的 actor 叫旧策略：

$$
\pi_{\mathrm{old}}
$$

更新时 actor 参数已经在变化，叫新策略：

$$
\pi_\theta
$$

同一个 token 在旧策略和新策略下各有一个概率：

$$
\pi_{\mathrm{old}}(a_t \mid s_t)
$$

$$
\pi_\theta(a_t \mid s_t)
$$

PPO 不直接用新策略 logprob，而是用新旧概率比：

$$
r_t(\theta)
=
\frac{
\pi_\theta(a_t \mid s_t)
}{
\pi_{\mathrm{old}}(a_t \mid s_t)
}
$$

在代码里通常用 logprob 更稳定：

$$
r_t(\theta)
=
\exp\left(
\log \pi_\theta(a_t \mid s_t)
-
\log \pi_{\mathrm{old}}(a_t \mid s_t)
\right)
$$

源码对应：

```python
log_ratio = mb_resp_logp - old_resp_logp[inds]
ratio = torch.exp(log_ratio)
```

`ratio` 的含义：

```text
ratio = 1: 新旧 actor 对这个 token 的概率一样
ratio > 1: 新 actor 更倾向这个 token
ratio < 1: 新 actor 更不倾向这个 token
```

为什么要保存 `old_logp`？

```text
因为 response 是 old actor 采样出来的。
如果只看 new_logp，就不知道新策略相对采样策略改了多少。
```

### 2.4 PPO clip：让 policy gradient 变成受限更新

如果只用：

$$
r_t(\theta) A_t
$$

当 `A_t > 0` 时，优化器会倾向把 `ratio` 推得越大越好。

当 `A_t < 0` 时，优化器会倾向把 `ratio` 压得越小越好。

这会让新策略离采样这批数据的旧策略太远，导致训练不稳定。

PPO 的核心处理是 clipped surrogate objective：

$$
\mathcal{L}^{\mathrm{clip}}(\theta)
=
-\mathbb{E}_t
\left[
\min\left(
r_t(\theta) A_t,
\mathrm{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) A_t
\right)
\right]
$$

这个公式的行为要分两种情况理解：

```text
如果 A_t > 0：
  希望提高该 token 概率，所以希望 ratio 变大；
  但 ratio 超过 1 + epsilon 后，不再继续给额外收益。

如果 A_t < 0：
  希望降低该 token 概率，所以希望 ratio 变小；
  但 ratio 低于 1 - epsilon 后，不再继续给额外收益。
```

源码是最小化 loss，所以把 `-min(...)` 写成等价的 `max(-..., -...)`：

$$
\mathrm{policy\_loss}
=
\mathbb{E}\left[
\max\left(
-A_t r_t,
-A_t \mathrm{clip}(r_t, 1-\varepsilon, 1+\varepsilon)
\right)
\right]
$$

MiniMind 再加上 reference KL penalty：

$$
\mathrm{actor\_loss}
=
\mathrm{policy\_loss}
+
\beta \mathrm{KL}(\pi_\theta, \pi_{\mathrm{ref}})
$$

### 2.5 critic 和 GAE：advantage 从哪里来

reward model 给的是整条 response 的分数，不是每个 token 的分数。

但 policy loss 要每个 response token 一个 `A_t`。

所以 PPO 引入 critic：

$$
V_\phi(s_t)
\approx
\mathbb{E}[\text{future return} \mid s_t]
$$

critic 的作用是估计：

```text
从这个 token 位置往后，平均能拿到多少回报。
```

有了 `V(s_t)` 后，先算 TD residual：

$$
\delta_t
= r_t + \gamma V(s_{t+1}) - V(s_t)
$$

再用 GAE 从后往前累计：

$$
A_t
= \delta_t + \gamma \lambda A_{t+1}
$$

最后 critic 的训练目标是：

$$
\mathrm{return}_t
=
A_t + V_{\mathrm{old}}(s_t)
$$

MiniMind 的做法是：

```text
token_rewards 全部为 0
-> 最后一个有效 response token 加上整条 response reward
-> critic 估计每个 response token 位置的 value
-> GAE 从后往前算每个 token 的 advantage
-> returns = advantages + old_values
```

这里要注意：

```text
reward 是整条 response 的标量；
advantage 是每个 response token 的训练信号。
```

### 2.6 LLM PPO 为什么还要 reference KL

经典 PPO 可以只讲 actor、critic、environment reward。

但 LLM RLHF 还多一个问题：

```text
reward model 不完美；
如果 actor 只追 reward，可能学到奇怪输出、格式投机、重复或偏离 SFT 行为。
```

所以 MiniMind 像常见 RLHF PPO 一样保留 reference model。

reference model 通常是训练前的 SFT 模型：

```text
pi_ref = frozen SFT policy
```

训练时额外惩罚 actor 偏离 reference：

$$
\beta \cdot \mathrm{KL}(\pi_\theta \| \pi_{\mathrm{ref}})
$$

MiniMind 的 `kl_ref_penalty` 用的是 token 级近似形式：

```python
torch.exp(ref_resp_logp - mb_resp_logp) - (ref_resp_logp - mb_resp_logp) - 1.0
```

写成数学形式，先定义第 `b` 条样本第 `t` 个 response token：

$$
\ell^{\theta}_{b,t}
=
\log \pi_{\theta}(a_{b,t} \mid s_{b,t})
$$

$$
\ell^{\mathrm{ref}}_{b,t}
=
\log \pi_{\mathrm{ref}}(a_{b,t} \mid s_{b,t})
$$

代码里的差值是：

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

单个 token 的 penalty 是：

$$
k_{b,t}
=
\exp(\Delta_{b,t})
-
\Delta_{b,t}
-
1
$$

也就是：

$$
k_{b,t}
=
\frac{
\pi_{\mathrm{ref}}(a_{b,t} \mid s_{b,t})
}{
\pi_{\theta}(a_{b,t} \mid s_{b,t})
}
-
\log
\frac{
\pi_{\mathrm{ref}}(a_{b,t} \mid s_{b,t})
}{
\pi_{\theta}(a_{b,t} \mid s_{b,t})
}
-
1
$$

带上 response mask 后，MiniMind 实际优化的是 masked mean：

$$
\mathrm{KL}_{\mathrm{ref}}
=
\frac{
\sum_{b,t} m_{b,t} k_{b,t}
}{
\sum_{b,t} m_{b,t}
}
$$

其中 `m_{b,t}` 对应源码里的 `resp_policy_mask`。

这个形式可以看作对：

$$
\mathrm{KL}(\pi_{\theta} \| \pi_{\mathrm{ref}})
$$

的 token 级采样估计。直观上，actor 给某个 token 的概率越偏离 reference，这个 penalty 越大。

这不是 DPO 里的 chosen/rejected reference logratio。这里的 reference 作用是：

```text
约束 actor 在 PPO 更新时不要离原始 SFT policy 太远。
```

### 2.7 PPO 完整算法伪代码

把上面连起来，PPO 的训练框架是：

```text
initialize actor pi_theta
initialize critic V_phi
initialize frozen reference pi_ref
initialize reward model R

for each training step:
    prompts = sample prompts

    with no_grad:
        responses = actor.generate(prompts)
        old_logps = log pi_old(response_tokens | prompts)
        rewards = R(prompts, responses)
        old_values = V_phi(prompts + responses)
        advantages, returns = GAE(rewards, old_values)

    for ppo_epoch in range(K):
        for minibatch in rollout_batch:
            new_logps = log pi_theta(response_tokens | prompts)
            values = V_phi(prompts + responses)

            ratio = exp(new_logps - old_logps)
            policy_loss = clipped_surrogate(ratio, advantages)
            kl_loss = KL(pi_theta, pi_ref)
            value_loss = clipped_mse(values, returns)

            loss = policy_loss + beta * kl_loss + vf_coef * value_loss
            update actor and critic

    sync latest actor into rollout engine
```

这个框架里最容易混的地方是：

```text
old_logps: 采样 response 时的 actor 概率，固定不变
new_logps: 当前正在更新的 actor 概率，每次 forward 会变
advantages: 这批 response token 好不好，由 reward 和 critic 决定
ratio: 新 actor 相对旧 actor 对同一 token 改了多少
clip: 限制 ratio，防止离 old actor 太远
```

### 2.8 PPO 公式到 MiniMind 变量映射

| PPO 符号 | 含义 | MiniMind 变量 |
|---|---|---|
| `s_t` | prompt + 已生成 response 前缀 | `gen_out` 中 response token 之前的上下文 |
| `a_t` | 第 `t` 个 response token | `completion_ids[:, t]` / `labels.gather(...)` |
| `pi_old` | rollout 时的 actor | `rollout_engine.policy_model` |
| `log pi_old(a_t|s_t)` | 生成时 token logprob | `old_resp_logp` |
| `pi_theta` | 当前正在更新的 actor | `actor_model` / `actor_unwrapped` |
| `log pi_theta(a_t|s_t)` | 更新时重新计算的 token logprob | `mb_resp_logp` |
| `r_t(theta)` | 新旧策略概率比 | `ratio = torch.exp(log_ratio)` |
| `R(x, y)` | 整条 response reward | `rewards` |
| `r_t` | token 级 reward | `token_rewards` |
| `V(s_t)` | critic value | `old_resp_values`, `mb_resp_values` |
| `A_t` | advantage | `advantages` |
| `return_t` | value target | `returns` |
| `epsilon` | PPO clip 范围 | `args.clip_epsilon` |
| `pi_ref` | frozen reference policy | `ref_model` |
| `KL(pi_theta, pi_ref)` | reference 约束 | `kl_ref_penalty` |

<a id="l23-reading-order"></a>
## 3. 源码阅读顺序图

建议按这个顺序读：

```text
1. dataset/lm_dataset.py:195-224
   看 PPO 数据集为什么只返回 prompt。

2. trainer/train_ppo.py:365-389
   看 actor、reference、critic、reward model、rollout engine 如何初始化。

3. trainer/rollout_engine.py:23-92
   看 rollout 如何生成 response，并保存 old per-token logprob。

4. trainer/train_ppo.py:84-128
   看一个 batch 如何从 prompt 变成 gen_out、completion_ids、mask、logp_pos。

5. trainer/train_ppo.py:130-151
   看 critic、reference、reward 和 GAE 如何产生 advantages/returns。

6. trainer/train_ppo.py:164-203
   看 PPO mini-batch 更新里的 ratio、clip、KL、value loss。

7. trainer/train_ppo.py:246
   看 actor 更新后如何同步给 rollout engine。
```

和前几课的关系：

```text
第 21 课：reference model 和 token logprob
第 22 课：sequence logprob 和 DPO loss
第 23 课：PPO rollout、advantage、clip update
第 24 课：GRPO/CISPO，减少或替代 PPO 里的 critic/value 组件
```

<a id="l23-source-walkthrough"></a>
## 4. MiniMind 源码走读

### 4.1 PPO 数据集只提供 prompt

#### 源码证据：RLAIFDataset

文件：`dataset/lm_dataset.py:195-224`

看它是为了理解：PPO 不是读取一个固定 answer 来训练，而是只拿 prompt 给 actor 生成。

代码摘录：

```python
class RLAIFDataset(Dataset):
    def create_chat_prompt(self, conversations):
        conversations = pre_processing_chat(conversations)
        use_thinking = random.random() < self.thinking_ratio
        return self.tokenizer.apply_chat_template(
            conversations[:-1],
            tokenize=False,
            open_thinking=use_thinking,
            add_generation_prompt=True
        )
    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt,
            'answer': ""
        }
```

这段代码说明：

- `conversations[:-1]` 表示不把最后的标准回复当训练标签。
- `add_generation_prompt=True` 让 prompt 结束在 assistant 待生成位置。
- `answer` 是空字符串，真正的 response 来自后面的 rollout。

理解到这里就够：

```text
PPO 的训练数据不是 labels，而是让 actor 生成 response 的上下文。
```

暂时不要看：

```text
thinking_ratio 的课程外效果；
真实 RLAIF 数据如何收集；
reward model 的训练过程。
```

### 4.2 四类模型的分工

#### 源码证据 A：CriticModel

文件：`trainer/train_ppo.py:36-49`

看它是为了理解：critic 不是输出 vocab logits，而是每个 token 一个 value。

代码摘录：

```python
class CriticModel(MiniMindForCausalLM):
    def __init__(self, params):
        super().__init__(params)
        self.value_head = nn.Linear(params.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = self.model.norm(outputs[0])
        values = self.value_head(hidden_states).squeeze(-1)
        return values
```

这段代码说明：

- critic 复用 MiniMind backbone。
- `value_head` 把 hidden state 映射成一个标量。
- 返回 shape 是 `[batch, seq]`，不是 `[batch, seq, vocab]`。

#### 源码证据 B：训练入口初始化四类组件

文件：`trainer/train_ppo.py:365-389`

看它是为了理解：PPO 同时需要 actor、reference、critic、reward model 和 rollout engine。

代码摘录：

```python
actor_model, tokenizer = init_model(lm_config, base_weight, device=args.device)
ref_model, _ = init_model(lm_config, base_weight, device=args.device)
ref_model = ref_model.eval().requires_grad_(False)

critic_model = CriticModel(lm_config)
critic_model.load_state_dict(state_dict, strict=False)
critic_model = critic_model.to(args.device)
reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)

rollout_engine = create_rollout_engine(
    engine_type=args.rollout_engine,
    policy_model=actor_model,
    tokenizer=tokenizer,
    device=args.device,
    autocast_ctx=autocast_ctx,
)
train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=(args.max_seq_len + args.max_gen_len), thinking_ratio=args.thinking_ratio)
```

这段代码说明：

- actor 是要训练的策略模型。
- reference 从同一个 base weight 初始化，但冻结。
- critic 从 base checkpoint 加载 backbone，再加 `value_head`。
- reward model 是外部打分器。
- rollout engine 持有当前 actor，用来生成 response。

#### 源码证据 C：reward model wrapper

文件：`trainer/trainer_utils.py:160-177`

看它是为了理解：源码中的 reward model 输出一个被截断到 `[-3, 3]` 的标量。

代码摘录：

```python
class LMForRewardModel:
    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
        self.model = self.model.to(device).eval()

    @torch.no_grad()
    def get_score(self, messages, response):
        eval_messages = [
            {"role": "user", "content": message_context},
            {"role": "assistant", "content": response}
        ]
        score = self.model.get_score(self.tokenizer, eval_messages)
        return max(min(score, 3.0), -3.0)
```

这段代码说明：

- reward model 不参与反向传播。
- 输入是整理后的用户上下文和 actor response。
- 输出是整条 response 的标量分数。

### 4.3 rollout 生成 response，并记录 old logprob

#### 源码证据 A：计算每个 completion token 的 logprob

文件：`trainer/rollout_engine.py:23-36`

看它是为了理解：PPO 需要保存生成当时的 `old_resp_logp`。

代码摘录：

```python
def compute_per_token_logps(model, input_ids: Tensor, n_keep: int, attention_mask: Optional[Tensor] = None) -> Tensor:
    logits = unwrapped(input_ids, attention_mask=attention_mask, logits_to_keep=n_keep + 1).logits[:, :-1, :]
    per_token_logps = []
    for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
        per_token_logps.append(
            torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1)
        )
    return torch.stack(per_token_logps)
```

这段代码说明：

- `n_keep` 是 completion 长度。
- logits 先去掉最后一个位置，因为最后一个 logits 没有下一个 token 标签。
- `input_ids[:, -n_keep:]` 是 completion token ids。
- `gather` 取出 actor 对自己生成 token 的 logprob。

注意：

```text
这里不是求 vocab 上所有 token 的概率，而是只取 response 真实 token 的 logprob。
```

#### 源码证据 B：Torch rollout 返回完整训练所需字段

文件：`trainer/rollout_engine.py:71-92`

看它是为了理解：rollout 的产物不只是文本，还包括 old logprob 和 completion ids。

代码摘录：

```python
output_ids = model.generate(
    input_ids=prompt_ids.repeat_interleave(num_generations, dim=0),
    attention_mask=attention_mask.repeat_interleave(num_generations, dim=0),
    max_new_tokens=max_new_tokens,
    do_sample=True,
    temperature=temperature,
)
prompt_len = prompt_ids.size(1)
completion_ids = output_ids[:, prompt_len:]
full_mask = (output_ids != self.tokenizer.pad_token_id).long()
per_token_logps = compute_per_token_logps(self.policy_model, output_ids, completion_ids.size(1), attention_mask=full_mask)
return RolloutResult(output_ids, completion_ids, per_token_logps, completions, ...)
```

这段代码说明：

- `output_ids` 是 prompt 和 response 拼接后的完整序列。
- `completion_ids` 只保留 response 区域。
- `per_token_logps` 对应 completion token。
- PPO 更新时不能丢掉这些 old logprob。

### 4.4 只在 response token 上建 mask

#### 源码证据：response 位置和 mask

文件：`trainer/train_ppo.py:117-128`

看它是为了理解：PPO loss 只作用在有效 response token，不作用在 prompt 和 padding。

代码摘录：

```python
full_mask = (gen_out != tokenizer.pad_token_id).long()
labels = gen_out[:, 1:].clone()
B = len(prompts)
resp_labels = completion_ids
resp_idx = torch.arange(resp_labels.size(1), device=gen_out.device).unsqueeze(0)
logp_pos = prompt_lens.unsqueeze(1) - 1 + resp_idx
resp_pad_mask = rollout_result.completion_mask.to(args.device).bool()
resp_lengths = resp_pad_mask.sum(dim=1)
eos_mask = resp_labels.eq(tokenizer.eos_token_id) & resp_pad_mask
has_eos = eos_mask.any(dim=1)
eos_pos = torch.argmax(eos_mask.int(), dim=1)
resp_lengths = torch.where(has_eos, eos_pos + 1, resp_lengths).long().clamp(min=1)
resp_policy_mask = ((resp_idx < resp_lengths.unsqueeze(1)) & resp_pad_mask).float()
resp_value_mask = resp_policy_mask.clone()
```

这段代码说明：

- `labels = gen_out[:, 1:]` 是 next-token 标签。
- 第 `j` 个 completion token 的 logits 位置是 `prompt_len - 1 + j`。
- `resp_policy_mask` 同时处理 padding 和 eos 后 token。
- policy loss 和 value loss 都用 response mask。

容易错的地方：

```text
completion_ids 的第 0 个 token，不对应 logits 的 prompt_len 位置；
它对应 logits[:, prompt_len - 1]，因为 logits 预测下一个 token。
```

### 4.5 reward 放在最后一个有效 response token

#### 源码证据 A：规则 reward + reward model

文件：`trainer/train_ppo.py:52-76`

看它是为了理解：MiniMind 的 reward 是整条 response 的标量，不是每个 token 的监督标签。

代码摘录：

```python
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

- reward 由启发式规则和 reward model 分数组成。
- reward 的 shape 是 `[batch]`。
- 它评价整条 response。

#### 源码证据 B：把 response reward 写到最后一个 token

文件：`trainer/train_ppo.py:136-138`

看它是为了理解：GAE 需要 token 级 reward，所以源码把整条回复分数放到末 token。

代码摘录：

```python
token_rewards = torch.zeros_like(old_resp_logp)
last_idx = resp_lengths - 1
token_rewards[torch.arange(B, device=args.device)[valid_resp], last_idx[valid_resp]] += rewards[valid_resp]
```

这段代码说明：

- `token_rewards` shape 和 `old_resp_logp` 一样，都是 `[batch, response_len]`。
- 除最后一个有效 response token 外，其它 token reward 为 0。
- 后续 GAE 会把末端 reward 从后往前传播到 earlier tokens。

### 4.6 GAE 计算 advantages 和 returns

#### 源码证据：从后往前递推 advantage

文件：`trainer/train_ppo.py:130-151`

看它是为了理解：PPO 的 policy loss 用 `advantages`，critic 的 value loss 用 `returns`。

代码摘录：

```python
values_seq = critic_for_rollout(input_ids=gen_out, attention_mask=full_mask)
old_resp_values = values_seq.gather(1, logp_pos) * resp_value_mask
...
gen_len = old_resp_values.size(1)
lastgaelam = torch.zeros(B, device=args.device)
advs_rev = []
for t in reversed(range(gen_len)):
    nv = old_resp_values[:, t + 1] if t < gen_len - 1 else 0.0
    delta = token_rewards[:, t] + args.gamma * nv - old_resp_values[:, t]
    lastgaelam = delta + args.gamma * args.lam * lastgaelam
    advs_rev.append(lastgaelam)
advantages = torch.stack(advs_rev[::-1], dim=1)
returns = advantages + old_resp_values

adv_mean = (advantages * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
adv_var = ((advantages - adv_mean) ** 2 * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
advantages = (advantages - adv_mean) * torch.rsqrt(adv_var + 1e-8) * resp_policy_mask
```

这段代码说明：

- critic 输出 `[batch, full_seq]`，再用 `logp_pos` 取 response 位置。
- GAE 从最后一个 response token 往前算。
- `returns = advantages + old_resp_values` 给 critic 作为目标。
- advantages 会在有效 response token 上标准化。

### 4.7 PPO update：ratio、clip、KL、value loss

#### 源码证据 A：重新 forward 得到 new logprob

文件：`trainer/train_ppo.py:171-181`

看它是为了理解：rollout 时保存 old logprob，更新时重新算 new logprob。

代码摘录：

```python
mb_values_seq = critic_unwrapped(input_ids=gen_out[inds], attention_mask=full_mask[inds])
mb_resp_values = mb_values_seq.gather(1, logp_pos[inds])

res = actor_unwrapped(input_ids=gen_out[inds], attention_mask=full_mask[inds])
mb_resp_logp = F.log_softmax(res.logits[:, :-1], dim=-1).gather(
    2, labels[inds].unsqueeze(-1)
).squeeze(-1).gather(1, logp_pos[inds])

log_ratio = mb_resp_logp - old_resp_logp[inds]
```

这段代码说明：

- `old_resp_logp` 不参与当前 actor forward。
- `mb_resp_logp` 来自当前 actor。
- 两者相减得到 `log_ratio`。

#### 源码证据 B：PPO clip policy loss

文件：`trainer/train_ppo.py:191-199`

看它是为了理解：README 公式里的 `ratio`、`clip`、`KL` 在源码里如何落地。

代码摘录：

```python
ratio = torch.exp(log_ratio)
clipfrac = ((((ratio - 1.0).abs() > args.clip_epsilon).float() * resp_policy_mask[inds]).sum()
            / resp_policy_mask[inds].sum().clamp(min=1))
kl_ref_penalty = ((torch.exp(ref_resp_logp[inds] - mb_resp_logp) - (ref_resp_logp[inds] - mb_resp_logp) - 1.0)
                  * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
policy_loss = ((torch.max(-advantages[inds] * ratio,
                          -advantages[inds] * torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon))
               * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
               + args.kl_coef * kl_ref_penalty)
```

这段代码说明：

- `ratio` 比较新旧 actor 对同一批 response token 的概率。
- `clipfrac` 统计有多少有效 token 超过裁剪范围。
- `kl_ref_penalty` 约束 actor 不要离 reference 太远。
- `policy_loss` 只在 `resp_policy_mask` 有效位置求平均。

#### 源码证据 C：critic value loss

文件：`trainer/train_ppo.py:200-203`

看它是为了理解：critic 也用 clip，避免 value 估计一步跳太大。

代码摘录：

```python
value_loss = 0.5 * (torch.max((mb_resp_values - returns[inds]) ** 2,
                              (torch.clamp(mb_resp_values, old_resp_values[inds] - args.cliprange_value,
                                           old_resp_values[inds] + args.cliprange_value) - returns[inds]) ** 2)
                    * resp_value_mask[inds]).sum() / resp_value_mask[inds].sum().clamp(min=1)
```

这段代码说明：

- `returns` 是 critic 的训练目标。
- `old_resp_values` 用来裁剪 critic 更新幅度。
- value loss 和 policy loss 使用同一片 response 区域。

### 4.8 更新 rollout engine 的 policy

#### 源码证据：训练后同步 actor

文件：`trainer/train_ppo.py:226-246`

看它是为了理解：PPO 是 on-policy，所以 rollout engine 必须定期拿到最新 actor。

代码摘录：

```python
if grad_accum_step % args.accumulation_steps == 0:
    clip_grad_norm_(actor_model.parameters(), args.grad_clip)
    clip_grad_norm_(critic_model.parameters(), args.grad_clip)
    actor_optimizer.step()
    critic_optimizer.step()
    actor_scheduler.step()
    critic_scheduler.step()
    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()

if step % args.save_interval == 0 or step == iters:
    rollout_engine.update_policy(actor_model)
```

这段代码说明：

- actor 和 critic 都会更新。
- rollout engine 不能永远用旧 actor。
- 如果使用 SGLang，`update_policy` 会把新权重同步到推理服务。

<a id="l23-must-write"></a>
## 5. 本节必须会写 / 暂时不要求

### 5.1 必须会写

本节的教学版核心不是完整训练脚本，而是下面 5 个小函数：

```python
def response_logp_positions(prompt_lens: torch.Tensor, response_len: int) -> torch.Tensor:
    ...

def masked_gae(
    token_rewards: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...

def ppo_policy_loss(
    new_logps: torch.Tensor,
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_epsilon: float,
) -> torch.Tensor:
    ...

def reference_kl_penalty(
    policy_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    ...

def clipped_value_loss(
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    cliprange_value: float,
) -> torch.Tensor:
    ...
```

它们分别对应：

```text
train_ppo.py:121-122
train_ppo.py:140-151
train_ppo.py:180-199
train_ppo.py:194-195
train_ppo.py:200-203
```

### 5.2 暂时不要求

本节先不要求手写：

```text
真实 reward model 加载；
SGLang rollout engine；
DDP 同步和 early_stop_kl 的通信细节；
checkpoint、scheduler、wandb/swanlab；
真实长文本采样质量；
reward hacking 的完整治理。
```

这些是工程外围或更大的 RLHF 话题。第 23 课先把核心张量链路讲清楚。

<a id="l23-handwrite"></a>
## 6. 手写模块

本节对应教学版建议文件：

```text
course/impl/core/ppo.py
```

### 6.1 补 `response_logp_positions`

接口：

```python
def response_logp_positions(prompt_lens: torch.Tensor, response_len: int) -> torch.Tensor:
    ...
```

对齐源码：

```text
trainer/train_ppo.py:121-122
```

行为要求：

```text
输入:
  prompt_lens: [batch]
  response_len: int

输出:
  logp_pos: [batch, response_len]

规则:
  第 j 个 response token 对应 logits[:, prompt_len - 1 + j]
```

### 6.2 补 `masked_gae`

接口：

```python
def masked_gae(
    token_rewards: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

对齐源码：

```text
trainer/train_ppo.py:140-151
```

行为要求：

```text
输入:
  token_rewards: [batch, response_len]
  values:        [batch, response_len]
  mask:          [batch, response_len]

输出:
  advantages: [batch, response_len]
  returns:    [batch, response_len]
```

注意：

```text
advantages 要只在 mask 区域标准化；
padding 位置最终应为 0。
```

### 6.3 补 `ppo_policy_loss`

接口：

```python
def ppo_policy_loss(
    new_logps: torch.Tensor,
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_epsilon: float,
) -> torch.Tensor:
    ...
```

对齐源码：

```text
trainer/train_ppo.py:180-199
```

行为要求：

```text
log_ratio = new_logps - old_logps
ratio = exp(log_ratio)
loss = masked mean(max(-advantages * ratio, -advantages * clipped_ratio))
```

这个函数只负责 PPO clipped surrogate。MiniMind 的完整 actor loss 还要加：

```text
kl_coef * reference_kl_penalty(...)
```

### 6.4 补 `reference_kl_penalty`

接口：

```python
def reference_kl_penalty(
    policy_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    ...
```

对齐源码：

```text
trainer/train_ppo.py:194-195
```

行为要求：

```text
delta = ref_logps - policy_logps
penalty = exp(delta) - delta - 1
loss = masked mean(penalty)
```

对应公式：

$$
\Delta_{b,t}
=
\log \pi_{\mathrm{ref}}(a_{b,t} \mid s_{b,t})
-
\log \pi_{\theta}(a_{b,t} \mid s_{b,t})
$$

$$
\mathrm{reference\_kl\_penalty}
=
\frac{
\sum_{b,t} m_{b,t}
\left[
\exp(\Delta_{b,t}) - \Delta_{b,t} - 1
\right]
}{
\sum_{b,t} m_{b,t}
}
$$

理解重点：

```text
这不是 DPO 的 reference logratio；
这是 RLHF PPO 里防止 actor 偏离 frozen reference policy 的约束项。
```

### 6.5 补 `clipped_value_loss`

接口：

```python
def clipped_value_loss(
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    cliprange_value: float,
) -> torch.Tensor:
    ...
```

对齐源码：

```text
trainer/train_ppo.py:200-203
```

行为要求：

```text
unclipped = (values - returns) ** 2
clipped_values = clamp(values, old_values - cliprange_value, old_values + cliprange_value)
clipped = (clipped_values - returns) ** 2
loss = 0.5 * masked mean(max(unclipped, clipped))
```

<a id="l23-experiment"></a>
## 7. 实验验证

本节实验不加载真实 reward model，也不启动 SGLang。它用随机 tiny MiniMind 模型构造一批 synthetic rollout，验证 PPO 核心张量链路。

实验文件：

```text
course/labs/trace_ppo_batch_flow.py
```

命令：

```bash
cd /home/sun/minimind
python course/labs/trace_ppo_batch_flow.py
```

这个实验验证：

```text
1. completion token 的 old logprob shape 是 [batch, response_len]。
2. logp_pos 使用 prompt_len - 1 + response_idx。
3. response mask 会去掉 padding/eos 后 token。
4. 标量 reward 会写到最后一个有效 response token。
5. GAE 输出 advantages/returns。
6. ratio、clipfrac、KL penalty、policy loss、value loss 都只在 response mask 上计算。
```

记录输出：

```text
gen_out.shape =
completion_ids.shape =
old_resp_logp.shape =
logp_pos =
resp_policy_mask =
advantages.shape =
returns.shape =
masked_adv_mean =
ratio_min =
ratio_max =
approx_kl =
policy_loss =
value_loss =
```

输出含义：

```text
ratio_min/ratio_max 接近 1，说明 old 和 new actor 此时还没真正更新；
approx_kl 接近 0，说明新旧策略几乎一致；
masked_adv_mean 接近 0，说明 advantages 在有效 response token 上做了标准化；
policy_loss 和 value_loss 是可反向传播的标量。
```

如果你后续补了 `course/impl/core/ppo.py`，可以新增对齐测试：

```text
course/impl/tests/test_ppo.py
```

测试内容：

```text
response_logp_positions 对齐 train_ppo.py 的 logp_pos；
masked_gae 对齐 train_ppo.py 的 GAE 递推；
ppo_policy_loss 对齐 train_ppo.py 的 clip policy loss；
reference_kl_penalty 对齐 train_ppo.py 的 reference KL penalty；
clipped_value_loss 对齐 train_ppo.py 的 value loss。
```

<a id="l23-stage-assembly"></a>
## 8. 阶段组装

PPO/GRPO 阶段的教学版最小闭环可以这样组装：

```text
CoursePromptDataset
-> actor.generate 得到 completion_ids
-> old_logps = response token logprob
-> reward_fn 给每条 response 一个标量 reward
-> critic 得到 response values
-> masked_gae 得到 advantages/returns
-> ppo_policy_loss 更新 actor
-> reference_kl_penalty 约束 actor 不要偏离 reference
-> clipped_value_loss 更新 critic
```

### 8.1 本节验收命令

```bash
cd /home/sun/minimind
python course/labs/trace_ppo_batch_flow.py
```

如果环境没有 GPU，也可以强制 CPU：

```bash
python course/labs/trace_ppo_batch_flow.py --device cpu
```

### 8.2 源码差异

教学版先省略：

```text
SGLang 服务；
真实 reward model；
多轮 rollout 采样；
DDP 通信；
checkpoint/resume；
大 batch 和梯度累积调度。
```

这些省略不改变 PPO 核心：

```text
old_logp -> new_logp -> ratio -> advantage -> clipped policy loss
reference_logp -> reference KL penalty
```

### 8.3 Portfolio 记录

完成本节后，可以在 `course/portfolio/experiments.md` 记录：

```text
PPO batch trace:
- 构造 synthetic prompt/completion。
- 用 tiny actor 计算 old response logprob。
- 用 tiny critic 计算 response value。
- 把 scalar reward 写到最后一个有效 response token。
- 用 GAE 得到 advantages/returns。
- 手算 ratio、clip policy loss、reference KL penalty 和 clipped value loss。
```

在 `course/notes/mistakes.md` 记录容易错的点：

```text
1. 把 old_logp 和 new_logp 混成同一个东西。
2. 忘记 completion 第 0 个 token 对应 logits 的 prompt_len - 1 位置。
3. 把 prompt token 也算进 policy loss。
4. 对 padding token 做 advantage 标准化。
5. 忘记 actor 更新后同步 rollout engine。
```

<a id="l23-check"></a>
## 9. 本节检查

1. 在 LLM PPO 里，`state`、`action`、`trajectory`、`reward` 分别对应什么？
2. PPO 为什么不能像 DPO 一样只读静态 chosen/rejected 数据？
3. policy gradient 里的 `advantage` 决定什么更新方向？
4. `old_resp_logp` 为什么必须在 rollout 时保存？
5. `ratio = exp(new_logp - old_logp)` 大于 1 代表什么？
6. PPO clip 对 `A_t > 0` 和 `A_t < 0` 分别限制了什么？
7. `advantages` 和 `returns` 分别给哪个 loss 使用？
8. MiniMind 的 reference KL penalty 和 DPO 的 reference logratio 有什么区别？
9. 第 `j` 个 completion token 为什么对应 `prompt_len - 1 + j` 的 logits 位置？
10. actor 更新后为什么要调用 `rollout_engine.update_policy(actor_model)`？

<a id="l23-next"></a>
## 10. 下一课

第 24 课进入 [GRPO 与 CISPO](24_grpo_cispo.md)。

下一课要解决：

```text
GRPO 和 PPO 的核心差异是什么；
为什么 GRPO 可以减少对 critic/value model 的依赖；
MiniMind 的 train_grpo.py 如何组织 multiple generations；
group-relative reward/advantage 如何进入 loss；
CISPO 和普通 GRPO 的 loss_type 在源码里如何分叉。
```
