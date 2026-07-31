# 03-进阶 / 02 - DPO 与强化学习

> **TL;DR**：minimind 把 DPO / PPO / GRPO / CISPO / Agentic RL 全部从 0 实现了一遍，且统一在一个框架下：`J_PO = E[ f(r_t)·g(A_t) - h(KL_t) ]`。本篇带你跑通 4 个训练脚本 + 一个 Agentic RL，并讲清它们的差异。
> **心理学技巧**：**统一视角降低认知负荷** —— 与其记 5 个算法，不如记 1 个公式和 3 个组件的不同实例化。心理学研究表明，把离散信息组织进统一框架能显著提升记忆和迁移能力。

## ✅ 你将能

- 说出 RLHF 和 RLAIF 的区别
- 写出 PO 算法的统一公式 `J_PO = E[f(r_t)·g(A_t) - h(KL_t)]`
- 解释 DPO / PPO / GRPO / CISPO 在三个组件上的不同实例化
- 跑通 minimind 的 4 个 RL 训练脚本
- 知道 Agentic RL 与普通 RL 的差异（多轮、延迟奖励、工具执行）
- 用断点续训从中断处继续训练

---

## 步骤 0：先理解 RLHF vs RLAIF（5 分钟）

| | RLHF | RLAIF |
|---|---|---|
| **反馈来源** | 人类标注偏好 | AI / 规则 / 环境 |
| **典型算法** | DPO | PPO / GRPO / CISPO |
| **数据形态** | 静态偏好对（chosen/rejected） | 在线 rollout 后打分 |
| **成本** | 高、扩展性差 | 低、可海量生成 |
| **能改的能力** | 偏好/安全对齐 | 推理、工具使用、可验证任务 |
| **minimind 脚本** | `train_dpo.py` | `train_ppo.py` / `train_grpo.py` / `train_agent.py` |

> 参见 `README.md:891-907`。简单说：DPO 用人类标好的"好/坏答案对"训练；PPO/GRPO 用 AI/规则在模型生成的回答上打分。RLAIF 更现代、更可扩展，也是 DeepSeek-R1 等推理模型的核心。

---

## 步骤 1：PO 算法的统一视角（核心）

`README.md:910-938` 提出所有 Policy Optimization 算法本质都在优化同一个期望：

$$\mathcal{J}_{PO} = \mathbb{E}_{q \sim P(Q),\, o \sim \pi(O|q)} \left[ \underbrace{f(r_t)}_{\text{策略项}} \cdot \underbrace{g(A_t)}_{\text{优势项}} - \underbrace{h(\text{KL}_t)}_{\text{正则项}} \right]$$

训练时最小化 `L_PO = -J_PO`。三个核心组件：

| 组件 | 含义 | 通俗解释 |
|---|---|---|
| **策略项** `f(r_t)` | 怎么用概率比 `r_t = π_θ/π_ref` | 新策略比旧策略"探索"了多少，要不要信这个梯度 |
| **优势项** `g(A_t)` | 怎么算优势 `A_t` | 这个 token 比平均好还是坏，好的鼓励、坏的抑制 |
| **正则项** `h(KL_t)` | 怎么约束偏离 | 防止跑偏太远 / 防止管太死 |

**不同算法只是对这三个组件的不同实例化**。看最终的对照表（来自 `README.md:1269-1274`）：

| 算法 | 策略项 `f(r_t)` | 优势项 `g(A_t)` | 正则项 `h(KL_t)` | 训练模型数 |
|---|---|---|---|---|
| **DPO** | `log r_w - log r_l` | 无显式优势项 | 隐含在 `β` 中 | 1（前向 2） |
| **PPO** | `min(r, clip(r))` | `R - V(s)` | `β·E[KL]` | 2（actor + critic） |
| **GRPO** | `min(r, clip(r))` | `(R - μ)/σ` | `β·KL_t` | 1 |
| **CISPO** | `clip(r, 0, ε_max)·A_t·log π_θ` | `(R - μ)/σ` | `β·KL_t` | 1 |

记住这张表，下面 4 个算法都只是在回答："三件事怎么设计？"

---

## 步骤 2：DPO —— 直接偏好优化（最简单的 RL）

### 2.1 原理

DPO（[Rafailov et al., 2023]）从 PPO 的 KL 约束目标**数学推导**出一个解析解，**直接最大化 chosen/rejected 的对数概率差**，无需训练 reward model 和 critic。

损失（`README.md:951`）：

$$\mathcal{L}_{DPO} = -\mathbb{E}\left[\log \sigma\left(\beta \left[\log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right]\right)\right]$$

- **策略项**：`log r_w - log r_l`（chosen vs rejected 概率比之差）
- **优势项**：无显式（通过偏好对比隐式体现）
- **正则项**：隐含在 `β` 中

### 2.2 数据格式

`dataset/dpo.jsonl`（见 `dataset/lm_dataset.py:122` 的 `DPODataset`）：

```json
{
  "chosen": [
    {"role": "user", "content": "什么是光合作用？"},
    {"role": "assistant", "content": "光合作用是植物利用光能..."}
  ],
  "rejected": [
    {"role": "user", "content": "什么是光合作用？"},
    {"role": "assistant", "content": "不知道，问别人去。"}
  ]
}
```

同一问题，好答案 vs 坏答案。

### 2.3 关键代码（`trainer/train_dpo.py`）

DPO 的核心是计算 chosen/rejected 的对数概率之差，看 `train_dpo.py:34-50`：

```python
def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    # 在 chosen/rejected 上各自求和（前半是 chosen，后半是 rejected）
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)

    chosen_ref, reject_ref = ref_log_probs[:B//2], ref_log_probs[B//2:]
    chosen_policy, reject_policy = policy_log_probs[:B//2], policy_log_probs[B//2:]

    pi_logratios = chosen_policy - reject_policy       # π 部分
    ref_logratios = chosen_ref - reject_ref            # ref 部分
    logits = pi_logratios - ref_logratios             # 减去 ref 是关键
    loss = -F.logsigmoid(beta * logits)                # -log σ(β·logits)
    return loss.mean()
```

**关键设计**：
1. 一次 forward 同时跑 chosen + rejected（cat 在 batch 维，`train_dpo.py:65-67`）
2. 需要 ref_model 计算 ref log_probs，但 ref 不反传梯度（`train_dpo.py:74` 用 `torch.no_grad()`）
3. `mask` 只在 assistant 部分（`DPODataset.generate_loss_mask`，`dataset/lm_dataset.py:176-192`）

### 2.4 训练

```bash
cd trainer
python train_dpo.py \
  --from_weight full_sft \
  --save_weight dpo \
  --data_path ../dataset/dpo.jsonl \
  --epochs 1 \
  --batch_size 4 \
  --learning_rate 4e-8 \
  --beta 0.15
```

> ⚠️ **学习率必须极小**（4e-8）。DPO 是在 SFT 之上做偏好微调，学习率稍大就会把 SFT 学到的能力打没。`train_dpo.py:137` 注释里也写了"建议<=5e-8避免遗忘"。

✅ **验证**：训练每隔 `save_interval`（默认 100 步）产出 `out/dpo_768.pth`。

推理：

```bash
python eval_llm.py --load_from ./model --weight dpo
```

### 2.5 DPO 的局限

- **off-policy**：用静态偏好数据集，不做在线探索
- 只能学"偏好/安全"，对"能不能做对题"的智力提升有限
- 需要人类标注成本高

这就是为什么有了下面的 RLAIF 系列。

---

## 步骤 3：Rollout 引擎 —— RLAIF 的基础设施

PPO/GRPO/CISPO/Agentic RL 都需要先让模型"采样生成"回答，再打分。这部分逻辑被解耦到 `trainer/rollout_engine.py`，可插拔替换：

```
                ┌───────────── RolloutEngine ──────────────┐
                │                                          │
  prompt_ids →  │  → model.generate / SGLang HTTP → output_ids
                │  → compute_per_token_logps → per_token_logps
                │  → batch_decode → completions (text)      │
                └──────────────────────────────────────────┘
                                  ↓
                          RolloutResult
                                  ↓
                    calculate_rewards → advantages
                                  ↓
                        policy gradient update
```

### 3.1 两种引擎

`trainer/rollout_engine.py:209-224` 的工厂：

```python
def create_rollout_engine(engine_type="torch", policy_model, tokenizer, device, ...):
    if engine_type == "torch":
        return TorchRolloutEngine(...)        # 用 model.generate
    elif engine_type == "sglang":
        return SGLangRolloutEngine(...)       # 用 SGLang HTTP API
```

| 引擎 | 适用场景 | 启动方式 |
|---|---|---|
| `torch` | 单卡训练、调试、学习 | 默认，无需额外启动 |
| `sglang` | 大批量生成、生产 | 需先启 SGLang server |

启动 SGLang（`trainer/rollout_engine.py:1-3`）：

```bash
python -m sglang.launch_server --model-path ./minimind-3 \
  --attention-backend triton --host 0.0.0.0 --port 8998
```

> SGLang 是高性能推理引擎，比原生 PyTorch `generate` 快 3-5 倍，但需要单独服务进程。训练时通过 `--rollout_engine sglang` 切换。

### 3.2 关键函数 `compute_per_token_logps`

`trainer/rollout_engine.py:24-36`：给一段已生成的 `output_ids`，算出每个 token 在当前策略下的 log 概率。这就是 `r_t = π_θ / π_ref` 里 `π_θ` 的来源。

```python
def compute_per_token_logps(model, input_ids, n_keep, attention_mask):
    logits = unwrapped(input_ids, ..., logits_to_keep=n_keep + 1).logits[:, :-1, :]
    per_token_logps = torch.gather(logits.log_softmax(-1), 1, ids.unsqueeze(1)).squeeze(1)
    return torch.stack(per_token_logps)
```

### 3.3 权重同步 `update_policy`

训练若干步后，actor 模型权重变了，rollout 引擎里的"采样模型"也要更新。`rollout_engine.py:94-95`（torch 引擎）只是简单地替换引用：

```python
def update_policy(self, model):
    self.policy_model = model   # 训练循环每 N 步调用一次
```

SGLang 引擎则把新权重写到磁盘并触发服务器 reload（`rollout_engine.py:175-194`）。

---

## 步骤 4：PPO —— 经典 RLHF 算法

### 4.1 原理

PPO（[Schulman et al., 2017]）是 RLHF 的经典算法，三件事的标准答案：

| 组件 | PPO 的答案 |
|---|---|
| 策略项 | `min(r, clip(r, 1-ε, 1+ε))`（对称 clip） |
| 优势项 | `A = R - V(s)`（需 Critic 网络估 V） |
| 正则项 | `β·E[KL]` |

需要训练 2 个网络：**Actor**（策略模型）+ **Critic**（价值模型）。

### 4.2 minimind 的 PPO 实现（`trainer/train_ppo.py`）

#### 4.2.1 Critic 模型

`train_ppo.py:36-48` 自定义一个 Critic（继承自 `MiniMindForCausalLM`，把 `lm_head` 换成输出单一价值的 `value_head`）：

```python
class CriticModel(MiniMindForCausalLM):
    def __init__(self, params):
        super().__init__(params)
        self.value_head = nn.Linear(params.hidden_size, 1)   # 输出 V(s)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = self.model.norm(outputs[0])
        values = self.value_head(hidden_states).squeeze(-1)   # [B, L]
        return values
```

#### 4.2.2 奖励计算

`train_ppo.py:51-75` 的 `calculate_rewards`，奖励是**多源混合**：

```python
rewards[i] += 0.5 if 20 <= len(response) <= 800 else -0.5    # 长度奖励
if 'Ỡ' in response:
    thinking, answer = response.split('Ỡ', 1)
    rewards[i] += 1.0 if 20 <= len(thinking) <= 300 else -0.5  # 思考长度
    rewards[i] += 0.25 if response.count('Ỡ') == 1 else -0.25 # 思考闭合
rewards[i] -= rep_penalty(answer)                              # 重复惩罚
score = reward_model.get_score(messages, answer)              # 奖励模型分
rewards += reward_model_scores
```

> 默认用 `InternLM2-1.8B-Reward`（`train_ppo.py:391`，路径通过 `--reward_model_path` 指定）。需放在项目同级目录。

#### 4.2.3 GAE（广义优势估计）

`train_ppo.py:139-150` 实现 GAE：

```python
gen_len = old_resp_values.size(1); lastgaelam = torch.zeros(B)
advs_rev = []
for t in reversed(range(gen_len)):
    nv = old_resp_values[:, t + 1] if t < gen_len - 1 else 0.0
    delta = token_rewards[:, t] + args.gamma * nv - old_resp_values[:, t]
    lastgaelam = delta + args.gamma * args.lam * lastgaelam
    advs_rev.append(lastgaelam)
advantages = torch.stack(advs_rev[::-1], dim=1)
returns = advantages + old_resp_values
# 优势归一化
advantages = (advantages - adv_mean) * rsqrt(adv_var + 1e-8) * mask
```

#### 4.2.4 PPO loss（`train_ppo.py:204-216`）

```python
ratio = torch.exp(log_ratio)                                    # r_t
policy_loss = torch.max(-advantages * ratio,
                        -advantages * torch.clamp(ratio, 1-ε, 1+ε))  # 对称 clip
value_loss = 0.5 * torch.max((mb_values - returns)**2, ...)    # value 也 clip
loss = policy_loss + args.vf_coef * value_loss + args.kl_coef * kl_ref_penalty
```

#### 4.2.5 早停机制

`train_ppo.py:201-202`：

```python
if approx_kl_val > args.early_stop_kl:    # 默认 0.25
    stop_ppo = True
```

KL 太大说明策略已经跑偏，强行更新会破坏模型。这是个工程保险。

### 4.3 训练

```bash
cd trainer
python train_ppo.py \
  --from_weight full_sft \
  --save_weight ppo_actor \
  --data_path ../dataset/rlaif.jsonl \
  --batch_size 2 \
  --learning_rate 3e-7 \
  --critic_learning_rate 5e-7 \
  --thinking_ratio 0.9 \
  --rollout_engine torch
```

关键参数（`train_ppo.py:309-353`）：

| 参数 | 默认 | 含义 |
|---|---|---|
| `--learning_rate` | 3e-7 | Actor 学习率（远小于 SFT） |
| `--critic_learning_rate` | 5e-7 | Critic 学得快一点（先让 V 收敛） |
| `--clip_epsilon` | 0.2 | PPO clip 范围 |
| `--vf_coef` | 0.5 | value loss 权重 |
| `--kl_coef` | 0.02 | KL 惩罚系数 |
| `--gamma` / `--lam` | 1.0 / 0.95 | GAE 折扣因子 |
| `--ppo_update_iters` | 2 | 同一批 rollout 重复几次更新 |
| `--early_stop_kl` | 0.25 | KL 阈值，超过就早停 |
| `--thinking_ratio` | 0.9 | 90% 概率开启 thinking 注入 |

✅ **验证**：训练每隔 `save_interval`（默认 10 步）产出 `out/ppo_actor_768.pth`，日志会打印 Reward / KL / AvgLen。

### 4.4 PPO 的痛点

`README.md:1118` 总结：
- **reward 提升缓慢**：Critic 收敛慢拖累 Actor，两者相互依赖
- **显存 1.5-2 倍**：要同时存 Actor + Critic + Ref + Reward

这就是 GRPO 的动机。

---

## 步骤 5：GRPO —— DeepSeek-R1 用的算法

### 5.1 原理

GRPO（[Shao et al., 2024]，DeepSeekMath）的核心创新：**用组内相对优势替代 Critic**。同一问题生成 N 个回答，组内平均奖励作为 baseline，高于平均的鼓励、低于平均的抑制，无需训练 critic 网络。

| 组件 | GRPO 的答案 |
|---|---|
| 策略项 | `min(r, clip(r, 1-ε, 1+ε))`（同 PPO） |
| 优势项 | `A = (R - μ_group) / σ_group`（组内归一化，**无需 Critic**） |
| 正则项 | `β·KL_t`（token 级 KL） |

### 5.2 关键代码（`trainer/train_grpo.py`）

#### 5.2.1 分组生成

`train_grpo.py:80-86` 用 `num_generations=6` 一次生成 6 个回答：

```python
rollout_result = rollout_engine.rollout(
    prompt_ids=prompt_inputs["input_ids"],
    attention_mask=prompt_inputs["attention_mask"],
    num_generations=args.num_generations,   # 默认 6
    max_new_tokens=args.max_gen_len,
    temperature=0.8,
)
```

#### 5.2.2 组内归一化优势（`train_grpo.py:121-124`）

```python
grouped_rewards = rewards.view(-1, args.num_generations)     # [B, num_gen]
mean_r = grouped_rewards.mean(dim=1).repeat_interleave(num_gen)   # 组内均值
std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(num_gen)
advantages = (rewards - mean_r) / (std_r + 1e-4)             # 组内归一化
```

这就是"用组内均值代替 V(s)"的核心。**没有 critic 网络，省一半显存。**

#### 5.2.3 GRPO loss（`train_grpo.py:139-143`）

```python
clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
per_token_loss1 = ratio * advantages.unsqueeze(1)
per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)
```

### 5.3 训练

```bash
cd trainer
python train_grpo.py \
  --from_weight full_sft \
  --save_weight grpo \
  --data_path ../dataset/rlaif.jsonl \
  --batch_size 2 \
  --num_generations 6 \
  --learning_rate 3e-7 \
  --loss_type grpo
```

✅ **验证**：产出 `out/grpo_768.pth`。日志会打印 `Reward / KL_ref / Adv Std / Avg Response Len`。

### 5.4 GRPO 的痛点

`README.md:1135`：**退化组（Degenerate Groups）** —— 如果某个问题 6 个回答奖励都差不多，组内 σ≈0，学习信号接近 0。超小模型上尤其明显。

---

## 步骤 6：CISPO —— 稳定版 GRPO

### 6.1 原理

CISPO（[HuggingFace, 2025]）解决了 PPO/GRPO 的一个长期痛点：**ratio 被 clip 后梯度也被硬截断**。

`README.md:1158-1170` 的核心思想：把策略项从 `min(r·A, clip(r)·A)` 改写成 `clip(r, 0, ε_max)·A·log π_θ`。这样 ratio 即使被截断（作为常数权重），梯度路径仍然通过 `log π_θ` 保留。

| 组件 | CISPO 的答案 |
|---|---|
| 策略项 | `clip(r, 0, ε_max)·A_t·log π_θ`（ratio 只作裁剪权重，**梯度走 log π_θ**） |
| 优势项 | `(R - μ)/σ`（同 GRPO） |
| 正则项 | `β·KL_t`（同 GRPO） |

$$\mathcal{L}_{CISPO} = -\mathbb{E}\left[\min(r_t, \varepsilon_{max}) \cdot A_t \cdot \log \pi_\theta(a_t|s) - \beta \cdot \text{KL}_t\right]$$

### 6.2 关键代码（`train_grpo.py:135-137`）

CISPO 是 GRPO 的 loss 变体，同一个脚本切换：

```python
if args.loss_type == "cispo":
    clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()   # ratio 不传梯度
    per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)
    #                                  ↑ log π_θ                      ← 梯度从这里走
```

**对比 GRPO**：

```python
else:  # grpo
    clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
    per_token_loss1 = ratio * advantages.unsqueeze(1)
    per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
    per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)
```

### 6.3 训练

```bash
cd trainer
python train_grpo.py \
  --from_weight full_sft \
  --save_weight grpo \
  --loss_type cispo \
  --epsilon_high 5.0 \
  --beta 0.1
```

只需把 `--loss_type` 从 `grpo` 改成 `cispo`，其他参数沿用 GRPO 配置。

---

## 步骤 7：Agentic RL —— 多轮 Tool-Use 场景

### 7.1 与普通 RL 的区别

| | 普通 GRPO/CISPO | Agentic RL |
|---|---|---|
| **轨迹** | 单轮 `prompt → answer` | 多轮 `prompt → tool_call → observation → ... → final_answer` |
| **奖励** | 即时（对回答打分） | 延迟（对整条轨迹打分） |
| **数据** | `rlaif.jsonl` | `agent_rl.jsonl` / `agent_rl_math.jsonl`（含 `gt`） |
| **核心问题** | 答得好不好 | 会不会用工具 + 用得对不对 |

数据格式（`dataset/lm_dataset.py:226-252` 的 `AgentRLDataset`）：

```json
{
  "conversations": [...],   // 最后一个 assistant 留空
  "gt": [59, "20.5"]         // ground truth，用于 RLVR 校验
}
```

> **RLVR（RL from Verifiable Rewards）**：奖励信号来自可验证规则（如数学题最终答案是否等于 gt），不是 AI 主观打分。

### 7.2 多轮 Rollout

`trainer/train_agent.py:98-157` 的 `rollout_single` 是核心：

```python
for turn in range(max_turns):                                    # 最多 3 轮
    context = tokenizer.apply_chat_template(messages, ..., tools=tools, open_thinking=open_thinking)
    rollout_result = rollout_engine.rollout(prompt_ids=..., max_new_tokens=...)
    new_text = rollout_result.completions[0]
    all_outputs.append(new_text)
    calls = parse_tool_calls(new_text)                           # 解析 `` 里的 JSON
    if not calls: break                                          # 模型直答，结束
    messages.append({"role": "assistant", "content": new_text})
    for call in calls:
        result = execute_tool(name, raw)                        # 执行工具
        messages.append({"role": "tool", "content": result_str})  # 把结果拼回
    # 把 observation token 也拼进 response_ids，但 mask=0（不学工具结果）
```

**关键技巧**：observation（工具返回结果）的 token 会被拼进 `response_ids` 但 `response_mask=0`（`train_agent.py:150-152`），策略梯度不在这部分算 loss。

### 7.3 工具集

`train_agent.py:40-47` 定义了 6 个工具（calculate_math / unit_converter / get_current_weather / get_current_time / get_exchange_rate / translate_text），用 `MOCK_RESULTS`（`train_agent.py:57-64`）模拟执行。

### 7.4 奖励组成

`train_agent.py:188-239` 的 `calculate_rewards`：

$$R(\tau) = R_{\text{answer}} + R_{\text{tool}} + R_{\text{format}} + R_{\text{rm}} - R_{\text{unfinished}}$$

| 项 | 含义 |
|---|---|
| `R_tool` | 工具调用合法性（`tool_gap = abs(有效调用数 - len(gt))`，`train_agent.py:230-231`） |
| `R_gt` | gt 命中数（`2.5 × verified / len(gt)`，`train_agent.py:235`）|
| `R_format` | 思考闭合、长度合理性 |
| `R_rm` | Reward Model 分（无工具调用时才用） |
| `R_unfinished` | 超过 max_turns 未完成扣 0.5 |

### 7.5 训练

```bash
cd trainer
python train_agent.py \
  --from_weight full_sft \
  --save_weight agent \
  --data_path ../dataset/agent_rl.jsonl \
  --batch_size 2 \
  --num_generations 4 \
  --max_turns 3 \
  --thinking_ratio 0.1 \
  --loss_type cispo
```

数学专项训练：

```bash
python train_agent.py \
  --data_path ../dataset/agent_rl_math.jsonl \
  --save_weight agent_math \
  --thinking_ratio 0.5
```

✅ **验证**：产出 `out/agent_768.pth`。日志会打印 `Reward / KL / GrpStd / AdvStd / AvgLen`。

### 7.6 SGLang 加速

`train_agent.py` 的 rollout 是串行的，每个样本多轮生成 + 工具执行很慢。可用 SGLang 加速：

```bash
# 启动 SGLang server
python -m sglang.launch_server --model-path ./minimind-3 \
  --attention-backend triton --host 0.0.0.0 --port 8998

# 训练时切换引擎
cd trainer
python train_agent.py \
  --rollout_engine sglang \
  --sglang_base_url http://localhost:8998 \
  --sglang_shared_path ./sglang_ckpt_agent \
  --data_path ../dataset/agent_rl_math.jsonl
```

---

## 步骤 8：断点续训

所有 4 个脚本都支持 `--from_resume 1` 自动续训：

```bash
# 中断后重启
python train_grpo.py --from_resume 1
```

机制（以 `train_grpo.py:300-307` 为例）：

```python
ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume == 1 else None
# ...
if ckp_data:
    model.load_state_dict(ckp_data['model'])
    optimizer.load_state_dict(ckp_data['optimizer'])
    scheduler.load_state_dict(ckp_data['scheduler'])
    start_epoch = ckp_data['epoch']
    start_step = ckp_data.get('step', 0)
```

`SkipBatchSampler`（`trainer_utils.py`）会跳过已训的 step，避免重复。

PPO 还会恢复 critic 和 scheduler（`train_ppo.py:415-423`）：

```python
critic_model.load_state_dict(ckp_data['critic_model'])
critic_optimizer.load_state_dict(ckp_data['critic_optimizer'])
critic_scheduler.load_state_dict(ckp_data['critic_scheduler'])
```

✅ **验证**：日志开头会打印 `Epoch [X/Y]: 跳过前 N 个 step，从 step N+1 开始`。

---

## 🧯 踩坑提示

### Q1：PPO/GRPO 训练不收敛，reward 一直降
最常见是**奖励信号问题**：检查 reward_model 路径（`--reward_model_path`，默认 `../../internlm2-1_8b-reward`，应在项目同级目录），或用 `--debug_mode` 打印每个样本的 reward 分项看哪项在拖后腿。

### Q2：Agentic RL 训练时工具调用陷入循环
`train_agent.py:88-95` 用 `signal.alarm(1)` 给工具执行加 1 秒超时，防止死循环。若仍循环，降 `max_turns`、提高 `thinking_ratio` 让模型先思考。

### Q3：PPO 显存 OOM
PPO 显存是 GRPO 的 1.5-2 倍（actor + critic + ref + reward）。降 `batch_size`、`max_gen_len`，或改用 GRPO。

### Q4：DPO 学完反而能力下降
学习率太高。`train_dpo.py:137` 默认 `4e-8`，若你改成 `1e-6` 几乎必崩。DPO 是微调的微调，要极度保守。

### Q5：GRPO 训练出现退化组
某些问题 6 个回答奖励都一样（σ≈0），优势全 0。这是 `README.md:1135` 的已知问题。可提高 `temperature` 让回答更多样，或换数据集。

### Q6：CISPO 和 GRPO 切换后结果差很多
CISPO 的 `epsilon_high`（默认 5.0）很关键。太大退化成纯 GRPO，太小梯度信号弱。先用默认值，稳定后再调。

### Q7：rollout engine 报 `update_policy failed`
SGLang 引擎需要训练进程能写共享存储路径（`--sglang_shared_path`），且 SGLang server 要能访问。检查路径权限和 SGLang health：`curl http://localhost:8998/health`。

### Q8：多卡训练 DDP 死锁
PPO 在 `train_ppo.py:197-199` 用 `dist.all_reduce(approx_kl_val)` 同步各卡 KL 防止某卡 break 而其他卡继续导致死锁。如果改 PPO 代码不要去掉这个同步。

---

## ✅ 本篇完成自检

<details>
<summary>点开自检（先想 30 秒）</summary>

1. RLHF 和 RLAIF 的本质区别是什么？
   - 反馈来源不同：RLHF 是人类标注的偏好对，RLAIF 是 AI/规则/环境自动生成的反馈信号。RLHF 更贴人类偏好但成本高，RLAIF 自动化可扩展。

2. PO 算法统一公式的三个组件是什么？分别对应什么？
   - 策略项 f(r_t)：怎么用概率比 r_t；优势项 g(A_t)：怎么算优势 A_t；正则项 h(KL_t)：怎么约束偏离。

3. GRPO 相比 PPO 省了什么？怎么做到的？
   - 省了 Critic 网络。用同问题生成 N 个回答的组内均值替代 V(s) 作为 baseline，组内归一化得优势。

4. CISPO 解决了 GRPO 的什么问题？
   - ratio 被 clip 后梯度也被硬截断。CISPO 把策略项改写成 `clip(r,0,ε_max)·A·log π_θ`，ratio 只作常数权重，梯度走 `log π_θ` 不断。

5. DPO 为什么学习率要极小（4e-8）？
   - DPO 在 SFT 之上做偏好微调，本质是微调的微调。学习率稍大就会破坏 SFT 学到的能力，发生灾难性遗忘。

6. Agentic RL 的 rollout 为什么要把 observation token 拼进 response_ids 但 mask=0？
   - 拼进去是为了让 actor 看到工具结果继续生成；mask=0 是因为工具结果是环境给的，不该让模型"学"这些 token 的概率，只在模型自己生成的部分算策略梯度。

7. PPO 的早停机制是怎么工作的？为什么需要？
   - 计算 approx_kl（新旧策略 KL 散度），超过阈值（默认 0.25）说明策略已经跑偏太远，强行更新会破坏模型。早停是工程保险。

8. CISPO 在 minimind 里和 GRPO 是同一个脚本吗？为什么？
   - 是同一个脚本 `train_grpo.py`，通过 `--loss_type` 切换。CISPO 在 GRPO 基础上只改了 loss 写法，其他流程（分组采样、奖励计算、优势构造）完全一致。

9. Agentic RL 的奖励是即时的还是延迟的？包含哪些项？
   - 延迟的，对整条多轮轨迹联合打分。包含 R_answer + R_tool + R_format + R_rm - R_unfinished。

10. 断点续训时 SkipBatchSampler 解决了什么问题？
    - 跳过已训过的 step，避免重复训练。配合 `start_step` 让 dataloader 从中断点继续，而不是从 epoch 开头重训。

</details>

下一篇：[03-工具调用与思考](./03-工具调用与思考.md) —— Tool Call 数据格式、Adaptive Thinking 与 OpenAI API 集成。
