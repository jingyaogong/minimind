# Lesson 23 笔记：PPO 中 Response 采样与梯度问题

## 目录

- [核心问题：为什么 y 不能直接求导](#核心问题为什么-y-不能直接求导)
- [普通 CE Loss 的梯度怎么传](#普通-ce-loss-的梯度怎么传)
- [PPO 的问题：y 是"采样"出来的](#ppo-的问题y-是采样出来的)
- [DPO 的解决方案](#dpo-的解决方案)
- [PPO 的解决方案：Reinforce 估计](#ppo-的解决方案reinforce-估计)
- [一句话总结](#一句话总结)

---

## 核心问题：为什么 y 不能直接求导

> response y 是模型采样出来的离散 token 序列，不能像普通 CE loss 那样直接对 token 选择过程求导。

---

## 普通 CE Loss 的梯度怎么传

```python
# 普通 SFT
logits = model(input_ids)           # 模型输出每个 token 的 logits
loss = F.cross_entropy(logits, labels)  # 直接拿 ground truth labels 算 loss
loss.backward()                      # 梯度直接流回 logits → embedding
```

**关键：** labels 是**固定的 ground truth**，不是从模型里采样出来的。

| 步骤 | 说明 |
|------|------|
| 1 | logits = model(input_ids) |
| 2 | loss = cross_entropy(logits, y_fixed) |
| 3 | backward → 梯度流回 logits |

---

## PPO 的问题：y 是"采样"出来的

```python
# PPO 里
y_w = torch.multinomial(probs, num_samples)  # 从模型分布中采样
```

| 问题 | 说明 |
|------|------|
| 采样是随机的 | `torch.multinomial` 每次结果可能不同 |
| 采样不可导 | autograd 无法跟踪随机采样过程 |

**梯度流断在这里：**

```
logits → softmax → sample → y (随机!)
                        ↑
                  这个随机采样过程没有梯度
```

---

## DPO 的解决方案

DPO **绕过了这个问题**，不采样，而是直接用 model 输出的 log_probs：

```python
# DPO 直接用模型输出的概率分布
policy_log_probs = model(x_chosen).log_probs  # 可导
ref_log_probs = ref_model(x_chosen).log_probs  # 可导

# 计算 ratio 时，不需要对采样求导
log_ratio = policy_log_probs - ref_log_probs
```

---

## PPO 的解决方案：Reinforce 估计

PPO 需要对采样过程求导，用的是 **Reinforce 梯度估计**：

```
∇θ E_{y~πθ}[r(y)] = E_{y~πθ}[r(y) * ∇θ log πθ(y)]
```

| 步骤 | 说明 |
|------|------|
| 1 | 从 π_θ(y|x) 采样 y |
| 2 | 计算 reward r(y) |
| 3 | 用 ∇ log πθ(y) 把 reward 的信号传回去 |

**关键技巧：** 用 log probability 的梯度来"引导"随机采样的方向。

---

## PPO + KL 约束公式：k = r - log r - 1

### 公式拆解

```python
k = π_ref(a|s) / π_θ(a|s) - log(π_ref(a|s) / π_θ(a|s)) - 1
```

设 `r = π_ref / π_θ`，则：

```
k(r) = r - log r - 1
```

### 核心性质

| 性质 | 说明 |
|------|------|
| **最小值在 r = 1** | k(1) = 1 - 0 - 1 = 0，policy = reference 时无惩罚 |
| **永远 ≥ 0** | 经典不等式 r - log r - 1 ≥ 0 |
| **偏离越大，惩罚越大** | r >> 1 或 r << 1 时，k 增大 |

### 为什么这样设计

> 限制 policy 不要离 reference 太远，但又不能太死

| 公式项 | 含义 |
|--------|------|
| `r = π_ref / π_θ` | policy 相对于 reference 的概率比值 |
| `r - log r - 1` | KL divergence 的单 token surrogate |

**作用：**
- 保证非负
- 在 r=1 处最小
- 对大偏离进行更强惩罚
- 实现稳定的策略更新约束

---

## Thinking 机制：open_thinking 的作用

### RLAIFDataset 中的 thinking 控制

```python
def create_chat_prompt(self, conversations):
    use_thinking = random.random() < self.thinking_ratio
    return self.tokenizer.apply_chat_template(
        conversations[:-1],
        tokenize=False,
        open_thinking=use_thinking,
        add_generation_prompt=True
    )
```

| 模式 | 效果 |
|------|------|
| `open_thinking=True` | 模型学会"先想再答"，输出包含 `<think>` 推理过程 |
| `open_thinking=False` | 模型学会"直接回答"，不输出推理过程 |

### chat template 中的实现

```jinja
{%- if open_thinking is defined and open_thinking is true %}
    {{- '<think>\n' }}           {# 只打开，不关闭，让模型自己生成 </think> #}
{%- else %}
    {{- '<think>\n\n</think>\n\n' }}  {# 完整空 thinking 块 #}
{%- endif %}
```

| open_thinking | 模板输出 |
|----------------|----------|
| True | `<|im_start|>assistant\n<think>`（打开但没关闭） |
| False | `<|im_start|>assistant\n<think>\n\n</think>\n\n`（完整空块） |

### 之前阶段有没有训练过 thinking

| 阶段 | 有 thinking 训练数据吗 |
|------|----------------------|
| Pretrain | 没有 |
| SFT | 有字段但**实际数据没有**（`reasoning_content` 是空的） |
| RLAIFDataset | **有** — 通过 `open_thinking` 参数控制 |

### SFT 中也有 thinking 相关机制

SFTDataset 的数据 features 定义了 `reasoning_content` 字段，chat template 对 assistant 消息会做：

```jinja
<|im_start|>assistant
<think>
{reasoning_content}
</think>
{content}<|im_end|>
```

另外 `post_processing_chat()` 会随机移除空 thinking 块：

```python
if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
    prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
```

### 总结：thinking 训练流程

| 阶段 | 作用 |
|------|------|
| **SFT** | 让模型见过 `<think>`...`</think>` 格式，也见过空 thinking 或被移除的情况 |
| **RLAIF/PPO** | 用 `thinking_ratio` 随机决定 rollout 时"打开 thinking"还是"空 thinking 直答" |
| **推理** | 用户用 `open_thinking` 开关控制模型偏向"显式思考"还是"直接回答" |

---

## PPO 中四个模型的角色

| 模型 | 来源 | 训练方式 |
|------|------|----------|
| **actor_model** | 从 full_sft 来 | PPO 训练它 |
| **ref_model** | 从 full_sft 来 | 冻结，当参考模型 |
| **critic_model** | 从 full_sft backbone 来，加 value_head | PPO 训练它 |
| **reward_model** | 外部奖励模型 | 只负责给 response 打分 |

---

## SGLang 详解

### 一句话定性

> HTTP API 的作用：把"模型生成能力"变成一个"可调用的远程服务接口"。本质不是为了"网络"，而是为了：解耦 + 调度 + 并发 + 工程化部署。

---

### model.generate vs SGLang

| 方式 | model.generate | SGLang |
|------|---------------|--------|
| 形式 | 函数调用 | HTTP / RPC |
| 执行 | 当前进程 | 独立推理服务 |
| batch | 固定 batch | Continuous batching |
| cache | 本地函数里 | server 统一管理 |

---

### SGLang 额外做了什么

**没有发生的事：**
- ❌ 改变模型结构
- ❌ 改变 forward 计算
- ❌ 替换 model.generate 的算法本质

**发生的事：**
- 把"generate 循环"从 Python 进程搬到一个高性能推理系统里执行

---

### Continuous Batching（关键）

| 方式 | 行为 |
|------|------|
| HF generate | 等所有请求到齐，固定 batch 一起生成 |
| SGLang | 动态拼 batch，完成就退出，新请求随时加入 |

```
t=0: prompt A, B
t=1: prompt C 加进来
t=2: prompt D 加进来
→ GPU 利用率更高
```

---

### HTTP API 的核心作用

| 作用 | 说明 |
|------|------|
| **解耦** | 训练和推理完全分开，可独立升级/扩展 |
| **服务化** | 变成标准接口，不关心内部怎么生成 |
| **并发调度入口** | 统一入口供 scheduler 做 continuous batching |
| **分布式扩展** | 多训练 worker 共用一个推理服务 |

---

### RLHF 场景为什么特别需要

| 需求 | 为什么需要 HTTP API |
|------|---------------------|
| rollout 量极大 | 一个 step = 100~1000 prompts，必须高吞吐入口 |
| 训练/推理资源分离 | Trainer 用一批 GPU，Rollout 用另一批，必须跨进程通信 |
| 动态调度 | prompt 随时进来，generation 长度不一致，必须统一调度 |

---

### RLHF pipeline 中的位置

```
Prompt dataset
      ↓
HTTP API（SGLang）
      ↓
Generated trajectories
      ↓
Reward model
      ↓
PPO / GRPO update
```

---

### 一句话总结（考试级）

> SGLang 并没有替代 model.generate，而是把"生成过程（forward + sampling + KV cache loop）"从本地函数调用，升级为可并发调度的推理服务；模型参数 θ 完全不变，只是执行环境和调度方式变了。

---

## approx_kl 详解

### 公式

```python
approx_kl = (
    0.5 * (log_ratio ** 2) * resp_policy_mask[inds]
).sum() / resp_policy_mask[inds].sum().clamp(min=1)
```

展开：

```
approx_kl = Σ_{b,t} m_{b,t} * 0.5 * (log π_θ(a_{b,t}|s_{b,t}) - log π_old(a_{b,t}|s_{b,t}))²
            / Σ_{b,t} m_{b,t}
```

其中 `m_{b,t} = resp_policy_mask[b,t]`

### resp_policy_mask 的作用

| 排除的内容 | 说明 |
|-----------|------|
| prompt tokens | 只统计 response |
| padding | 只统计有效 token |
| EOS 之后 | 只统计有效 response 区域 |

### 为什么叫 approx_kl

PPO 需要监控：当前 actor `π_θ` 离 rollout 时的 old actor `π_old` 有多远。

KL 本来是衡量两个分布距离的量。当新旧策略很接近时，可以用二阶近似：

```
KL(π_old || π_θ) ≈ 0.5 * (log π_θ - log π_old)²
```

所以源码用 `0.5 * (log_ratio ** 2)` 来近似新旧策略的 KL。

### 直观判断

| log_ratio | 含义 | approx_kl |
|-----------|------|-----------|
| 接近 0 | new actor 和 old actor 很接近 | 小 |
| 绝对值变大 | new actor 和 old actor 差得更远 | 大 |

---

## 一句话总结

| 概念 | 说明 |
|------|------|
| **普通 CE** | y 是 ground truth 常数 → 梯度直接流回 |
| **DPO** | y 是分布比较，不需要采样 → 直接用 log_probs |
| **PPO** | y 必须采样 → 用 Reinforce 估计器把 reward 信号传回去 |
| **KL 约束** | k = r - log r - 1，保证 policy 不偏离 reference 太远 |
| **approx_kl** | 0.5 * (log_ratio)²，新旧策略 KL 散度的近似 |
| **Thinking** | 通过 `open_thinking` 控制模型是否显式输出推理过程 |
| **SGLang** | 生成服务化，支持并发调度和连续 batching |
