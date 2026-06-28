# Lesson 22 笔记：DPO 中 Reference Model 的作用

## 目录

- [一句话核心作用](#一句话核心作用)
- [它到底在"限制"什么](#它到底在限制什么)
- [直觉理解](#直觉理解)
- [为什么 reference 能起作用](#为什么-reference-能起作用)
- [数学本质](#数学本质)
- [reference 如何"限制漂移"](#reference-如何限制漂移)
- [更本质解释](#更本质解释)
- [物理类比](#物理类比)
- [为什么必须 freeze reference](#为什么必须-freeze-reference)
- [一句话总结](#一句话总结)

---

## 一句话核心作用

> reference model 提供"行为基准"，让新模型只能在它的基础上做有限改进，防止策略分布偏移过大。

---

## 它到底在"限制"什么

不是限制"输出答案"，而是限制**概率分布（policy）的变化幅度**：

| 限制内容 | 说明 |
|----------|------|
| token 选择概率 | 不能乱变 |
| 整体语言行为 | 不能漂移 |

---

## 直觉理解

把模型想成一个人：

| 角色 | 含义 |
|------|------|
| reference model | 原来的你（SFT 模型） |
| new model | 现在训练的你 |

**DPO 在做：**

> "你可以变得更会回答问题，但不能变成另一个完全不同的人"

---

## 为什么 reference 能起作用

DPO 的目标不是绝对优化，而是**"差值问题"**：

```
(new model 的偏好优势) - (reference model 的偏好优势)
```

**关键点：** 它把优化变成"差值问题"。

---

## 数学本质（简化记）

DPO 核心：

```
(π_θ(y_w) - π_θ(y_l)) - (π_ref(y_w) - π_ref(y_l))
```

**含义：** 新模型必须"比 reference 更偏好 winner"。

---

## reference 如何"限制漂移"

| 如果模型乱跑 | 结果 |
|-------------|------|
| π_θ 会偏离 π_ref 很远 | reference 项变得很"显著" |

→ 在 loss 中形成**"拉回力"**

---

## 更本质解释

> reference model = 隐式 KL 正则的基准点

| 等价关系 | 说明 |
|----------|------|
| PPO | 显式 KL penalty |
| DPO | 隐式 KL constraint |

---

## 物理类比

| 项 | 类比 |
|----|------|
| reward / preference | 推你往前走 |
| reference model | 弹簧 / 锚点 |

**结果：** 你可以变好，但不会飞走。

---

## 为什么必须 freeze reference

reference 必须是"固定标准"，否则约束失效。

如果 reference 也在训练，它的"基准"会不断移动，约束就不存在了。

---

---

## DPO 中 mask 和 sum 的作用

### 代码

```python
def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)
```

### 原始 shape

```
[B, T]
```

| 维度 | 含义 |
|------|------|
| B | batch |
| T | sequence length |
| 每个值 | log π(token_t \| context) |

### mask 的作用

| token 类型 | mask 值 | 是否参与训练 |
|------------|---------|-------------|
| prompt | 0 | ❌ 不训练 |
| assistant | 1 | ✅ 训练 |
| padding | 0 | ❌ 不训练 |

如果不 mask，会优化"模型怎么复述 prompt"，完全错误。

### 为什么要 sum

**DPO 的比较对象不是 token，而是整条回答（sequence）**

DPO 真正优化的是：
```
log π(y|x) = Σ_t log π(y_t | x, y_<t)
```

sum 是把"逐 token 的概率"还原成"整句概率"。

### 为什么必须是 sequence-level

因为 DPO 比的是 `y_w vs y_l`（winner vs loser）。

- 如果不 sum → token 级别对比，无法比较整句优劣
- sum 后 → 得到 `log π(y_w|x) - log π(y_l|x)`

### 一句话总结

> log-prob 是 token-level 的，但优化目标是 sequence-level（整段回答概率），因此需要通过 mask 过滤无关 token，并在 sequence 维度上求和，得到完整 response 的 log-prob。

---

## F.logsigmoid 详解

### 一句话解释

> logsigmoid(x) = sigmoid(x) 的 log 版本

```
log(σ(x))
```

### 数学定义

sigmoid：
```
σ(x) = 1 / (1 + e^(-x))
```

logsigmoid：
```
logσ(x) = -log(1 + e^(-x))
```

### 它在干嘛（直觉）

| x | sigmoid(x) | logsigmoid(x) |
|---|------------|---------------|
| +∞ | 1 | 0 |
| 0 | 0.5 | -0.69 |
| -∞ | 0 | -∞ |

### 为什么不用 log(sigmoid(x))

| 问题 | 原因 |
|------|------|
| ❌ 数值不稳定 | 会 overflow / underflow |
| ✅ 正确做法 | 用 `F.logsigmoid(x)` |

`F.logsigmoid(x)` 是**稳定实现**，避免了直接取 log 带来的数值问题。

---

## 一句话总结

> reference model 在 DPO 中作为固定行为基准，通过与当前模型的对比隐式形成 KL 约束，使优化目标从"绝对提升偏好"变为"相对参考模型的改进"，从而防止策略分布发生过大漂移。
