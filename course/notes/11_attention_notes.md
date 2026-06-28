# 第11节：Attention 实现（Flash vs Fallback）

> 你问的 AI 回答，核心问题已标注

---

## 📚 本节目录

1. [整体在干什么](#1-整体在干什么)
2. [Flash Attention 触发条件](#2-flash-attention-触发条件)
3. [causal mask 为什么是 -seq_len](#3-causal-mask-为什么是--seq_len)
4. [scores 的 shape](#4-scores-的-shape)
5. [softmax 为什么用 float()](#5-softmax-为什么用-float)
6. [fallback 完整流程](#6-fallback-完整流程)
7. [两条路径对比](#7-两条路径对比)

---

## 1. 整体在干什么

根据输入条件选择 attention 实现：
- **Flash 路径**：快 + 省显存（GPU fused kernel）
- **fallback 路径**：完整手写 QK^T + mask + softmax + V

---

## 2. Flash Attention 触发条件

```python
if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
```

| 条件 | 含义 |
|------|------|
| `self.flash` | 系统 + config 都允许 Flash Attention |
| `seq_len > 1` | 单 token 不需要 attention（退化） |
| `not self.is_causal` | 非 causal（encoder）可以 |
| `past_key_value is None` | 没有 KV cache（prefill 阶段）可以 |
| `attention_mask is None or 全1` | 不能有 padding mask，不能有复杂 mask |

**总结**：Flash 只在"干净、标准、无 padding、无 decode 冲突"时使用

**❌ decode + KV cache 不走 flash**：因为 causal mask 作用在完整 KV 上，Flash 无法处理

---

## 3. causal mask 为什么是 `-seq_len`

```python
scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
```

**关键点**：K 可能是 "past + current"（KV cache）

```text
K = [past_K, current_K]
     ↓       ↓
     长度不确定  当前 seq_len 个

scores[:, :, :, -seq_len:] 只对"当前新增 token 的 K 部分"做 causal mask
```

`triu(1)` = 上三角（不含主对角线），设为 -inf：

```
     K0  K1  K2  K3(新)
Q0 [ 0  -inf -inf -inf ]
Q1 [ 0   0  -inf -inf ]
Q2 [ 0   0   0  -inf ]
Q3 [ 0   0   0   0  ]  ← 当前 token 不能看未来
```

---

## 4. scores 的 shape

```python
scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
```

**shape = [B, H, Q_len, K_len]**

```
B = batch size
H = num heads
Q_len = query 长度
K_len = key 长度（可能包含 KV cache）
```

---

## 5. softmax 为什么用 float()

```python
F.softmax(scores.float(), dim=-1)
```

**原因**：防止 FP16 softmax 数值爆炸

FP16 数值范围小，softmax 计算中间值可能溢出。

---

## 6. fallback 完整流程

```
QK^T / √d
    ↓
causal mask（防未来）
    ↓
padding mask（忽略 pad）
    ↓
softmax
    ↓
dropout
    ↓
× V
```

---

## 7. 两条路径对比

| 路径 | 使用条件 | 特点 |
|------|---------|------|
| Flash | 干净输入、无 padding、无 KV cache 冲突 | 快 / fused kernel / 省显存 |
| fallback | 任意情况 | 稳 / 可控 / 完整手写 |
