# 第7节：混合精度 + 梯度累积 + 梯度裁剪

> 训练流程核心机制笔记

---

## 📚 本节目录

1. [整体训练流程](#1-整体训练流程核心顺序)
2. [每一步在干什么](#2-每一步在干什么)
3. [三大核心机制总结](#3-三大核心机制总结)
4. [一句话总理解](#4-一句话总理解)

---

## 1. 整体训练流程（核心顺序）

```python
scaler.scale(loss).backward()

if step % accumulation_steps == 0:
    scaler.unscale_(optimizer)
    clip_grad_norm_(model.parameters(), max_norm)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

---

## 2. 每一步在干什么

### (1) `scaler.scale(loss).backward()`

**作用**：AMP 防止 FP16 下溢

- loss 先乘一个 scale（放大）
- backward 计算梯度

**目的**：防止 FP16 数值太小 → 梯度变 0

---

### (2) 梯度累积（accumulation）

**每 N 步才更新一次参数**

**作用**：
- 模拟大 batch training
- 节省显存

---

### (3) `scaler.unscale_(optimizer)`

**作用**：梯度"还原"

因为之前 loss 被放大了：`grad = grad × scale`

所以这里要把梯度恢复到真实数值（为了梯度裁剪）

---

### (4) `clip_grad_norm_()`

**作用**：防止梯度爆炸

**数学本质**：

先算整体梯度范数：
$$\|g\| = \sqrt{\sum_i g_i^2}$$

- 如果 `norm ≤ max_norm`：不变
- 如果 `norm > max_norm`：缩放 → `g := g × max_norm / norm`

**直觉**：不改变方向，只限制"更新步子大小"

---

### (5) `scaler.step(optimizer)`

**作用**：真正更新参数

- 如果检测到 NaN/Inf → 自动跳过
- 否则正常 optimizer.step()

---

### (6) `scaler.update()`

**作用**：动态调整 loss scale

| 情况 | 动作 |
|------|------|
| 出现 NaN/Inf | scale ↓（变小，更保守） |
| 训练稳定 | scale ↑（变大，提高精度） |

**本质**：自动调节 FP16 "放大倍率"，保证稳定 + 高效

---

### (7) `optimizer.zero_grad(set_to_none=True)`

**作用**：清空梯度

`set_to_none=True` → 更省显存 + 更快

---

## 3. 三大核心机制总结

### AMP（scaler）

👉 解决 FP16 数值问题

- scale loss（防 underflow）
- 自动调 scale（防 overflow）

### 梯度累积

👉 解决显存限制

- 多 step 累积梯度
- 再统一更新

### 梯度裁剪

👉 解决梯度爆炸

- 控制 norm 上限
- 保持方向不变

---

## 4. 一句话总理解

**这套流程本质是在做三件事**：

1. 用 AMP 提速
2. 用 accumulation 放大 batch
3. 用 clipping 保稳定
