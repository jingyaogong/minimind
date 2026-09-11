# Lesson 20 笔记：蒸馏 Loss（KL + Temperature）详解

## 目录

- [蒸馏 Loss 完整流程](#蒸馏-loss-完整流程)
- [Step 1：teacher 分布（soft target）](#step-1teacher-分布soft-target)
- [Step 2：student 分布](#step-2student-分布)
- [Step 3：KL Loss](#step-3kl-loss)
- [Q1：为什么除 T 会变"更均匀"](#q1为什么除-t-会变更均匀)
- [Q2：KL 为什么可能负？为什么不取绝对值](#q2kl-为什么可能负为什么不取绝对值)
- [Step 4：为什么要乘 T²](#step-4为什么要乘-t²)
- [一句话总结](#一句话总结)

---

## 蒸馏 Loss 完整流程

```python
def distillation_loss(student_logits, teacher_logits, temperature=1.0):
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction='batchmean'
    )
    return (temperature ** 2) * kl
```

---

## Step 1：teacher 分布（soft target）

```python
teacher_probs = softmax(teacher_logits / T).detach()
```

做了三件事：

| 步骤 | 操作 | 作用 |
|------|------|------|
| ① | `logits / T` | 压缩 logits 差距 |
| ② | `softmax` | 变成概率分布 |
| ③ | `detach` | teacher 不参与反向传播 |

---

## Step 2：student 分布

```python
student_log_probs = log_softmax(student_logits / T)
```

做了：
- `logits / T`（同样变软）
- `log_softmax`（为了计算 KL）

---

## Step 3：KL Loss

```python
kl = F.kl_div(student_log_probs, teacher_probs)
```

PyTorch 的 `F.kl_div` 计算的是：

```
KL(P_teacher || P_student) = Σ P_teacher(x) * log(P_teacher(x) / P_student(x))
```

---

## Q1：为什么除 T 会变"更均匀"

**本质：softmax 对 logits 差值是指数放大的，除以 T 就是在"压缩差距"**

| T | logits 差距 | 分布 |
|---|------------|------|
| 1 | 原始 | 很尖（winner takes all） |
| > 1 | 被压缩 | 更平 |

**直觉记忆：T 越大 → "winner takes all" 越弱**

```
T=1:  softmax([3.0, 1.0, 0.0]) → [0.90, 0.07, 0.03]   很尖
T=2:  softmax([1.5, 0.5, 0.0]) → [0.58, 0.31, 0.11]   更平
T=10: softmax([0.3, 0.1, 0.0]) → [0.35, 0.32, 0.33]   很平
```

---

## Q2：KL 为什么可能负？为什么不取绝对值

**KL 不是距离，而是"用 Q 编码 P 的额外信息成本"**

| 性质 | 说明 |
|------|------|
| ❌ 不是距离 | 不对称，KL(P\|\|Q) ≠ KL(Q\|\|P) |
| ✅ 整体 ≥ 0 | Jensen 不等式保证 |
| ⚠️ 单项可能为负 | 如果 Q(x) > P(x)，则 log(P/Q) < 0 |

**直觉理解：**

就算某些 token "Q 猜得更好"，整体编码策略仍然不可能比真实分布更优。

---

## Step 4：为什么要乘 T²

```python
return (temperature ** 2) * kl
```

**原因：softmax(z/T) 会让梯度整体缩小约 1/T²**

发生了两次缩放：

| 步骤 | 梯度缩放 |
|------|----------|
| logits / T | scale ↓ 1/T |
| softmax gradient | 再 ↓ 1/T |
| **合计** | **≈ 1/T²** |

所以要乘 T² 补偿，保持训练强度。

---

## 一句话总结

| 概念 | 一句话 |
|------|--------|
| **蒸馏本质** | teacher 用 softmax(z/T) 生成"软标签"，student 用 log_softmax(z/T) 拟合 teacher 分布 |
| **T 的作用** | T > 1 → logits 差距压缩 → 分布更平 → 信息更丰富 |
| **KL 本质** | 用 Q 逼近 P 时的信息损失（额外编码成本），不是距离 |
| **乘 T²** | temperature 让梯度缩小约 T² 倍，需要放大回来 |
