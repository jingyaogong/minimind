# 第12节：RoPE（旋转位置编码）

> 核心两部分：频率来源 + 旋转为何产生相对位置

---

## 📚 本节目录

1. [第一部分：θ 和 ω 的来源](#第一部分θ-和-ω-的来源)
2. [第二部分：θ 角度怎么用到 Q/K 上](#第二部分θ-角度怎么用到-qk-上旋转的实现)
3. [第三部分：为什么旋转 + 点积 = 相对位置](#第三部分为什么旋转-点积-相对位置)
4. [第四部分：precompute_freqs_cis 源码解析](#第四部分precompute_freqs_cis-源码解析生成-sincos-表)
5. [第五部分：apply_rotary_pos_emb 源码解析](#第五部分apply_rotary_pos_emb-源码解析旋转-qk)
6. [一页总结](#一页总结最重要)
7. [一句话终极总结](#一句话终极总结)

---

## 第一部分：θ 和 ω 的来源

### 1. 频率定义（每个维度对 k）

$$\omega_k = 10000^{-2k/d}$$

- k = 第 k 个"二维维度对"
- d = head_dim

**含义**：每一对 2D 维度都有一个固定"旋转速度" ω_k

### 2. 位置角定义

对 token 位置 i / j：

$$\theta_i^k = i \cdot \omega_k$$
$$\theta_j^k = j \cdot \omega_k$$

### 3. 关键关系（核心）

因为 ω_k 相同：

$$\theta_i^k - \theta_j^k = i\omega_k - j\omega_k = (i-j)\omega_k$$

**结论**：

| 符号 | 含义 |
|------|------|
| i | query 位置 |
| j | key 位置 |
| ω_k | 共享的（同一个维度对） |
| 差值 | 自然变成 (i - j) |

---

## 第二部分：θ 角度怎么用到 Q/K 上（旋转的实现）

### 1. θ 不是"存着不用"，而是直接改 Q/K

RoPE 做的是：

```
原始 Q/K：
    Q_i, K_j

加 RoPE 后变成：
    Q_i' = rotate(Q_i, θ_i)
    K_j' = rotate(K_j, θ_j)
```

### 2. 旋转公式（对每一对 2D 维度）

$$\begin{aligned}
x' &= x \cos\theta - y \sin\theta \\
y' &= x \sin\theta + y \cos\theta
\end{aligned}$$

**θ 的作用**：控制"这个 token 向量旋转多少度"

### 3. attention 真正用的是"旋转后的 Q/K"

```
不是：
    Q_i · K_j   ❌（原始）

而是：
    Q_i'(θ_i) · K_j'(θ_j)  ✅（旋转后）
```

### 4. 关键变化

attention 变成：

$$(R(\theta_i) Q_i) \cdot (R(\theta_j) K_j)$$

**θ 不是"附加信息"，而是直接改变 Q/K 的数值。**

---

## 第三部分：为什么旋转 + 点积 = 相对位置

### 1. RoPE 定义

$$Q_i' = R(i\omega) Q_i$$
$$K_j' = R(j\omega) K_j$$

### 2. attention 变形

$$Q_i' \cdot K_j' = (R(i\omega)Q_i) \cdot (R(j\omega)K_j)$$

### 3. 线代关键性质

| 性质 | 公式 |
|------|------|
| 转置 | $R(a)^T = R(-a)$ |
| 乘法 | $R(a)R(b) = R(a+b)$ |

### 4. 推导核心

$$Q_i^T R(i\omega)^T R(j\omega) K_j$$
$$= Q_i^T R(-i\omega) R(j\omega) K_j$$
$$= Q_i^T R((j-i)\omega) K_j$$

### 5. 最终结论

**attention 只依赖**：$(j-i)\omega$

**第二部分结论**：旋转矩阵的"可加性" + "正交性"，让 attention 从**绝对位置** → 变成**相对位置**

---

## 第四部分：precompute_freqs_cis 源码解析（生成 sin/cos 表）

### 0. 一句话总览

**预先生成"每个位置 × 每个维度"的 RoPE sin/cos 表（可选 YaRN 缩放），供 attention 直接使用。**

### 1. Step 1：构造频率 ω

```python
freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2).float() / dim))
```

生成：$\omega_i = base^{-i/d}$

**直觉**：
- 前面维度 → 高频（变化快）
- 后面维度 → 低频（变化慢）
- 这是 RoPE 的"多尺度位置编码"

### 2. Step 2：生成位置 θ

```python
t = torch.arange(end)
freqs = torch.outer(t, freqs)
```

变成：$\theta_{t,i} = t \cdot \omega_i$

**直觉**：
| position t | dim i |
|------------|-------|
| t=0 | 0 |
| t=1 | ω |
| t=2 | 2ω |

### 3. Step 3：转 sin / cos（核心）

```python
freqs_cos = torch.cat([cos(freqs), cos(freqs)], dim=-1)
freqs_sin = torch.cat([sin(freqs), sin(freqs)], dim=-1)
```

**关键**：把 (dim/2) 的频率扩展回 dim

RoPE 是每 2 维共享一个 θ：
```
[θ0, θ1, θ2] → [θ0, θ0, θ1, θ1, θ2, θ2]
```

### 4. YaRN 扩展（长上下文）

**解决的问题**：RoPE 在长上下文（32k+）会失效

**核心思想**：不同频率维度，用不同缩放策略

```python
if end / orig_max > 1:  # 推理长度 > 训练长度
    # 构造 ramp: 0→1 线性过渡带
    ramp = clamp((i - low) / (high - low))

    # 修改频率：低频不变，高频压缩
    freqs = freqs * (1 - ramp + ramp / factor)
```

| dim 区域 | 处理方式 |
|----------|----------|
| low 以下 | 不变 |
| middle | 逐渐变化 |
| high 以上 | 完全缩放 (1/factor) |

### 5. 完整心智模型

```
1. dim → frequency ω
2. position t → angle θ = t·ω
3. θ → sin/cos table
4. (可选) YaRN 调整 ω
5. attention 时用 cos/sin 做旋转
```

---

## 第五部分：apply_rotary_pos_emb 源码解析（旋转 Q/K）

### 0. 一句话总览

**把 Q 和 K 的每一对维度当成二维向量，然后按 position 做旋转。**

### 1. 整体结构

```python
q_embed = q * cos + rotate_half(q) * sin
k_embed = k * cos + rotate_half(k) * sin
```

Q 和 K 做完全一样的操作。

### 2. rotate_half 到底干嘛

```python
rotate_half(x):
    x1, x2, x3, x4, x5, x6 → -x2, -x3, -x4, -x5, -x6, x1
```

**更本质理解**：把 embedding 的"后半段"旋转到前面，并取负号

**等价数学**：对每一对 (x, y)：
```
(x, y) → (-y, x)
```
这一步 = "90° 旋转基向量"

### 3. 主公式拆解

只看 Q（K 完全一样）：

```
q_embed = q * cos + rotate_half(q) * sin
```

对每一对 (x, y)：

```
q = (x, y)
rotate_half(q) = (-y, x)
代入：
x' = x·cos + (-y)·sin = x·cos - y·sin
y' = y·cos + x·sin
```

### 4. 这就是标准二维旋转

$$\begin{pmatrix} x' \\ y' \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix}$$

### 5. unsqueeze_dim = 1 的作用

```python
cos.unsqueeze(unsqueeze_dim)  # [T, D] → [1, T, 1, D] → broadcast to [B, T, H, D]
```

对齐 head 和 batch 维度。

### 6. Q 和 K 为什么要同样变换？

**非常关键**：RoPE 是"对称变换"，不是 query/key 分开设计

### 7. 为什么这样就有相对位置？

```
Q_i 用 θ_i 旋转
K_j 用 θ_j 旋转
点积 → f(θ_i - θ_j) = f((i-j)·ω)
```

位置差自然出现！

### 8. 完整终极心智模型

```
1. position → θ
2. θ → sin/cos（预计算表）
3. q/k → split 成二维 pair
4. rotate_half = 90° 基向量
5. cos + sin 组合 = 任意角旋转
6. attention = 相对角度差
```

---

## 一页总结（最重要）

### 🌟 RoPE 核心三件事

| 序号 | 内容 |
|------|------|
| ① | 每个维度对有频率 ω_k |
| ② | 位置变角度 θ_i = iω_k, θ_j = jω_k |
| ③ | 旋转 + 点积 → 相对位置 $Q_i' \cdot K_j' \Rightarrow f(i-j)$ |

### 💡 几何直觉

```
θ_i = iω  →  位置 i 的旋转角度
θ_j = jω  →  位置 j 的旋转角度

旋转后点积：
Q_i' · K_j' = Q_i · R((j-i)ω) · K_j

只和 (j-i) 有关！
```

---

## 一句话终极总结

**RoPE 通过"按频率旋转 Q/K"，利用旋转矩阵的代数性质，使 attention 从依赖绝对位置 i/j 变成只依赖相对位置 (i-j)。**
