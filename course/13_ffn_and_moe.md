# 第 13 课：FFN 与 MoE

这一课只解决一个问题：Attention 汇总上下文之后，MiniMind 如何用 FFN 或 MoE 对每个 token 的 hidden 向量做非线性变换。

## 目录

- [0. 本节主线](#l13-mainline)
- [1. 原理讲解](#l13-principle)
- [2. 源码阅读顺序图](#l13-reading-order)
- [3. MiniMind 源码走读](#l13-source-walkthrough)
- [4. 本节必须会写 / 暂时不要求](#l13-must-write)
- [5. 手写模块](#l13-handwrite)
- [6. 对齐测试](#l13-alignment-test)
- [7. 阶段组装](#l13-stage-assembly)
- [8. 本节检查](#l13-check)
- [9. 下一课](#l13-next)

<a id="l13-mainline"></a>
## 0. 本节主线

Transformer block 里，Attention 和 FFN/MoE 分工不同：

```text
Attention:
让每个 token 从上下文 token 里取信息。

FFN / MoE:
对每个 token 自己的 hidden 向量做非线性变换。
```

MiniMind block 的后半段是：

```text
attention 输出后的 hidden_states
-> post_attention_layernorm
-> FeedForward 或 MOEFeedForward
-> residual add
-> 下一个 block
```

Dense FFN 的核心公式可以写成：

$$
\begin{aligned}
u &= W_{\text{up}}x \\
g &= W_{\text{gate}}x \\
h &= \mathrm{SiLU}(g) \odot u \\
y &= W_{\text{down}}h
\end{aligned}
$$

也就是：先扩维得到候选特征 `u` 和门控特征 `g`，门控后再降回 `hidden_size`。

MoE 的核心公式可以写成：

$$
\begin{aligned}
r &= W_{\text{router}}x \\
p &= \mathrm{softmax}(r) \\
S &= \mathrm{TopK}(p, k) \\
\alpha_e &= \frac{p_e}{\sum_{j \in S} p_j}, \quad e \in S \\
y &= \sum_{e \in S}\alpha_e \cdot \mathrm{Expert}_e(x)
\end{aligned}
$$

也就是：router 先给每个 expert 一个概率，只选 top-k 个 expert 计算，再把被选中的概率归一化成权重 $\alpha_e$，最后加权求和。

一句话：

```text
FFN 是所有 token 共享同一个专家；MoE 是每个 token 先经过 router 选择少数专家，再加权汇总专家输出。
```

<a id="l13-principle"></a>
## 1. 原理讲解

### 1.1 FFN 到底干什么

第 11 课讲 Attention 时，我们一直在看 token 之间的信息流：

```text
位置 i 的 token
-> 用 Q 去看上下文 K
-> 用注意力权重汇总上下文 V
```

这解决的是“token 之间怎么交流”。

但拿到上下文信息后，每个 token 自己的向量还要继续加工。这个加工主要由 FFN 完成。

FFN 有一个重要特点：

```text
它不混合不同位置的 token。
```

如果输入是：

```text
hidden_states: [batch, seq, hidden_size]
```

那么 FFN 对每个位置独立做同一套变换：

```text
hidden_states[:, 0, :] -> FFN -> output[:, 0, :]
hidden_states[:, 1, :] -> FFN -> output[:, 1, :]
hidden_states[:, 2, :] -> FFN -> output[:, 2, :]
```

所以可以这样理解：

```text
Attention 负责横向：token 与 token 之间的信息交换。
FFN 负责纵向：单个 token 的 hidden 维度内部加工。
```

### 1.2 MiniMind 的 FFN 为什么有三个线性层

普通 MLP 可能写成：

```text
down( activation(up(x)) )
```

MiniMind 用的是 SwiGLU 风格。对单个 token 的 hidden 向量 `x`，可以写成：

$$
\begin{aligned}
u &= W_{\text{up}}x \\
g &= W_{\text{gate}}x \\
h &= \mathrm{SiLU}(g) \odot u \\
y &= W_{\text{down}}h
\end{aligned}
$$

如果放回一个 batch，形状是：

| 变量 | 形状 | 含义 |
|---|---|---|
| $B$ | 标量 | batch size |
| $S$ | 标量 | sequence length |
| $H$ | 标量 | `hidden_size` |
| $I$ | 标量 | `intermediate_size` |
| $x$ | `[B, S, H]` | FFN 输入，每个 token 一个 hidden 向量 |
| $u$ | `[B, S, I]` | $W_{\text{up}}x$，候选特征 |
| $g$ | `[B, S, I]` | $W_{\text{gate}}x$，门控特征 |
| $h$ | `[B, S, I]` | $\mathrm{SiLU}(g) \odot u$，门控后的中间特征 |
| $y$ | `[B, S, H]` | $W_{\text{down}}h$，FFN 输出 |

这里 $\odot$ 表示逐元素相乘。对应到源码就是：

```text
W_up    -> up_proj
W_gate  -> gate_proj
W_down  -> down_proj
```

这里有三条线性投影：

```text
gate_proj: hidden_size -> intermediate_size
up_proj:   hidden_size -> intermediate_size
down_proj: intermediate_size -> hidden_size
```

`gate_proj` 经过 `silu` 后像一个门：

$$
\mathrm{SiLU}(W_{\text{gate}}x)
$$

它会和 `up_proj(x)` 逐元素相乘：

$$
\mathrm{SiLU}(W_{\text{gate}}x) \odot (W_{\text{up}}x)
$$

这一步的意思是：

```text
up_proj 产生候选特征；
gate_proj 决定哪些特征更应该通过。
```

最后 `down_proj` 把维度从 `intermediate_size` 降回 `hidden_size`，这样才能和 residual 主干相加。

### 1.3 FFN 为什么必须回到 hidden_size

Transformer block 的主干 shape 一直是：

```text
[batch, seq, hidden_size]
```

FFN 中间可以扩维：

```text
[batch, seq, hidden_size]
-> gate/up
-> [batch, seq, intermediate_size]
```

但输出必须回到：

```text
[batch, seq, hidden_size]
```

因为 block 里还有 residual：

```text
hidden_states = hidden_states + mlp(norm(hidden_states))
```

如果 MLP 输出不是 `hidden_size`，就无法和原来的 `hidden_states` 相加。

### 1.4 MoE 是什么

MoE 是 Mixture of Experts，中文可以理解成“专家混合”。

Dense FFN 是：

```text
每个 token 都走同一个 FFN。
```

MoE 是：

```text
准备多个 FFN expert；
每个 token 先通过 router 打分；
只选择 top-k 个 expert 计算；
把 expert 输出按 router 权重加权求和。
```

这里的 `expert` 不是一个抽象名字，而是一个真正的子网络。

在 MiniMind 里，一个 expert 本质上就是一个独立的 `FeedForward`：

```text
expert 0 = FeedForward(...)
expert 1 = FeedForward(...)
expert 2 = FeedForward(...)
expert 3 = FeedForward(...)
```

它们结构一样，但参数各自独立。也就是说，`Expert_0(x)` 和 `Expert_2(x)` 都是 FFN 计算，但用的是两套不同的权重。

如果有 4 个 experts，top-k=1，那么某个 token 可能走：

```text
token A -> expert 2
token B -> expert 0
token C -> expert 2
token D -> expert 3
```

MoE 的核心取舍是：

```text
总参数量变大，因为有多个 experts；
每个 token 激活的参数量不一定同比变大，因为只走少数 experts。
```

这就是 README 里 `198M-A64M` 这种写法的含义：总参数更多，但单 token 激活参数接近较小模型。

### 1.5 Router 干什么

Router 是一个线性层。对单个 token 的 hidden 向量 `x`，它先产生每个 expert 的 logit：

$$
r = W_{\text{router}}x
$$

其中：

$$
r \in \mathbb{R}^{E}
$$

这里 $E$ 表示 expert 数量，也就是 `num_experts`。

然后用 softmax 变成概率：

$$
p_e = \frac{\exp(r_e)}{\sum_{j=1}^{E}\exp(r_j)}
$$

如果 `num_experts=4`，可能得到：

```text
p = [0.05, 0.10, 0.80, 0.05]
```

top-1 routing 会选择 expert 2。

如果 `num_experts_per_tok=2`，可能选择 expert 2 和 expert 1，再按权重加权：

$$
\begin{aligned}
S &= \mathrm{TopK}(p, 2) = \{2, 1\} \\
\alpha_2 &= \frac{0.80}{0.80 + 0.10} \\
\alpha_1 &= \frac{0.10}{0.80 + 0.10} \\
y &= \alpha_2 \cdot \mathrm{Expert}_2(x) + \alpha_1 \cdot \mathrm{Expert}_1(x)
\end{aligned}
$$

MiniMind 默认主线是 `4 experts / top-1`，所以先把 top-1 跑通最重要。

把它放回源码里的 batch 计算，形状是：

| 变量 | 形状 | 含义 |
|---|---|---|
| $B$ | 标量 | batch size |
| $S$ | 标量 | sequence length |
| $H$ | 标量 | `hidden_size` |
| $N$ | 标量 | 展平后的 token 数，$N = B \times S$ |
| $E$ | 标量 | expert 总数，`num_experts` |
| $K$ | 标量 | 每个 token 选择几个 expert，`num_experts_per_tok` |
| $x$ | `[B, S, H]` | MoE 输入 |
| $x_{\text{flat}}$ | `[N, H]` | 把 batch 和 seq 展平成 token 维 |
| $r$ | `[N, E]` | router logits，softmax 前的分数 |
| $p$ / `scores` | `[N, E]` | router softmax 概率 |
| $p_{t,e}$ | 标量 | 第 `t` 个 token 分给第 `e` 个 expert 的概率，即 `scores[t, e]` |
| `topk_idx` | `[N, K]` | 每个 token 选中的 expert id |
| `topk_weight` | `[N, K]` | 每个 token 选中 expert 的权重 |
| $y_{\text{flat}}$ | `[N, H]` | 展平后的 MoE 输出 |
| $y$ | `[B, S, H]` | reshape 回原始 batch/seq 的输出 |

### 1.6 MoE 为什么需要 aux loss

Router 有一个常见问题：它可能总是把大部分 token 分给少数 expert。

比如：

```text
expert 0: 95% token
expert 1:  2% token
expert 2:  2% token
expert 3:  1% token
```

这样其他 expert 学不到东西，MoE 退化成“一个很忙的 expert + 几个闲置 expert”。

所以 MoE 常用一个负载均衡辅助损失。先把符号说清楚：

| 符号 | 形状 | 含义 | 对应源码 |
|---|---|---|---|
| $t$ | 标量下标 | 第 `t` 个 token。源码里先把 `[B, S]` 展平成 `N = B * S` 个 token。 | `x_flat = x.view(-1, hidden_dim)` |
| $e$ | 标量下标 | 第 `e` 个 expert。比如 `e=2` 就是 `self.experts[2]`。 | `for i, expert in enumerate(self.experts)` |
| $E$ | 标量 | expert 总数。 | `config.num_experts` |
| $K$ | 标量 | 每个 token 选几个 expert。MiniMind 默认是 `1`。 | `config.num_experts_per_tok` |
| $p$ / `scores` | `[N, E]` | 所有 token 对所有 expert 的 router 概率。 | `scores = F.softmax(...)` |
| $p_{t,e}$ | 标量 | 第 `t` 个 token 被 router 分给第 `e` 个 expert 的 softmax 概率。 | `scores[t, e]` |
| $\mathrm{load}_e$ | 标量 | top-k 选择后，实际有多少比例的 token 选中了 expert `e`。 | `load = F.one_hot(topk_idx, ...).float().mean(0)` |
| $\mathrm{prob}_e$ | 标量 | router 对 expert `e` 的平均偏好，即所有 token 的 $p_{t,e}$ 平均值。 | `scores.mean(0)` |
| $L_{\text{aux}}$ | 标量 | 辅助损失，用来惩罚 router 总偏向少数 expert。 | `self.aux_loss = ...` |

其中最容易混的是 $p_{t,e}$。它不是最终输出，也不是标签，而是 router 的“分配概率”：

$$
p_{t,e} = \mathrm{softmax}(W_{\text{router}}x_t)_e
$$

如果第 3 个 token 的 router 分数是：

```text
scores[3] = [0.05, 0.10, 0.80, 0.05]
```

那么：

```text
p_{3,0} = 0.05
p_{3,1} = 0.10
p_{3,2} = 0.80
p_{3,3} = 0.05
```

如果是 top-1 routing，这个 token 会选择 expert 2。  
但是 aux loss 不只看“最终选了谁”，也看 router 对所有 expert 的平均概率分布。

设一个 batch 展平后有 `N` 个 token，`E` 个 experts。MiniMind 里的负载均衡量可以这样理解：

$$
\begin{aligned}
\mathrm{load}_e
&= \frac{1}{N}\sum_{t=1}^{N}\mathbf{1}[\text{token } t \text{ 选择 expert } e] \\
\mathrm{prob}_e
&= \frac{1}{N}\sum_{t=1}^{N}p_{t,e} \\
L_{\text{aux}}
&= E \cdot \lambda_{\text{router}} \sum_{e=1}^{E}\mathrm{load}_e \cdot \mathrm{prob}_e
\end{aligned}
$$

这里 $\lambda_{\text{router}}$ 对应源码里的 `router_aux_loss_coef`。

这里的 $L_{\text{aux}}$ 不是语言模型的 next-token loss。语言模型主 loss 关心“下一个 token 预测对不对”；$L_{\text{aux}}$ 只关心“MoE 的专家使用是否太偏”。

训练时真正优化的是：

$$
L_{\text{train}} = L_{\text{lm}} + L_{\text{aux}}
$$

所以它的作用是给 router 一个轻微约束：在不破坏主任务的前提下，尽量不要让所有 token 都挤到同一个 expert。

注意一个源码细节：`F.one_hot(topk_idx, E)` 的形状是 `[N, K, E]`，MiniMind 接着 `.mean(0)` 后得到 `[K, E]`。默认 `K=1` 时，它等价于上面公式里的每个 expert 一个 $\mathrm{load}_e$；如果 `K>1`，源码会保留 top-k 每个位置的负载，再和 `scores.mean(0)` 广播相乘，最后 `sum()` 成一个标量 `aux_loss`。

<a id="l13-reading-order"></a>
## 2. 源码阅读顺序图

这节源码按这个顺序读：

```text
MiniMindConfig
-> MiniMindBlock
-> FeedForward
-> MOEFeedForward
-> MiniMindModel.forward 里的 aux_loss 汇总
```

对应文件都在：

```text
model/model_minimind.py
```

先看 config，是为了知道 dense FFN 和 MoE 由哪些参数控制。  
再看 block，是为了知道 FFN/MoE 插在 Transformer block 的哪个位置。  
然后看 dense FFN 和 MoE 的具体计算。  
最后看 aux loss 如何被模型 forward 返回。

<a id="l13-source-walkthrough"></a>
## 3. MiniMind 源码走读

### 第 1 步：Config 决定 FFN 和 MoE 尺寸

File: `model/model_minimind.py:25-45`

Read this to understand: FFN/MoE 的维度、专家数量、top-k 和 aux loss 系数都来自 config。

Code/config/template excerpt:

```python
self.hidden_act = kwargs.get("hidden_act", 'silu')
self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
...
self.num_experts = kwargs.get("num_experts", 4)
self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)
```

This code shows:

- `hidden_act='silu'` 决定 FFN 激活函数。
- `intermediate_size` 是 dense FFN 的中间维度。
- `num_experts` 是 MoE expert 数量。
- `num_experts_per_tok` 是每个 token 选择几个 expert。
- `router_aux_loss_coef` 控制 MoE 辅助损失权重。

### 第 2 步：Block 决定用 dense FFN 还是 MoE

File: `model/model_minimind.py:178-194`

Read this to understand: MoE 不是额外插入一层，而是替代普通 FFN 的位置。

Code/config/template excerpt:

```python
self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)
...
hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
```

This code shows:

- `config.use_moe=False` 时，block 使用普通 `FeedForward`。
- `config.use_moe=True` 时，block 使用 `MOEFeedForward`。
- 两者都必须输入 `[batch, seq, hidden_size]`，输出也必须是 `[batch, seq, hidden_size]`。
- 因为输出要和 `hidden_states` 做 residual add。

### 第 3 步：Dense FeedForward 是 SwiGLU

File: `model/model_minimind.py:136-146`

Read this to understand: MiniMind 的 FFN 不是简单两层 MLP，而是 gate/up/down 三投影。

Code/config/template excerpt:

```python
class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

这段源码对应的数学公式是：

$$
y = W_{\text{down}}\left(\mathrm{SiLU}(W_{\text{gate}}x) \odot W_{\text{up}}x\right)
$$

This code shows:

- `gate_proj` 和 `up_proj` 都把 `hidden_size` 扩到 `intermediate_size`。
- `return self.down_proj(...)` 是源码写法，不是课程里的数学公式写法。
- `down_proj` 把维度降回 `hidden_size`。
- 输入输出 shape 不变，中间 shape 变大。

### 第 4 步：MoE 先把 token 展平

File: `model/model_minimind.py:156-160`

Read this to understand: Router 是按 token 路由的，所以先把 `[batch, seq]` 合并成 token 维。

Code/config/template excerpt:

```python
batch_size, seq_len, hidden_dim = x.shape
x_flat = x.view(-1, hidden_dim)
scores = F.softmax(self.gate(x_flat), dim=-1)
topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
```

This code shows:

- `x_flat.shape = [batch * seq, hidden_dim]`。
- `self.gate(x_flat).shape = [batch * seq, num_experts]`。
- `scores.shape = [batch * seq, num_experts]`，是每个 token 对每个 expert 的概率。
- `topk_idx.shape = [batch * seq, num_experts_per_tok]`，是每个 token 选择的 expert id。
- `topk_weight.shape = [batch * seq, num_experts_per_tok]`，是对应 expert 的权重。

### 第 5 步：MoE 对每个 expert 收集 token 并写回

File: `model/model_minimind.py:161-170`

Read this to understand: MiniMind 的 MoE 实现是按 expert 循环，找到分给这个 expert 的 token，再把 expert 输出加回对应 token 位置。

Code/config/template excerpt:

```python
if self.config.norm_topk_prob: topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
y = torch.zeros_like(x_flat)
for i, expert in enumerate(self.experts):
    mask = (topk_idx == i)
    if mask.any():
        token_idx = mask.any(dim=-1).nonzero().flatten()
        weight = topk_weight[mask].view(-1, 1)
        y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
```

This code shows:

- `norm_topk_prob` 会把被选中的 top-k 权重重新归一化。
- `mask.shape = [batch * seq, num_experts_per_tok]`，找出哪些 token 选择了当前 expert。
- `token_idx.shape = [M]`，其中 `M` 是当前 expert 实际接收到的 token 数。
- `x_flat[token_idx].shape = [M, hidden_dim]`，这是当前 expert 的输入。
- `expert(x_flat[token_idx]).shape = [M, hidden_dim]`，这是当前 expert 的输出。
- `weight.shape = [M, 1]`，用于给当前 expert 的输出加权。
- `index_add_` 把 `[M, hidden_dim]` 的 expert 输出加回 `y.shape = [batch * seq, hidden_dim]` 的原 token 行号。

top-1 时，每个 token 只会写回一次。top-k 大于 1 时，同一个 token 会从多个 expert 收到输出，所以要加权相加。

### 第 6 步：MoE 计算 aux loss

File: `model/model_minimind.py:171-176`

Read this to understand: aux loss 是 router 的负载均衡约束，不是语言模型主 loss。

Code/config/template excerpt:

```python
if self.training and self.config.router_aux_loss_coef > 0:
    load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
    self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
else:
    self.aux_loss = scores.new_zeros(1).squeeze()
return y.view(batch_size, seq_len, hidden_dim)
```

This code shows:

- 只有训练模式才计算 aux loss。
- `F.one_hot(topk_idx, num_experts).shape = [batch * seq, num_experts_per_tok, num_experts]`。
- `load = ...mean(0)` 的源码形状是 `[num_experts_per_tok, num_experts]`。
- MiniMind 默认 `num_experts_per_tok=1` 时，可以把它理解成每个 expert 被选中的比例，也就是公式里的 $\mathrm{load}_e$。
- `scores.mean(0).shape = [num_experts]`，表示 router 给每个 expert 的平均概率，也就是公式里的 $\mathrm{prob}_e$。
- `(load * scores.mean(0)).sum()` 最后变成一个标量，也就是 `aux_loss`。
- 输出 reshape 回 `[batch, seq, hidden_dim]`。

### 第 7 步：MiniMindModel 汇总所有 MoE 层的 aux loss

File: `model/model_minimind.py:230-232`

Read this to understand: 每层 MoE 都有自己的 aux loss，最终会加总后返回。

Code/config/template excerpt:

```python
hidden_states = self.norm(hidden_states)
aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
return hidden_states, presents, aux_loss
```

This code shows:

- dense FFN 没有 aux loss。
- 只有 `MOEFeedForward` 层的 `aux_loss` 会被汇总。
- CausalLM forward 会把 `aux_loss` 返回出去，后续训练脚本再把它加到训练 loss 里。

### 第 8 步：训练脚本把 aux loss 加到主 loss

File: `trainer/train_pretrain.py:35-38`

Read this to understand: `L_aux` 最终是怎么参与反向传播的。

Code/config/template excerpt:

```python
with autocast_ctx:
    res = model(input_ids, labels=labels)
    loss = res.loss + res.aux_loss
    loss = loss / args.accumulation_steps
```

This code shows:

- `res.loss` 是 causal LM 的 next-token loss，也就是 $L_{\text{lm}}$。
- `res.aux_loss` 是 MoE router 的负载均衡损失，也就是 $L_{\text{aux}}$。
- `loss = res.loss + res.aux_loss` 才是最终反向传播的训练损失。

<a id="l13-must-write"></a>
## 4. 本节必须会写 / 暂时不要求

必须会写：

1. Dense FeedForward：

$$
\begin{aligned}
u &= W_{\text{up}}x \\
g &= W_{\text{gate}}x \\
h &= \mathrm{SiLU}(g) \odot u \\
y &= W_{\text{down}}h
\end{aligned}
$$

2. Router 与 top-k 加权：

$$
\begin{aligned}
r &= W_{\text{router}}x \\
p &= \mathrm{softmax}(r) \\
S &= \mathrm{TopK}(p, k) \\
y &= \sum_{e \in S}\alpha_e \cdot \mathrm{Expert}_e(x)
\end{aligned}
$$

3. MoE token dispatch：

```text
找到每个 expert 对应的 token
只对这些 token 调用 expert
用 index_add_ 写回原位置
```

4. MoE aux loss：

$$
L_{\text{aux}}
= E \cdot \lambda_{\text{router}} \sum_{e=1}^{E}\mathrm{load}_e \cdot \mathrm{prob}_e
$$

暂时不要求：

```text
1. fused MoE kernel
2. expert parallel / 多机专家并行
3. capacity factor / token dropping
4. DeepSpeed-MoE / Megatron-LM 工程实现
5. 更复杂的 router 正则化
```

本节重点是对齐 MiniMind 的 PyTorch 原生实现。

<a id="l13-handwrite"></a>
## 5. 手写模块

本节你要补的是：

```text
course/impl/core/model_parts.py
```

### 5.1 补 `FeedForward`

接口已经在骨架里：

```python
class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str = "silu"):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

你要实现的行为：

- 输入：`x: [batch, seq, hidden_size]`
- 输出：`y: [batch, seq, hidden_size]`
- 核心公式：

$$
\begin{aligned}
u &= W_{\text{up}}x \\
g &= W_{\text{gate}}x \\
h &= \mathrm{SiLU}(g) \odot u \\
y &= W_{\text{down}}h
\end{aligned}
$$

对齐源码：

```text
model/model_minimind.py:136-146
```

注意：

- 层名字最好保持 `gate_proj`、`up_proj`、`down_proj`，方便测试脚本拷贝原模型权重。
- `bias=False`，和原源码一致。
- `hidden_act` 先支持 `silu` 即可。

### 5.2 补 `MOEFeedForward`

本节已经在骨架里加入 `MOEFeedForward`。你要实现的行为：

```text
输入:
x: [batch, seq, hidden_size]

中间:
x_flat: [batch * seq, hidden_size]
scores: [batch * seq, num_experts]
topk_idx: [batch * seq, num_experts_per_tok]
topk_weight: [batch * seq, num_experts_per_tok]

输出:
y: [batch, seq, hidden_size]
```

对齐源码：

```text
model/model_minimind.py:148-176
```

实现时先保证 top-1 能跑通，再保证 `num_experts_per_tok > 1` 时能用 `index_add_` 加权累加。

<a id="l13-alignment-test"></a>
## 6. 对齐测试

本节新增对齐测试：

```text
course/impl/tests/test_ffn_moe_alignment.py
```

运行命令：

```bash
cd /home/sun/minimind
python course/impl/tests/test_ffn_moe_alignment.py
```

现在还没有实现时，这个测试会因为 `NotImplementedError` 失败。等你补完 `FeedForward` 和 `MOEFeedForward` 后，它应该打印：

```text
feedforward_max_abs_diff=...
moe_eval_max_abs_diff=...
moe_train_max_abs_diff=...
moe_aux_loss_abs_diff=...
```

验收标准：

```text
所有 max_abs_diff 都应该接近 0。
一般小于 1e-6 就可以认为对齐。
```

测试做了什么：

```text
1. 实例化 MiniMind 原版 FeedForward / MOEFeedForward。
2. 实例化 course/impl 里的手写版本。
3. 把原版权重复制到手写版本。
4. 用同一个输入 x 前向。
5. 比较输出和 aux_loss。
```

这比单纯看 shape 更严格：它要求你的实现和原源码数值一致。

<a id="l13-stage-assembly"></a>
## 7. 阶段组装

本节完成后，模型结构阶段会多出两个可复用模块：

```text
course/impl/core/model_parts.py::FeedForward
course/impl/core/model_parts.py::MOEFeedForward
```

它们以后会接到教学版 block 中：

```text
hidden_states
-> Attention
-> residual
-> RMSNorm
-> FeedForward 或 MOEFeedForward
-> residual
```

当前模型结构阶段还缺：

```text
RMSNorm
Attention
RoPE
KV cache
MiniMindBlock
CourseMiniMindForCausalLM
```

这些会在后续阶段组装或回补时补齐。第 13 课先把 FFN/MoE 这块手写清楚。

<a id="l13-check"></a>
## 8. 本节检查

1. Attention 和 FFN 在 Transformer block 里的分工有什么区别？
2. MiniMind 的 FFN 为什么有 `gate_proj`、`up_proj`、`down_proj` 三个线性层？
3. 为什么 FFN 中间可以是 `intermediate_size`，但最终必须回到 `hidden_size`？
4. MoE 的 router 输出的 `scores` 表示什么？
5. top-1 MoE 和 dense FFN 的计算路径有什么区别？
6. 如果输入 `x.shape = [B, S, H]`，写出 `x_flat`、`scores`、`topk_idx`、`topk_weight`、`y` 的形状。
7. 为什么 MoE 需要 aux loss？它和 causal LM loss 是一回事吗？
8. `index_add_` 在 MoE 里解决了什么问题？

<a id="l13-next"></a>
## 9. 下一课

第 14 课进入训练机制：学习率、梯度累积和混合精度。

从下一课开始，我们会从模型结构转向训练脚本。你要看懂的不只是 `loss.backward()`，还要看清：

```text
一个 batch 的 loss 如何累积；
什么时候 optimizer.step；
学习率怎么随 step 变化；
混合精度如何影响 forward/backward。
```
