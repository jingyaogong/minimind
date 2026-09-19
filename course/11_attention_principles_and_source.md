# 第 11 课：Attention 原理与源码

这一课只解决一个问题：一个 token 如何通过 Attention 从上下文 token 里选择并汇总信息，然后仍然回到 `[batch, seq, hidden_size]` 主干。

## 目录

- [0. 本节主线](#l11-mainline)
- [1. 本节要懂的 5 个层次](#l11-levels)
- [2. Attention 一次完整原理讲解](#l11-complete-principle)
- [3. Attention.forward 一次完整源码走读](#l11-forward-walkthrough)
- [4. 源码变量字典](#l11-variable-dict)
- [5. 关键源码对照](#l11-source-map)
- [6. 实验验证](#l11-experiment)
- [7. 本节检查](#l11-check)
- [8. 下一课](#l11-next)

<a id="l11-mainline"></a>
## 0. 本节主线

Attention 的完整逻辑是：

```text
每个 token 先有自己的 hidden 向量
-> 每个 hidden 向量生成 Q/K/V 三种角色
-> 当前 token 的 Q 和所有可见 token 的 K 计算匹配分数
-> causal mask 把未来 token 屏蔽掉
-> softmax 把分数变成注意力权重
-> 用注意力权重对所有可见 token 的 V 加权求和
-> 得到当前 token 汇总上下文后的新表示
-> 所有 token、所有 head 并行做这件事
-> 合并多头并用 o_proj 回到 hidden_size
```

一句话：

```text
Q/K 决定看谁，V 决定拿什么，softmax 决定拿多少，causal mask 决定不能看未来，o_proj 决定回到主干。
```

这节课分成两大块：

```text
先完整讲一遍 Attention 原理；
再完整走一遍 MiniMind 的 Attention.forward 源码。
```

<a id="l11-levels"></a>
## 1. 本节要懂的 5 个层次

| 层次 | 要理解什么 | 源码位置 |
|---|---|---|
| Attention 解决什么问题 | token 需要从上下文 token 里取信息 | `model/model_minimind.py:178-194` |
| 一次完整 Attention 怎么算 | Q/K 打分，softmax，汇总 V | `model/model_minimind.py:111-134` |
| Q/K/V 从哪里来 | 同一份 hidden_states 投影成三种角色 | `model/model_minimind.py:91-116` |
| mask 和多头怎么进来 | causal mask 禁止看未来，多头并行做 attention | `model/model_minimind.py:123-131` |
| 怎么回到主干 | repeat_kv 对齐 K/V heads，o_proj 输出回 hidden_size | `model/model_minimind.py:86-89`, `model/model_minimind.py:132-134` |

学完本节，你应该能不看源码说出一遍 Attention 的完整计算过程，然后再能对着 `Attention.forward` 逐行解释每个变量的 shape 和作用。

<a id="l11-complete-principle"></a>
## 2. Attention 一次完整原理讲解

先不要看源码。我们用一个序列来理解 Attention：

```text
位置:   0   1   2   3
token:  A   B   C   D
```

假设现在看位置 2，也就是 token `C`。

因为这是 causal language model，`C` 只能看：

```text
A, B, C
```

不能看未来的 `D`。

### 2.1 每个 token 先有 hidden 向量

第 9 课讲过，进入 block 时，每个 token 都已经有自己的 hidden 向量：

```text
h_A, h_B, h_C, h_D
```

这些向量都在同一个 hidden 空间里，比如每个都是 64 维或 768 维。

如果模型只对每个 `h_i` 单独做 MLP，它就不能直接知道别的 token 里有什么信息。Attention 要补上的就是：

```text
让 h_C 能从 h_A、h_B、h_C 里按需取信息。
```

### 2.2 为什么要有 Q/K/V

Attention 把每个 hidden 向量投影成三种角色：

```text
q_i = W_Q h_i
k_i = W_K h_i
v_i = W_V h_i
```

对 token `C` 来说：

```text
q_C：C 当前想找什么信息
k_A/k_B/k_C：A/B/C 各自提供什么可匹配的特征
v_A/v_B/v_C：A/B/C 真正能贡献给 C 的内容
```

所以 Q/K/V 不是三个不同输入，而是同一批 hidden 向量的三种投影角色：

```text
Q/K 用来决定“看谁”；
V 用来提供“拿什么”。
```

### 2.3 Q 和 K 怎么决定“看谁”

`C` 要决定看 A、B、C 各多少，就用 `q_C` 分别和它们的 key 做点积：

```text
score(C, A) = q_C · k_A / sqrt(head_dim)
score(C, B) = q_C · k_B / sqrt(head_dim)
score(C, C) = q_C · k_C / sqrt(head_dim)
```

点积越大，表示越匹配。这里的分数不是概率，只是未归一化的匹配分数。

除以 `sqrt(head_dim)` 是为了缩放分数尺度。否则 `head_dim` 越大，点积越容易变得很大，softmax 会过早变得极端。

### 2.4 causal mask 为什么出现

如果不加限制，`C` 也会和 `D` 算：

```text
score(C, D)
```

但训练语言模型时，位置 `C` 不能提前看到未来 token `D`。否则第 10 课讲的 next-token prediction 就会作弊。

所以 causal mask 会把未来位置变成：

```text
score(C, D) = -inf
```

softmax 之后：

```text
weight(C, D) = 0
```

也就是未来 token 完全不参与信息汇总。

### 2.5 softmax 后为什么乘 V

对 A、B、C 的分数经过 softmax 后，变成权重：

```text
weight(C, A), weight(C, B), weight(C, C)
```

这些权重表示：

```text
C 应该从 A/B/C 各拿多少信息。
```

然后真正汇总的是 value：

```text
new_C =
  weight(C, A) * v_A
+ weight(C, B) * v_B
+ weight(C, C) * v_C
```

这就是 Attention 的核心公式：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(head_dim) + mask) V
```

### 2.6 从一个 token 推广到所有 token

上面只讲了 `C` 如何看 A/B/C。实际模型会对每个位置都做同样的事：

```text
A 看 A
B 看 A/B
C 看 A/B/C
D 看 A/B/C/D
```

所以 attention scores 会形成一个矩阵：

```text
[query_position, key_position]
```

如果有 batch 和 heads，就变成：

```text
[batch, heads, query_seq, key_seq]
```

这就是为什么实验里 `scores.shape=(2, 4, 5, 5)`。

### 2.7 多头 attention 是什么

如果只用一个 head，模型只有一种“看上下文”的方式。

多头 attention 是把 hidden 向量拆成多个子空间：

```text
hidden_size = num_heads * head_dim
```

每个 head 都独立算一套：

```text
QK 分数 -> softmax -> 加权 V
```

最后把多个 head 的结果合并，再映射回 `hidden_size`。

所以多头不是多了几层，也不是复制 batch；它是在 hidden 维度内部并行做多个 attention 视角。

### 2.8 这一节原理到此为止

到这里你应该已经有完整观念：

```text
Attention 不是神秘模块；
它就是让每个 token 用 Q 去匹配可见 token 的 K，
再按匹配权重汇总这些 token 的 V。
```

下一步再看源码，源码只是把这套过程用张量并行实现。

<a id="l11-forward-walkthrough"></a>
## 3. Attention.forward 一次完整源码走读

这一节先完整走一遍源码，不拆成零散片段。

### 源码证据 A：Attention.forward 全流程

文件：`model/model_minimind.py:111-134`

看它是为了理解：MiniMind 如何把上一节的完整 Attention 原理实现成张量计算。

源码：

```python
def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
    bsz, seq_len, _ = x.shape
    xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
    xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
    xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
    xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
    xq, xk = self.q_norm(xq), self.k_norm(xk)
    cos, sin = position_embeddings
    xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
    if past_key_value is not None:
        xk = torch.cat([past_key_value[0], xk], dim=1)
        xv = torch.cat([past_key_value[1], xv], dim=1)
    past_kv = (xk, xv) if use_cache else None
    xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
    if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
        output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
    else:
        scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.is_causal: scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
        if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
        output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
    output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
    output = self.resid_dropout(self.o_proj(output))
    return output, past_kv
```

这段源码可以分成 8 步读。

### 第 1 步：确认输入 shape

```python
bsz, seq_len, _ = x.shape
```

这里的 `x` 是 block 传进 Attention 的 hidden states，来自：

```python
self.input_layernorm(hidden_states)
```

它的 shape 是：

```text
[batch, seq, hidden_size]
```

这一步只取出 batch 和 seq 长度，为后面 reshape 做准备。

### 第 2 步：从 x 投影出 Q/K/V

```python
xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
```

这里对应原理里的：

```text
q_i = W_Q h_i
k_i = W_K h_i
v_i = W_V h_i
```

同一份 `x` 通过三组 Linear 变成三种角色：

```text
xq: query
xk: key
xv: value
```

### 第 3 步：把 Q/K/V reshape 成多头

```python
xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
```

这一步把最后一维拆开：

```text
Q: [batch, seq, num_attention_heads, head_dim]
K: [batch, seq, num_key_value_heads, head_dim]
V: [batch, seq, num_key_value_heads, head_dim]
```

MiniMind 允许 Q heads 和 K/V heads 不同，这是 GQA 的设计。

### 第 4 步：Q/K norm 和 RoPE

```python
xq, xk = self.q_norm(xq), self.k_norm(xk)
cos, sin = position_embeddings
xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
```

`q_norm` 和 `k_norm` 对每个 head 的向量做 RMSNorm，稳定 Q/K 的尺度。

RoPE 作用在 Q/K 上，是为了让 Q/K 匹配时带上位置信息。注意 V 不参与 RoPE，因为位置信息主要影响“怎么匹配”，不是“拿回什么内容”。

RoPE 细节放第 12 课，这里只记住：

```text
RoPE 修改 Q/K，不修改 V。
```

### 第 5 步：KV cache 先跳过细节

```python
if past_key_value is not None:
    xk = torch.cat([past_key_value[0], xk], dim=1)
    xv = torch.cat([past_key_value[1], xv], dim=1)
past_kv = (xk, xv) if use_cache else None
```

这段用于生成阶段的 KV cache。

训练时通常一次性输入整段序列，可以先不看。第 12 课会专门讲：

```text
为什么生成时要缓存历史 K/V；
为什么不用每步重算全部历史。
```

本节只要知道：

```text
cache 缓存的是 K/V，不是 Q。
```

### 第 6 步：转成 attention 计算布局，并 repeat K/V

```python
xq, xk, xv = (
    xq.transpose(1, 2),
    repeat_kv(xk, self.n_rep).transpose(1, 2),
    repeat_kv(xv, self.n_rep).transpose(1, 2)
)
```

原来多头 shape 是：

```text
[batch, seq, heads, head_dim]
```

attention 矩阵乘法更方便的布局是：

```text
[batch, heads, seq, head_dim]
```

所以要 `transpose(1, 2)`。

K/V 先经过 `repeat_kv`，是因为 MiniMind 使用 GQA：

```text
Q heads 多；
K/V heads 少；
计算前把 K/V heads 复制到和 Q heads 一样多。
```

### 第 7 步：计算 scores、mask、softmax、加权 V

普通分支是最容易理解的：

```python
scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
if self.is_causal:
    scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
if attention_mask is not None:
    scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
```

这几行就是 Attention 原理公式：

```text
scores = QK^T / sqrt(head_dim)
weights = softmax(scores + mask)
output = weights V
```

shape 是：

```text
xq:      [batch, heads, query_seq, head_dim]
xk^T:    [batch, heads, head_dim, key_seq]
scores:  [batch, heads, query_seq, key_seq]
xv:      [batch, heads, key_seq, head_dim]
output:  [batch, heads, query_seq, head_dim]
```

causal mask 的作用是：

```text
未来 key position 的分数变成 -inf；
softmax 后未来位置权重变成 0。
```

flash attention 分支：

```python
F.scaled_dot_product_attention(...)
```

做的是同一件事，只是调用 PyTorch 优化实现。

### 第 8 步：合并 heads，o_proj 回到 hidden_size

```python
output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
output = self.resid_dropout(self.o_proj(output))
return output, past_kv
```

attention 内部输出是：

```text
[batch, heads, seq, head_dim]
```

先转回：

```text
[batch, seq, heads, head_dim]
```

再合并：

```text
[batch, seq, heads * head_dim]
```

最后 `o_proj` 输出：

```text
[batch, seq, hidden_size]
```

这样 Attention 才能回到 `MiniMindBlock` 的 residual stream：

```python
hidden_states += residual
```

### 完整源码走读到此为止

你读 `Attention.forward` 时，不要把它看成很多互不相关的小技巧。它就是这条链：

```text
x
-> Q/K/V
-> 多头 shape
-> Q/K 加位置
-> K/V cache 可选拼接
-> QK 打分
-> mask
-> softmax 权重
-> 汇总 V
-> 合并 heads
-> o_proj 回主干
```

<a id="l11-variable-dict"></a>
## 4. 源码变量字典

这一节把 `Attention.forward` 里的关键变量统一列出来，方便你回看源码。

| 变量 | shape | 含义 |
|---|---|---|
| `x` | `[batch, seq, hidden_size]` | Attention 输入 hidden states |
| `xq` | `[batch, seq, q_heads, head_dim]` 后转 `[batch, q_heads, seq, head_dim]` | query，用来匹配 key |
| `xk` | `[batch, seq, kv_heads, head_dim]` 后转 `[batch, q_heads, seq, head_dim]` | key，被 query 匹配 |
| `xv` | `[batch, seq, kv_heads, head_dim]` 后转 `[batch, q_heads, seq, head_dim]` | value，被权重加权汇总 |
| `scores` | `[batch, heads, query_seq, key_seq]` | 每个 query 位置看每个 key 位置的分数 |
| `attention_mask` | 可广播到 scores | padding 等外部 mask |
| `output` | 先 `[batch, heads, seq, head_dim]`，最后 `[batch, seq, hidden_size]` | Attention 汇总后的结果 |
| `past_kv` | `(xk, xv)` | 生成时缓存历史 K/V |

<a id="l11-source-map"></a>
## 5. 关键源码对照

前面已经完整走读过源码。这里只把几个容易混的点再对照一次。

### 5.1 Q/K/V 从哪里来

文件：`model/model_minimind.py:91-116`

看它是为了理解：Q/K/V 是同一份 `x` 的三套投影，不是三个外部输入。

源码：

```python
self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
...
xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
```

这段代码说明：

- 输入都是同一个 `x`。
- Q/K/V 是三套投影参数学出来的三种表示。
- Q/K 负责匹配，V 负责被汇总。

### 5.2 为什么 scores 是 `[batch, heads, seq, seq]`

文件：`model/model_minimind.py:123-128`

看它是为了理解：query position 和 key position 如何形成二维注意力矩阵。

源码：

```python
xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
```

这段代码说明：

- `xq` 是 `[batch, heads, query_seq, head_dim]`。
- `xk.transpose(-2, -1)` 是 `[batch, heads, head_dim, key_seq]`。
- 相乘后得到 `[batch, heads, query_seq, key_seq]`。

### 5.3 causal mask 为什么加在 scores 上

文件：`model/model_minimind.py:128-131`

看它是为了理解：未来 token 为什么不会进入 softmax 权重。

源码：

```python
scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
if self.is_causal: scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
```

这段代码说明：

- causal mask 是在 softmax 前加到 scores 上。
- 未来位置变成 `-inf`。
- softmax 后未来位置权重为 0。
- 最后才用权重乘 V。

### 5.4 GQA 和 o_proj 如何收尾

文件：`model/model_minimind.py:86-89`, `model/model_minimind.py:132-134`

看它是为了理解：K/V heads 如何对齐 Q heads，以及多头结果如何回到主干。

源码：

```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))

output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
output = self.resid_dropout(self.o_proj(output))
return output, past_kv
```

这段代码说明：

- `repeat_kv` 把较少的 K/V heads 扩到 Q heads 数量。
- 多头输出合并后经过 `o_proj`。
- 最终输出回到 `[batch, seq, hidden_size]`。

<a id="l11-experiment"></a>
## 6. 实验验证

### 实验 A：追踪完整 Attention 流程

这个实验验证：

```text
Q/K/V 来自同一个 hidden_states；
scores 是 QK 匹配矩阵；
causal mask 让未来位置权重为 0；
softmax(scores) @ V 后再 o_proj；
手算结果和模块 forward 完全一致。
```

运行：

```bash
cd /home/sun/minimind
PYTHONDONTWRITEBYTECODE=1 python course/labs/trace_attention_shapes.py
```

记录：

```text
x.shape =
q_proj(x).shape =
k_proj(x).shape =
v_proj(x).shape =
scores.shape =
attention weights for batch 0, head 0 =
future_attention_mass_for_batch0_head0 =
manual_vs_module_max_abs_diff =
```

你应该看到：

```text
x.shape=(2, 5, 64)
q_proj(x).shape=(2, 5, 64)
k_proj(x).shape=(2, 5, 32)
v_proj(x).shape=(2, 5, 32)
scores.shape=(2, 4, 5, 5)
future_attention_mass_for_batch0_head0=0.0000000000
manual_vs_module_max_abs_diff=0.000000000000
```

重点看输出的 attention weights 矩阵。它应该是下三角有效：

```text
第 0 行只能看第 0 列；
第 1 行只能看第 0、1 列；
第 2 行只能看第 0、1、2 列；
未来位置权重为 0。
```

### 实验 B：改变 head 配置

运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python course/labs/trace_attention_shapes.py --hidden_size 96 --num_attention_heads 6 --num_key_value_heads 3
```

记录：

```text
head_dim =
q_proj(x).shape =
k_proj(x).shape =
xq.shape =
xk after repeat shape =
scores.shape =
```

你应该看到：

```text
head_dim=16
q_proj(x).shape=(2, 5, 96)
k_proj(x).shape=(2, 5, 48)
xq.shape=(2, 5, 6, 16)
xk after repeat shape=(2, 5, 6, 16)
scores.shape=(2, 6, 5, 5)
```

这说明：

- head 配置改变的是 Attention 内部多头形状。
- 输出仍然要回到 `[batch, seq, hidden_size]`。

<a id="l11-check"></a>
## 7. 本节检查

如果你真懂了本节，应该能不看答案说清楚：

1. Attention 要解决的核心问题是什么。
2. 从一个 token 的角度，Attention 是如何从上下文拿信息的。
3. Q/K/V 为什么要分成三种角色。
4. QK scores 的每个元素表示什么。
5. 为什么 softmax 后乘的是 V，不是 K。
6. causal mask 和第 10 课的 next-token loss 是什么关系。
7. `Attention.forward` 从 `x` 到 `output` 的完整执行顺序是什么。
8. 为什么最终必须用 `o_proj` 回到 `[batch, seq, hidden_size]`。

<a id="l11-next"></a>
## 8. 下一课

第 12 课进入 `RoPE 与 KV cache`：我们会讲位置编码如何作用在 Q/K 上，以及生成时 past key/value 如何让模型不用每一步重算全部历史。
