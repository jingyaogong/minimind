# 第 9 课：Embedding、RMSNorm 与残差 Block

这一课只解决一个问题：`input_ids` 进入 MiniMind 后，如何变成 `hidden_states`，并在每个 block 里保持 `[batch, seq, hidden_size]` 这条主干 shape。

## 目录

- [0. 本节主线](#l09-mainline)
- [1. 本节要懂的 5 个原理](#l09-principles)
- [2. 变量流转](#l09-flow)
- [3. 原理一：Embedding 把 token id 变成向量](#l09-embedding)
- [4. 原理二：主干 shape 必须保持不变](#l09-main-shape)
- [5. 原理三：RMSNorm 只归一化最后一维](#l09-rmsnorm)
- [6. 原理四：残差连接要求 shape 对齐](#l09-residual)
- [7. 原理五：final norm 和 lm_head 完成输出转换](#l09-final-norm-lm-head)
- [8. 实验验证](#l09-experiment)
- [9. 本节检查](#l09-check)
- [10. 下一课](#l09-next)

<a id="l09-mainline"></a>
## 0. 本节主线

MiniMind 主干数据流是：

```text
input_ids [batch, seq]
-> embed_tokens
-> hidden_states [batch, seq, hidden_size]
-> input_layernorm
-> self_attn
-> residual add
-> post_attention_layernorm
-> mlp
-> residual add
-> 下一个 block
-> final norm
-> lm_head
-> logits [batch, seq, vocab_size]
```

所以本节的核心不是 Attention 公式，而是先看懂 Transformer block 的外壳：

```text
Embedding 负责升维；
RMSNorm 负责稳定每个 token 的 hidden 向量；
Attention/MLP 负责变换 hidden_states；
Residual 负责把变换结果加回主干；
整个 block 输入输出 shape 都保持 [batch, seq, hidden_size]。
```

<a id="l09-principles"></a>
## 1. 本节要懂的 5 个原理

| 原理 | 要理解什么 | 源码证据 |
|---|---|---|
| Embedding 把 token id 变成向量 | `input_ids` 是整数，进入模型后变成 `hidden_states` | `model/model_minimind.py:196-204`, `model/model_minimind.py:209-215` |
| 主干 shape 必须保持不变 | block 输入输出都要是 `[batch, seq, hidden_size]`，后面才能层层相接 | `model/model_minimind.py:186-194`, `model/model_minimind.py:220-230` |
| RMSNorm 只归一化最后一维 | 它不改变 batch/seq/hidden shape，只稳定 hidden 向量尺度 | `model/model_minimind.py:50-60` |
| 残差连接要求 shape 对齐 | Attention 和 MLP 输出必须能加回原来的 `hidden_states` | `model/model_minimind.py:186-194` |
| final norm 和 lm_head 完成输出转换 | 主干最后仍是 hidden，再映射成词表 logits | `model/model_minimind.py:230-232`, `model/model_minimind.py:245-249` |

学完本节，你应该能读懂 `MiniMindModel.forward` 的主干流程，并能说清楚为什么每个 block 都要保持同一个 hidden shape。

<a id="l09-flow"></a>
## 2. 变量流转

最小 shape 流：

```text
input_ids
shape = [batch_size, seq_len]
dtype = torch.long
含义 = token id

embed_tokens(input_ids)
shape = [batch_size, seq_len, hidden_size]
含义 = 每个 token 的向量表示

MiniMindBlock(hidden_states)
shape = [batch_size, seq_len, hidden_size]
含义 = 被 Attention 和 MLP 改写过的 token 表示

lm_head(hidden_states)
shape = [batch_size, seq_len, vocab_size]
含义 = 每个位置对下一个 token 的预测分数
```

这一节先记住一条规则：

```text
只要还在 Transformer 主干里，shape 基本保持 [batch, seq, hidden_size]。
```

<a id="l09-embedding"></a>
## 3. 原理一：Embedding 把 token id 变成向量

### 原理讲解

tokenizer 输出的 `input_ids` 只是整数序列，例如：

```text
[1, 345, 1024, 2]
```

这些数字本身没有连续空间里的几何意义。模型第一步要做的是查表：

```text
token id -> hidden_size 维向量
```

这个查表矩阵就是 embedding：

```text
embed_tokens.weight: [vocab_size, hidden_size]
```

如果 `input_ids.shape = [2, 6]`，`hidden_size=64`，那么 embedding 后：

```text
hidden_states.shape = [2, 6, 64]
```

从这一刻开始，模型不再处理整数 token id，而是在处理每个 token 的向量表示。

### 源码证据 A：embedding 矩阵在哪里创建

文件：`model/model_minimind.py:196-204`

看它是为了理解：模型如何根据 config 创建输入 embedding。

源码：

```python
class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

这段代码说明：

- `nn.Embedding(config.vocab_size, config.hidden_size)` 创建 token 查表矩阵。
- embedding 的行数等于词表大小。
- embedding 的列数等于模型 hidden 维度。
- `layers` 和 `norm` 都是在 embedding 后继续处理 hidden_states 的模块。

### 源码证据 B：input_ids 如何变成 hidden_states

文件：`model/model_minimind.py:209-215`

看它是为了理解：forward 开始时变量含义如何变化。

源码：

```python
def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
    batch_size, seq_length = input_ids.shape
    if hasattr(past_key_values, 'layers'): past_key_values = None
    past_key_values = past_key_values or [None] * len(self.layers)
    start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
    hidden_states = self.dropout(self.embed_tokens(input_ids))
```

这段代码说明：

- 进入 forward 时，`input_ids` 只有 `[batch, seq]` 两维。
- `self.embed_tokens(input_ids)` 后，多出最后一维 `hidden_size`。
- 变量名从 `input_ids` 变成 `hidden_states`，说明后面处理的是向量表示。

### 理解到这一步就够

你应该能说清楚：

```text
input_ids 是 token 的编号；
hidden_states 是 token 的向量；
embedding 是编号到向量的查表。
```

暂时不用看：

- embedding 初始化分布。
- embedding 语义空间如何形成。
- 词向量可视化。

<a id="l09-main-shape"></a>
## 4. 原理二：主干 shape 必须保持不变

### 原理讲解

Transformer block 是可以一层接一层堆起来的，前提是：

```text
第 N 层输出 shape = 第 N+1 层输入 shape
```

MiniMind 主干里这个 shape 是：

```text
[batch_size, seq_len, hidden_size]
```

Attention 会改写每个 token 对上下文的吸收方式，MLP 会改写每个 token 自身的非线性表示，但它们最终都要回到 `hidden_size` 维。否则下一层 block 就接不上。

这就是为什么你读源码时要一直盯住这条主干：

```text
hidden_states
```

它像一条流水线：每个 block 接过来，处理完，再交给下一个 block。

### 源码证据 A：block 输入输出都是 hidden_states

文件：`model/model_minimind.py:186-194`

看它是为了理解：一个 block 如何接收并返回同一个主干变量。

源码：

```python
def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
    residual = hidden_states
    hidden_states, present_key_value = self.self_attn(
        self.input_layernorm(hidden_states), position_embeddings,
        past_key_value, use_cache, attention_mask
    )
    hidden_states += residual
    hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
    return hidden_states, present_key_value
```

这段代码说明：

- block 输入变量叫 `hidden_states`。
- Attention 的输出仍然赋值给 `hidden_states`。
- MLP 处理后仍然回到 `hidden_states`。
- block 最后返回新的 `hidden_states` 给下一层。

### 源码证据 B：多层 block 串起来

文件：`model/model_minimind.py:220-230`

看它是为了理解：同一个主干变量如何穿过所有层。

源码：

```python
presents = []
for layer, past_key_value in zip(self.layers, past_key_values):
    hidden_states, present = layer(
        hidden_states,
        position_embeddings,
        past_key_value=past_key_value,
        use_cache=use_cache,
        attention_mask=attention_mask
    )
    presents.append(present)
hidden_states = self.norm(hidden_states)
```

这段代码说明：

- 每一层都接收上一层的 `hidden_states`。
- 每一层都返回更新后的 `hidden_states`。
- 循环结束后再做 final RMSNorm。

### 理解到这一步就够

你应该能画出：

```text
hidden_states -> block0 -> hidden_states -> block1 -> hidden_states -> final norm
```

暂时不用看：

- `present_key_value` 的缓存内容。
- RoPE 的位置编码细节。
- Attention 内部矩阵乘法。

<a id="l09-rmsnorm"></a>
## 5. 原理三：RMSNorm 只归一化最后一维

### 原理讲解

RMSNorm 是一种归一化层。它处理的对象是每个 token 的 hidden 向量。

如果：

```text
hidden_states.shape = [batch, seq, hidden_size]
```

RMSNorm 做的是沿最后一维计算均方根：

```text
sqrt(mean(x^2 over hidden_size))
```

然后把这个 token 的 hidden 向量缩放到更稳定的尺度。

重要的是：

```text
RMSNorm 不改变 shape
```

输入是 `[batch, seq, hidden_size]`，输出仍然是 `[batch, seq, hidden_size]`。

MiniMind 使用的是 Pre-Norm 结构，也就是在两个子步骤里都先 norm，再送进对应模块。注意这不是两条并列路径，而是同一个 block 里的顺序执行：

```text
hidden_states -> RMSNorm -> Attention -> residual add
hidden_states -> RMSNorm -> MLP -> residual add
```

### 源码证据 A：RMSNorm 的实现

文件：`model/model_minimind.py:50-60`

看它是为了理解：RMSNorm 为什么只影响数值尺度，不改变 shape。

源码：

```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)
```

这段代码说明：

- `mean(-1, keepdim=True)` 表示沿最后一维计算。
- `keepdim=True` 保留维度，所以可以广播回原 shape。
- `self.weight` 的长度是 `hidden_size`，用于每个 hidden 维度的缩放。
- `type_as(x)` 会把输出转回输入 dtype。

### 源码证据 B：RMSNorm 放在 block 的两个位置

文件：`model/model_minimind.py:178-184`

看它是为了理解：一个 block 里为什么有两个 norm。

源码：

```python
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)
```

这段代码说明：

- `input_layernorm` 在 Attention 前。
- `post_attention_layernorm` 在 MLP 前。
- 两个 norm 的维度都是 `hidden_size`。

### 理解到这一步就够

你应该能说清楚：

```text
RMSNorm 是对每个 token 的 hidden 向量做尺度归一化；
它不会改变 batch、seq、hidden 三个维度。
```

暂时不用看：

- RMSNorm 和 LayerNorm 的严格差异。
- 归一化为什么有助于深层训练稳定。

<a id="l09-residual"></a>
## 6. 原理四：残差连接要求 shape 对齐

### 原理讲解

残差连接就是把模块输出加回原来的主干：

```text
new_hidden = module(norm(hidden)) + old_hidden
```

它的好处是：每个模块不需要从零重写全部表示，而是在原表示上增加一个变化量。

但残差相加有一个硬要求：

```text
两个张量 shape 必须一致
```

所以 Attention 输出必须是 `[batch, seq, hidden_size]`，MLP 输出也必须是 `[batch, seq, hidden_size]`。这也是为什么 Attention 末尾有 `o_proj` 映射回 `hidden_size`，FFN 末尾有 `down_proj` 映射回 `hidden_size`。这些细节后面会继续展开。

### 源码证据 A：Attention 残差

文件：`model/model_minimind.py:186-193`

看它是为了理解：Attention 输出如何加回 block 输入。

源码：

```python
def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
    residual = hidden_states
    hidden_states, present_key_value = self.self_attn(
        self.input_layernorm(hidden_states), position_embeddings,
        past_key_value, use_cache, attention_mask
    )
    hidden_states += residual
```

这段代码说明：

- `residual` 保存的是进入 Attention 前的主干。
- `self_attn(...)` 输出新的 `hidden_states`。
- `hidden_states += residual` 把 Attention 输出加回原主干。

### 源码证据 B：MLP 残差

文件：`model/model_minimind.py:192-194`

看它是为了理解：MLP 输出如何加回 Attention 后的主干。

源码：

```python
hidden_states += residual
hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
return hidden_states, present_key_value
```

这段代码说明：

- Attention 残差后，`hidden_states` 继续作为 MLP 的输入主干。
- MLP 前先做 `post_attention_layernorm`。
- MLP 输出再次加回 `hidden_states`。

### 理解到这一步就够

你应该能说清楚：

```text
残差连接不是拼接，也不是替换；
它是相同 shape 的张量相加。
```

暂时不用看：

- 残差连接的梯度流数学证明。
- DeepNorm、ReZero 等残差变体。

<a id="l09-final-norm-lm-head"></a>
## 7. 原理五：final norm 和 lm_head 完成输出转换

### 原理讲解

所有 block 处理完后，模型还没有直接输出 token 概率。

这时的主干仍然是：

```text
hidden_states: [batch, seq, hidden_size]
```

MiniMind 先做一次 final RMSNorm，然后 `MiniMindForCausalLM` 用 `lm_head` 把最后一维从 `hidden_size` 映射成 `vocab_size`：

```text
[batch, seq, hidden_size]
-> lm_head
-> [batch, seq, vocab_size]
```

这个 `[batch, seq, vocab_size]` 就是 logits。后面 causal LM loss 会把 logits 和 labels shift 后做 cross entropy。

### 源码证据 A：主干最后的 final norm

文件：`model/model_minimind.py:220-232`

看它是为了理解：所有 block 之后，模型返回的还是 hidden_states。

源码：

```python
for layer, past_key_value in zip(self.layers, past_key_values):
    hidden_states, present = layer(
        hidden_states,
        position_embeddings,
        past_key_value=past_key_value,
        use_cache=use_cache,
        attention_mask=attention_mask
    )
    presents.append(present)
hidden_states = self.norm(hidden_states)
aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
return hidden_states, presents, aux_loss
```

这段代码说明：

- block 循环结束后做 final norm。
- `MiniMindModel` 返回的是 hidden_states，不是 logits。
- logits 是外层 `MiniMindForCausalLM` 继续生成的。

### 源码证据 B：lm_head 把 hidden 映射成 logits

文件：`model/model_minimind.py:245-249`

看它是为了理解：什么时候从 hidden_size 变成 vocab_size。

源码：

```python
def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
    hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :])
    loss = None
```

这段代码说明：

- `self.model(...)` 只负责主干 hidden 表示。
- `self.lm_head(...)` 负责变成词表 logits。
- logits 的最后一维是 `vocab_size`。

### 理解到这一步就够

你应该能说清楚：

```text
MiniMindModel 输出 hidden_states；
MiniMindForCausalLM 再用 lm_head 输出 logits。
```

暂时不用看：

- `logits_to_keep` 的推理优化用途。
- loss shift 的细节，第 10 课会集中讲。

<a id="l09-experiment"></a>
## 8. 实验验证

### 实验 A：跟踪一个 tiny 模型的 block shape

这个实验验证：

```text
input_ids
-> embedding
-> layernorm
-> attention
-> residual add
-> layernorm
-> mlp
-> residual add
-> final norm
-> logits
```

运行：

```bash
cd /home/sun/minimind
PYTHONDONTWRITEBYTECODE=1 python course/labs/trace_block_shapes.py
```

记录：

```text
input_ids.shape =
embed_tokens.weight.shape =
after embed_tokens + dropout shape =
block input shape =
after input_layernorm shape =
self_attn output shape =
after attention residual add shape =
mlp output shape =
block output shape =
logits.shape =
```

你应该看到：

```text
input_ids.shape = (2, 6)
after embed_tokens + dropout shape = (2, 6, 64)
block input shape = (2, 6, 64)
block output shape = (2, 6, 64)
logits.shape = (2, 6, 6400)
```

这说明：

- embedding 把 `[batch, seq]` 升成 `[batch, seq, hidden_size]`。
- block 内部多次变换，但主干 shape 不变。
- `lm_head` 最后把 hidden 映射到 `vocab_size`。

### 实验 B：改变 hidden_size 看主干 shape

运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python course/labs/trace_block_shapes.py --hidden_size 96 --num_hidden_layers 3
```

记录：

```text
after embed_tokens + dropout shape =
每个 block output shape =
logits.shape =
```

你应该看到：

```text
hidden_states 的最后一维从 64 变成 96；
logits 的最后一维仍然是 vocab_size=6400。
```

这说明：

- `hidden_size` 控制主干向量宽度。
- `vocab_size` 控制最终输出类别数。
- block 数量变多，只是主干多走几层，不改变每层输入输出 shape。

<a id="l09-check"></a>
## 9. 本节检查

如果你真懂了本节，应该能不看答案说清楚：

1. `input_ids` 和 `hidden_states` 的 shape 和含义分别是什么。
2. embedding 为什么会让张量从二维变成三维。
3. RMSNorm 沿哪一维做归一化，为什么不改变 shape。
4. 残差连接为什么要求 Attention/MLP 输出都是 `[batch, seq, hidden_size]`。
5. `MiniMindModel` 输出的是 logits 还是 hidden_states。
6. logits 的最后一维为什么是 `vocab_size`。

<a id="l09-next"></a>
## 10. 下一课

第 10 课进入 `Causal LM forward 与 loss shift`：我们会集中看 `MiniMindForCausalLM.forward`，把 `hidden_states -> logits -> shifted logits/labels -> cross entropy` 这条链路讲清楚。
