# 第 12 课：RoPE 与 KV cache

这一课只解决一个问题：MiniMind 如何给 Q/K 注入位置信息，并在生成时复用历史 K/V，做到每步只计算新 token 但仍然和整段 forward 对齐。

## 目录

- [0. 本节主线](#l12-mainline)
- [1. 本节要懂的 6 个层次](#l12-levels)
- [2. RoPE 与 KV cache 一次完整原理讲解](#l12-complete-principle)
- [3. 从 generate 到 Attention 的完整源码走读](#l12-source-walkthrough)
- [4. 源码变量字典](#l12-variable-dict)
- [5. 关键源码对照](#l12-source-evidence)
- [6. 实验验证](#l12-experiment)
- [7. 本节检查](#l12-check)
- [8. 下一课](#l12-next)

<a id="l12-mainline"></a>
## 0. 本节主线

RoPE 与 KV cache 的完整逻辑是：

```text
hidden_states
-> 投影成 Q/K/V
-> 对 Q/K 做 norm
-> 按当前位置取出 cos/sin
-> 用 RoPE 旋转 Q/K，让 attention 打分带上位置信息
-> Attention 用带位置的 Q/K 算分数，用 V 汇总内容
-> use_cache=True 时保存当前层的 K/V
-> 下一步生成只输入新 token
-> 用历史 cache 长度得到 start_pos
-> 新 token 的 Q/K 用正确位置的 RoPE
-> 新 K/V 拼到旧 K/V 后继续 attention
```

一句话：

```text
RoPE 解决“当前位置怎么进入 attention 打分”，KV cache 解决“历史 token 的 K/V 不要每步重算”，二者靠 start_pos 对齐。
```

这节课不是单独背 RoPE 公式，也不是单独看 cache 变量。你要形成一个完整观念：

```text
如果生成第 100 个 token，只输入这个新 token，
它的 RoPE 位置仍然必须是 99，
并且它要和 cache 里的 0..98 位置 K/V 一起做 attention。
```

<a id="l12-levels"></a>
## 1. 本节要懂的 6 个层次

| 层次 | 要理解什么 | 源码位置 |
|---|---|---|
| 位置为什么进 attention | 语言模型不仅要知道 token 是什么，还要知道 token 在哪里 | `model/model_minimind.py:111-124` |
| RoPE 表怎么预先算好 | 每个位置、每个 head 维度都有对应的 cos/sin | `model/model_minimind.py:62-78`, `model/model_minimind.py:205-207` |
| RoPE 怎么作用到 Q/K | RoPE 旋转 Q/K，不旋转 V | `model/model_minimind.py:80-84`, `model/model_minimind.py:117-119` |
| cache 时怎么选正确位置 | `start_pos` 来自历史 K 的长度 | `model/model_minimind.py:211-219` |
| KV cache 存什么 | 每层保存已经算好的 K/V，下一步和新 K/V 拼接 | `model/model_minimind.py:120-123`, `model/model_minimind.py:221-232` |
| generate 如何使用 cache | 每轮只把未缓存的新 token 送进 forward | `model/model_minimind.py:257-288`, `eval_llm.py:82-87` |

学完本节，你应该能说明：RoPE 为什么必须跟 cache 长度配合；`past_key_values[0][0].shape[1]` 代表什么；为什么 cache 里存 K/V 而不是存所有 hidden_states。

<a id="l12-complete-principle"></a>
## 2. RoPE 与 KV cache 一次完整原理讲解

先不要看源码。先把两个问题分开：

```text
RoPE 解决的问题：
Attention 怎么知道 token 的位置关系？

KV cache 解决的问题：
生成时为什么不用每一步重复计算所有历史 token？
```

它们是两件事，但在生成阶段必须配合：

```text
RoPE 负责位置正确；
KV cache 负责计算复用；
start_pos 负责让“只输入新 token”时的位置仍然正确。
```

### 2.1 Attention 为什么需要位置

Attention 的核心打分是：

```text
score(i, j) = q_i · k_j / sqrt(head_dim)
```

这个公式只说：

```text
位置 i 的 query 和位置 j 的 key 有多匹配。
```

但如果 Q/K 里没有位置信息，模型很难知道：

```text
A 在 C 前面两个位置
A 在 C 前面十个位置
A 在 C 后面一个位置
```

这些不是同一件事。语言模型不只需要知道 token 是什么，还要知道 token 的顺序和距离。比如：

```text
我 喜欢 吃 苹果
苹果 喜欢 吃 我
```

这两句话 token 很接近，但位置关系变了，意思就变了。

一种做法是在 embedding 上加位置向量。MiniMind 用的是 RoPE：不直接给 `hidden_states` 加位置，而是在 attention 打分前，把位置信息放进 Q/K。

### 2.2 RoPE 是什么

RoPE 是 Rotary Position Embedding，中文可以理解成“旋转位置编码”。

它做的事情很具体：

```text
给不同位置的 Q/K 做不同角度的旋转。
```

先看二维向量：

```text
x = [a, b]
```

如果 token 在位置 `pos`，RoPE 会根据 `pos` 得到一个角度 `theta_pos`，然后把 `[a, b]` 旋转成：

```text
a' = a * cos(theta_pos) - b * sin(theta_pos)
b' = a * sin(theta_pos) + b * cos(theta_pos)
```

高维向量也是同样逻辑。一个 head 里的向量维度是 `head_dim`，RoPE 会把它拆成多组维度，每组按不同频率旋转：

```text
低频维度：适合表达较长距离的位置变化
高频维度：适合表达较短距离的位置变化
```

所以 RoPE 不是给 token 附加一个位置编号，而是让 Q/K 的方向随着位置变化。

MiniMind 代码里的批量写法是：

```text
x_rope = x * cos + rotate_half(x) * sin
```

这就是上面二维旋转公式的向量化实现。

### 2.3 RoPE 干了什么

第 11 课讲过，Attention 是用 Q/K 点积决定“看谁”：

```text
score(i, j) = q_i · k_j / sqrt(head_dim)
```

加了 RoPE 之后，真正参与点积的不是原始 Q/K，而是旋转后的 Q/K：

```text
q_i_rope = R(i) q_i
k_j_rope = R(j) k_j

score(i, j) = q_i_rope · k_j_rope / sqrt(head_dim)
            = (R(i) q_i) · (R(j) k_j) / sqrt(head_dim)
```

这里的 `R(i)` 可以理解成“位置 i 对应的旋转矩阵”。

关键点在这里：

```text
(R(i) q_i) · (R(j) k_j)
等价于
q_i 和 k_j 做点积时，中间带上了 i 与 j 的相对位置关系。
```

直观说：

```text
位置 i 的 token 要看位置 j 的 token 时，
attention 分数不只取决于 q_i 和 k_j 的内容，
还取决于 i 和 j 隔多远、方向如何。
```

这就是 RoPE 最重要的作用：

```text
RoPE 让 Q/K 的匹配分数带有相对位置信息。
```

不用死记矩阵推导，但要记住这个结论。它比“RoPE 把向量旋转一下”更重要。

### 2.4 为什么 RoPE 只作用在 Q/K 上

Attention 里三种角色是：

```text
Q：当前位置想找什么
K：每个位置提供什么可匹配特征
V：每个位置真正贡献什么内容
```

位置最直接影响的是“匹配关系”：

```text
当前位置 i 的 Q，要和历史位置 j 的 K 做匹配。
匹配分数应该知道 i 和 j 的相对位置。
```

所以 MiniMind 对 Q/K 做 RoPE：

```text
xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
```

但 V 不旋转。

原因是：

```text
Q/K 负责决定“看谁、看多少”；
V 负责提供“拿回来什么内容”。
```

先按 Q/K 算出注意力权重，再用权重去加权 V。位置主要影响权重，不直接改 V 的内容。

### 2.5 KV cache 是什么

现在换到生成阶段。

假设 prompt 是：

```text
位置:   0   1   2
token:  A   B   C
```

模型第一次 forward 处理 `A B C`，这一步通常叫 prefill。

在每一层 Attention 里，模型会算出：

```text
Q_A, Q_B, Q_C
K_A, K_B, K_C
V_A, V_B, V_C
```

其中 Q 是当前 token 做 query 时才用的。prefill 结束后，如果下一个 token 是 `D`，下一步只需要：

```text
D 的 Q
历史 A/B/C 的 K/V
```

历史 A/B/C 的 Q 不再需要，因为 A/B/C 作为 query 的输出已经算完了。

所以 KV cache 存的是每一层的历史 K/V：

```text
past_key_values[layer] = (past_k, past_v)
```

形状是：

```text
past_k: [batch, past_seq, num_key_value_heads, head_dim]
past_v: [batch, past_seq, num_key_value_heads, head_dim]
```

在 MiniMind 里要注意：

```text
cache 里的 K 是已经做过 RoPE 的 K；
cache 里的 V 是没有做 RoPE 的 V。
```

因为源码顺序是：

```text
Q/K 做 RoPE
-> 拼接历史 K/V
-> 保存 past_kv
```

### 2.6 KV cache 干了什么

不用 cache 时，每生成一步都把完整序列重新送进模型：

```text
已知 A B C，要预测 D：
forward(A B C)

生成 D 后，要预测 E：
forward(A B C D)

生成 E 后，要预测 F：
forward(A B C D E)
```

问题是：`A B C` 在后面每一步都会被重复计算。

用 cache 时，时间线变成：

```text
prefill:
输入 A B C
算出 A/B/C 的 K/V cache
用 C 位置 logits 预测 D

decode 1:
把 D 追加到 input_ids
本轮只输入 D
复用 A/B/C 的 K/V cache
只计算 D 的 Q/K/V
用 D 位置 logits 预测 E

decode 2:
把 E 追加到 input_ids
本轮只输入 E
复用 A/B/C/D 的 K/V cache
只计算 E 的 Q/K/V
用 E 位置 logits 预测 F
```

所以 KV cache 的作用不是改变模型会说什么，而是改变计算方式：

```text
不使用 cache：每步重复算整个历史序列
使用 cache：历史 K/V 复用，每步只算新 token
```

### 2.7 为什么 RoPE 和 cache 必须靠 start_pos 对齐

这里是本节最容易错的地方。

prefill 处理 `A B C` 后，cache 里保存的是：

```text
K_A(pos0), K_B(pos1), K_C(pos2)
V_A,       V_B,       V_C
```

现在生成出新 token `D`，下一轮 decode 只输入 `D`：

```text
本轮 input_ids 形状看起来是 [batch, 1]
seq_length = 1
```

但 `D` 在完整序列里的真实位置不是 0，而是 3：

```text
位置:   0   1   2   3
token:  A   B   C   D
```

所以给 `D` 算 RoPE 时，不能用位置 0 的 cos/sin，必须用位置 3 的 cos/sin。

MiniMind 用 cache 长度得到这个位置：

```text
start_pos = past_key_values[0][0].shape[1]
```

如果历史 cache 长度是 3，那么：

```text
start_pos = 3
position_embeddings = freqs[3:4]
```

这样新 token `D` 的 Q/K 才会带上位置 3。

如果错误地用位置 0，那么：

```text
D 会被当成一个新序列的第 0 个 token
它和 A/B/C 的相对位置关系会错
cache 生成就不再等价于整段 forward
```

### 2.8 判断 cache 是否正确的标准

正确的 KV cache 不应该改变模型输出。它只是省计算。

所以可以用下面这个等价关系检查：

```text
不用 cache：
forward(A B C D)
-> 取 D 位置 logits 预测 E

使用 cache：
forward(A B C, use_cache=True)
-> 得到 cache(A,B,C)
forward(D, past_key_values=cache, use_cache=True)
-> 取 D 位置 logits 预测 E
```

如果 RoPE 的 `start_pos` 正确，K/V 拼接正确，那么两边 D 位置的 logits 应该非常接近。

本节可以先记住这句话：

```text
RoPE 让 Q/K 打分知道相对位置；
KV cache 保存历史 K/V，避免重复计算；
start_pos 让只输入新 token 时仍然使用真实位置。
```

实验会验证这一点。

<a id="l12-source-walkthrough"></a>
## 3. 从 generate 到 Attention 的完整源码走读

这一节沿着真实执行路径走，不先拆小片段。

```text
eval_llm.py
-> model.generate(...)
-> MiniMindForCausalLM.generate
-> MiniMindForCausalLM.forward
-> MiniMindModel.forward
-> MiniMindBlock.forward
-> Attention.forward
```

### 第 1 步：CLI 推理调用 generate

File: `eval_llm.py:82-87`

Read this to understand: CLI 推理时不是手写循环，而是把 prompt 的 `input_ids` 交给模型自己的 `generate`。

Code/config/template excerpt:

```python
generated_ids = model.generate(
    inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
    max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    top_p=args.top_p, temperature=args.temperature, repetition_penalty=1
)
```

This code shows:

- `inputs["input_ids"]` 是 prompt token。
- `generate` 负责逐 token 生成。
- 默认是否用 cache 不在这里显式传，因为 `MiniMindForCausalLM.generate` 的 `use_cache=True` 是默认值。

### 第 2 步：generate 初始化 input_ids 和 past_key_values

File: `model/model_minimind.py:257-263`

Read this to understand: 生成循环开始前，模型准备好完整 prompt、空 cache 和结束标记。

Code/config/template excerpt:

```python
def generate(..., use_cache=True, ...):
    input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
    attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
    past_key_values = kwargs.pop("past_key_values", None)
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
    if streamer: streamer.put(input_ids.cpu())
```

This code shows:

- `past_key_values` 初始通常是 `None`。
- `use_cache=True` 是默认生成方式。
- `input_ids` 会一直增长，包含 prompt 和已经生成出的 token。

### 第 3 步：每一轮只把未缓存部分送进 forward

File: `model/model_minimind.py:263-266`

Read this to understand: KV cache 真正省计算的入口是 `input_ids[:, past_len:]`。

Code/config/template excerpt:

```python
for _ in range(max_new_tokens):
    past_len = past_key_values[0][0].shape[1] if past_key_values else 0
    outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
    attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
```

This code shows:

- 第一次循环 `past_key_values=None`，所以 `past_len=0`，整段 prompt 会进入 forward。
- 后面循环 `past_len` 等于 cache 中已经保存的序列长度。
- `input_ids[:, past_len:]` 只取新 token，历史 token 不再重复送入模型。
- `attention_mask` 仍然随着完整序列增长，因为新 query 要能看到完整 key/value 序列。

### 第 4 步：CausalLM forward 把任务交给 MiniMindModel

File: `model/model_minimind.py:245-253`

Read this to understand: `MiniMindForCausalLM.forward` 主要负责调用 backbone、接 lm_head、返回 cache。

Code/config/template excerpt:

```python
def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
    hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :])
    ...
    return MoeCausalLMOutputWithPast(..., logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
```

This code shows:

- cache 是从 `self.model(...)` 返回的。
- `lm_head` 只负责把 hidden_states 变成 logits，不负责位置和 cache。
- 返回对象里的 `past_key_values` 会被 `generate` 下一轮继续使用。

### 第 5 步：MiniMindModel 用 cache 长度计算 start_pos

File: `model/model_minimind.py:209-219`

Read this to understand: RoPE 与 KV cache 在这里真正接上。

Code/config/template excerpt:

```python
batch_size, seq_length = input_ids.shape
if hasattr(past_key_values, 'layers'): past_key_values = None
past_key_values = past_key_values or [None] * len(self.layers)
start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
hidden_states = self.dropout(self.embed_tokens(input_ids))
...
position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])
```

This code shows:

- `seq_length` 是本次 forward 输入的长度，不一定是完整上下文长度。
- `start_pos` 是历史 cache 的长度。
- `position_embeddings` 从 `start_pos` 开始切，保证新 token 用它在完整序列中的位置。

这是本节最关键的一段源码。

### 第 6 步：每层 block 接收自己的 past_key_value

File: `model/model_minimind.py:221-232`

Read this to understand: cache 是逐层保存的，不是全模型只存一份。

Code/config/template excerpt:

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
...
return hidden_states, presents, aux_loss
```

This code shows:

- 每个 transformer layer 都有自己的 K/V cache。
- `past_key_value` 是当前层的历史 K/V。
- `present` 是当前层更新后的 K/V。
- `presents` 会返回给 `generate`，成为下一轮的 `past_key_values`。

### 第 7 步：Attention 里先算 Q/K/V，再对 Q/K 做 RoPE

File: `model/model_minimind.py:111-119`

Read this to understand: RoPE 进入 attention 的位置在 Q/K projection 之后、attention 打分之前。

Code/config/template excerpt:

```python
bsz, seq_len, _ = x.shape
xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
xq, xk = self.q_norm(xq), self.k_norm(xk)
cos, sin = position_embeddings
xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
```

This code shows:

- RoPE 作用在 reshape 成 head 之后的 Q/K 上。
- `xv` 没有进入 `apply_rotary_pos_emb`。
- `position_embeddings` 是上一步按 `start_pos` 切出来的 cos/sin。

### 第 8 步：Attention 把历史 K/V 和新 K/V 拼起来

File: `model/model_minimind.py:120-124`

Read this to understand: KV cache 的主体就是拼接历史 K/V 和当前 K/V。

Code/config/template excerpt:

```python
if past_key_value is not None:
    xk = torch.cat([past_key_value[0], xk], dim=1)
    xv = torch.cat([past_key_value[1], xv], dim=1)
past_kv = (xk, xv) if use_cache else None
xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
```

This code shows:

- `dim=1` 是序列长度维度。
- 拼接后的 `xk/xv` 包含历史 token 和当前 token。
- `past_kv` 保存的是 repeat 之前的 K/V，shape 仍然是 `[batch, seq, num_key_value_heads, head_dim]`。
- `repeat_kv` 是为了 GQA，让 K/V head 数对齐 attention head 数。

### 第 9 步：generate 更新 cache 并追加 next_token

File: `model/model_minimind.py:267-288`

Read this to understand: 每轮生成后，`input_ids` 变长，cache 也变长。

Code/config/template excerpt:

```python
logits = outputs.logits[:, -1, :] / temperature
...
next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
...
input_ids = torch.cat([input_ids, next_token], dim=-1)
past_key_values = outputs.past_key_values if use_cache else None
...
return input_ids
```

This code shows:

- 每轮只用最后一个位置的 logits 选出 `next_token`。
- `input_ids` 追加新 token。
- `past_key_values` 更新为本轮 forward 返回的新 cache。
- 下一轮的 `past_len` 会更大，所以只会处理更新后的新 token。

<a id="l12-variable-dict"></a>
## 4. 源码变量字典

| 变量 | 形状或类型 | 含义 |
|---|---|---|
| `freqs_cos` | `[max_position_embeddings, head_dim]` | 每个位置、每个 head 维度对应的 cos |
| `freqs_sin` | `[max_position_embeddings, head_dim]` | 每个位置、每个 head 维度对应的 sin |
| `position_embeddings` | `(cos_slice, sin_slice)` | 当前 forward 需要的 RoPE 位置片段 |
| `start_pos` | int | 当前输入在完整序列中的起始位置，来自 cache 长度 |
| `xq` | `[batch, seq, num_attention_heads, head_dim]` | 当前输入 token 的 query |
| `xk` | `[batch, seq, num_key_value_heads, head_dim]` | 当前输入 token 的 key，RoPE 后会进入 cache |
| `xv` | `[batch, seq, num_key_value_heads, head_dim]` | 当前输入 token 的 value |
| `past_key_values` | list | 每层一个 `(past_k, past_v)` |
| `past_key_value` | tuple or None | 当前层的历史 K/V |
| `present` | tuple or None | 当前层更新后的 K/V |
| `past_len` | int | `generate` 里已经缓存的 token 数 |
| `input_ids[:, past_len:]` | tensor | 本轮还没有进入 cache 的新 token |

两个维度最容易混：

```text
seq_length: 本次 forward 输入长度
past_len/start_pos: 之前已经缓存的历史长度
```

不用 cache 时：

```text
seq_length = 完整上下文长度
start_pos = 0
```

用 cache 解码时：

```text
seq_length = 1
start_pos = 历史上下文长度
```

<a id="l12-source-evidence"></a>
## 5. 关键源码对照

这里不是重新走完整源码，而是把几个关键点单独拎出来，方便复习时定位。

### 5.1 RoPE 的配置来自模型 config

File: `model/model_minimind.py:27-39`

Read this to understand: RoPE 的最大位置、频率底数和推理外推配置都在 config 里。

Code/config/template excerpt:

```python
self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
self.rope_theta = kwargs.get("rope_theta", 1e6)
self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
self.rope_scaling = {
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 16,
    "original_max_position_embeddings": 2048,
    "attention_factor": 1.0,
    "type": "yarn"
} if self.inference_rope_scaling else None
```

This code shows:

- `max_position_embeddings` 决定预计算 RoPE 表的长度。
- `rope_theta` 影响不同维度的旋转频率。
- `inference_rope_scaling` 是推理外推相关，先知道它会影响 RoPE 表即可。

### 5.2 预计算 cos/sin 表

File: `model/model_minimind.py:62-78`

Read this to understand: MiniMind 不是每次 attention 临时算 RoPE 表，而是在模型初始化时预先算出 cos/sin。

Code/config/template excerpt:

```python
def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    ...
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin
```

This code shows:

- `dim` 对应 `head_dim`。
- `end` 对应最大位置数量。
- `torch.outer(t, freqs)` 得到每个位置、每个频率的角度。
- `freqs_cos/freqs_sin` 最终扩展到 `[end, head_dim]`。

### 5.3 RoPE 表注册成 buffer

File: `model/model_minimind.py:205-207`

Read this to understand: RoPE 表不是训练参数，但会跟着模型移动设备。

Code/config/template excerpt:

```python
freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
self.register_buffer("freqs_cos", freqs_cos, persistent=False)
self.register_buffer("freqs_sin", freqs_sin, persistent=False)
```

This code shows:

- RoPE 表由 `config.head_dim` 和 `config.max_position_embeddings` 决定。
- `register_buffer` 表示它属于模型状态，但不是可训练参数。
- `persistent=False` 表示它不作为普通权重持久保存。

### 5.4 apply_rotary_pos_emb 的关键公式

File: `model/model_minimind.py:80-84`

Read this to understand: MiniMind 用 `rotate_half` 实现批量旋转。

Code/config/template excerpt:

```python
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed
```

This code shows:

- 输入是 Q/K，不包含 V。
- `cos.unsqueeze(1)` 让 cos/sin 从 `[seq, head_dim]` 变成可以广播到 `[batch, seq, heads, head_dim]`。
- 返回的 `q_embed/k_embed` shape 不变，只是数值带上了位置旋转。

### 5.5 已下载 transformers 模型也带 use_cache 配置

File: `minimind-3/config.json:23-35`

Read this to understand: transformers 格式模型配置里也记录了位置长度、RoPE 参数和默认 cache 行为。

Code/config/template excerpt:

```json
"max_position_embeddings": 32768,
"num_attention_heads": 8,
"num_hidden_layers": 8,
"num_key_value_heads": 4,
"rope_scaling": null,
"rope_theta": 1000000.0,
"use_cache": true,
```

This code shows:

- 下载好的 `minimind-3` 模型也使用 RoPE 参数。
- `use_cache: true` 说明推理时默认倾向于使用 KV cache。
- 这里是 transformers 配置；本节主要读的是项目自定义 `model/model_minimind.py` 的实现。

<a id="l12-experiment"></a>
## 6. 实验验证

本节实验不需要真实权重。它用 tiny 随机模型验证三个事实：

```text
1. RoPE cos/sin 的 shape 和 Q/K 对齐。
2. KV cache 的 K/V 长度会随着生成增长。
3. 整段 forward 的最后 token logits 与 cache 增量 forward 的 logits 接近一致。
```

命令：

```bash
cd /home/sun/minimind
python course/labs/trace_rope_kv_cache.py
```

重点记录这些输出：

```text
freqs_cos.shape =
q_after_rope.shape =
prefix_layer0_k.shape =
step_layer0_k.shape =
full_vs_incremental_last_logits_max_abs_diff =
cache_and_no_cache_generate_match =
```

你应该看到：

```text
q_after_rope.shape 和 q_before_rope.shape 一样
prefix_layer0_k 的 seq 维度等于 prefix 长度
step_layer0_k 的 seq 维度等于 prefix 长度 + 1
full_vs_incremental_last_logits_max_abs_diff 非常接近 0
cache_and_no_cache_generate_match=True
```

如果 `full_vs_incremental_last_logits_max_abs_diff` 很大，通常说明：

```text
start_pos 错了；
RoPE 位置切片错了；
cache 拼接维度错了；
或者生成时没有正确传 past_key_values。
```

<a id="l12-check"></a>
## 7. 本节检查

1. RoPE 为什么作用在 Q/K 上，而不是作用在 V 上？
2. `freqs_cos[start_pos:start_pos + seq_length]` 里的 `start_pos` 为什么不能总是 0？
3. `past_key_values[0][0].shape[1]` 表示什么？
4. `generate` 里为什么用 `input_ids[:, past_len:]`，而不是每轮都把完整 `input_ids` 送进 forward？
5. KV cache 为什么存 K/V，而不是存 Q/K/V？
6. 使用 KV cache 应该改变模型输出结果，还是只改变计算效率？为什么？

<a id="l12-next"></a>
## 8. 下一课

第 13 课进入 `FFN 与 MoE`。

Attention 负责让 token 从上下文中取信息；FFN 负责对每个 token 的 hidden 向量做非线性变换。MiniMind 里还有 dense FFN 和 MoE FFN 两条路径，下一课会先讲普通 FFN，再看 MoE 的 router、expert 和 aux loss。
