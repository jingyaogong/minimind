# 第 8 课：MiniMindConfig 与参数规模

这一课只解决一个问题：`MiniMindConfig` 里的配置项，如何决定模型结构、张量 shape、参数量和权重兼容性。

## 目录

- [0. 本节主线](#l08-mainline)
- [1. 本节要懂的 6 个原理](#l08-principles)
- [2. 变量流转](#l08-flow)
- [3. 原理一：Config 是模型结构合同](#l08-config-contract)
- [4. 原理二：`vocab_size * hidden_size` 是 token 矩阵大小](#l08-vocab-hidden)
- [5. 原理三：`num_hidden_layers` 是 block 的复制次数](#l08-layers)
- [6. 原理四：attention head 决定 Q/K/V 的拆分方式](#l08-attention-heads)
- [7. 原理五：`intermediate_size` 主要影响 FFN 参数量](#l08-intermediate-size)
- [8. 原理六：改 config 会影响权重加载](#l08-weight-loading)
- [9. 实验验证](#l08-experiment)
- [10. 本节检查](#l08-check)
- [11. 下一课](#l08-next)

<a id="l08-mainline"></a>
## 0. 本节主线

模型配置的本质是：

```text
MiniMindConfig
-> 决定 vocab_size / hidden_size / num_hidden_layers / heads / intermediate_size
-> 创建 embedding、N 个 block、norm、lm_head
-> 决定每个参数矩阵的 shape
-> 决定总参数量
-> 决定某个 .pth 权重能不能加载进来
```

所以这节课开始正式进入模型本体。不要把 `hidden_size=768`、`num_hidden_layers=8` 当成普通超参数，它们会直接改变模型里矩阵的形状。

<a id="l08-principles"></a>
## 1. 本节要懂的 6 个原理

| 原理 | 要理解什么 | 源码证据 |
|---|---|---|
| Config 是模型结构合同 | `MiniMindConfig` 保存的不是训练技巧，而是模型骨架参数 | `model/model_minimind.py:10-45`, `minimind-3/config.json:8-37` |
| `vocab_size * hidden_size` 是 token 矩阵大小 | embedding 和 lm_head 的 shape 都受词表和隐藏维度控制 | `model/model_minimind.py:196-204`, `model/model_minimind.py:234-243` |
| `num_hidden_layers` 是 block 的复制次数 | 每一层 block 都有自己独立的 Attention、Norm、MLP 参数 | `model/model_minimind.py:178-185`, `model/model_minimind.py:196-204` |
| attention head 决定 Q/K/V 的拆分方式 | `num_attention_heads`、`num_key_value_heads`、`head_dim` 决定投影矩阵和中间 shape | `model/model_minimind.py:22-24`, `model/model_minimind.py:91-116` |
| `intermediate_size` 主要影响 FFN 参数量 | FFN 有 3 个大矩阵，通常是每层 block 的重要参数来源 | `model/model_minimind.py:25-26`, `model/model_minimind.py:136-145` |
| 改 config 会影响权重加载 | `.pth` 里的 tensor shape 必须和当前模型 shape 对得上 | `trainer/trainer_utils.py:119-127` |

学完本节，你应该能解释：为什么 MiniMind 的 `64M` 不是一个魔法数字，而是由词表、隐藏维度、层数、Attention 和 FFN 矩阵共同堆出来的。

<a id="l08-flow"></a>
## 2. 变量流转

从 config 到模型参数：

```text
vocab_size, hidden_size
-> nn.Embedding(vocab_size, hidden_size)
-> input_ids [batch, seq]
-> hidden_states [batch, seq, hidden_size]

num_hidden_layers
-> ModuleList([MiniMindBlock(...) for _ in range(num_hidden_layers)])
-> hidden_states 依次经过 N 个 block

num_attention_heads, num_key_value_heads, head_dim
-> q_proj / k_proj / v_proj / o_proj
-> attention 内部拆成多个 head

intermediate_size
-> FeedForward 的 gate_proj / up_proj / down_proj

hidden_size, vocab_size
-> lm_head(hidden_states)
-> logits [batch, seq, vocab_size]
```

最重要的三个 shape：

```text
input_ids:     [batch_size, seq_len]
hidden_states: [batch_size, seq_len, hidden_size]
logits:        [batch_size, seq_len, vocab_size]
```

<a id="l08-config-contract"></a>
## 3. 原理一：Config 是模型结构合同

### 原理讲解

`MiniMindConfig` 不是“训练参数表”，而是“模型结构合同”。

训练参数像 `learning_rate`、`batch_size`，改变的是训练方式；结构参数像 `hidden_size`、`num_hidden_layers`、`vocab_size`，改变的是模型里参数矩阵的形状。

比如：

```text
hidden_size=768
```

表示每个 token 在模型内部会被表示成 768 维向量。

```text
num_hidden_layers=8
```

表示模型主体里有 8 个 `MiniMindBlock`。

```text
vocab_size=6400
```

表示模型最后每个位置要在 6400 个 token id 上预测概率。

所以 config 一旦变了，模型结构也变了。结构变了，旧权重通常就不能直接加载。

### 源码证据 A：MiniMind 自己的配置类

文件：`model/model_minimind.py:10-45`

看它是为了理解：哪些字段会成为模型结构的一部分。

源码：

```python
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
```

这段代码说明：

- `hidden_size` 默认是 768。
- `num_hidden_layers` 默认是 8。
- `vocab_size` 默认是 6400。
- `head_dim` 默认由 `hidden_size // num_attention_heads` 推出来。
- `intermediate_size` 默认由 `hidden_size` 推出来。

### 源码证据 B：下载好的 `minimind-3` 配置

文件：`minimind-3/config.json:8-37`

看它是为了理解：真实下载模型使用的是哪组结构参数。

配置片段：

```json
{
  "head_dim": 96,
  "hidden_size": 768,
  "intermediate_size": 2432,
  "max_position_embeddings": 32768,
  "num_attention_heads": 8,
  "num_hidden_layers": 8,
  "num_key_value_heads": 4,
  "tie_word_embeddings": true,
  "vocab_size": 6400
}
```

这段配置说明：

- 你下载的 `minimind-3` 是 `hidden_size=768`、`num_hidden_layers=8`。
- `head_dim=96`，所以 `8 heads * 96 = 768`。
- `vocab_size=6400`，和 MiniMind tokenizer 的词表规模对齐。
- `tie_word_embeddings=true`，表示 embedding 和输出头共享同一组 token 矩阵。

### 理解到这一步就够

你应该能说清楚：

```text
Config 决定模型结构；
训练参数决定怎么训练；
二者不是一回事。
```

暂时不用看：

- YaRN/RoPE 长上下文扩展细节。
- `PretrainedConfig` 在 Transformers 里的完整机制。

<a id="l08-vocab-hidden"></a>
## 4. 原理二：`vocab_size * hidden_size` 是 token 矩阵大小

### 原理讲解

LLM 里有一个很大的矩阵负责把 token id 变成向量：

```text
embedding matrix: [vocab_size, hidden_size]
```

如果：

```text
vocab_size = 6400
hidden_size = 768
```

那么这个矩阵参数量就是：

```text
6400 * 768 = 4,915,200
```

也就是约 `4.9M` 参数。

模型输出时还需要一个矩阵把 hidden state 映射回词表：

```text
lm_head: [vocab_size, hidden_size]
```

MiniMind 默认把 embedding 和 lm_head 绑定成同一组权重，所以这块参数通常只算一份。这叫 `tie_word_embeddings`。

### 源码证据 A：输入 embedding

文件：`model/model_minimind.py:196-204`

看它是为了理解：`input_ids` 进入模型后，第一块参数矩阵由谁创建。

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

- `embed_tokens` 的 shape 由 `vocab_size` 和 `hidden_size` 决定。
- `input_ids` 经过 embedding 后会变成 `hidden_states`。
- `num_hidden_layers` 同时在这里决定 block 数量。

### 源码证据 B：输出 lm_head 和权重绑定

文件：`model/model_minimind.py:234-243`

看它是为了理解：模型最后为什么能输出 `[batch, seq, vocab_size]` 的 logits。

源码：

```python
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings: self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()
```

这段代码说明：

- `lm_head` 把 `hidden_size` 映射到 `vocab_size`。
- `tie_word_embeddings=True` 时，`embed_tokens.weight` 和 `lm_head.weight` 是同一组参数。
- 如果取消绑定，token 相关参数大约会多一份 `vocab_size * hidden_size`。

### 理解到这一步就够

你应该能回答：

```text
为什么 tokenizer 的 vocab_size 会影响模型大小？
因为 embedding 和 lm_head 都要覆盖整个词表。
```

暂时不用看：

- logits softmax 的数值稳定细节。
- 不同 tokenizer 的 BPB/PPL 对比。

<a id="l08-layers"></a>
## 5. 原理三：`num_hidden_layers` 是 block 的复制次数

### 原理讲解

Transformer 主体不是一个单独的大函数，而是很多层 block 叠起来。

MiniMind 里一个 block 大致是：

```text
RMSNorm
-> Attention
-> 残差
-> RMSNorm
-> FFN 或 MoE
-> 残差
```

`num_hidden_layers=8` 的含义不是“同一个 block 循环 8 次并共享参数”，而是创建 8 个独立的 `MiniMindBlock`。每一层都有自己的 Attention、Norm、MLP 参数。

所以层数增加时，参数量通常接近线性增加：

```text
总 block 参数量 ≈ num_hidden_layers * 单个 block 参数量
```

### 源码证据 A：一个 block 里有什么

文件：`model/model_minimind.py:178-185`

看它是为了理解：每一层 block 里包含哪些子模块。

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

- 每个 block 都有一个 Attention。
- 每个 block 有两个 RMSNorm。
- 每个 block 的 MLP 可以是普通 FFN，也可以是 MoE。

### 源码证据 B：创建多层 block

文件：`model/model_minimind.py:196-204`

看它是为了理解：`num_hidden_layers` 如何变成多层模型。

源码：

```python
self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
```

这段代码说明：

- `range(self.num_hidden_layers)` 循环几次，就创建几个 block。
- `ModuleList` 里的每个 block 都是独立模块。
- 改 `num_hidden_layers` 会改变模型参数数量和权重 key 数量。

### 理解到这一步就够

你应该能说清楚：

```text
num_hidden_layers 控制模型深度；
hidden_size 控制每层内部向量宽度；
二者都会影响参数量，但方式不同。
```

暂时不用看：

- 每一层学到的语义差异。
- 深层模型的训练稳定性问题。

<a id="l08-attention-heads"></a>
## 6. 原理四：attention head 决定 Q/K/V 的拆分方式

### 原理讲解

`hidden_size` 是 token 向量总宽度，attention 会把这个向量拆成多个 head。

在 `minimind-3` 里：

```text
hidden_size = 768
num_attention_heads = 8
head_dim = 96
```

所以：

```text
8 * 96 = 768
```

这表示 Query 被拆成 8 个 head，每个 head 96 维。

MiniMind 还用了 `num_key_value_heads=4`。这表示 K/V head 比 Q head 少，属于 Grouped Query Attention 的思路。直观理解是：Q 有 8 组，但 K/V 只有 4 组，后面再复用到 8 组 query 上。这样可以减少 K/V 相关参数和 cache 体积。

本节只要理解配置如何决定 shape，不展开 Attention 公式。Attention 的计算细节放到后面的专门课。

### 源码证据 A：head 配置从哪里来

文件：`model/model_minimind.py:22-24`

看它是为了理解：`head_dim` 默认如何计算。

源码：

```python
self.num_attention_heads = kwargs.get("num_attention_heads", 8)
self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
```

这段代码说明：

- 默认有 8 个 attention heads。
- 默认有 4 个 key/value heads。
- 默认 `head_dim = hidden_size // num_attention_heads`。

### 源码证据 B：Q/K/V/O 投影矩阵

文件：`model/model_minimind.py:91-103`

看它是为了理解：head 配置如何变成 Linear 层的输入输出维度。

源码：

```python
class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
```

这段代码说明：

- `q_proj` 输出维度是 `num_attention_heads * head_dim`。
- `k_proj` 和 `v_proj` 输出维度是 `num_key_value_heads * head_dim`。
- `o_proj` 再把多头结果映射回 `hidden_size`。
- K/V head 更少时，K/V 投影参数会比 Q 投影少。

### 源码证据 C：forward 里的 head shape

文件：`model/model_minimind.py:111-116`

看它是为了理解：Linear 输出后如何 reshape 成多头格式。

源码：

```python
bsz, seq_len, _ = x.shape
xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
```

这段代码说明：

- attention 输入 `x` 的 shape 是 `[batch, seq, hidden_size]`。
- Q 被 reshape 成 `[batch, seq, num_attention_heads, head_dim]`。
- K/V 被 reshape 成 `[batch, seq, num_key_value_heads, head_dim]`。

### 理解到这一步就够

你应该能说清楚：

```text
num_attention_heads 不是层数；
它是在每层 attention 内部把 hidden 向量拆成多少组。
```

暂时不用看：

- scaled dot-product attention 的完整公式。
- causal mask 和 flash attention 分支。
- KV cache 的拼接过程。

<a id="l08-intermediate-size"></a>
## 7. 原理五：`intermediate_size` 主要影响 FFN 参数量

### 原理讲解

每个 Transformer block 除了 Attention，还有 FFN。

MiniMind 的普通 FFN 有三个矩阵：

```text
gate_proj: hidden_size -> intermediate_size
up_proj:   hidden_size -> intermediate_size
down_proj: intermediate_size -> hidden_size
```

所以 FFN 参数量大致是：

```text
3 * hidden_size * intermediate_size
```

在很多 Transformer 里，FFN 是每层参数量的重要来源。`hidden_size` 增大时，`intermediate_size` 也会跟着变大，所以参数量会明显上升。

MoE 也是从这里切入的：如果 `use_moe=True`，block 里的 `mlp` 不再是一个普通 FFN，而是多个 expert FFN 加一个 router。本课只先知道开关在哪里，MoE 细节后面讲到 FFN 与 MoE 时再展开。

### 源码证据 A：`intermediate_size` 的默认计算

文件：`model/model_minimind.py:25-26`

看它是为了理解：FFN 中间维度默认不是随便写死的。

源码：

```python
self.hidden_act = kwargs.get("hidden_act", 'silu')
self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
```

这段代码说明：

- 激活函数默认是 `silu`。
- `intermediate_size` 默认由 `hidden_size` 推导。
- hidden 越宽，FFN 中间层通常也越宽。

### 源码证据 B：普通 FFN 的三个矩阵

文件：`model/model_minimind.py:136-145`

看它是为了理解：FFN 参数量为什么和 `hidden_size * intermediate_size` 相关。

源码：

```python
class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
```

这段代码说明：

- FFN 有 `gate_proj`、`up_proj`、`down_proj` 三个 Linear。
- 这三个矩阵都和 `hidden_size`、`intermediate_size` 直接相关。
- 改 `hidden_size` 会同时影响 Attention 和 FFN。

### 源码证据 C：MoE 只先看开关

文件：`model/model_minimind.py:148-153`

看它是为了理解：MoE 是如何接在 FFN 位置上的。

源码：

```python
class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])
```

这段代码说明：

- MoE 在 FFN 位置引入多个 expert。
- `gate` 负责给 token 分配 expert。
- 本节先不展开 router、top-k、aux loss，只知道它会显著改变 FFN 部分的参数结构。

### 理解到这一步就够

你应该能回答：

```text
为什么 hidden_size 增大时，参数量不是只小幅增加？
因为 embedding、attention、FFN 都会被 hidden_size 影响，其中 FFN 还有 intermediate_size 放大。
```

暂时不用看：

- SwiGLU 的数学细节。
- MoE 的 token dispatch 和负载均衡 loss。

<a id="l08-weight-loading"></a>
## 8. 原理六：改 config 会影响权重加载

### 原理讲解

`.pth` 权重文件里保存的是一堆 tensor。

每个 tensor 都有名字和 shape，例如：

```text
model.embed_tokens.weight: [6400, 768]
model.layers.0.self_attn.q_proj.weight: [768, 768]
```

如果你把当前模型改成：

```text
hidden_size=512
```

那么新模型期待的 shape 就会变成另一套。旧权重里的 `[6400, 768]` 通常就装不进新模型的 `[6400, 512]`。

这里要注意：`strict=False` 可以容忍缺 key 或多 key，但不能神奇地把 shape 不一致的 tensor 变成可加载。结构参数变了，权重兼容性就要重新检查。

### 源码证据 A：加载模型时先按当前 config 创建结构

文件：`trainer/trainer_utils.py:119-127`

看它是为了理解：加载权重之前，模型结构已经由当前 `lm_config` 决定了。

源码：

```python
def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model', save_dir='../out', device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)

    if from_weight!= 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)
```

这段代码说明：

- `MiniMindForCausalLM(lm_config)` 先创建当前结构。
- `torch.load` 只是读出旧权重。
- `load_state_dict` 要把旧权重塞进当前结构。
- 如果当前结构和旧权重 shape 对不上，就会失败。

### 理解到这一步就够

你应该能说清楚：

```text
改 hidden_size / vocab_size / layers 不是普通调参；
它会改变权重 shape，因此影响已有权重能不能加载。
```

暂时不用看：

- `state_dict` 里每个 key 的完整命名规则。
- 模型结构迁移和权重插值。

<a id="l08-experiment"></a>
## 9. 实验验证

### 实验 A：看 tiny config 的参数分解

这个实验验证：

```text
config
-> embedding/lm_head shape
-> block 参数量
-> attention 参数量
-> FFN 参数量
```

运行：

```bash
cd /home/sun/minimind
PYTHONDONTWRITEBYTECODE=1 python course/labs/inspect_model_config_params.py
```

记录：

```text
hidden_size =
num_hidden_layers =
vocab_size =
embed_tokens.weight =
lm_head.weight =
total_unique_params =
token_matrix_params_once =
block0.self_attn =
block0.mlp =
```

你应该看到：

```text
embed_tokens.weight = (6400, 64)
lm_head.weight = (6400, 64)
embedding_and_lm_head_are_same_parameter = True
hidden_states = [batch_size, seq_len, 64]
```

这说明 token 矩阵的 shape 直接来自 `vocab_size` 和 `hidden_size`。

### 实验 B：比较不同 config 对参数量的影响

运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python course/labs/inspect_model_config_params.py --compare
```

记录：

```text
base total =
wider_hidden total =
deeper_layers total =
larger_vocab total =
untied_lm_head total =
```

你应该观察：

- `wider_hidden` 会让 block、attention、FFN 都变大。
- `deeper_layers` 主要增加 block 数量。
- `larger_vocab` 主要增加 token 矩阵。
- `untied_lm_head` 会多出一份输出头参数。

### 实验 C：查看真实 `minimind-3` 规模

运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python course/labs/inspect_model_config_params.py --preset minimind3 --device meta
```

这里用 `meta` 设备是为了只看参数 shape 和数量，不真正分配完整权重内存。

记录：

```text
hidden_size =
num_hidden_layers =
intermediate_size =
total_unique_params =
token_matrix_params_once =
block0_total =
```

你应该看到总参数量接近 `64M`。这就是 `minimind-3` 这个小模型参数规模的大致来源。

<a id="l08-check"></a>
## 10. 本节检查

如果你真懂了本节，应该能不看答案说清楚：

1. `hidden_size=768` 具体表示什么。
2. `vocab_size=6400` 为什么会影响 embedding 和 lm_head。
3. `tie_word_embeddings=True` 对参数量有什么影响。
4. `num_hidden_layers=8` 是共享一个 block 循环 8 次，还是创建 8 个独立 block。
5. `num_attention_heads`、`num_key_value_heads`、`head_dim` 三者是什么关系。
6. 为什么改 `hidden_size` 后，旧的 `.pth` 权重通常不能直接加载。

<a id="l08-next"></a>
## 11. 下一课

第 9 课进入 `Embedding、RMSNorm 与残差 Block`：我们会跟踪 `input_ids` 如何变成 `hidden_states`，以及一个 block 如何保持 `[batch, seq, hidden_size]` 这个主干 shape。
