# 01-理论 / 02 - Transformer 核心架构

> **TL;DR**：逐行拆解 `model/model_minimind.py` 的 288 行，从 RMSNorm 到 MoE 到 generate。
> 这是 **整套教程最重要的一篇**。读完你就读懂了一个对齐 Qwen3 的 LLM。
>
> **心理学技巧**：先讲故事再讲原理。先把整体流程图放在脑子里，再逐行看代码，不会迷失。

## ✅ 你将能

- 默写出 Decoder-Only Block 的 forward 流程
- 解释 RMSNorm / Pre-Norm / RoPE / GQA / SwiGLU / MoE 每一个的作用
- 读懂 `model/model_minimind.py` 全部 288 行
- 知道为什么 minimind-3 对齐 Qwen3

---

## 一、先看整体：Decoder-Only 长这样

minimind 是 **Decoder-Only Transformer**（GPT 系列、Llama、Qwen 都是这种）：

```
input_ids [B, L]
    │
    ▼
embed_tokens [B, L, D]            ← 词表查表
    │  (RoPE 注入位置信息到 Q/K)
    ▼
┌─────────────────────────────────────────────────┐
│  Block × n_layers (默认 8 层)                    │
│  ─────────────────────────────────────────────  │
│  x ──┬───────────────────┐                     │
│      ▼                    │                     │
│  input_layernorm (RMSNorm)│                     │
│      ▼                    │                     │
│  Attention (GQA + RoPE)    │                     │
│      ▼                    │                     │
│  ──── + ──────────────────┘  residual 1         │
│      │                                          │
│      ▼                                          │
│  post_attention_layernorm (RMSNorm)             │
│      │                                          │
│      ▼                                          │
│  FeedForward (SwiGLU) 或 MOEFeedForward         │
│      ▼                                          │
│  ──── + ──────────────────  residual 2          │
│      │                                          │
└──────┼──────────────────────────────────────────┘
       ▼
norm (RMSNorm)
       ▼
lm_head [B, L, vocab_size]       ← 输出 logits
       │
       ▼
cross_entropy (labels = input_ids 偏移一位)
```

> 主 README 也有结构图 `images/LLM-structure.jpg`，对照看更清晰。

## 二、组件逐个拆解

### 2.1 Config：所有超参一个地方管

`model/model_minimind.py:10` `MiniMindConfig`

```python
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        ...
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)   # ← GQA
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
```

**关键配置（默认 minimind-3 Dense）**：

| 参数 | 值 | 含义 |
|---|---|---|
| `hidden_size` (D) | 768 | 模型宽度 |
| `num_hidden_layers` | 8 | Transformer 层数 |
| `vocab_size` | 6400 | 词表 |
| `num_attention_heads` (q) | 8 | Q 头数 |
| `num_key_value_heads` (kv) | 4 | KV 头数（GQA：q/kv=2） |
| `head_dim` | 96 | 每个头的维度 (768/8) |
| `max_position_embeddings` | 32768 | 最大位置 |
| `rope_theta` | 1e6 | RoPE 基频 |
| `intermediate_size` | ≈2432 | FFN 中间维度 (≈ D × π / 64 × 64) |

**为什么 dim=768, layers=8？** 见主 README"模型配置"章节：MobileLLM 研究表明，小模型在固定参数下"深窄"比"矮胖"好，但 d_model<512 会模式崩溃，>1536 时加层比加宽更划算。768×8 是工程甜点。

### 2.2 RMSNorm：比 LayerNorm 更省

`model/model_minimind.py:50`

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        self.weight = nn.Parameter(torch.ones(dim))
    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)
```

- LayerNorm = 减均值 + 除标准差 + 仿射
- RMSNorm = **不减均值**，只除 RMS（均方根），加可学习缩放

省了均值计算，速度快、效果几乎一样。Llama / Qwen 都用 RMSNorm。

**Pre-Norm**：归一化在残差**之前**做（看 Block 流程图）。Pre-Norm 训练更稳定，是现代 LLM 标配。

### 2.3 RoPE：旋转位置编码（核心）

`model/model_minimind.py:62-84`

核心思想：**通过旋转矩阵把位置信息注入 Q 和 K**，让相对位置自然表达。

数学上：
- 给位置 m 的向量乘以旋转矩阵 $R_m$
- $q_m^T k_n = (R_m q)^T (R_n k) = q^T R_{n-m}^T R_m^T R_n k = q^T R_{n-m} k$
- 内积只依赖**相对位置 n−m**

代码细节：
```python
def precompute_freqs_cis(dim, end, rope_base, rope_scaling):
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim//2)].float() / dim))
    # ... YaRN 外推分支（见下文）
    t = torch.arange(end)
    freqs = torch.outer(t, freqs)
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin
```

**YaRN 外推**（`rope_scaling` 分支）：当推理长度超过训练长度时，把高频段频率缩放，让模型"自适应"更长上下文。minimind-3 默认 `original_max_position_embeddings=2048, factor=16`，理论上可外推到 32768。推理时加 `--inference_rope_scaling` 即可启用。

### 2.4 Attention（GQA + Flash）

`model/model_minimind.py:91-134` —— 这是最值得精读的一段。

```python
class Attention(nn.Module):
    def __init__(self, config):
        self.q_proj = nn.Linear(hidden, num_heads * head_dim, bias=False)        # 8 头
        self.k_proj = nn.Linear(hidden, num_kv_heads * head_dim, bias=False)     # 4 头 ← GQA
        self.v_proj = nn.Linear(hidden, num_kv_heads * head_dim, bias=False)     # 4 头
        self.o_proj = nn.Linear(num_heads * head_dim, hidden, bias=False)
        self.q_norm = RMSNorm(head_dim)     # ← QK 归一化（Qwen3 风格）
        self.k_norm = RMSNorm(head_dim)
```

**GQA（Grouped-Query Attention）**：
- 传统 MHA：每头都有独立的 K/V → 显存爆
- MQA：所有头共享一份 K/V → 效果差
- **GQA**：折中，4 个 KV 头被 8 个 Q 头共享（每 2 个 Q 头用 1 个 KV 头）
- 显存省一半，效果几乎不掉。Llama 3 / Qwen3 都用 GQA

**QK Norm**：对 Q 和 K 各做一次 RMSNorm，提升数值稳定性。Qwen3 / Gemma2 引入。

**forward 流程**：
```python
def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
    xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
    # 重塑成 [B, L, heads, head_dim]
    xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
    xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
    xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
    xq, xk = self.q_norm(xq), self.k_norm(xk)             # ← QK Norm
    xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)       # ← RoPE
    # KV cache 拼接
    if past_key_value is not None:
        xk = torch.cat([past_key_value[0], xk], dim=1)
        xv = torch.cat([past_key_value[1], xv], dim=1)
    # GQA：把 KV 复制到 Q 头数（repeat_kv）
    xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
    xv = repeat_kv(xv, self.n_rep).transpose(1, 2)
    # 优先用 Flash Attention
    if self.flash and ...:
        output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
    else:
        scores = (xq @ xk.transpose(-2, -1)) / sqrt(head_dim)
        # 因果 mask
        scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), -inf).triu(1)
        output = softmax(scores) @ xv
    output = self.o_proj(output.transpose(1, 2).reshape(bsz, seq_len, -1))
    return output, past_kv
```

**Flash Attention**：`F.scaled_dot_product_attention` 是 PyTorch 2.x 的 fused kernel，比手写循环快几倍，且省显存（不实体化 attention matrix）。

**KV cache**：自回归生成时，已算过的 K/V 缓存下来不重算，每步只算新 token 的 Q。这就是为什么推理 batch=1 时显存占用远小于训练。

### 2.5 SwiGLU FeedForward

`model/model_minimind.py:136-146`

```python
class FeedForward(nn.Module):
    def __init__(self, config, intermediate_size=None):
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj  = nn.Linear(intermediate, hidden, bias=False)
        self.up_proj    = nn.Linear(hidden, intermediate, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]   # silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

公式：$FFN(x) = (Silu(xW_g) \odot xW_u) W_d$

- 传统 FFN：$x → W_1 → act → W_2 → out$（两个矩阵）
- SwiGLU：**三个矩阵 + 门控**，多一个矩阵换来更好的表达力
- Llama / Qwen 全用 SwiGLU

**intermediate_size**：默认 `math.ceil(hidden_size × π / 64) × 64` ≈ 2432（≈3.17×D），这是为了让参数量和"两个矩阵的传统 FFN"持平。

### 2.6 MoE：以更小激活参数换更大容量

`model/model_minimind.py:148-176`

```python
class MOEFeedForward(nn.Module):
    def __init__(self, config):
        self.gate = nn.Linear(hidden, num_experts, bias=False)      # 路由
        self.experts = nn.ModuleList([FeedForward(config) for _ in range(num_experts)])
    def forward(self, x):
        scores = F.softmax(self.gate(x_flat), dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=num_experts_per_tok, sorted=False)
        # 每个 token 只激活 top-k 个专家
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)
            if mask.any():
                y.index_add_(0, token_idx, expert(x_flat[token_idx]) * weight)
        # aux_loss：负载均衡损失
        self.aux_loss = (load * scores.mean(0)).sum() * num_experts * router_aux_loss_coef
```

minimind-3-moe 配置：4 experts / top-1 routing，总参数 198M 但激活只用 64M。

**aux_loss（负载均衡损失）**：如果所有 token 都跑去用同一个专家，其他专家训不动。aux_loss 惩罚这种不均衡，让 token 均匀分布到各专家。Qwen3-MoE 也用同样的设计。

> 主 README 提到：minimind 没用 kernel-fused 算子（如 Triton / DeepSpeed-MoE），原生实现下 4 experts/top-1 比 dense 慢约 50%。

### 2.7 Block：把上面拼起来

`model/model_minimind.py:178-194`

```python
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id, config):
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),    # ← Pre-Norm
            position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual                  # ← residual 1
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)   # ← Pre-Norm + residual 2
        )
        return hidden_states, present_key_value
```

注意：**两次 RMSNorm 是分开的两个权重**，input_layernorm 和 post_attention_layernorm 不共享。

### 2.8 整体模型 + generate

`model/model_minimind.py:196-253` 是 `MiniMindModel` + `MiniMindForCausalLM`：
- `MiniMindModel`：embed → N 层 Block → final norm，返回 hidden_states + aux_loss
- `MiniMindForCausalLM`：在 hidden_states 上加 lm_head 输出 logits + 算 cross_entropy loss

`generate` 方法（`model/model_minimind.py:257-288`）是**从 0 实现**的采样循环：
```python
def generate(self, inputs, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, ...):
    input_ids = inputs
    for _ in range(max_new_tokens):
        outputs = self.forward(input_ids[:, past_len:], ..., use_cache=True)
        logits = outputs.logits[:, -1, :] / temperature              # 温度
        if top_k > 0: logits[logits < topk] = -inf                    # top_k
        if top_p < 1.0: # nucleus sampling                            # top_p
        next_token = torch.multinomial(softmax(logits), num_samples=1)  # 采样
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if finished.all(): break
```

每个采样参数的作用：
| 参数 | 作用 | 调大效果 |
|---|---|---|
| `temperature` | 控制概率分布尖锐度 | 更随机、更"创意" |
| `top_k` | 只在 logits 最高的 k 个里选 | 更保守、更"安全" |
| `top_p` | 在累计概率达 p 的最小集合里选 | 平衡多样性与合理性 |
| `repetition_penalty` | 对已出现 token 加惩罚 | 减少重复 |

**KV cache 让自回归生成高效**：每步只 forward 1 个新 token，而不是重新算整个序列。

## 三、与 Qwen3 的对齐点

| 组件 | Qwen3 | minimind-3 | 是否对齐 |
|---|---|---|---|
| RMSNorm | ✅ | ✅ | ✓ |
| Pre-Norm | ✅ | ✅ | ✓ |
| RoPE | ✅ | ✅ | ✓ |
| YaRN 外推 | ✅ | ✅ | ✓ |
| GQA | ✅ | ✅ (q=8/kv=4) | ✓ |
| QK Norm | ✅ | ✅ | ✓ |
| SwiGLU | ✅ | ✅ | ✓ |
| MoE w/o shared expert | ✅ | ✅ | ✓ |
| tie_word_embeddings | ✅ | ✅ | ✓ |

**这就是 minimind-3 能直接用 transformers / vllm / ollama 加载的原因**：结构完全对齐 Qwen3，转换权重几乎零成本。

## 四、动手验证

```bash
cd /path/to/minimind
python -c "
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
cfg = MiniMindConfig()
model = MiniMindForCausalLM(cfg)
total = sum(p.numel() for p in model.parameters()) / 1e6
print(f'总参数: {total:.1f}M')
print(f'层数: {cfg.num_hidden_layers}')
print(f'dim: {cfg.hidden_size}')
print(f'vocab: {cfg.vocab_size}')
print(f'GQA: q={cfg.num_attention_heads}, kv={cfg.num_key_value_heads}')
"
```

✅ **验证**：应输出 `总参数: ~64M`

## 五、关键代码对照表（务必记住）

| 概念 | 代码位置 |
|---|---|
| Config | `model/model_minimind.py:10` |
| RMSNorm | `model/model_minimind.py:50` |
| RoPE 预计算 | `model/model_minimind.py:62` |
| RoPE 应用 | `model/model_minimind.py:80` |
| GQA + repeat_kv | `model/model_minimind.py:86` |
| Attention forward | `model/model_minimind.py:111` |
| Flash Attention 分支 | `model/model_minimind.py:125-126` |
| KV cache 拼接 | `model/model_minimind.py:120-122` |
| SwiGLU FFN | `model/model_minimind.py:136-146` |
| MoE FFN + aux_loss | `model/model_minimind.py:148-176` |
| Decoder Block | `model/model_minimind.py:178-194` |
| 整体 model + 前向 | `model/model_minimind.py:196-253` |
| 自实现 generate | `model/model_minimind.py:257-288` |
| tie weights | `model/model_minimind.py:242` |

## ✅ 本篇完成自检

<details>
<summary>点开自检（先想 30 秒）</summary>

1. Pre-Norm 和 Post-Norm 区别？为什么现代 LLM 用 Pre-Norm？
   - Pre-Norm 在残差前归一化。训练更稳定，深层不会梯度爆炸。
2. GQA 比 MHA 省了多少显存？为什么不全用 MQA？
   - GQA 省 KV 头数对应的 K/V 缓存，MHA 8 头→GQA 4 头省一半。MQA 太极端（只 1 头）效果会掉。
3. RoPE 为什么能外推？
   - 旋转矩阵的相对位置性质，加 YaRN 频率缩放后可扩展到训练未见过的长度。
4. SwiGLU 为什么用 3 个矩阵而不是 2 个？
   - 引入门控，让 FFN 输出依赖输入（动态），表达能力更强；多一个矩阵但 intermediate_size 取较小值保持参数量持平。
5. MoE 的 aux_loss 解决什么问题？
   - 负载不均衡：所有 token 都跑去同一个专家，其他专家训不动。aux_loss 强制均衡分布。
6. KV cache 在 generate 时如何省算力？
   - 已算过的 K/V 不重算，每步只算 1 个新 token 的 Q。生成从 O(L²) 降到 O(L)。
7. `tie_word_embeddings=True` 节省多少？
   - embedding (vocab×hidden) 和 lm_head (hidden×vocab) 共享一份，省一份 ≈ 4.9M（minimind-3）。

</details>

恭喜，**你已经读懂了一个对齐 Qwen3 的 LLM**。下一篇：[03-训练数据格式](./03-训练数据格式.md) —— 理解数据如何变成 input_ids + labels。
