import math, torch, torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast


# ════════════════════════════════════════════════════════════════════════════════
#                              整体架构协作关系
# ════════════════════════════════════════════════════════════════════════════════
#
# MiniMindForCausalLM (最顶层，训练和推理的入口)
#   ├── MiniMindModel (Transformer 主干)
#   │     ├── embed_tokens (Embedding层: token_id → 向量)
#   │     ├── MiniMindBlock × N (N=8 层 Transformer 块，核心计算单元)
#   │     │     ├── input_layernorm (RMSNorm: 注意力前的归一化)
#   │     │     ├── Attention (GQA多头注意力 + RoPE位置编码)
#   │     │     │     ├── q_proj, k_proj, v_proj (线性投影)
#   │     │     │     ├── q_norm, k_norm (QK归一化，防止注意力分数爆炸)
#   │     │     │     ├── apply_rotary_pos_emb (RoPE旋转位置编码)
#   │     │     │     ├── repeat_kv (GQA: 复制KV头以匹配Q头数量)
#   │     │     │     ├── Flash Attention 或 手动注意力计算
#   │     │     │     └── o_proj (输出投影)
#   │     │     ├── post_attention_layernorm (RMSNorm: FFN前的归一化)
#   │     │     └── mlp (前馈网络，二选一):
#   │     │           ├── FeedForward (Dense版: SwiGLU激活)
#   │     │           └── MOEFeedForward (MoE版: Router + 多个Expert)
#   │     └── norm (最终RMSNorm)
#   └── lm_head (Linear: 隐藏向量 → 词表logits，与embed_tokens共享权重)
#
# 数据流 (训练时):
#   input_ids [B, S] → Embedding → [B, S, 768] → 8层Block → RMSNorm → lm_head
#   → logits [B, S, 6400] → CrossEntropy(logits[:-1], labels[1:]) → loss
#
# 数据流 (推理/generate时):
#   每步只输入1个新token → 用KV Cache避免重复计算 → 取最后位置logits
#   → temperature/top_k/top_p采样 → 输出next_token → 循环直到EOS
#
# ════════════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════════════
#                               MiniMind Config
# ════════════════════════════════════════════════════════════════════════════════
# 模型的所有超参数配置，对齐 Qwen3 生态
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        # ---- 基础结构参数 ----
        self.hidden_size = hidden_size              # 隐藏层维度 d_model，整个模型的"宽度"
        self.num_hidden_layers = num_hidden_layers  # Transformer 层数，模型的"深度"
        self.use_moe = use_moe                      # 是否使用 MoE (Mixture of Experts)
        self.dropout = kwargs.get("dropout", 0.0)   # Dropout率，默认0(不丢弃)

        # ---- 词表与特殊token ----
        self.vocab_size = kwargs.get("vocab_size", 6400)      # 词表大小(极小词表，压缩参数量)
        self.bos_token_id = kwargs.get("bos_token_id", 1)     # 句子开始token
        self.eos_token_id = kwargs.get("eos_token_id", 2)     # 句子结束token

        # ---- 注意力参数 ----
        self.flash_attn = kwargs.get("flash_attn", True)                    # 是否启用Flash Attention加速
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)      # Q头数量(8个)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)      # KV头数量(4个，GQA模式: 每2个Q头共享1个KV头)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)  # 每个头的维度 = 768/8 = 96

        # ---- FFN参数 ----
        self.hidden_act = kwargs.get("hidden_act", 'silu')    # 激活函数 (SiLU，用于SwiGLU)
        # intermediate_size = ceil(768 * π / 64) * 64 = 2432，FFN的中间维度(约hidden_size的3.14倍)
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)

        # ---- 位置编码参数 ----
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)  # 最大位置数(32K)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)                         # RMSNorm的epsilon防除零
        self.rope_theta = kwargs.get("rope_theta", 1e6)                              # RoPE基础频率(越大支持越长序列)

        # ---- 权重绑定 ----
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)  # embedding和lm_head共享权重，省一份参数

        # ---- YaRN长文本外推 ----
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,                              # 高频维度边界(这些维度不缩放)
            "beta_slow": 1,                               # 低频维度边界(这些维度做最大缩放)
            "factor": 16,                                 # 缩放因子(目标长度/原始长度)
            "original_max_position_embeddings": 2048,     # 原始训练长度
            "attention_factor": 1.0,                      # 注意力缩放系数
            "type": "yarn"
        } if self.inference_rope_scaling else None

        # ---- MoE专用参数 (use_moe=False时忽略) ----
        self.num_experts = kwargs.get("num_experts", 4)                   # Expert总数
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)   # 每个token激活几个Expert (top-1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)  # 每个Expert的FFN中间维度
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)          # 是否归一化路由概率
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)  # 负载均衡loss系数

# ════════════════════════════════════════════════════════════════════════════════
#                              基础组件
# ════════════════════════════════════════════════════════════════════════════════
# ---- RMSNorm: 比LayerNorm更简单高效的归一化 ----
# LayerNorm: (x - mean) / std * γ + β  (需要均值和方差)
# RMSNorm:   x / RMS(x) * γ            (只需均方根，省掉均值和偏置)
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习的缩放参数 γ

    def norm(self, x):
        # RMS(x) = sqrt(mean(x²))，rsqrt = 1/sqrt
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 先转float32算归一化(防精度损失)，再转回原dtype
        return (self.weight * self.norm(x.float())).type_as(x)

# ---- RoPE: 旋转位置编码 (预计算频率表) ----
# 核心思想: 不把位置"加"到embedding上，而是把Q和K向量"旋转"一个与位置相关的角度
# 这样 Q_i · K_j 的点积自然编码了相对距离 |i-j|
def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    # 每个维度对应一个频率: freq_i = 1 / (θ^(2i/d))
    # 低维度 → 高频(捕捉局部位置)，高维度 → 低频(捕捉远距离位置)
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0

    # YaRN外推: 当推理长度 > 训练长度时，对不同维度做差异化频率缩放
    # 高频维度(局部位置信息): 保持不变
    # 低频维度(远距离位置信息): 频率除以factor缩放
    # 中间维度: 线性插值过渡
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:  # 只有推理长度超出训练长度时才缩放
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            # ramp: 0→1的线性斜坡，控制缩放程度
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            # 最终频率 = 原频率 * (1-ramp + ramp/factor)
            # ramp=0 → 保持原频率(高频维度)
            # ramp=1 → 频率/factor(低频维度)
            freqs = freqs * (1 - ramp + ramp / factor)

    # 生成位置-频率矩阵: [max_pos, dim//2]
    t = torch.arange(end, device=freqs.device)       # 位置序列 [0, 1, 2, ..., end-1]
    freqs = torch.outer(t, freqs).float()             # 外积: 每个位置 × 每个频率
    # 计算cos和sin，拼接成完整维度 [max_pos, dim]
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

# ---- 应用RoPE旋转到Q和K ----
# 公式: q_rot = q * cos(θ) + rotate_half(q) * sin(θ)
# 其中 rotate_half 将向量前后半部分交换并取负: [a,b] → [-b,a] (相当于复数乘法)
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed

# ---- GQA辅助: 复制KV头以匹配Q头数量 ----
# GQA (Grouped Query Attention): Q=8头, KV=4头，每2个Q头共享1对KV
# 需要把4个KV头复制一份变成8个，才能和8个Q头做注意力计算
# 相比MHA(Q=K=V=8头)节省50%的KV Cache显存
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x  # MHA模式(Q头数=KV头数)，无需复制
    # [B, S, kv_heads, dim] → [B, S, kv_heads, n_rep, dim] → [B, S, q_heads, dim]
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))

# ════════════════════════════════════════════════════════════════════════════════
#                              注意力层
# ════════════════════════════════════════════════════════════════════════════════
# 实现 GQA (Grouped Query Attention) + RoPE + KV Cache + Flash Attention
#
# 计算流程:
#   x → Q,K,V投影 → QK归一化 → RoPE旋转 → KV Cache拼接
#   → GQA复制KV → 注意力计算(Flash或手动) → 输出投影
class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads      # Q头数 = 8
        self.n_local_kv_heads = self.num_key_value_heads     # KV头数 = 4
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 每个KV头对应几个Q头 = 2
        self.head_dim = config.head_dim                      # 每个头的维度 = 96
        self.is_causal = True                                # 因果注意力(只能看到前面的token)
        # Q/K/V/O 线性投影 (无偏置，现代LLM标配)
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)    # 768 → 8*96=768
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)      # 768 → 4*96=384
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)      # 768 → 4*96=384
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)    # 768 → 768
        # QK归一化 (Qwen3的设计，防止注意力分数随训练爆炸)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        # 检测是否支持Flash Attention (PyTorch 2.0+的SDPA)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape

        # Step 1: 线性投影得到Q, K, V
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # reshape成多头格式: [B, S, n_heads, head_dim]
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)      # [B, S, 8, 96]
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)   # [B, S, 4, 96]
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)   # [B, S, 4, 96]

        # Step 2: QK归一化 (防止注意力分数随维度增大而爆炸)
        xq, xk = self.q_norm(xq), self.k_norm(xk)

        # Step 3: 应用RoPE旋转位置编码
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # Step 4: KV Cache (推理加速，拼接历史K/V)
        # 训练时: past_key_value=None，每次处理完整序列
        # 推理时: 拼接之前步骤缓存的K/V，只需计算新token
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)  # 拼接历史K
            xv = torch.cat([past_key_value[1], xv], dim=1)  # 拼接历史V
        past_kv = (xk, xv) if use_cache else None

        # Step 5: GQA头复制 + 转置为注意力计算格式
        # Q: [B, 8, S, 96], K: [B, 8, S, 96](从4头复制到8头), V: 同K
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))

        # Step 6: 注意力计算 (两条路径)
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            # 路径A: Flash Attention (O(N)显存，更快)
            # 条件: 非单token推理 + 无自定义mask时可用
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        else:
            # 路径B: 手动注意力 (KV Cache推理 或 有padding mask时)
            # scores = Q @ K^T / sqrt(d)
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # 因果mask: 上三角设为-inf，确保位置i只能attend到<=i的位置
            if self.is_causal: scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            # padding mask: 被padding的位置设为-1e9
            if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            # softmax → dropout → 与V相乘
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv

        # Step 7: 合并多头 + 输出投影
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)  # [B, S, 768]
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


# ════════════════════════════════════════════════════════════════════════════════
#                           前馈网络 (FFN)
# ════════════════════════════════════════════════════════════════════════════════
# ---- SwiGLU FFN (Dense版) ----
# 标准FFN:  output = down(ReLU(up(x)))           两个矩阵
# SwiGLU:   output = down(SiLU(gate(x)) * up(x)) 三个矩阵，效果更好
# gate_proj 和 up_proj 分别投影到中间维度，gate走激活函数后与up逐元素相乘，再down投影回来
class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size  # 默认2432
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)  # 768 → 2432 (门控)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)  # 2432 → 768 (降维)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)    # 768 → 2432 (上投影)
        self.act_fn = ACT2FN[config.hidden_act]  # SiLU = x * sigmoid(x)

    def forward(self, x):
        # SwiGLU: down(act(gate(x)) ⊙ up(x))
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
# ---- MoE FFN (稀疏版) ----
# 每个token被Router分配到top-k个Expert(这里k=1)处理
# 相比Dense: 总参数量大(4个Expert)，但每个token只激活1个Expert
# 效果: 用更多参数量换更强表达力，而计算量(FLOPs)基本不变
#
# 流程: x → Router(gate) → 选出top-1 Expert → Expert(x) * weight → 输出
class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)  # Router: 768 → 4
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])  # 4个独立的FFN
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)  # [B*S, 768] 展平为token序列

        # Step 1: Router计算每个token对每个Expert的分数
        scores = F.softmax(self.gate(x_flat), dim=-1)  # [B*S, 4] 概率分布

        # Step 2: 选出每个token的top-k Expert (k=1)
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)  # [B*S, 1]
        # 归一化权重 (top-1时其实就是1.0，top-k>1时确保权重和为1)
        if self.config.norm_topk_prob: topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)

        # Step 3: 分发token到各Expert并加权聚合
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)  # 哪些token被分配到Expert i
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()     # 被选中的token索引
                weight = topk_weight[mask].view(-1, 1)               # 对应的路由权重
                # Expert处理 → 乘权重 → 累加到输出
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                # 训练时: 即使Expert未被选中，也让它参与计算图(0*params)
                # 否则DDP会报错"某些参数没有梯度"
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())

        # Step 4: 负载均衡辅助loss (鼓励token均匀分布到各Expert)
        # 如果没有这个loss，Router倾向于把所有token都发给同一个Expert("赢者通吃")
        if self.training and self.config.router_aux_loss_coef > 0:
            # load: 每个Expert实际接收的token比例 [num_experts]
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            # aux_loss = (实际负载分布 · 路由概率分布) * num_experts * coef
            # 当所有Expert负载均匀时，此loss最小
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()

        return y.view(batch_size, seq_len, hidden_dim)


# ════════════════════════════════════════════════════════════════════════════════
#                        Transformer Block (单层)
# ════════════════════════════════════════════════════════════════════════════════
# 一个标准的 Pre-Norm Transformer Decoder Block:
#
#   x ─────────────────────┐
#   │                      │ (残差连接1)
#   ├→ RMSNorm → Attention ─┘
#   │                      │
#   ├──────────────────────┤ (残差连接2)
#   └→ RMSNorm → FFN/MoE ──┘
#   │
#   output
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)          # 注意力前归一化
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # FFN前归一化
        # 根据配置选择Dense FFN或MoE FFN
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # 残差连接1: Attention
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,  # Pre-Norm: 先归一化再进注意力
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual  # x + Attention(Norm(x))

        # 残差连接2: FFN
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))  # x + FFN(Norm(x))
        return hidden_states, present_key_value


# ════════════════════════════════════════════════════════════════════════════════
#                      MiniMindModel (Transformer 主干)
# ════════════════════════════════════════════════════════════════════════════════
# 职责: token_ids → hidden_states (不含最终的lm_head投影)
#
# 流程: Embedding → Dropout → N层Block → 最终RMSNorm → hidden_states
class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)  # 6400 × 768
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])  # 8层
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最终归一化

        # 预计算RoPE频率表(注册为buffer，不参与训练但随模型保存/移动)
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape

        # 兼容transformers 5.x的cache格式
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)

        # 确定当前序列的起始位置 (用于KV Cache场景下取正确的RoPE)
        # 训练时: start_pos=0 (处理完整序列)
        # 推理时: start_pos=已生成token数 (只处理新token)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # Embedding: token_id → 向量
        hidden_states = self.dropout(self.embed_tokens(input_ids))  # [B, S] → [B, S, 768]

        # 获取当前位置段的RoPE (处理transformers>=5.x meta-device初始化导致buffer丢失)
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])

        # 逐层前向传播
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

        # 最终归一化
        hidden_states = self.norm(hidden_states)

        # 汇总所有MoE层的辅助loss
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss


# ════════════════════════════════════════════════════════════════════════════════
#                   MiniMindForCausalLM (最顶层: 训练+推理入口)
# ════════════════════════════════════════════════════════════════════════════════
# 在MiniMindModel基础上加了lm_head(hidden → vocab logits)和loss计算
# 继承GenerationMixin以兼容HuggingFace的generate接口
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}  # 声明权重绑定关系

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)                                           # Transformer主干
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)  # 768 → 6400
        # 权重绑定: lm_head和embedding共享同一份权重矩阵，减少参数量
        if self.config.tie_word_embeddings: self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()  # HuggingFace的初始化后处理

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        # Step 1: 通过Transformer主干得到hidden_states
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)

        # Step 2: lm_head投影到词表空间
        # logits_to_keep: 优化——只计算最后N个位置的logits(节省显存)
        # 推理时只需最后1个位置，RL训练时只需completion部分
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])  # [B, S, 6400]

        # Step 3: 计算loss (仅训练时有labels)
        loss = None
        if labels is not None:
            # 自回归: 用位置t的logits预测位置t+1的token
            # logits[:-1] 预测 labels[1:]
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)  # -100位置不算loss

        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)

    # ════════════════════════════════════════════════════════════════════════════
    #                        自回归生成 (推理)
    # ════════════════════════════════════════════════════════════════════════════
    # 自定义generate，不用HuggingFace的默认实现
    # 支持: KV Cache / temperature / top_k / top_p / repetition_penalty / streaming
    #
    # 每步循环:
    #   1. 只输入新token(利用KV Cache)
    #   2. 取最后位置logits
    #   3. temperature缩放 → repetition penalty → top_k截断 → top_p截断
    #   4. 采样得到next_token
    #   5. 拼接到input_ids，重复直到EOS或达到max长度
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        # 支持多条序列并行生成
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)  # 跟踪每条序列是否已结束
        if streamer: streamer.put(input_ids.cpu())

        for _ in range(max_new_tokens):
            # 利用KV Cache: 只输入past_len之后的新token
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            # 扩展attention_mask以包含新生成的位置
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None

            # 取最后一个位置的logits，除以temperature
            # temperature < 1 → 更确定(偏向高概率token)
            # temperature > 1 → 更随机(概率分布更平坦)
            logits = outputs.logits[:, -1, :] / temperature

            # Repetition Penalty: 降低已出现token的概率
            # 正分数的token: score / penalty (概率降低)
            # 负分数的token: score * penalty (概率更低)
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i]); score = logits[i, seen]; logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)

            # Top-K 截断: 只保留概率最高的K个token
            if top_k > 0:
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')

            # Top-P (Nucleus) 截断: 保留累积概率刚好超过P的最小token集合
            # 动态调整候选数量——高置信时候选少，低置信时候选多
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0  # 保证至少保留1个token
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')

            # 采样或贪心选择
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)

            # 已结束的序列强制输出EOS (防止padding位置产生垃圾token)
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)

            # 拼接新token
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())

            # 检查是否所有序列都已生成EOS
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break

        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids
