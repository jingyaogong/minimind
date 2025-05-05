# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

# MiniMind模型配置类，继承自transformers的PretrainedConfig
# 包含了模型的所有超参数设置，包括基础参数和MoE（混合专家）相关参数

from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig):
    # 模型类型标识符
    model_type = "minimind"

    def __init__(
            self,
            # 基础模型参数
            dropout: float = 0.0,  # dropout比率，用于防止过拟合
            bos_token_id: int = 1,  # 序列开始标记的ID
            eos_token_id: int = 2,  # 序列结束标记的ID
            hidden_act: str = 'silu',  # 激活函数类型
            hidden_size: int = 512,  # 隐藏层维度
            intermediate_size: int = None,  # 前馈网络中间层维度
            max_position_embeddings: int = 32768,  # 最大位置编码长度
            num_attention_heads: int = 8,  # 注意力头数量
            num_hidden_layers: int = 8,  # Transformer层数
            num_key_value_heads: int = 2,  # KV注意力头数量（用于分组查询注意力）
            vocab_size: int = 6400,  # 词表大小
            rms_norm_eps: float = 1e-05,  # RMSNorm层的epsilon值
            rope_theta: int = 1000000.0,  # 旋转位置编码的theta参数
            flash_attn: bool = True,  # 是否使用Flash Attention优化
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            use_moe: bool = False,
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.1,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.flash_attn = flash_attn
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率


# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

import math
import torch
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(torch.nn.Module):
    """RMSNorm层实现
    相比LayerNorm，RMSNorm去掉了均值归一化，只保留方差归一化，计算更简单且效果相当
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps  # 数值稳定性参数
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习的缩放参数

    def _norm(self, x):
        # 计算RMS（均方根）归一化
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 应用缩放并保持输入数据类型
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    """预计算旋转位置编码（RoPE）的三角函数值
    
    Args:
        dim: 隐藏层维度
        end: 最大序列长度
        theta: 位置编码的缩放因子
    
    Returns:
        freqs_cos: 余弦位置编码
        freqs_sin: 正弦位置编码
    """
    # 计算频率基础值
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    # 计算外积得到位置编码矩阵
    freqs = torch.outer(t, freqs).float()
    # 生成最终的三角函数值
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """应用旋转位置编码（RoPE）到查询和键向量
    
    通过将位置信息编码到注意力的查询(Q)和键(K)向量中，使模型能够感知token的相对位置关系
    """
    def rotate_half(x):
        # 将向量的后半部分取负并与前半部分交换位置
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    # 应用旋转变换到查询和键向量
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """重复键值向量以匹配注意力头的数量
    
    在分组查询注意力中，键值向量的头数少于查询向量，需要将键值向量重复多次
    
    Args:
        x: 输入张量，形状为[batch_size, seq_len, num_kv_heads, head_dim]
        n_rep: 重复次数
    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # 扩展并重塑张量以实现重复
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """多头注意力机制实现
    
    支持分组查询注意力（GQA）和Flash Attention优化
    分组查询注意力通过减少键值头的数量来降低计算和内存开销
    """
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        # 设置键值头数量，支持分组查询注意力
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads = args.num_attention_heads  # 查询头总数
        self.n_local_kv_heads = self.num_key_value_heads  # 键值头数量
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 每个键值头需要重复的次数
        self.head_dim = args.hidden_size // args.num_attention_heads  # 每个注意力头的维度
        
        # 定义线性变换层
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)  # 查询变换
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)  # 键变换
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)  # 值变换
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)  # 输出变换
        
        # Dropout层
        self.attn_dropout = nn.Dropout(args.dropout)  # 注意力权重的dropout
        self.resid_dropout = nn.Dropout(args.dropout)  # 残差连接的dropout
        self.dropout = args.dropout
        
        # 是否使用Flash Attention优化
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 位置编码的cos和sin值
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # KV缓存
                use_cache=False,  # 是否使用KV缓存
                attention_mask: Optional[torch.Tensor] = None):  # 注意力掩码
        """前向传播函数
        
        实现多头注意力的核心计算逻辑，包括：
        1. 线性变换生成查询、键、值向量
        2. 应用旋转位置编码
        3. 处理KV缓存（用于加速自回归生成）
        4. 执行注意力计算（支持Flash Attention优化）
        """
        bsz, seq_len, _ = x.shape
        # 生成查询、键、值向量
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # 重塑张量维度以适应多头注意力格式
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # 应用旋转位置编码
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        # 处理KV缓存，用于加速自回归生成
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)  # 拼接历史键向量
            xv = torch.cat([past_key_value[1], xv], dim=1)  # 拼接历史值向量
        past_kv = (xk, xv) if use_cache else None

        # 调整维度顺序并重复键值向量以匹配查询头数量
        xq, xk, xv = (
            xq.transpose(1, 2),  # [batch, num_heads, seq_len, head_dim]
            repeat_kv(xk, self.n_rep).transpose(1, 2),  # 重复键向量
            repeat_kv(xv, self.n_rep).transpose(1, 2)  # 重复值向量
        )

        if self.flash and seq_len != 1:
            dropout_p = self.dropout if self.training else 0.0
            attn_mask = None
            if attention_mask is not None:
                attn_mask = attention_mask.view(bsz, 1, 1, -1).expand(bsz, self.n_local_heads, seq_len, -1)
                attn_mask = attn_mask.bool() if attention_mask is not None else None

            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=True)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)  # scores+mask

            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    """前馈神经网络实现
    
    使用SwiGLU激活函数的变体，包含门控机制，可以更好地控制信息流动
    结构：输入 -> (门控路径 + 上投影路径) -> 下投影 -> 输出
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        # 如果未指定中间层维度，则根据隐藏层维度计算
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)  # 设置为隐藏层维度的8/3倍
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)  # 调整为64的倍数
        
        # 定义三个线性变换层
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)  # 门控路径
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)  # 降维投影
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)    # 升维投影
        
        self.dropout = nn.Dropout(config.dropout)  # dropout层
        self.act_fn = ACT2FN[config.hidden_act]    # 激活函数（默认为SiLU）

    def forward(self, x):
        # 实现SwiGLU的变体：激活(门控路径) * 上投影 -> 下投影 -> dropout
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class MoEGate(nn.Module):
    """混合专家（MoE）的门控机制实现
    
    负责为每个token选择最合适的专家组合，包括：
    1. 计算每个token与专家的相关性分数
    2. 选择top-k个专家处理每个token
    3. 计算辅助损失以平衡专家的使用
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = config.n_routed_experts  # 可路由的专家总数

        self.scoring_func = config.scoring_func  # 专家选择的评分函数
        self.alpha = config.aux_loss_alpha  # 辅助损失的权重
        self.seq_aux = config.seq_aux  # 是否在序列级别计算辅助损失

        self.norm_topk_prob = config.norm_topk_prob  # 是否对top-k概率进行归一化
        self.gating_dim = config.hidden_size  # 用于门控的特征维度
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))  # 门控权重矩阵
        self.reset_parameters()  # 初始化参数

    def reset_parameters(self) -> None:
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = 0
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    """混合专家（MoE）前馈网络实现
    
    包含多个专家网络和一个门控机制，每个token的处理会被动态路由到最合适的专家组合
    可选地包含共享专家，它们会处理所有token
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        # 初始化可路由的专家网络
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        # 初始化门控机制
        self.gate = MoEGate(config)
        # 可选：初始化共享专家网络
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=torch.float16)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)  # 确保类型一致
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok
        # 当tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
        # 且token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
        # 意味token_idxs[:6] -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token（每个token有可能被多个专家处理，这取决于num_experts_per_tok）
        # 接下来9个位置token_idxs[6:15] -> [4,  5,  6, 10, 11, 12...]属于专家1处理的token...依此类推
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    """MiniMind模型的主体实现
    
    一个基于Transformer的语言模型，支持以下特性：
    1. 分组查询注意力（GQA）降低计算开销
    2. 旋转位置编码（RoPE）处理位置信息
    3. 混合专家机制（MoE）提升模型容量
    4. RMSNorm归一化提升训练稳定性
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        # 基础配置
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        # 词嵌入层
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        # 初始化Transformer层
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        # 最终的归一化层
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 预计算旋转位置编码的三角函数值
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, theta=config.rope_theta)
        # 注册为非持久化缓冲区（不会被保存到模型文件中）
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        batch_size, seq_length = input_ids.shape
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )

        return hidden_states, presents, aux_loss


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight
        self.OUT = CausalLMOutputWithPast()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        h, past_kvs, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(h[:, slice_indices, :])
        self.OUT.__setitem__('last_hidden_state', h)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT
