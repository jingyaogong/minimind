#########################################
#           MiniMind Config
#########################################
from transformers import PretrainedConfig

class MiniMindConfig(PretrainedConfig):
    """
    MiniMind Config, 继承自 HuggingFace 的 PretrainedConfig
    用于设置和管理模型的各种超参数和结构设置
    """
    model_type = "minimind"

    def __init__(
            self,
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = 'silu',
            hidden_size: int = 512,
            intermediate_size: int = None,  # FFN 中间层大小, 推荐不用设置, 会自动计算
            max_position_embeddings: int = 32768,
            num_attention_heads: int = 8,   # 注意力头数, 也是 Query 的头数
            num_hidden_layers: int = 8,
            num_key_value_heads: int = 2,   # Key / Value 的头数, 如果未设定则等于 num_attention_heads
            vocab_size: int = 6400,
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000.0,
            inference_rope_scaling: bool = False,
            flash_attn: bool = True,
            ####################################################
            # MOE 相关配置
            # 当 use_moe = false 时, 以下配置将无效
            ####################################################
            use_moe: bool = False,
            num_experts_per_tok: int = 2,   # 每个 token 选择的专家数量
            n_routed_experts: int = 4,      # 总的专家数量
            n_shared_experts: int = 1,      # 共享专家
            scoring_func: str = 'softmax',  # 评分函数, 默认 'softmax'
            aux_loss_alpha: float = 0.01,   # 辅助损失的 alpha 参数
            seq_aux: bool = True,           # 是否在序列级别上计算辅助损失
            norm_topk_prob: bool = True,    # 是否标准化 top-k 概率, 推荐启用
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
        self.inference_rope_scaling = inference_rope_scaling
        # 外推长度 = factor * original_max_position_embeddings = 32768
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        self.flash_attn = flash_attn
        ####################################################
        # MOE 相关配置
        # 当 use_moe = false 时, 以下配置将无效
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok      # 每个 token 选择的专家数量
        self.n_routed_experts = n_routed_experts            # 总的专家数量
        self.n_shared_experts = n_shared_experts            # 共享专家
        self.scoring_func = scoring_func                    # 评分函数, 默认 'softmax'
        self.aux_loss_alpha = aux_loss_alpha                # 辅助损失的 alpha 参数
        self.seq_aux = seq_aux                              # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob                # 是否标准化 top-k 概率, 推荐启用



#########################################
#           MiniMind Model
#########################################
import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
# https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(torch.nn.Module):
    """
    RMSNorm (Root Mean Square Normalization) 标准化层
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        # 数值稳定性的小常数
        self.eps = eps
        # 可学习的权重参数
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # 计算 RMS (Root Mean Square) 标准化
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 应用 RMS 标准化并乘以权重
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: Optional[dict] = None):
    """
    预计算 Rotary Position Embedding (RoPE) 的频率
    """
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
            # YaRN 缩放公式, 用于扩展位置编码的有效长度
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    应用 Rotary Position Embedding (RoPE) 到 query 和 key
    """
    def rotate_half(x):
        # 将张量的后半部分移到前半部分, 实现旋转
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    重复扩充 key/value 头的 dim, 使其数量与 query 的头数对齐, 用来支持 GQA (Grouped Query Attention)
    torch.repeat_interleave(x, dim=2, repeats=n_rep)
    https://docs.pytorch.ac.cn/docs/stable/generated/torch.repeat_interleave.html

    :param: x: 一批 tensor 数据, shape 为 (bs, seq_len, num_key_value_heads, head_dim)
    :param: n_rep:  重复次数; 如果 query 有 32 个头, key/value 只有 4 个头, 则 n_rep = 8
    """
    # 获取 x 的形状, 并分别赋予 bs, seq_len, num_key_value_heads, head_dim
    bs, slen, num_key_value_heads, head_dim = x.shape
    # n_rep = 1, 则不重复, 直接返回原数据
    if n_rep == 1:
        return x
    # n_rep != 1, 则重复 kv 指定次数
    else:
        return x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
        # #### 等效代码 ####
        # # 在第三维上增加一个维度, 即 (bs, slen, num_key_value_heads, head_dim) -> (bs, slen, num_key_value_heads, 1, head_dim)
        # x = x[:, :, :, None, :]
        # # 将这个新增的维度扩展成 n_rep 大小 -> (bs, slen, num_key_value_heads, n_rep, head_dim)
        # x = x.expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        # # 将维度重新调整为 -> (bs, slen, num_key_value_heads * n_rep, head_dim)
        # # 注意这里必须是 num_key_value_heads * n_rep, 不能反过来
        # # num_key_value_heads * n_rep 意味着 -> [kv0, kv0, kv1, kv1], 如果反过来就变为 -> [kv0, kv1, kv0, kv1]
        # # 这会导致 query0 找到了 kv0, 但同组的 query1 却跑去匹配了 kv1, 这违背了 GQA 的基本原理
        # x = x.reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
        # # 返回 x
        # return x
        # #################

class Attention(nn.Module):
    """
    Attention 模块, 支持 GQA (Grouped Query Attention)
    """
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        # 确定 kv 头的数量, 支持 Grouped Query Attention (GQA)
            # 如果 args.num_key_value_heads = None, 则 self.num_key_value_heads = args.num_attention_heads
            # 否则 self.num_key_value_heads = args.num_key_value_heads
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        # 断言 args.num_attention_heads 必须能被 self.num_key_value_heads 整除
        assert args.num_attention_heads % self.num_key_value_heads == 0
        # Query 头数
        self.n_local_heads = args.num_attention_heads
        # Key / Value 头数
        self.n_local_kv_heads = self.num_key_value_heads
        # 重复因子
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 每个头的维度
        self.head_dim = args.hidden_size // args.num_attention_heads
        # 投影层: 将隐藏状态映射到 query、key、value
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        # 检查是否支持Flash Attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn

    def forward(
            self,
            x: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],             # 接收 cos 和 sin -> cos, sin = position_embeddings
            past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # 用于缓存 kv 的变量
            use_cache=False,                                                    # 是否开启 kv 缓存功能
            attention_mask: Optional[torch.Tensor] = None                       # 注意力掩码矩阵, 形状为 (batch_size, seq_len)
        ):
            # 获取 x 的维度信息
            bsz, seq_len, _ = x.shape
            # 投影到 query、key、value 空间
            xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
            # 重塑为多头格式
            xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
            xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
            xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
            # 获取位置编码
            cos, sin = position_embeddings
            # 应用旋转位置编码 (qk需要, k不需要)
            xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

            # kv_cache 实现: 拼接过去的 key/value
            if past_key_value is not None:
                xk = torch.cat([past_key_value[0], xk], dim=1)
                xv = torch.cat([past_key_value[1], xv], dim=1)
            past_kv = (xk, xv) if use_cache else None

            # 转置为注意力计算格式: (batch, heads, seq_len, head_dim)
            xq, xk, xv = (
                xq.transpose(1, 2),
                repeat_kv(xk, self.n_rep).transpose(1, 2),
                repeat_kv(xv, self.n_rep).transpose(1, 2)
            )

            # 使用 Flash Attention
            if self.flash and (seq_len > 1) and (past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
                # https://docs.pytorch.ac.cn/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
                output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
            # 其余情况使用 "标注注意力"
            else:
                # 计算注意力分数
                scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
                # 因果掩码: 防止看到未来信息
                # https://docs.pytorch.ac.cn/docs/stable/generated/torch.triu.html
                    # torch.triu(input, diagonal) 
                    # 返回矩阵 (2D 张量) 或矩阵批次的上三角部分 input, 结果张量 out 的其他元素将设置为 0
                    # diagonal 控制要考虑的对角线
                        # 如果 diagonal = 0, 则保留主对角线及其上方的所有元素
                        # 正值会排除主对角线以上的相同数量的对角线
                        # 负值会包含主对角线以下的相同数量的对角线
                scores[:, :, :, -seq_len:] += torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1)

                # attention_mask shape: (batch_size, seq_len)
                # - 1 表示该位置是有效 token
                # - 0 表示该位置是 padding
                if attention_mask is not None:
                    # (bs, seq_len) -> (bs, 1, 1, seq_len)
                    # 为了让掩码能够广播 (broadcast) 到注意力分数矩阵的维度上
                    # 因为 scores shape: (bs, num_heads, seq_len, seq_len)
                    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                    # -1e9 是一个极大的负数(-1000000000.0), 用于屏蔽无效位置
                    extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                    # 应用掩码:
                    # - 如果 extended_attention_mask = 1, 应用后对应位置 mask = 0 -> scores 不变 (保留)
                    # - 如果 extended_attention_mask = 0, 应用后对应位置 mask 接近 -inf -> scores 分数会大幅拉低, 经过 Softmax ≈ 0
                    scores = scores + extended_attention_mask

                # Softmax 归一化
                scores = F.softmax(scores.float(), dim=-1).type_as(xq)
                # Dropout
                scores = self.attn_dropout(scores)
                # 计算注意力输出 (乘上 Value)
                output = scores @ xv

            # 重塑并输出
            output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
            output = self.resid_dropout(self.o_proj(output))
            return output, past_kv


class FeedForward(nn.Module):
    """
    前反馈神经网络 
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        # 计算 FFN 中间层大小, 确保是 64 的倍数, 是为了对 GPU 内存做对齐优化, 用来加速推理
        if config.intermediate_size is None:
            # SwiGLU FFN 中间层大小建议为 hidden_size * 8/3 (LLaMA 风格)
            intermediate_size = int(config.hidden_size * 8 / 3)
            # 对中间层大小进行 64 的倍数对齐: 向上取整到最近的 64 的倍数
            # - 公式: (x + n - 1) // n * n
            # - 加上 63 -> 对该值 "向下取整(//)" -> 乘上 64 
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        # SwiGLU FFN 的三个线性层
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)  # 门控层
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)  # 下投影层
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)    # 上投影层
        self.dropout = nn.Dropout(config.dropout)
        # 激活函数, 默认为 SiLU
        # https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
        # from transformers.activations import ACT2FN
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # SwiGLU: gate * up -> down
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class MoEGate(nn.Module):
    """
    MoE 门控网络

    - 根据输入的隐藏状态, 动态地为每个 token 选择 top-k 个最合适的专家, 并输出其权重
    - 同时可选地计算一个负载均衡辅助损失来防止专家 "垄断"

    输入: 
        - hidden_states: (batch_size, seq_len, hidden_size)
    输出:
        - topk_idx:     每个 token 选择的 top-k 专家索引, 形状为 (bsz * seq_len, k)
        - topk_weight:  对应的专家权重 (归一化后), 形状同上
        - aux_loss:     辅助损失项 (用于训练时均衡专家负载)
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok             # 每个 token 选择的专家数量
        self.n_routed_experts = config.n_routed_experts     # 路由专家总数

        self.scoring_func = config.scoring_func             # 评分函数, 仅支持 softmax
        self.alpha = config.aux_loss_alpha                  # 辅助损失权重, 常见为 0.01 ~ 0.1
        self.seq_aux = config.seq_aux                       # 是否使用序列级辅助损失 (False 比较常见)

        self.norm_topk_prob = config.norm_topk_prob         # 是否标准化 top-k 概率, 推荐启用
        self.gating_dim = config.hidden_size                # 门控维度, 注意维度和隐藏层维度相同
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))  # 门控权重
        self.reset_parameters()                             # Kaiming 初始化 self.weight 权重

    def reset_parameters(self) -> None:
        # Kaiming 初始化权重
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        """
        :param: hidden_states (bsz, seq_len, hidden_size)
        """
        # 1.记录原始形状并展平, 将所有 token 视为独立样本
        bsz, seq_len, h = hidden_states.shape
        # shape: (bsz * seq_len, hidden_size)
        hidden_states = hidden_states.view(-1, h) 

        # 2.计算路由分数 (Gating Scores)
        # logits: (bsz * seq_len, n_routed_experts)
        # 这一步计算 token 与每个专家的相似度
        # https://docs.pytorch.ac.cn/docs/stable/generated/torch.nn.functional.linear.html
        # - hidden_states: (bsz * seq_len, hidden_size)
        # - self.weight:   (n_routed_experts, hidden_size) -> self.gating_dim = config.hidden_size
        # - logits: hidden_states @ weight.T = (bsz * seq_len, n_routed_experts)
        logits = F.linear(hidden_states, self.weight, None)
        
        if self.scoring_func == 'softmax':
            # 使用 softmax 转换为概率分布, 和为1
            # scores: (bsz * seq_len, n_routed_experts)
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        # 3.选择 top-k 专家
        # https://docs.pytorch.ac.cn/docs/stable/generated/torch.topk.html
        # 给定 scores 张量在给定维度(dim=-1)上最大的 k 个元素 -> (values, indices)
        # - topk_weight: 对应专家概率 (bsz * seq_len, top_k)
        # - topk_idx:    对应专家索引 (bsz * seq_len, top_k)
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # 4.标准化 top-k 概率
        # 如果选择了多个专家, 则将它们的概率重新归一化, 使其总和为 1
        if self.top_k > 1 and self.norm_topk_prob:
            # 分母
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            # 权重再归一化后的概率
            topk_weight = topk_weight / denominator

        # 5.计算辅助损失 (用于负载均衡)
        if self.training and self.alpha > 0.0:  # self.training 继承于 torch.nn.Module
            # (bsz * seq_len, n_routed_experts)
            scores_for_aux = scores
            aux_topk = self.top_k
            # (bsz * seq_len, top_k) -> (bsz, seq_len * top_k)
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)

            # 分支1: 序列级辅助损失 (较少见)
            if self.seq_aux:
                # (bsz, seq_len, n_routed_experts)
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                # ce 是 Counter for Experts / Cumulative Expert usage count
                # ce shape (bsz, n_routed_experts)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)

                # 目标张量为 ce, 作用是统计每个 sequence 中各个专家被选中的频率
                # https://docs.pytorch.ac.cn/docs/stable/generated/torch.Tensor.scatter_add_.html
                # - Tensor.scatter_add_(dim, index, src)
                # - 把 src[i] 的值, 累加到 "目标张量" 中由 index[i] 指定的 dim 维度上的位置
                # - self, index 和 src 必须具有相同的维度数

                # div_(seq_len * aux_topk / self.n_routed_experts) 是归一化因子
                # 如果完美均衡, 每个专家应被选中 (seq_len * top_k) / n_routed_experts 次
                # 因此 ce.div_(xxx) 之后, ce[i][j] 变成 "归一化的期望频次"
                # https://docs.pytorch.ac.cn/docs/stable/generated/torch.div.html
                # - torch.div(input, other, *, rounding_mode=None, out=None)
                # - 将输入 input 的每个元素除以 other 的相应元素
                ce.scatter_add_(
                    # 在 "专家维度" 上累加
                    dim = 1, 
                    # 每个 token 的 top-k 专家编号
                    index = topk_idx_for_aux_loss, 
                    # 对每个选中操作, 这里为每次加 1
                    src = torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(seq_len * aux_topk / self.n_routed_experts)
                
                # 计算辅助损失
                # 目标是最小化专家负载的不均衡性, 即让 f_i ≈ 1/n
                # 公式: aux_loss = alpha * (P_i * f_i).sum()
                # - P_i: 第 i 个专家的平均预测概率, 来自 scores_for_seq_aux.mean(dim=1)
                # - f_i: 第 i 个专家的实际被选中频率 (归一化后), 来自 ce
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
                ######################################################
                # scores_for_seq_aux.mean(dim=1)
                # -> 每个样本中, 各个专家的平均预测概率 (P_i)
                # ce * scores_for_seq_aux.mean(dim=1)
                # -> 每个样本中, 对每个专家的 频率 × 预测概率 (P_i * f_i)
                # .sum(dim=1).mean()
                # -> 每个样本的负载不均衡程度总和, 并平均到整个 batch
                # self.alpha
                # -> 缩放, 避免主导主 loss
                ######################################################

            # 分支2: Token 级/Batch 级辅助损失 (默认)
            else:
                # 1.计算 expert 被选中的实际频率 f_i
                # - mask_ce shape: (bsz * seq_len * n_routed_experts, n_routed_experts)
                # https://docs.pytorch.ac.cn/docs/stable/generated/torch.nn.functional.one_hot.html
                # 接收形状为 (*) 的索引值的 LongTensor, 并返回形状为 (*, num_classes) 的张量
                # - 每行代表一个选择事件
                # - 每列代表一个专家 (num_classes=self.n_routed_experts)
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                # 每个专家在 batch 中被选中的比例
                # ce shape: (n_routed_experts,)
                ce = mask_ce.float().mean(0)
                # fi: 归一化后的频率, 理想情况下 fi 应接近 1
                fi = ce * self.n_routed_experts

                # 2.计算 expert 的平均路由概率 P_i
                # Pi shape: (n_routed_experts,)
                Pi = scores_for_aux.mean(0)

                # 3.计算辅助损失
                # 计算点积并求和: aux_loss = alpha * (P_i * f_i).sum()
                # 期望分布是均匀的, 这个 Loss 最小化时通常意味着均衡
                aux_loss = (Pi * fi).sum() * self.alpha

        # 不需要计算辅助损失
        else:
            aux_loss = scores.new_zeros(1).squeeze()
            
        # 返回结果
        # - topk_idx:       选中的专家索引: (bsz * seq_len, top_k)
        # - topk_weight:    选中的专家权重: (bsz * seq_len, top_k)
        # - aux_loss:       辅助损失, 用于负载均衡
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    """
    MOE 前反馈神经网络
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        # 创建路由专家列表, 只有被 Gate 选中的才会被计算
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        # 门控网络: 输入是 x, 输出是 top-k 的专家索引和对应权重 (还有辅助损失 aux_loss)
        self.gate = MoEGate(config)
        # 创建共享专家列表 (可选)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        """
        :param: x: (bsz, seq_len, hidden_size)
        """
        # 原始输入, 用于后续共享专家相加
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape

        # 通过 Gate 获取路由结果
        # - topk_idx:       选中的专家索引: (bsz * seq_len, top_k)
        # - topk_weight:    选中的专家概率: (bsz * seq_len, top_k)
        # - aux_loss:       辅助损失, 用于负载均衡
        topk_idx, topk_weight, aux_loss = self.gate(x)

        # 准备数据
        x = x.view(-1, x.shape[-1])         # (bsz * seq_len, hidden_size)
        flat_topk_idx = topk_idx.view(-1)   # (bsz * seq_len * top_k)

        # 训练模式: 使用专家并行处理
        if self.training:
            # 将 x 重复扩展以匹配专家
            # 如果 k=2, 每个 token 需要被处理两次
            # https://docs.pytorch.ac.cn/docs/stable/generated/torch.repeat_interleave.html
            # https://docs.pytorch.ac.cn/docs/stable/generated/torch.Tensor.repeat.html
            # - Tensor.repeat(*repeats)
            # - repeats: 每个元素重复的次数
            # - dim: 沿着指定维度进行重复
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)     # (bsz * seq_len * top_k, hidden_size)

            # https://docs.pytorch.ac.cn/docs/stable/generated/torch.empty_like.html
            # 预分配输出空间 (返回一个大小与 x 相同的未初始化张量)
            y = torch.empty_like(x, dtype=x.dtype)      # (bsz * seq_len * top_k, hidden_size)

            # 遍历每一个专家
            for i, expert in enumerate(self.experts):
                # 制作掩码: 找出所有分配给专家 i 的 token 位置
                mask = (flat_topk_idx == i)
                # x[mask]: 只属于该专家的输入
                expert_out = expert(x[mask])
                # 专家输出不为空时, 将专家输出写入到 y 中对应位置
                if expert_out.shape[0] > 0: 
                    y[mask] = expert_out.to(y.dtype)
                # 否则, 添加一个零梯度的项, 保证即使没有 token 被分配, 参数仍有梯度
                else: 
                    # 这里的 0 * sum(...) 确保该专家的参数在计算图中, 避免 DDP 报错 (unused parameters)
                    y[mask] = expert_out.to(y.dtype) + 0 * sum(p.sum() for p in expert.parameters())

            # 聚合结果
            # y 目前的形状是 (bsz * seq_len * top_k, hidden_size), 代表 "展开的" 专家输出
            # 需要将 y 加权合并为 -> (bsz * seq_len, hidden_size)
            # - topk_weight： (bsz * seq_len, top_k)
            # - y.view(*topk_weight.shape, -1): (bsz * seq_len, top_k, hidden_size)
            # - topk_weight.unsqueeze(-1):      (bsz * seq_len, top_k, 1)
            # - 广播乘法 (N, k, h) * (N, k, 1) = (N, k, h)
            # - 最后在 top_k 维度上求和 (dim=1): (bsz * seq_len, top_k, hidden_size) -> (bsz * seq_len, hidden_size)
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            # y 已经是 (bsz, seq_len, hidden_size) 了, 这里的 view 只是为了保险
            y = y.view(*orig_shape)

        # 推理模式: 使用优化的专家推理
        # - 使用优化后的串行处理 (减少显存占用, 避免大量 mask 操作)
        else:
            # flat_topk_idx: (bsz * seq_len * top_k)
            # topk_weight.view(-1, 1) 展平了所有维度: (bsz * seq_len * top_k, 1)
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)

        # 添加共享专家的输出
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss

        # (bsz, seq_len, hidden_size)
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """
        推理模式下高效 MoE 实现
        不使用 repeat_interleave 复制 x (省显存), 而是利用索引重排
        
        :param x: (bsz * seq_len, hidden_size)
        :param flat_expert_indices: (bsz * seq_len * top_k)     专家编号
        :param flat_expert_weights: (bsz * seq_len * top_k, 1)  对应的权重
        """
        # 结果缓存 (bsz * seq_len, hidden_size)
        expert_cache = torch.zeros_like(x)

        # 1.对所有 token 的任务分配进行排序
        # idxs 是排序后的索引, 能够把 "分配给专家0的任务", "分配给专家1的任务" 聚在一起
        idxs = flat_expert_indices.argsort()

        # 2.计算每个专家处理多少个 token
        # - bincount 统计每个专家出现的次数
            # https://docs.pytorch.ac.cn/docs/stable/generated/torch.bincount.html
        # - cumsum 返回在 dim 维度上的累积和, 用于切片
            # https://docs.pytorch.ac.cn/docs/stable/generated/torch.cumsum.html
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)

        # 3.计算排序后的索引对应原始的 token_id
        # - flat_expert_indices 长度是 N*k, 是展开的 (bsz * seq_len * top_k)
        # - // top_k 操作将 "展开后的索引" 映射回 "原始 token 的行号"
        token_idxs = idxs // self.config.num_experts_per_tok
        #####################################
        # 当 tokens_per_expert = [6, 15, 20, 26], tokens_per_expert.shape[0] 即为专家数量（此时为4）
        # 且 token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
        # 意味 token_idxs[:6] -> [3, 7, 19, 21, 24, 25] 这6个位置属于 "专家0" 处理的 token (每个token有可能被多个专家处理, 这取决于 num_experts_per_tok)
        # 接下来9个位置 token_idxs[6:15] -> [4,  5,  6, 10, 11, 12...] 属于 "专家1" 处理的 token...依此类推
        #####################################

        # 遍历每个专家 (根据 tokens_per_expert 切片)
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            # 该专家在这个 batch 没有任务
            if start_idx == end_idx:
                continue
            expert = self.experts[i]

            # 获取属于专家 i 的所有 token 的原始行号
            exp_token_idx = token_idxs[start_idx:end_idx]
            # 从 x 中取出这些 token
            expert_tokens = x[exp_token_idx]
            # 前向计算
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            # 乘上对应的权重
            # - idxs[start_idx:end_idx] 取出的是 "flat_expert_weights" 对应的索引
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])

            # 将结果累加回 expert_cache
            # scatter_add_ 处理这种情况: 如果一个 token 同时选择了专家 A 和 专家 B
            # 它的结果需要是 (OutA * wA) + (OutB * wB)
            # 这里利用 scatter_add 根据 token 索引将结果加回去
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)
            #####################################
            # https://docs.pytorch.ac.cn/docs/stable/generated/torch.Tensor.scatter_add_.html
            # - Tensor.scatter_add_(dim, index, src)
            # - 把 src[i] 的值, 累加到 "目标张量" 中由 index[i] 指定的 dim 维度上的位置
            # - self, index 和 src 必须具有相同的维度数
            #####################################

        # (bsz * seq_len, hidden_size)
        return expert_cache


class MiniMindBlock(nn.Module):
    """
    MiniMind 模型的一个 Transformer 块
    """
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        # 自注意力层 (Attention 模块)
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        # 输入前标准化
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) 
        # 注意力后标准化
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) 
        # 根据配置选择FFN类型: 普通FFN / MoE FFN
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        """
        :param hidden_states:           (batch_size, seq_len, hidden_size)
        :param position_embeddings:     Tuple[torch.Tensor, torch.Tensor]
        :param past_key_value:          Tuple[torch.Tensor, torch.Tensor]
        :param use_cache:               是否开启 kv 缓存功能
        :param attention_mask:          注意力掩码矩阵, 形状为 (batch_size, seq_len)
        """
        # 第一个残差连接：输入层标准化 -> 注意力 -> 残差连接
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), 
            position_embeddings,
            past_key_value, 
            use_cache, 
            attention_mask
        )
        hidden_states += residual

        # 第二个残差连接：注意力后标准化 -> FFN -> 残差连接
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))

        # 返回 输出和缓存的键值对
        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    """
    MiniMind 模型主类
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size                                         # 词表大小
        self.num_hidden_layers = config.num_hidden_layers                           # 隐藏层数量
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)     # 词嵌入层
        self.dropout = nn.Dropout(config.dropout)
        # 构建多个相同的 Transformer 块 (MiniMindBlock)
        # 每个块包含自注意力 + FFN, 通过 ModuleList 管理参数
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)            # 最终标准化层

        # 预计算旋转位置编码
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings, rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling
        )
        # 注册为缓冲区, 不会被优化
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
            self,
            # 输入 token 序列, 形状：(batch_size, seq_len)
            input_ids: Optional[torch.Tensor] = None,
            # 注意力掩码：1 表示有效 token, 0 表示 padding
            attention_mask: Optional[torch.Tensor] = None,
            # KV 缓存列表 (用于生成任务)
            # - 每个元素为 (key, value) 对
            # - 形状为 [(bs, seq_len_k, num_heads, head_dim), ...]
            past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
            # 是否启用缓存 (推理加速)
            use_cache: bool = False,
            **kwargs
        ):
        # 提取输入维度
        batch_size, seq_length = input_ids.shape

        # KV 缓存处理
        #######################################################################################
        # https://docs.python.org/zh-cn/3.14/library/functions.html#hasattr
        # - 如果字符串是对象的属性之一的名称, 则返回 True, 否则返回 False
        if hasattr(past_key_values, 'layers'): 
            # 如果传进来的 past_key_values 是一个带 .layers 属性的对象 (比如 HuggingFace 封装的输出)
            # 那就把它当作无效缓存处理, 清空为 None
            past_key_values = None
        
        # 对 past_key_values 进行默认值填充
        # 如果是 None、空列表 [] 或其他 "假值(falsy)"
        # 就用一个长度为 num_hidden_layers 的全 None 列表替代 -> [None] * len(self.layers)
        past_key_values = past_key_values or [None] * len(self.layers)
        #######################################################################################

        # 计算起始位置（用于KV缓存）
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # 词嵌入 + dropout
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # 获取位置编码
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        # 逐层前向传播
        presents = []
        # Transformer 块堆叠处理
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            # 每层 MiniMindBlock 接受如下参数
            # 并输出 "下一个时刻隐藏状态 hidden_states" 和 "KV 缓存对 present"
            hidden_states, present = layer(
                hidden_states,                  # 当前隐藏状态
                position_embeddings,            # 对应的 RoPE 位置编码 (cos, sin)
                past_key_value=past_key_value,  # 上一时刻缓存的 KV 缓存 (key, value) tuple
                use_cache=use_cache,            # 是否启用缓存
                attention_mask=attention_mask   # 注意力掩码
            )
            # 存储每层的 kv 缓存 
            presents.append(present)

        # 最终标准化
        hidden_states = self.norm(hidden_states)

        # 计算 MoE 辅助损失
        # 对所有使用了 MOEFeedForward 的层, 累加其 .aux_loss 属性 (属性来自 MOEFeedForward, 值来自 MoEGate)
        # https://docs.pytorch.ac.cn/docs/stable/generated/torch.Tensor.new_zeros.html
        # - Tensor.new_zeros(size, *, dtype=None, device=None, requires_grad=False, layout=torch.strided, pin_memory=False)
        # - 返回一个大小为 size 的、填充有 0 的 Tensor; 
        # - 默认情况下, 返回的 Tensor 具有与此 Tensor 相同的 torch.dtype 和 torch.device
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        ######################################################################################
        # 为什么用 sum(..., start) 而不是直接 sum(...)?
        # - 若无任何 MoE 层, 则列表为空 -> sum([]) 返回 int(0)
        # - 但模型其他部分使用 torch.Tensor, int 和 Tensor 相加会报错
        # - 因此必须显式指定一个与 hidden_states 同 device/dtype 的零标量作为初始值
        ######################################################################################
        # hidden_states.new_zeros(1).squeeze()
        # - hidden_states.new_zeros(1) 返回一个标量张量 [0.]
        # - squeeze() 删除维度 1, 变成标量 tensor: tensor(0.)
        #
        ##### 等效代码 ########################################################################
        # - zero_scalar = hidden_states.new_zeros(1).squeeze()  # 创建一个标量 tensor(0.)
        # - aux_loss = zero_scalar                              # aux_loss 初始值为 tensor(0.)
        # - for layer in self.layers:
        # -     if isinstance(layer.mlp, MOEFeedForward):       # 检查是否是 MoE 类型
        # -         aux_loss = aux_loss + layer.mlp.aux_loss    # 累加辅助损失
        ######################################################################################

        # 返回最终隐藏状态、缓存和 MoE 辅助损失
        return hidden_states, presents, aux_loss


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    """
    MiniMindForCausalLM 类, 用于构建因果语言模型
    - 将 MiniMindModel 主干与线性输出头 (lm_head) 组合, 实现完整的文本生成能力
    - 兼容 HuggingFace Transformers 的训练/推理接口 (如 model.generate())
    - 并支持自回归解码、KV 缓存、损失计算等标准功能

    # 参考链接
    https://huggingface.co/docs/transformers/main_classes/text_generation
    """
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        # 初始化配置项，如果未提供则使用默认参数初始化
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        # MiniMind 模型主体
        self.model = MiniMindModel(self.config)
        # 语言模型头 (输出头)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # 权重共享：词嵌入和语言模型头共享权重
        # - 减少参数量 ~15~20%（约 vocab_size * hidden_size）
        # - 提升泛化能力, 增强词向量与输出分布的一致性 (LLaMA、GPT 系列标准做法)
        # - 此操作需在 super().__init__() 后执行, 因为父类可能初始化参数
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
            use_cache: bool = False,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            **args
        ):
        # 主干模型的前向传播
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        # 计算logits (只保留最后 logits_to_keep 个位置以节省内存)
        # - 适用于长上下文推理中只关心生成末尾结果的场景
        # - 若 logits_to_keep = 0, 则保留全部序列
        # - 若为整数 N, 则取最后 N 个 token 的 logits
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # 标签对齐：采用标准的因果语言建模损失计算方式
            shift_logits = logits[..., :-1, :].contiguous()  # 去掉最后一个位置的预测
            shift_labels = labels[..., 1:].contiguous()      # 去掉第一个位置的标签
            # 交叉熵损失
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

        # 构建输出
        output = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
        output.aux_loss = aux_loss
        # CausalLMOutputWithPast 实例
        # https://huggingface.co/docs/transformers/v5.0.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast
        # - loss (torch.Tensor):   训练时计算的交叉熵损失; 推理时为 None
        # - logits (torch.Tensor): 预测得分, 形状: (batch_size, seq_len, vocab_size) 或截断后长度
        # - past_key_values (List[Tuple[torch.Tensor]]):      每层的 KV 缓存, 用于后续 token 生成
        # - hidden_states (torch.Tensor):        所有 Transformer 层输出, 默认不返回, 但继承类提供
        # - aux_loss (torch.Tensor, 自定义字段):                MoE 辅助损失, 仅在启用且训练时非零
        return output