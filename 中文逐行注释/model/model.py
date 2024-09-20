import math
import struct
import inspect
from .LMConfig import LMConfig
from typing import Any, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

# 定义 RMSNorm 类，实现一种归一化方法，类似于 LayerNorm，但计算方式不同
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps  # 设置 epsilon，防止除零错误
        self.weight = nn.Parameter(torch.ones(dim))  # 初始化权重参数

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)  # 计算 RMSNorm

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)  # 应用 RMSNorm
        return output * self.weight  # 乘以权重参数

# 定义 precompute_pos_cis 函数，用于预计算位置编码的复数形式
def precompute_pos_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))  # 计算频率
    t = torch.arange(end, device=freqs.device)  # 生成时间序列
    freqs = torch.outer(t, freqs).float()  # 计算外积
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # 计算复数形式的位置编码
    return pos_cis

# 定义 apply_rotary_emb 函数，用于应用旋转位置编码
def apply_rotary_emb(xq, xk, pos_cis):
    def unite_shape(pos_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert pos_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return pos_cis.view(*shape)

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))  # 将 xq 转换为复数形式
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))  # 将 xk 转换为复数形式
    pos_cis = unite_shape(pos_cis, xq_)  # 调整 pos_cis 的形状
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)  # 应用旋转位置编码
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)  # 应用旋转位置编码
    return xq_out.type_as(xq), xk_out.type_as(xk)  # 返回结果

# 定义 repeat_kv 函数，用于重复 KV 头的值
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

# 定义 Attention 类，实现自注意力机制
class Attention(nn.Module):
    def __init__(self, args: LMConfig):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads  # 设置 KV 头的数量
        assert args.n_heads % self.n_kv_heads == 0  # 确保 KV 头的数量是总头数的因数
        self.n_local_heads = args.n_heads  # 设置本地头的数量
        self.n_local_kv_heads = self.n_kv_heads  # 设置本地 KV 头的数量
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 计算重复次数
        self.head_dim = args.dim // args.n_heads  # 计算每个头的维度
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)  # 初始化 Q 矩阵
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)  # 初始化 K 矩阵
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)  # 初始化 V 矩阵
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)  # 初始化输出矩阵
        self.k_cache, self.v_cache = None, None  # 初始化 KV 缓存
        self.attn_dropout = nn.Dropout(args.dropout)  # 初始化注意力 dropout
        self.resid_dropout = nn.Dropout(args.dropout)  # 初始化残差 dropout
        self.dropout = args.dropout  # 设置 dropout 概率
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn  # 判断是否使用 Flash Attention

        if not self.flash:
            # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))  # 初始化掩码
            mask = torch.triu(mask, diagonal=1)  # 生成上三角掩码
            self.register_buffer("mask", mask)  # 注册掩码

    def forward(self, x: torch.Tensor, pos_cis: torch.Tensor, use_kv_cache=False):
        bsz, seqlen, _ = x.shape
        if use_kv_cache and self.eval():  # 如果使用 KV 缓存且在评估模式下
            if self.k_cache is None or self.k_cache.shape[1] != x.shape[1] - 1:
                xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)  # 计算 Q, K, V
            else:
                token = x[:, -1:, :]  # 获取最后一个 token
                xq = torch.cat((torch.zeros_like(x[:, :-1, :]), self.wq(token)), dim=1)  # 更新 Q
                xk = torch.cat((self.k_cache, self.wk(token)), dim=1)  # 更新 K
                xv = torch.cat((self.v_cache, self.wv(token)), dim=1)  # 更新 V

            self.k_cache, self.v_cache = xk, xv  # 更新 KV 缓存
        else:
            xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)  # 计算 Q, K, V

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)  # 调整 Q 的形状
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)  # 调整 K 的形状
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)  # 调整 V 的形状

        xq, xk = apply_rotary_emb(xq, xk, pos_cis)  # 应用旋转位置编码

        xk = repeat_kv(xk, self.n_rep)  # 重复 K 的值
        xv = repeat_kv(xv, self.n_rep)  # 重复 V 的值

        xq = xq.transpose(1, 2)  # 调整 Q 的形状
        xk = xk.transpose(1, 2)  # 调整 K 的形状
        xv = xv.transpose(1, 2)  # 调整 V 的形状

        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None,
                                                                      dropout_p=self.dropout if self.training else 0.0,
                                                                      is_causal=True)  # 使用 Flash Attention
        else:
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)  # 计算注意力分数
            assert hasattr(self, 'mask')
            scores = scores + self.mask[:, :, :seqlen, :seqlen]  # 应用掩码
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)  # 计算 softmax
            scores = self.attn_dropout(scores)  # 应用注意力 dropout
            output = torch.matmul(scores, xv)  # 计算输出

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)  # 调整输出的形状

        output = self.wo(output)  # 应用输出矩阵
        output = self.resid_dropout(output)  # 应用残差 dropout
        return output  # 返回输出

# 定义 FeedForward 类，实现前馈神经网络
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim  # 设置隐藏层维度
            hidden_dim = int(2 * hidden_dim / 3)  # 调整隐藏层维度
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)  # 调整隐藏层维度
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # 初始化第一层线性变换
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # 初始化第二层线性变换
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # 初始化第三层线性变换
        self.dropout = nn.Dropout(dropout)  # 初始化 dropout

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))  # 前向传播

# 定义 MoEGate 类，实现专家混合（MoE）的门控机制
class MoEGate(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok  # 设置每个 token 选择的专家数量
        self.n_routed_experts = config.n_routed_experts  # 设置路由专家的数量

        self.scoring_func = config.scoring_func  # 设置评分函数
        self.alpha = config.aux_loss_alpha  # 设置辅助损失的权重
        self.seq_aux = config.seq_aux  # 设置序列辅助损失

        self.norm_topk_prob = config.norm_topk_prob  # 设置是否归一化 top-k 概率
        self.gating_dim = config.dim  # 设置门控维度
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))  # 初始化权重参数
        self.reset_parameters()  # 重置参数

    def reset_parameters(self) -> None:
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # 使用 Kaiming 初始化权重

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape

        hidden_states = hidden_states.view(-1, h)  # 调整隐藏状态的形状
        logits = F.linear(hidden_states, self.weight, None)  # 计算 logits
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)  # 计算 softmax 评分
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)  # 选择 top-k 专家

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20  # 计算归一化分母
            topk_weight = topk_weight / denominator  # 归一化 top-k 概率

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
            aux_loss = None
        return topk_idx, topk_weight, aux_loss  # 返回 top-k 专家索引、权重和辅助损失

# 定义 MOEFeedForward 类，实现专家混合（MoE）的前馈神经网络
class MOEFeedForward(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            FeedForward(
                dim=config.dim,
                hidden_dim=config.hidden_dim,
                multiple_of=config.multiple_of,
                dropout=config.dropout,
            )
            for _ in range(config.n_routed_experts)
        ])  # 初始化专家列表

        self.gate = MoEGate(config)  # 初始化门控机制
        if config.n_shared_experts is not None:
            self.shared_experts = FeedForward(
                dim=config.dim,
                hidden_dim=config.hidden_dim,
                multiple_of=config.multiple_of,
                dropout=config.dropout,
            )  # 初始化共享专家

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape

        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)

        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)

        if self.training:
            # 训练模式下，重复输入数据
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=torch.float16)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i])
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            # 推理模式下，只选择最优专家
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)

        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)

        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok
        # 例如当tokens_per_expert=[6, 15, 20, 26, 33, 38, 46, 52]
        # 当token_idxs=[3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...]
        # 意味着当token_idxs[:6] -> [3,  7, 19, 21, 24, 25,  4]位置的token都由专家0处理，token_idxs[6:15]位置的token都由专家1处理......
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 使用 scatter_add_ 进行 sum 操作
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache

# 定义 TransformerBlock 类，实现 Transformer 的一个块，包括自注意力和前馈神经网络
class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: LMConfig):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)  # 初始化自注意力机制

        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)  # 初始化注意力归一化
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)  # 初始化前馈神经网络归一化

        if args.use_moe:
            self.feed_forward = MOEFeedForward(args)  # 初始化专家混合前馈神经网络
        else:
            self.feed_forward = FeedForward(
                dim=args.dim,
                hidden_dim=args.hidden_dim,
                multiple_of=args.multiple_of,
                dropout=args.dropout,
            )  # 初始化前馈神经网络

    def forward(self, x, pos_cis, use_kv_cache=False):
        h = x + self.attention(self.attention_norm(x), pos_cis, use_kv_cache)  # 计算自注意力
        out = h + self.feed_forward(self.ffn_norm(h))  # 计算前馈神经网络
        return out  # 返回输出

# 定义 Transformer 类，实现整个 Transformer 模型
class Transformer(PreTrainedModel):
    config_class = LMConfig
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: LMConfig = None):
        super().__init__(params)
        if not params:
            params = LMConfig()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
class Transformer(PreTrainedModel):
    config_class = LMConfig
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: LMConfig = None):
        super().__init__(params)
        if not params:
            params = LMConfig()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)  # 初始化词嵌入层
        self.dropout = nn.Dropout(params.dropout)  # 初始化 dropout 层
        self.layers = torch.nn.ModuleList()  # 初始化 Transformer 块列表
        for layer_id in range(self.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))  # 添加 Transformer 块
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)  # 初始化归一化层
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)  # 初始化输出层
        self.tok_embeddings.weight = self.output.weight  # 共享词嵌入和输出层的权重
        pos_cis = precompute_pos_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len)  # 预计算位置编码
        self.register_buffer("pos_cis", pos_cis, persistent=False)  # 注册位置编码缓冲区

        self.apply(self._init_weights)  # 初始化模型权重

        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * params.n_layers))  # 对特定权重进行初始化

        self.last_loss = None  # 初始化最后一个损失
        self.OUT = CausalLMOutputWithPast()  # 初始化输出对象

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # 初始化线性层的权重
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)  # 初始化线性层的偏置
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # 初始化嵌入层的权重

    def forward(self, tokens: Optional[torch.Tensor] = None, targets: Optional[torch.Tensor] = None,
                use_kv_cache=False, **keyargs):
        if 'input_ids' in keyargs:
            tokens = keyargs['input_ids']  # 如果传入了 input_ids，则使用 input_ids
        if 'attention_mask' in keyargs:
            targets = keyargs['attention_mask']  # 如果传入了 attention_mask，则使用 attention_mask

        _bsz, seqlen = tokens.shape  # 获取批量大小和序列长度
        h = self.tok_embeddings(tokens)  # 获取词嵌入
        h = self.dropout(h)  # 应用 dropout
        pos_cis = self.pos_cis[:seqlen]  # 获取对应序列长度的位置编码
        for idx, layer in enumerate(self.layers):
            h = layer(h, pos_cis, use_kv_cache)  # 逐层应用 Transformer 块

        h = self.norm(h)  # 应用归一化

        if targets is not None:
            logits = self.output(h)  # 计算 logits
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)  # 计算交叉熵损失
        else:
            logits = self.output(h[:, [-1], :])  # 计算最后一个 token 的 logits
            self.last_loss = None  # 没有目标时，损失为 None

        self.OUT.__setitem__('logits', logits)  # 设置输出对象的 logits
        self.OUT.__setitem__('last_loss', self.last_loss)  # 设置输出对象的 last_loss

        return self.OUT  # 返回输出对象

    @torch.inference_mode()  # 推理模式
    def generate(self, idx, eos, max_new_tokens, temperature=0.7, top_k=None, stream=True, repetition_penalty=1.,
                 use_kv_cache=True):
        index = idx.shape[1]  # 获取当前序列长度
        while idx.shape[1] < max_new_tokens - 1:  # 当生成的 token 数量小于最大数量时
            inference_res = self(idx, use_kv_cache=use_kv_cache)  # 进行前向传播
            logits = inference_res.logits  # 获取 logits
            logits = logits[:, -1, :]  # 获取最后一个 token 的 logits

            for token in set(idx.tolist()[0]):  # 对重复 token 进行惩罚
                logits[:, token] /= repetition_penalty

            if temperature == 0.0:  # 如果温度为 0，直接选择概率最高的 token
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature  # 调整 logits
                if top_k is not None:  # 如果设置了 top-k 采样
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')  # 将小于 top-k 的 logits 设为负无穷

                probs = F.softmax(logits, dim=-1)  # 计算概率
                idx_next = torch.multinomial(probs, num_samples=1, generator=None)  # 采样下一个 token

            if idx_next == eos:  # 如果生成的 token 是结束符，停止生成
                break

            idx = torch.cat((idx, idx_next), dim=1)  # 将生成的 token 添加到序列中
            if stream:  # 如果需要流式输出
                yield idx[:, index:]  # 返回生成的 token

        if not stream:  # 如果不需要流式输出
            yield idx[:, index:]  # 返回生成的 token

    @torch.inference_mode()  # 推理模式
    def eval_answer(self, idx):
        idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]  # 截取序列
        inference_res = self(idx_cond)  # 进行前向传播
        logits = inference_res.logits  # 获取 logits
        logits = logits[:, -1, :]  # 获取最后一个 token 的 logits
        return logits  # 返回 logits