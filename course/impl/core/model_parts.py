"""Core model parts to implement during the model-structure stage.

The functions and classes here intentionally start as skeletons. Course
lessons will ask the learner to fill them in and align behavior with
model/model_minimind.py.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from transformers.activations import ACT2FN


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """TODO: implement RoPE half-rotation.

    Align with: model/model_minimind.py::apply_rotary_pos_emb.rotate_half
    """
    raise NotImplementedError("Implement in the RoPE lesson.")


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TODO: apply RoPE to q and k.

    Align with: model/model_minimind.py::apply_rotary_pos_emb
    """
    raise NotImplementedError("Implement in the RoPE lesson.")


class RMSNorm(nn.Module):
    """TODO: implement RMSNorm.

    Align with: model/model_minimind.py::RMSNorm
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Implement in the RMSNorm lesson.")


class Attention(nn.Module):
    """TODO: implement a minimal MiniMind attention module.

    Align with: model/model_minimind.py::Attention
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.rms_norm_eps = rms_norm_eps

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        raise NotImplementedError("Implement across the Attention/RoPE/KV cache lessons.")


class FeedForward(nn.Module):
    """TODO: implement dense FFN.

    Align with: model/model_minimind.py::FeedForward
    """

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str = "silu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.gate_proj=nn.Linear(self.hidden_size,self.intermediate_size,bias=False)
        self.up_proj=nn.Linear(self.hidden_size,self.intermediate_size,bias=False)
        self.down_proj=nn.Linear(self.intermediate_size,self.hidden_size,bias=False)
        self.act_fn=ACT2FN[hidden_act]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x))*self.up_proj(x))


class MOEFeedForward(nn.Module):
    """TODO: implement a minimal MiniMind MoE FFN.

    Align with: model/model_minimind.py::MOEFeedForward
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 4,
        num_experts_per_tok: int = 1,
        hidden_act: str = "silu",
        norm_topk_prob: bool = True,
        router_aux_loss_coef: float = 5e-4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.hidden_act = hidden_act
        self.norm_topk_prob = norm_topk_prob
        self.router_aux_loss_coef = router_aux_loss_coef
        self.aux_loss = torch.zeros(())
        self.gate=nn.Linear(self.hidden_size,self.num_experts,bias=False)
        self.experts=nn.ModuleList(FeedForward(self.hidden_size,self.intermediate_size) for _ in range(self.num_experts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size,seq_len,hidden_size=x.shape
        x_flat=x.view(-1,hidden_size)
        scores=F.softmax(self.gate(x_flat),dim=-1)
        topk_weight,topk_idx=torch.topk(scores,k=self.num_experts_per_tok,dim=-1,sorted=False)#[B*S,num_experts_per_tok]
        if self.norm_topk_prob:topk_weight=topk_weight/(topk_weight.sum(dim=-1,keepdim=True)+1e-20)
        y=torch.zeros_like(x_flat)
        for i,expert in enumerate(self.experts):
            mask=(topk_idx==i)
            if mask.any():
                token_idx=mask.any(dim=-1).nonzero().flatten()#选出带有这个expert的每个token的id
                weight=topk_weight[mask].view(-1,1)#取出这些token对应的expert的权重
                y.index_add_(0,token_idx,(expert(x_flat[token_idx])*weight).to(y.dtype))#把经过计算的向量加回原先的位置
            elif self.training:
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        if self.training and self.router_aux_loss_coef > 0:
            load=F.one_hot(topk_idx,self.num_experts).float()#[B*S,num_experts_per_tok,self.num_experts]
            load=load.mean(0)#[num_experts_per_tok,self.num_experts]
            self.aux_loss=(load*scores.mean(dim=0)).sum()*self.router_aux_loss_coef*self.num_experts
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()#创建一个 shape = [1] 的全 0 tensor
        return y.view(batch_size,seq_len,hidden_size)
        
        
        raise NotImplementedError("Implement in the MoE lesson.")
