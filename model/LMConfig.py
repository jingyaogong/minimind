from transformers import PretrainedConfig
from typing import List


class LMConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
            self,
            dim: int = 512,
            n_layers: int = 8,
            n_heads: int = 8,
            n_kv_heads: int = 2,
            vocab_size: int = 6400,
            hidden_dim: int = None,
            multiple_of: int = 64,
            norm_eps: float = 1e-5,
            max_seq_len: int = 8192,
            rope_theta: int = 1e6,
            dropout: float = 0.0,
            flash_attn: bool = True,
            # MLA相关配置
            use_mla: bool = False,
            q_lora_rank: int = 720,  # query压缩的维度
            kv_lora_rank: int = 14,
            qk_nope_head_dim: int = 38,
            qk_rope_head_dim: int = 32,
            v_head_dim: int = 64,
            original_seq_len = 512,
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            use_moe: bool = False,
            ####################################################
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: bool = True,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.1,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            use_cache: bool = False,
            **kwargs,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.dropout = dropout
        self.flash_attn = flash_attn

        self.use_mla = use_mla
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.original_seq_len = original_seq_len
        self.mscale: float = 1.
        self.rope_factor = 20
        self.max_batch_size = 1
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
        self.use_cache = use_cache
        super().__init__(**kwargs)
