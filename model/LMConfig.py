from transformers import PretrainedConfig
from typing import List

# 定义 LMConfig 类，继承自 PretrainedConfig
class LMConfig(PretrainedConfig):
    model_type = "minimind"  # 设置模型类型为 "minimind"

    def __init__(
            self,
            dim: int = 512,  # 模型维度，默认为 512
            n_layers: int = 8,  # Transformer 层数，默认为 8
            n_heads: int = 16,  # 注意力头数，默认为 16
            n_kv_heads: int = 8,  # KV 头数，默认为 8
            vocab_size: int = 6400,  # 词汇表大小，默认为 6400
            hidden_dim: int = None,  # 隐藏层维度，默认为 None
            multiple_of: int = 64,  # 隐藏层维度的倍数，默认为 64
            norm_eps: float = 1e-5,  # 归一化层的 epsilon 值，默认为 1e-5
            max_seq_len: int = 512,  # 最大序列长度，默认为 512
            dropout: float = 0.0,  # Dropout 概率，默认为 0.0
            flash_attn: bool = True,  # 是否使用 Flash Attention，默认为 True
            ####################################################
            # 以下是 MOE（Mixture of Experts）的特定配置
            # 当 use_moe 为 False 时，以下配置无效
            ####################################################
            use_moe: bool = False,  # 是否使用 MOE，默认为 False
            num_experts_per_tok=2,  # 每个 token 选择的专家数量，默认为 2
            n_routed_experts=4,  # 总的专家数量，默认为 4
            n_shared_experts: bool = True,  # 是否使用共享专家，默认为 True
            scoring_func='softmax',  # 评分函数，默认为 'softmax'
            aux_loss_alpha=0.01,  # 辅助损失的 alpha 参数，默认为 0.01
            seq_aux=True,  # 是否在序列级别上计算辅助损失，默认为 True
            norm_topk_prob=True,  # 是否标准化 top-k 概率，默认为 True
            **kwargs,
    ):
        self.dim = dim  # 设置模型维度
        self.n_layers = n_layers  # 设置 Transformer 层数
        self.n_heads = n_heads  # 设置注意力头数
        self.n_kv_heads = n_kv_heads  # 设置 KV 头数
        self.vocab_size = vocab_size  # 设置词汇表大小
        self.hidden_dim = hidden_dim  # 设置隐藏层维度
        self.multiple_of = multiple_of  # 设置隐藏层维度的倍数
        self.norm_eps = norm_eps  # 设置归一化层的 epsilon 值
        self.max_seq_len = max_seq_len  # 设置最大序列长度
        self.dropout = dropout  # 设置 Dropout 概率
        self.flash_attn = flash_attn  # 设置是否使用 Flash Attention
        ####################################################
        # 以下是 MOE（Mixture of Experts）的特定配置
        # 当 use_moe 为 False 时，以下配置无效
        ####################################################
        self.use_moe = use_moe  # 设置是否使用 MOE
        self.num_experts_per_tok = num_experts_per_tok  # 设置每个 token 选择的专家数量
        self.n_routed_experts = n_routed_experts  # 设置总的专家数量
        self.n_shared_experts = n_shared_experts  # 设置是否使用共享专家
        self.scoring_func = scoring_func  # 设置评分函数
        self.aux_loss_alpha = aux_loss_alpha  # 设置辅助损失的 alpha 参数
        self.seq_aux = seq_aux  # 设置是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 设置是否标准化 top-k 概率
        super().__init__(**kwargs)  # 调用父类 PretrainedConfig 的初始化方法