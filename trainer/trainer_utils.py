import os
import sys

# 设置包名
__package__ = "trainer"
# 将项目根目录添加到系统路径, 确保能够导入同级目录的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer
from model.model_minimind import MiniMindForCausalLM

def get_model_params(model, config):
    """
    计算并打印模型参数量统计信息

    Args:
    - model:    待统计的 PyTorch 模型
    - config:   模型配置对象, 包含 use_moe, n_routed_experts, num_experts, num_experts_per_tok, n_shared_experts 等属性

    Returns:
    - 无返回值, 会在主进程打印模型参数量信息
    """
    # 计算模型总参数量, 单位为百万 (M)
    total = sum(p.numel() for p in model.parameters()) / 1e6

    # 从配置中获取 MoE 相关的专家数量配置
    # n_routed: 路由专家总数, n_active: 每个 token 激活的专家数, n_shared: 共享专家数
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)

    # 计算路由专家 (routed experts) 的参数量, 单位为百万 (M)
    # 通过匹配参数名中包含 'mlp.experts.0.' 的参数来统计
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6

    # 计算共享专家 (shared experts) 的参数量, 单位为百万 (M)
    # 通过匹配参数名中包含 'mlp.shared_experts.0.' 的参数来统计
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6

    # 计算基础参数量 (非专家部分的参数)
    base = total - (expert * n_routed) - (shared_expert * n_shared)

    # 计算激活的参数量 (基础参数 + 激活的专家参数)
    active = base + (expert * n_active) + (shared_expert * n_shared)

    # 如果存在未被激活的专家参数, 打印总参数量和激活参数量, 否则只打印总参数量
    if active < total:
        Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else:
        Logger(f'Model Params: {total:.2f}M')


def is_main_process():
    """
    判断当前进程是否为主进程

    在分布式训练环境中, 只有 rank 为 0 的进程是主进程
    如果分布式环境未初始化, 也认为当前是主进程

    Returns:
    - bool: 如果是主进程返回 True, 否则返回 False
    """
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    """
    线程安全的日志打印函数

    只能在主进程 (rank 0) 打印日志, 避免分布式训练时日志重复输出

    Args:
    - content: 要打印的内容, 可以是字符串或任何可打印对象
    """
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    """
    计算当前学习率, 使用余弦退火策略 (Cosine Annealing)

    学习率变化曲线: 从 0.55*lr 开始, 经过余弦变化, 最终回到 0.1*lr
    公式: lr * (0.1 + 0.45 * (1 + cos(pi * current_step / total_steps)))

    Args:
    - current_step:     当前训练步数 (从 0 开始)
    - total_steps:      总训练步数
    - lr:               基础学习率

    Returns:
    - float:            当前步的学习率值
    """
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode():
    """
    初始化分布式训练模式

    检查环境变量 RANK 是否存在来判断是否使用分布式训练
    如果未使用分布式训练, 返回 0 表示非 DDP 模式
    如果使用分布式训练, 初始化 NCCL 后端, 设置本地 GPU 设备

    Returns:
    - int: 本地 rank 编号, 非 DDP 模式下返回 0
    """
    # 检查是否需要初始化分布式训练
    # 如果环境变量 RANK 不存在或为 -1, 则表示不使用分布式训练
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非 DDP 模式

    # 初始化分布式训练进程组, 使用 NCCL 后端 (适用于 GPU 通信)
    dist.init_process_group(backend="nccl")

    # 从环境变量获取本地 rank, 并设置当前进程使用的 GPU 设备
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    return local_rank


def setup_seed(seed: int):
    """
    设置所有随机数种子, 确保实验可复现
    - 设置 Python random、numpy、torch 的随机种子
    - 以及 CUDA 相关 的随机种子和确定性设置

    Args:
    - seed: 随机数种子值, 建议使用整数
    """
    # 设置 Python 内置随机数模块的种子
    random.seed(seed)
    # 设置 NumPy 的随机数种子
    np.random.seed(seed)
    # 设置 PyTorch CPU 计算的随机种子
    torch.manual_seed(seed)
    # 设置 PyTorch 单个 GPU 的随机种子
    torch.cuda.manual_seed(seed)
    # 设置 PyTorch 所有 GPU 的随机种子 (多卡训练时需要)
    torch.cuda.manual_seed_all(seed)
    # 设置 cuDNN 为确定性模式, 禁用优化以确保可复现
    # 这会降低训练速度, 但能确保结果可复现
    torch.backends.cudnn.deterministic = True
    # 禁用 cuDNN 的自动调优功能, 避免因算法选择导致的不可复现
    torch.backends.cudnn.benchmark = False


def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    """
    模型检查点保存与加载函数

    支持两种模式:
    1. 保存模式: 当 model 参数不为 None 时, 保存模型权重、优化器状态、训练进度等信息
    2. 加载模式: 当 model 参数为 None 时, 从检查点文件恢复训练状态

    Args:
    - lm_config:    模型配置对象, 包含 use_moe, hidden_size 等属性
    - weight:       检查点名称前缀, 默认为 'full_sft'
    - model:        待保存的模型对象, 为 None 时表示加载模式
    - optimizer:    优化器对象, 用于保存优化器状态
    - epoch:        当前训练轮次
    - step:         当前训练步数
    - wandb:        Weights & Biases 日志对象, 用于获取运行 ID
    - save_dir:     检查点保存目录
    - **kwargs:     额外的可保存对象 (如 lr_scheduler 等), 需要具备 state_dict 方法

    Returns:
    - dict 或 None: 加载模式下返回检查点数据字典, 保存模式下返回 None
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 构建检查点文件名
    # 根据是否使用 MoE 模型添加不同的后缀
    moe_path = '_moe' if lm_config.use_moe else ''

    # 完整模型权重文件路径
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'

    # 恢复训练所需的完整检查点文件路径 (包含优化器状态等)
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    # ===== 保存模式 =====
    if model is not None:
        # 获取原始模型 (如果是 DDP 包装的模型, 需要解包)
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        # 处理 PEFT/Lora 等场景下的模型 (使用 _orig_mod 属性获取原始模型)
        raw_model = getattr(raw_model, '_orig_mod', raw_model)

        # 获取模型状态字典, 并将权重转换为半精度 (FP16) 后转移到 CPU
        state_dict = raw_model.state_dict()
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}

        # 使用临时文件避免保存过程中断导致文件损坏
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)

        # 获取 Weights & Biases 的运行 ID
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        # 构建恢复训练所需的完整数据字典
        resume_data = {
            'model': state_dict,                        # 模型权重
            'optimizer': optimizer.state_dict(),        # 优化器状态
            'epoch': epoch,                             # 当前训练轮次
            'step': step,                               # 当前训练步数
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,  # 分布式训练 world size
            'wandb_id': wandb_id                        # W&B 运行 ID
        }

        # 处理额外的可保存对象 (如学习率调度器等)
        for key, value in kwargs.items():
            if value is not None:
                # 如果对象是模型 (可能是 DDP 包装的), 先解包再获取 state_dict
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        # 保存完整检查点到临时文件, 然后原子替换
        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)

        # 清理内存
        del state_dict, resume_data
        torch.cuda.empty_cache()

    # ===== 加载模式 =====
    else:
        # 如果检查点文件存在, 加载恢复数据
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')

            # 处理 GPU 数量变化的情况, 自动调整 step 值
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1

            if saved_ws != current_ws:
                # 当 GPU 数量变化时, 按比例调整 step
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU 数量变化 ({saved_ws}→{current_ws}), step 已自动转换为 {ckp_data["step"]}')

            return ckp_data

        # 如果检查点文件不存在, 返回 None
        return None


def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model', save_dir='../out', device='cuda'):
    """
    初始化模型和分词器

    加载 Hugging Face 格式的分词器, 创建 MiniMind 模型实例
    可选择从预训练权重加载模型参数

    Args:
    - lm_config:        模型配置对象
    - from_weight:      预训练权重名称前缀, 默认为 'pretrain', 设置为 'none' 表示不加载权重
    - tokenizer_path:   分词器模型目录路径
    - save_dir:         权重文件保存目录
    - device:           模型运行设备, 默认为 'cuda'

    Returns:
    - tuple: (model, tokenizer) 模型和分词器对象
    """
    # 从预训练目录加载分词器
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 创建 MiniMind 因果语言模型实例
    model = MiniMindForCausalLM(lm_config)

    # 如果指定了加载权重, 从检查点文件加载模型参数
    if from_weight != 'none':
        # 根据是否使用 MoE 构建权重文件名
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'

        # 加载权重到 CPU, 然后加载到模型
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    # 打印模型参数量统计信息
    get_model_params(model, lm_config)

    # 打印可训练参数量 (需要计算梯度)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')

    # 将模型移动到指定设备并返回
    return model.to(device), tokenizer


class SkipBatchSampler(Sampler):
    """
    可跳过指定数量批次的采样器

    继承自 PyTorch Sampler, 在训练过程中跳过指定数量的初始批次
    常用于从检查点恢复训练时, 跳过已经训练过的批次
    """

    def __init__(self, sampler, batch_size, skip_batches=0):
        """
        初始化 SkipBatchSampler

        Args:
        - sampler:      基础采样器, 提供数据索引序列
        - batch_size:   每个批次的样本数量
        - skip_batches: 要跳过的批次数量, 默认为 0 (不跳过)
        """
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        """
        生成批次索引迭代器

        遍历基础采样器的索引, 组装成批次
        跳过指定数量的初始批次

        Yields:
        - list: 批次索引列表, 每个列表包含 batch_size 个样本索引
        """
        batch = []
        skipped = 0

        # 遍历基础采样器的所有索引
        for idx in self.sampler:
            batch.append(idx)

            # 当批次满时
            if len(batch) == self.batch_size:
                # 如果还有需要跳过的批次, 跳过当前批次
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue

                # 产出当前批次
                yield batch
                batch = []

        # 处理最后一个不完整的批次
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        """
        返回有效批次的总数量

        Returns:
        - int: 跳过 skip_batches 后的可用批次数量
        """
        # 计算总批次数, 向上取整
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size

        # 返回跳过 skip_batches 后的批次数量, 不能为负数
        return max(0, total_batches - self.skip_batches)