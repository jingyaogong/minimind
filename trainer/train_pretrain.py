import os
import sys

# 设置包名
__package__ = "trainer"
# 将项目根目录添加到系统路径, 确保能够导入同级目录的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import torch
# 分布式训练支持
import torch.distributed as dist

# 上下文管理器
# https://docs.python.org/zh-cn/3.14/library/contextlib.html#contextlib.nullcontext
from contextlib import nullcontext
from torch import optim, nn
# 分布式数据并行 DDP
from torch.nn.parallel import DistributedDataParallel
# 数据加载器和分布式采样器
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
# 预训练数据集
from dataset.lm_dataset import PretrainDataset 
# 导入训练工具函数：学习率计算、日志记录、检查点保存、分布式初始化等
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

# 忽略所有警告信息，减少输出干扰
import warnings
warnings.filterwarnings('ignore')

def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """
    执行单个训练 epoch

    Args:
    - epoch:        当前 epoch 编号
    - loader:       数据加载器
    - iters:        当前 epoch 的总迭代次数
    - start_step:   起始 step 编号 (用于断点续训)
    - wandb:        实验日志工具实例
    """
    # 记录当前 epoch 开始时间
    start_time = time.time()

    # 遍历数据加载器, 获取批次数据
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # 将 数据 & 标签 移动到指定设备
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        
        # 根据当前 step 计算当前学习率
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        # 更新优化器中所有参数组的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 使用混合精度上下文进行前向传播
        # - 在 __name__ == "__main__" 中定义
        # - autocast_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(dtype=dtype)
        with autocast_ctx:
            # 模型前向传播, 其中包含损失
            res = model(input_ids, labels=labels)
            # 合并主损失和辅助损失 (如 MoE 的负载均衡损失)
            loss = res.loss + res.aux_loss
            # 根据梯度累积步数进行缩放
            loss = loss / args.accumulation_steps

        # 反向传播计算梯度
        # - 在 __name__ == "__main__" 中定义
        # - scaler = torch.amp.GradScaler("cuda", enabled=(args.dtype == 'float16'))
        scaler.scale(loss).backward()

        # 当达到梯度累积步数时，执行优化器更新
        if (step + 1) % args.accumulation_steps == 0:
            # 取消梯度缩放, 为优化器步骤做准备
            scaler.unscale_(optimizer)
            # 对梯度进行裁剪, 防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # 执行优化器步骤更新参数
            scaler.step(optimizer)
            # 更新缩放器状态
            scaler.update()
            # 清空梯度, 设置为 None 以节省内存
            optimizer.zero_grad(set_to_none=True)

        # 达到日志打印间隔时, 记录训练状态
        if step % args.log_interval == 0 or step == iters - 1:
            # 计算已消耗时间
            spend_time = time.time() - start_time
            # 获取当前损失值
            current_loss = loss.item() * args.accumulation_steps
            # 获取辅助损失值
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            # 计算 logits 损失 (主损失)
            current_logits_loss = current_loss - current_aux_loss
            # 获取当前学习率
            current_lr = optimizer.param_groups[-1]['lr']
            # 估算剩余时间 (分钟)
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            # 打印训练日志
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            # 记录到 wandb (如果启用)
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # 达到保存间隔且为主进程时, 保存模型检查点
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            # 设置为评估模式
            model.eval()
            # 生成模型文件名后缀, 区分 MoE 和普通模型
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # 获取原始模型 (移除 DDP 包装)
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            # 获取模型状态字典
            state_dict = raw_model.state_dict()
            # 保存模型权重 (转换为半精度以节省空间)
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # 保存完整训练检查点 (包含配置、优化器状态等)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            # 恢复训练模式
            model.train()
            # 释放显存
            del state_dict

        # 清理当前批次的数据, 释放显存
        del input_ids, labels, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数 (建议 1 轮 zero 或 2-6 轮充分训练)")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度 (中文 1token≈1.5~1.7 字符)")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用 MoE 架构 (0=否, 1=是)")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练, 为 none 则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测 & 续训 (0=否, 1=是)")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用 wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb 项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用 torch.compile 加速 (0=否, 1=是)")
    args = parser.parse_args()

    # ========== 1.初始化环境和随机种子 ===========
    # 初始化分布式训练模式, 返回本地 GPU 编号
    local_rank = init_distributed_mode()
    # 如果已初始化分布式环境, 更新设备为对应 GPU
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 设置随机种子, 确保实验可复现 (不同进程使用不同种子避免同步)
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2.配置目录、模型参数、检查 ckp ===========
    # 创建保存目录 (如果不存在)
    os.makedirs(args.save_dir, exist_ok=True)
    # 根据命令行参数创建 MiniMind 模型配置
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # 如果启用断点续训, 尝试加载已有检查点信息
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3.设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    # 根据参数选择精度类型 (bfloat16 或 float16)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # https://docs.pytorch.ac.cn/docs/stable/amp
    # 新版废弃: autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=dtype)
    
    # ========== 4.配置 wandb ==========
    wandb = None
    # 仅主进程启用 wandb, 避免日志重复
    if args.use_wandb and is_main_process():
        # 导入日志工具
        import swanlab as wandb
        # 从检查点获取 wandb run ID (用于恢复实验)
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        # 设置恢复模式：如果有 ID 则恢复之前实验, 否则创建新实验
        resume = 'must' if wandb_id else None
        # 生成实验名称, 包含关键超参数
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        # 初始化 wandb 实验
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5.定义模型、数据、优化器 ==========
    # 初始化模型和分词器
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    # 如果启用 torch.compile, 对模型进行编译优化 (需要 PyTorch 2.0+ & Linux)
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    # 创建预训练数据集
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 如果使用分布式训练, 创建分布式采样器
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # 创建梯度缩放器 (用于混合精度训练)
    # https://docs.pytorch.ac.cn/docs/stable/amp.html#gradient-scaling
    # 新版废弃: scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    scaler = torch.amp.GradScaler("cuda", enabled=(args.dtype == 'float16'))
    # 创建 AdamW 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6.从 ckp 中恢复状态 ==========
    start_epoch, start_step = 0, 0
    # 如果有检查点数据, 恢复训练状态
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7.DDP 包装模型 ==========
    # 如果使用分布式训练, 使用包装模型为 DDP 模式
    if dist.is_initialized():
        # 忽略频率向量 (避免 DDP 重复广播)
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8.开始训练 ==========
    # 遍历所有 epoch
    for epoch in range(start_epoch, args.epochs):
        # 如果使用分布式采样器, 设置 epoch 以确保数据打乱
        train_sampler and train_sampler.set_epoch(epoch)
        # 设置随机种子并生成打乱的索引顺序
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # 计算需要跳过的 step 数 (断点续训时)
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # 创建批次采样器, 支持跳过指定数量的批次
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        # 创建数据加载器
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        # 如果有跳过, 跳过前 start_step 个 step
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前 {start_step} 个 step, 从 step {start_step + 1} 开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9.清理分布进程 ==========
    # 销毁分布式进程组, 释放资源
    if dist.is_initialized(): dist.destroy_process_group()