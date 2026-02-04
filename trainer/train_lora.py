import os
import sys

# 设置包名
__package__ = "trainer"
# 将项目根目录添加到系统路径, 确保能够导入同级目录的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import SFTDataset
from model.model_lora import save_lora, apply_lora
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

# 忽略所有警告信息, 保持输出整洁
import warnings
warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, lora_params, start_step=0, wandb=None):
    """
    训练单个 epoch 的函数

    执行 LoRA 微调的一个完整训练周期, 包括前向传播、反向传播、梯度更新和日志记录

    Args:
    - epoch:        当前训练轮次
    - loader:       数据加载器, 提供训练数据批次
    - iters:        当前 epoch 的总迭代次数
    - lora_params:  LoRA 可训练参数列表, 用于梯度裁剪
    - start_step:   起始步数 (用于断点续训时跳过已训练部分)
    - wandb:        Weights & Biases 日志对象, 可选
    """
    # 记录训练开始时间, 用于计算 ETA (预计完成时间)
    start_time = time.time()
    # 遍历数据加载器中的每个批次, 从 start_step + 1 开始计数
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # 将输入数据移动到指定设备 (GPU 或 CPU)
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        # 使用余弦退火策略计算当前步的学习率
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        # 更新优化器中的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 使用自动混合精度上下文进行前向传播 (仅在 GPU 时启用)
        with autocast_ctx:
            # 模型前向传播, 计算损失 (主损失 + 辅助损失, 如 MoE 负载均衡损失)
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            # 梯度累积: 将损失除以累积步数, 模拟大批量训练效果
            loss = loss / args.accumulation_steps

        # 使用梯度缩放器进行反向传播 (防止 FP16 梯度下溢)
        scaler.scale(loss).backward()

        # 每 accumulation_steps 步执行一次参数更新
        if (step + 1) % args.accumulation_steps == 0:
            # 反缩放梯度, 准备进行梯度裁剪
            scaler.unscale_(optimizer)
            # 对 LoRA 参数进行梯度裁剪, 防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
            # 执行优化器步骤, 更新参数
            scaler.step(optimizer)
            # 更新梯度缩放器的缩放因子
            scaler.update()
            # 清空梯度, 释放内存
            optimizer.zero_grad(set_to_none=True)

        # 每隔 log_interval 步或最后一步打印训练日志
        if step % args.log_interval == 0 or step == iters - 1:
            # 计算已花费时间
            spend_time = time.time() - start_time
            # 恢复原始损失值 (乘以累积步数)
            current_loss = loss.item() * args.accumulation_steps
            # 获取辅助损失值 (MoE 负载均衡损失)
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            # 计算主损失 (总损失减去辅助损失)
            current_logits_loss = current_loss - current_aux_loss
            # 获取当前学习率
            current_lr = optimizer.param_groups[-1]['lr']
            # 计算预计剩余时间 (ETA)
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            # 在主进程打印训练进度日志
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            # 如果使用 wandb, 记录训练指标
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # 每隔 save_interval 步或最后一步保存模型检查点 (仅主进程执行)
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            # 切换到评估模式, 准备保存模型
            model.eval()
            # 构建 LoRA 权重保存路径
            lora_save_path = f'{args.save_dir}/{args.lora_name}_{lm_config.hidden_size}.pth'
            # LoRA只保存LoRA权重
            save_lora(model, lora_save_path)
            # 保存完整的训练检查点 (包含模型、优化器、学习率调度器状态)
            lm_checkpoint(lm_config, weight=args.lora_name, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            # 切换回训练模式
            model.train()

        # 显式删除张量, 释放 GPU 内存
        del input_ids, labels, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind LoRA Fine-tuning")
    parser.add_argument("--save_dir", type=str, default="../out/lora", help="模型保存目录")
    parser.add_argument("--lora_name", type=str, default="lora_identity", help="LoRA权重名称(如lora_identity/lora_medical等)")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度 (中文 1 token ≈ 1.5~1.7 字符)")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构 (0=否, 1=是)")
    parser.add_argument("--data_path", type=str, default="../dataset/lora_identity.jsonl", help="LoRA训练数据路径")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练, 默认 full_sft")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训 (0=否, 1=是)")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-LoRA", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速 (0=否, 1=是)")
    args = parser.parse_args()

    # ========== 1.初始化环境和随机种子 ==========
    # 初始化分布式训练模式, 返回本地 GPU 编号
    local_rank = init_distributed_mode()
    # 如果启用了分布式训练, 将设备设置为当前进程对应的 GPU
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 设置随机种子, 不同进程使用不同的种子以避免数据重复 (基础种子 42 + 进程 rank)
    
    # ========== 2.配置目录、模型参数、检查ckp ==========
    # 确保模型保存目录存在
    os.makedirs(args.save_dir, exist_ok=True)
    # 创建 MiniMind 模型配置对象, 设置隐藏层维度、层数和是否使用 MoE
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # 如果启用了断点续训, 尝试从检查点加载训练状态
    ckp_data = lm_checkpoint(lm_config, weight=args.lora_name, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3.设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    # 根据参数选择数据类型: bfloat16 或 float16
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # https://docs.pytorch.ac.cn/docs/stable/amp
    # 新版废弃: autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=dtype)
    
    # ========== 4.配置 wandb ==========
    # 初始化 wandb 对象为 None, 如果不启用则保持为 None
    wandb = None
    # 仅在主进程且启用 wandb 时初始化
    if args.use_wandb and is_main_process():
        # 导入 swanlab 作为 wandb 的替代 (可能是国内用户适配)
        import swanlab as wandb
        # 从检查点恢复 wandb 运行 ID (如果存在)
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        # 如果存在 wandb_id, 则必须恢复原有运行; 否则创建新运行
        resume = 'must' if wandb_id else None
        # 构建 wandb 运行名称, 包含关键训练参数
        wandb_run_name = f"MiniMind-LoRA-{args.lora_name}-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        # 初始化 wandb 运行
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5.定义模型、应用LoRA、冻结非LoRA参数 ==========
    # 初始化 MiniMind 模型和分词器, 从指定权重加载预训练参数
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    # 如果启用 torch.compile, 编译模型以加速训练 (PyTorch 2.0+ 特性)
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    # 应用 LoRA (Low-Rank Adaptation) 到模型, 添加可训练的低秩矩阵
    apply_lora(model)
    
    # 统计模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    # 统计 LoRA 相关参数量 (参数名中包含 'lora' 的参数)
    lora_params_count = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)
    # 打印模型参数量统计信息
    Logger(f"LLM 总参数量: {total_params / 1e6:.3f} M")
    Logger(f"LoRA 参数量: {lora_params_count / 1e6:.3f} M")
    Logger(f"LoRA 参数占比: {lora_params_count / total_params * 100:.2f}%")
    
    # 冻结非 LoRA 参数, 收集 LoRA 参数
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name:
            # 启用 LoRA 参数的梯度计算
            param.requires_grad = True
            # 将 LoRA 参数添加到可训练参数列表
            lora_params.append(param)
        else:
            # 冻结非 LoRA 参数 (原始模型参数不更新)
            param.requires_grad = False
    
    # ========== 6.定义数据和优化器 ==========
    # 创建指令微调数据集 (SFTDataset), 加载 JSONL 格式的训练数据
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 如果启用分布式训练, 使用 DistributedSampler 确保数据不重复; 否则为 None
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # 创建梯度缩放器, 仅在 float16 模式下启用 (用于混合精度训练, 防止梯度下溢)
    # https://docs.pytorch.ac.cn/docs/stable/amp.html#gradient-scaling
    # 新版废弃: scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    scaler = torch.amp.GradScaler("cuda", enabled=(args.dtype == 'float16'))
    # 创建 AdamW 优化器, 仅优化 LoRA 参数
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
    
    # ========== 7.从检查点恢复状态 ==========
    # 初始化起始 epoch 和 step 为 0 (从头训练)
    start_epoch, start_step = 0, 0
    # 如果存在检查点数据, 恢复训练状态
    if ckp_data:
        # 加载模型权重 (strict=False 允许部分加载, 适应 LoRA 场景)
        model.load_state_dict(ckp_data['model'], strict=False)
        # 恢复优化器状态 (包括动量等)
        optimizer.load_state_dict(ckp_data['optimizer'])
        # 恢复梯度缩放器状态
        scaler.load_state_dict(ckp_data['scaler'])
        # 恢复训练轮次
        start_epoch = ckp_data['epoch']
        # 恢复训练步数 (默认为 0)
        start_step = ckp_data.get('step', 0)
    
    # ========== 8.DDP包装模型 ==========
    # 如果启用分布式训练, 使用 DDP 包装模型
    if dist.is_initialized():
        # 设置 DDP 忽略同步的参数 (位置编码相关的缓存, 不需要梯度同步)
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        # 使用 DDP 包装模型, 指定当前进程使用的 GPU 设备
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 9.开始训练 ==========
    # 遍历从 start_epoch 到总 epoch 数的每个轮次
    for epoch in range(start_epoch, args.epochs):
        # 如果存在分布式采样器, 设置当前 epoch (确保数据打乱顺序在不同 epoch 不同)
        train_sampler and train_sampler.set_epoch(epoch)
        # 设置当前 epoch 的随机种子, 并生成随机打乱的索引列表
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # 如果是恢复训练的第一个 epoch 且需要跳过部分 step, 设置 skip; 否则为 0
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # 创建 SkipBatchSampler, 支持跳过指定数量的批次 (断点续训用)
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        # 创建 DataLoader, 使用自定义的 batch_sampler
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        # 如果需要跳过部分 step, 打印提示信息并调用 train_epoch
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, lora_params, start_step, wandb)
        else:
            # 正常训练, 从 step 0 开始
            train_epoch(epoch, loader, len(loader), lora_params, 0, wandb)
    
    # ========== 10.清理分布式进程 ==========
    # 如果启用了分布式训练, 销毁进程组, 释放资源
    if dist.is_initialized(): dist.destroy_process_group()