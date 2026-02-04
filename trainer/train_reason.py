# 导入标准库
import os
import sys

# 设置包名, 确保能够正确导入同级目录的模块
__package__ = "trainer"
# 将项目根目录添加到系统路径
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
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

# 忽略警告信息, 减少控制台输出干扰
import warnings
warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, tokenizer, lm_config, start_step=0, wandb=None):
    """
    单个 epoch 的训练函数

    执行模型的推理蒸馏训练, 包括前向传播、损失计算、梯度累积和反向传播
    使用特殊 token (<think>, </think>, <answer>, </answer>) 对思考过程和答案部分进行加权

    Args:
    - epoch:        当前训练轮次
    - loader:       数据加载器, 提供 (input_ids, labels) 批次
    - iters:        当前 epoch 的总迭代步数
    - tokenizer:    分词器对象, 用于获取特殊 token 的 ID
    - lm_config:    模型配置对象, 包含 use_moe 等属性
    - start_step:   起始步数, 用于断点续训时跳过已训练的步数
    - wandb:        Weights & Biases 日志对象, 可选
    """
    # 获取特殊 token 的 ID, 用于在损失计算中对思考过程加权
    # <think> 和 </think> 标记思考过程的开始和结束
    start_of_think_ids = tokenizer(' <think>').input_ids
    end_of_think_ids = tokenizer('</think>').input_ids
    # <answer> 和 </answer> 标记答案的开始和结束
    start_of_answer_ids = tokenizer('<answer>').input_ids
    end_of_answer_ids = tokenizer('</answer>').input_ids
    # 使用不 reduction 的交叉熵损失, 以便后续应用自定义 loss mask
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    # 记录 epoch 开始时间, 用于计算训练速度和 ETA
    start_time = time.time()

    # 遍历数据加载器, 从 start_step + 1 开始计数
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # 将输入数据移动到训练设备
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        # 计算当前步的学习率, 使用余弦退火策略
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        # 更新优化器的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 使用自动混合精度上下文进行前向传播
        with autocast_ctx:
            # 模型前向传播, 获取输出结果
            res = model(input_ids)
            # 对 logits 和 labels 进行移位, 以进行因果语言建模
            # shift_logits: 预测下一个 token 的 logits
            # shift_labels: 实际的下一个 token 的标签
            shift_logits = res.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # 计算每个位置的交叉熵损失
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(shift_labels.size())

            # 构建损失掩码, 标记有效标签位置 (非 -100 的 padding)
            loss_mask = (shift_labels != -100).float()
            # 找出特殊 token 的位置 (思考过程和答案的标记)
            sp_ids = torch.isin(
                shift_labels.view(-1),
                torch.tensor(start_of_think_ids + end_of_think_ids + start_of_answer_ids + end_of_answer_ids).to(args.device)
            )
            # 对 loss mask 进行加权: 特殊 token 位置权重设为 10, 增强对推理过程的学习
            loss_mask_flat = loss_mask.view(-1)
            loss_mask_sum = loss_mask_flat.sum()
            loss_mask_flat[sp_ids] = 10
            loss_mask = loss_mask_flat.view(shift_labels.size())
            # 计算加权后的 logits 损失
            logits_loss = (loss * loss_mask).sum() / loss_mask_sum
            # 总损失 = 加权 logits 损失 + MoE 辅助损失 (如使用 MoE)
            loss = logits_loss + res.aux_loss
            # 梯度累积: 将损失除以累积步数, 模拟大批量训练
            loss = loss / args.accumulation_steps

        # 使用梯度缩放器进行反向传播, 防止 FP16/BF16 梯度下溢
        scaler.scale(loss).backward()

        # 当达到梯度累积步数时, 执行优化器更新
        if (step + 1) % args.accumulation_steps == 0:
            # 取消梯度缩放, 以便进行梯度裁剪
            scaler.unscale_(optimizer)
            # 梯度裁剪, 防止梯度爆炸, 保持训练稳定
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # 执行优化器更新
            scaler.step(optimizer)
            # 更新梯度缩放器的缩放因子
            scaler.update()
            # 清空梯度, set_to_none=True 节省内存
            optimizer.zero_grad(set_to_none=True)

        # 按日志间隔打印训练状态, 或在最后一个 step 打印
        if step % args.log_interval == 0 or step == iters - 1:
            # 计算已花费的时间和各项损失值
            spend_time = time.time() - start_time
            # 恢复原始损失值 (之前除以了 accumulation_steps)
            current_loss = loss.item() * args.accumulation_steps
            # 获取 MoE 辅助损失 (如果存在)
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = logits_loss.item()
            current_lr = optimizer.param_groups[-1]['lr']
            # 计算预计剩余时间 (分钟)
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            # 打印训练日志
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            # 如果使用 wandb, 记录训练指标
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # 按保存间隔保存模型检查点, 仅在主进程执行
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            # 切换到评估模式, 准备保存模型
            model.eval()
            # 根据是否使用 MoE 确定文件名后缀
            moe_suffix = '_moe' if lm_config.use_moe else ''
            # 构建模型保存路径
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # 如果是 DDP 模型, 解包获取原始模型
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            # 处理 torch.compile 包装的情况
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            # 获取模型状态字典
            state_dict = raw_model.state_dict()
            # 将权重转换为半精度 (FP16) 并保存到 CPU, 减小文件大小
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # 保存完整检查点 (包含优化器状态等, 用于断点续训)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            # 切换回训练模式
            model.train()
            # 释放状态字典内存
            del state_dict

        # 删除批次数据, 释放 GPU 内存
        del input_ids, labels, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Reasoning Distillation")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='reason', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=720, type=int, help="训练的最大截断长度 (中文 1 token ≈ 1.5 ~ 1.7 字符)")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构 (0=否, 1=是)")
    parser.add_argument("--data_path", type=str, default="../dataset/r1_mix_1024.jsonl", help="推理蒸馏数据路径")
    parser.add_argument('--from_weight', default='dpo', type=str, help="基于哪个权重训练, 默认dpo")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训 (0=否, 1=是)")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Reasoning", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速 (0=否, 1=是)")
    args = parser.parse_args()

    # ========== 1.初始化分布式环境和随机种子 ==========
    # 初始化分布式训练模式, 获取本地 rank
    local_rank = init_distributed_mode()
    # 如果分布式训练已初始化, 根据 local_rank 设置设备
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 设置随机种子, 分布式环境下每个 rank 使用不同的种子
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2.配置目录、模型参数、检查检查点 ==========
    # 创建保存目录 (如果不存在)
    os.makedirs(args.save_dir, exist_ok=True)
    # 初始化模型配置
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # 如果启用断点续训, 尝试加载检查点
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None

    # ========== 3.设置混合精度训练 ==========
    # 确定设备类型
    device_type = "cuda" if "cuda" in args.device else "cpu"
    # 根据参数设置数据类型 (BF16 或 FP16)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # https://docs.pytorch.ac.cn/docs/stable/amp
    # 新版废弃: autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=dtype)

    # ========== 4.配置 wandb 日志 ==========
    wandb = None
    # 仅在主进程且启用 wandb 时初始化
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        # 断点续训时恢复 wandb 运行 ID
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        # 构建运行名称
        wandb_run_name = f"MiniMind-Reasoning-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5.定义模型、数据集、优化器 ==========
    # 初始化模型和分词器
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    # 如果使用 torch.compile 加速 (PyTorch 2.0+ 特性)
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    # 创建训练数据集
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 如果是分布式训练, 创建分布式采样器
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # 创建梯度缩放器, 仅在 float16 模式下启用 (用于混合精度训练, 防止梯度下溢)
    # https://docs.pytorch.ac.cn/docs/stable/amp.html#gradient-scaling
    # 新版废弃: scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    scaler = torch.amp.GradScaler("cuda", enabled=(args.dtype == 'float16'))
    # 创建 AdamW 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6.从检查点恢复训练状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        # 加载模型权重
        model.load_state_dict(ckp_data['model'])
        # 加载优化器状态
        optimizer.load_state_dict(ckp_data['optimizer'])
        # 加载梯度缩放器状态
        scaler.load_state_dict(ckp_data['scaler'])
        # 恢复训练轮次和步数
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7.DDP 包装模型 ==========
    # 如果使用分布式训练, 用 DistributedDataParallel 包装模型
    if dist.is_initialized():
        # 设置不需要同步的参数 (RoPE 位置编码的频率参数)
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8.开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # 分布式采样器设置 epoch, 确保每个 epoch 的数据打乱顺序不同
        train_sampler and train_sampler.set_epoch(epoch)
        # 设置当前 epoch 的随机种子, 并打乱数据顺序
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # 如果是恢复训练的第一个 epoch, 计算需要跳过的步数
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # 创建支持跳过批次的采样器
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        # 创建数据加载器
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        # 执行训练 epoch
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step, 从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, tokenizer, lm_config, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), tokenizer, lm_config, 0, wandb)

    # ========== 9.清理分布式进程 ==========
    # 训练完成后销毁进程组, 释放资源
    if dist.is_initialized(): dist.destroy_process_group()