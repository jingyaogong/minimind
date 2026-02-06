import os
import sys

# 设置包名
__package__ = "trainer"
# 将项目根目录添加到系统路径，确保能够导入同级目录的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

# 忽略所有警告信息, 避免训练过程中输出过多警告
import warnings
warnings.filterwarnings('ignore')


def distillation_loss(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    """
    计算知识蒸馏的 KL 散度损失

    使用 Softmax 和 KL Divergence 衡量学生模型和教师模型预测分布的差异
    温度参数用于控制分布的平滑程度, 温度越高分布越平滑

    Args:
    - student_logits:    学生模型的原始输出 logits, shape 为 [batch, seq_len, vocab_size]
    - teacher_logits:    教师模型的原始输出 logits, shape 为 [batch, seq_len, vocab_size]
    - temperature:       蒸馏温度系数, 用于平滑 softmax 分布, 默认为 1.0
    - reduction:         损失聚合方式, 默认为 'batchmean' (按批次平均)

    Returns:
    - float:             计算得到的蒸馏损失值, 已乘以温度平方
    """
    # 计算教师模型在温度下的概率分布, 并 detach 避免梯度传播到教师模型
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()

    # 计算学生模型在温度下的对数概率分布
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # 计算 KL 散度损失
    kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction=reduction
    )
    # 乘以温度平方进行缩放, 保持梯度大小与标准交叉熵一致
    return (temperature ** 2) * kl


def train_epoch(epoch, loader, iters, teacher_model, lm_config_student, start_step=0, wandb=None, alpha=0.0, temperature=1.0):
    """
    训练一个 epoch

    执行知识蒸馏训练的前向传播、损失计算和反向传播更新
    包含标准交叉熵损失和蒸馏损失两种监督信号

    Args:
    - epoch:             当前训练轮次 (从 0 开始计数)
    - loader:            数据加载器, 提供批次数据
    - iters:             当前 epoch 的总迭代次数
    - teacher_model:     教师模型, 用于生成软标签进行蒸馏
    - lm_config_student: 学生模型配置, 用于判断是否为 MoE 架构
    - start_step:        起始步数, 用于从检查点恢复训练
    - wandb:             Weights & Biases 日志对象, 用于记录训练指标
    - alpha:             损失权重, 平衡 CE 损失和蒸馏损失, 总损失 = alpha*CE + (1-alpha)*KL
    - temperature:       蒸馏温度, 用于控制分布平滑程度
    """
    start_time = time.time()
    
    # 设置教师模型为评估模式, 禁用 dropout 等随机操作
    if teacher_model is not None:
        teacher_model.eval()
        # 冻结教师模型参数, 不计算梯度
        teacher_model.requires_grad_(False)

    # 遍历数据加载器中的每个批次
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # 将数据移动到指定设备
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        # 创建损失掩码, 标记哪些位置需要计算损失 (ignore_index=-100 的位置为 0)
        loss_mask = (labels[..., 1:] != -100).float()
        # 使用余弦退火策略计算当前学习率
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        # 更新优化器中所有参数组的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 前向传播 (学生模型)
        # - 在 __name__ == "__main__" 中定义
        # - autocast_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(dtype=dtype)
        with autocast_ctx:
            res = model(input_ids)
            # 提取学生模型的 logits, 去掉最后一个位置的预测 (因为标签向右偏移了一位)
            student_logits = res.logits[..., :-1, :].contiguous()

        # 教师模型前向传播 (只在 eval & no_grad 模式下)
        if teacher_model is not None:
            with torch.no_grad():
                teacher_logits = teacher_model(input_ids).logits[..., :-1, :].contiguous()
                # 如果学生模型和教师模型的词表大小不同, 只取教师模型前 vocab_size_student 个
                vocab_size_student = student_logits.size(-1)
                teacher_logits = teacher_logits[..., :vocab_size_student]

        # 计算损失
        # 1) Ground-Truth CE Loss (标准交叉熵损失)
        shift_labels = labels[..., 1:].contiguous()
        loss_mask_flat = loss_mask.view(-1)
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction='none'
        )
        # 计算加权平均损失, 忽略被 mask 的位置
        ce_loss_raw = torch.sum(ce_loss * loss_mask_flat) / (loss_mask_flat.sum() + 1e-8)

        # 如果是 MoE 模型, 加上辅助损失
        if lm_config_student.use_moe: 
            ce_loss = ce_loss_raw + res.aux_loss
        else: 
            ce_loss = ce_loss_raw

        # 2) Distillation Loss (蒸馏损失)
        if teacher_model is not None:
            # 只在被 mask 的位置上计算蒸馏损失
            distill_loss = distillation_loss(
                student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],
                teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask_flat == 1],
                temperature=temperature
            )
        else:
            # 如果没有教师模型, 蒸馏损失为 0
            distill_loss = torch.tensor(0.0, device=args.device)

        # 3) 总损失 = alpha * CE + (1-alpha) * Distill，再除以梯度累积步数
        loss = (alpha * ce_loss + (1 - alpha) * distill_loss) / args.accumulation_steps

        # 反向传播
        scaler.scale(loss).backward()

        # 梯度累积步数达到后, 执行优化器更新
        if (step + 1) % args.accumulation_steps == 0:
            # 取消梯度缩放, 准备更新
            scaler.unscale_(optimizer)
            # 梯度裁剪, 防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # 执行优化器更新
            scaler.step(optimizer)
            # 更新缩放器
            scaler.update()
            # 清空梯度, 将 None 赋值给梯度以节省内存
            optimizer.zero_grad(set_to_none=True)

        # 记录日志
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            # 恢复实际的损失值 (乘以累积步数)
            current_loss = loss.item() * args.accumulation_steps
            current_ce_loss = ce_loss_raw.item()
            current_aux_loss = res.aux_loss.item() if lm_config_student.use_moe else 0.0
            current_lr = optimizer.param_groups[-1]['lr']
            # 估算剩余时间
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            
            # 打印训练日志
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, ce: {current_ce_loss:.4f}, aux_loss: {current_aux_loss:.4f}, distill: {distill_loss.item():.4f}, learning_rate: {current_lr:.8f}, epoch_time: {eta_min:.3f}min')
            
            # 记录到 wandb
            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "ce_loss": current_ce_loss,
                    "aux_loss": current_aux_loss,
                    "distill_loss": distill_loss.item() if teacher_model is not None else 0.0,
                    "learning_rate": current_lr,
                    "epoch_time": eta_min
                })

        # 保存检查点
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            # 根据是否为 MoE 模型添加后缀
            moe_suffix = '_moe' if lm_config_student.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config_student.hidden_size}{moe_suffix}.pth'
            # 获取原始模型 (解包 DDP)
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            # 保存模型权重为半精度格式
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # 保存完整检查点 (包含优化器状态等)
            lm_checkpoint(lm_config_student, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            # 恢复训练模式
            model.train()
            # 清理内存
            del state_dict

        # 清理本批次的数据, 释放显存
        del input_ids, labels, loss_mask, res, student_logits, ce_loss, distill_loss, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Knowledge Distillation")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='full_dist', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=6, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument("--max_seq_len", type=int, default=340, help="训练的最大截断长度 (中文1token≈1.5~1.7字符)")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_mini_512.jsonl", help="训练数据路径")
    parser.add_argument('--student_hidden_size', default=512, type=int, help="学生模型隐藏层维度")
    parser.add_argument('--student_num_layers', default=8, type=int, help="学生模型隐藏层数量")
    parser.add_argument('--teacher_hidden_size', default=768, type=int, help="教师模型隐藏层维度")
    parser.add_argument('--teacher_num_layers', default=16, type=int, help="教师模型隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构 (0=否, 1=是)")
    parser.add_argument('--from_student_weight', default='full_sft', type=str, help="学生模型基于哪个权重")
    parser.add_argument('--from_teacher_weight', default='full_sft', type=str, help="教师模型基于哪个权重")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训 (0=否, 1=是)")
    parser.add_argument('--alpha', default=0.5, type=float, help="CE损失权重, 总损失=alpha*CE+(1-alpha)*KL")
    parser.add_argument('--temperature', default=1.5, type=float, help="蒸馏温度 (推荐范围1.0-2.0)")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Distillation", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速 (0=否, 1=是)")
    args = parser.parse_args()

    # ========== 1.初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2.配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config_student = MiniMindConfig(hidden_size=args.student_hidden_size, num_hidden_layers=args.student_num_layers, use_moe=bool(args.use_moe))
    lm_config_teacher = MiniMindConfig(hidden_size=args.teacher_hidden_size, num_hidden_layers=args.teacher_num_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config_student, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3.配置混合精度训练 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # https://docs.pytorch.ac.cn/docs/stable/amp
    # 新版废弃: autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=dtype)
    
    # ========== 4.配置 wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Distill-S{args.student_hidden_size}T{args.teacher_hidden_size}-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5.定义学生和教师模型 ==========
    # 学生模型和分词器
    model, tokenizer = init_model(lm_config_student, args.from_student_weight, device=args.device)
    # 使用 torch.compile 加速模型 (如果支持)
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    Logger(f'学生模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
    # 教师模型
    teacher_model, _ = init_model(lm_config_teacher, args.from_teacher_weight, device=args.device)
    teacher_model.eval()
    teacher_model.requires_grad_(False)
    Logger(f'教师模型总参数量：{sum(p.numel() for p in teacher_model.parameters()) / 1e6:.3f} M')

    # 创建训练数据集
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 创建分布式采样器 (如果在分布式环境下)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # 创建梯度缩放器 (混合精度训练需要)
    # https://docs.pytorch.ac.cn/docs/stable/amp.html#gradient-scaling
    # 新版废弃: scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    scaler = torch.amp.GradScaler("cuda", enabled=(args.dtype == 'float16'))
    # 创建优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6.从检查点恢复训练状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7.使用 DDP 包装模型 (如果在分布式环境下) ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8.开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # 设置分布式采样器的 epoch (打乱数据)
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        # 随机打乱数据集索引
        indices = torch.randperm(len(train_ds)).tolist()
        # 计算需要跳过的批次数量
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # 创建批次采样器
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        # 如果有跳过的批次, 打印提示信息并恢复训练
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, teacher_model, lm_config_student, start_step, wandb, args.alpha, args.temperature)
        else:
            train_epoch(epoch, loader, len(loader), teacher_model, lm_config_student, 0, wandb, args.alpha, args.temperature)
    
    # ========== 9.清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()