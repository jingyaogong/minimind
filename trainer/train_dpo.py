import os
import sys

# 设置包名
__package__ = "trainer"
# 将项目根目录添加到系统路径, 确保能够导入同级目录的模块
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
from dataset.lm_dataset import DPODataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

# 忽略所有警告信息, 保持输出整洁
import warnings
warnings.filterwarnings('ignore')


def logits_to_log_probs(logits, labels):
    """
    将 logits 转换为对数概率

    对每个位置的 logits 计算 log_softmax, 然后根据 labels 提取对应的对数概率
    这是 DPO 训练中的基础操作, 用于计算策略模型和参考模型的 log probabilities

    Args:
    - logits:   模型输出的 logits, shape: (batch_size, seq_len, vocab_size)
    - labels:   目标 token 序列, shape:   (batch_size, seq_len)

    Returns:
    - log_probs_per_token: 每个 token 的对数概率, shape: (batch_size, seq_len)
    """
    # logits shape: (batch_size, seq_len, vocab_size)
    # labels shape: (batch_size, seq_len)
    # log_probs shape: (batch_size, seq_len)
    log_probs = F.log_softmax(logits, dim=2)
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    """
    计算 DPO (Direct Preference Optimization) 损失

    DPO 是一种直接根据人类偏好数据优化语言模型的方法, 不需要训练奖励模型
    通过比较策略模型和参考模型在 chosen/rejected 样本上的对数概率比率来计算损失

    Args:
    - ref_log_probs:        参考模型对数概率, shape: (batch_size, seq_len)
    - policy_log_probs:     策略模型对数概率, shape: (batch_size, seq_len)
    - mask:                 有效 token 掩码, shape: (batch_size, seq_len)
    - beta:                 DPO 温度参数, 控制模型偏离参考模型的程度

    Returns:
    - loss:                 平均 DPO 损失值
    """
    # ref_log_probs 和 policy_log_probs 都是 shape: (batch_size, seq_len)
    # https://github.com/jingyaogong/minimind/issues/298
    # 计算每个序列的有效长度, 用于对 log probs 进行归一化
    seq_lengths = mask.sum(dim=1, keepdim=True).clamp_min(1e-8)  # 防止零长度 mask 导致除零 NaN
    ref_log_probs = (ref_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    policy_log_probs = (policy_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()

    # 将 chosen 和 rejected 数据分开, 批次的前半部分是 chosen, 后半部分是 rejected
    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]

    # 计算对数概率比率, 衡量策略模型相比参考模型的改进程度
    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()


def train_epoch(epoch, loader, iters, ref_model, lm_config, start_step=0, wandb=None, beta=0.1):
    """
    执行一个 epoch 的 DPO 训练

    Args:
    - epoch:            当前训练轮次
    - loader:           数据加载器
    - iters:            当前 epoch 的总迭代次数
    - ref_model:        参考模型 (冻结参数, 不更新)
    - lm_config:        模型配置对象
    - start_step:       从此步骤开始训练 (用于恢复训练)
    - wandb:            Weights & Biases 日志对象
    - beta:             DPO 温度参数
    """
    start_time = time.time()
    
    for step, batch in enumerate(loader, start=start_step + 1):
        # 从批次中提取 chosen 和 rejected 数据, 并移动到目标设备
        x_chosen = batch['x_chosen'].to(args.device)
        x_rejected = batch['x_rejected'].to(args.device)
        y_chosen = batch['y_chosen'].to(args.device)
        y_rejected = batch['y_rejected'].to(args.device)
        mask_chosen = batch['mask_chosen'].to(args.device)
        mask_rejected = batch['mask_rejected'].to(args.device)
        
        # 将 chosen 和 rejected 数据拼接在一起, 形成完整的批次
        # 批次的前半部分是 chosen (用户偏好的回复), 后半部分是 rejected (用户不喜欢的回复)
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        # 根据当前步骤动态计算学习率 (余弦退火策略)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            # 参考模型前向传播, 不计算梯度 (frozen)
            with torch.no_grad():
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits
            ref_log_probs = logits_to_log_probs(ref_logits, y)
            
            # 策略模型前向传播, 计算 logits 和 log probabilities
            outputs = model(x)
            logits = outputs.logits
            policy_log_probs = logits_to_log_probs(logits, y)
            
            # 计算 DPO 损失和辅助损失 (MoE 负载均衡损失)
            dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
            loss = dpo_loss_val + outputs.aux_loss
            loss = loss / args.accumulation_steps

        # 使用梯度缩放器进行反向传播, 支持混合精度训练
        scaler.scale(loss).backward()

        # 当累积达到指定步数时, 执行参数更新
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # 按指定间隔打印日志
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_dpo_loss = dpo_loss_val.item()
            current_aux_loss = outputs.aux_loss.item()
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, dpo_loss: {current_dpo_loss:.4f}, aux_loss: {current_aux_loss:.4f}, learning_rate: {current_lr:.8f}, epoch_time: {eta_min:.3f}min')
            
            if wandb: wandb.log({"loss": current_loss, "dpo_loss": current_dpo_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # 按指定间隔保存模型检查点, 仅主进程执行
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # 解包 DDP 包装的模型以获取原始模型
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        # 删除中间变量以释放内存
        del x_chosen, x_rejected, y_chosen, y_rejected, mask_chosen, mask_rejected, x, y, mask
        del ref_outputs, ref_logits, ref_log_probs, outputs, logits, policy_log_probs, loss


if __name__ == "__main__":
    # 创建参数解析器, 定义 DPO 训练的所有配置参数
    parser = argparse.ArgumentParser(description="MiniMind DPO (Direct Preference Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='dpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=4e-8, help="初始学习率 (建议<=5e-8避免遗忘)")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="训练的最大截断长度 (中文1token≈1.5~1.7字符)")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构 (0=否, 1=是)")
    parser.add_argument("--data_path", type=str, default="../dataset/dpo.jsonl", help="DPO训练数据路径")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训 (0=否, 1=是)")
    parser.add_argument('--beta', default=0.1, type=float, help="DPO中的beta参数")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-DPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速 (0=否, 1=是)")
    args = parser.parse_args()

    # ========== 1.初始化环境和随机种子 ==========
    # 初始化分布式训练模式, 返回本地 rank (非 DDP 模式下返回 0)
    local_rank = init_distributed_mode()
    # 如果处于分布式训练环境, 根据 local_rank 设置当前设备
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 设置随机种子, 分布式训练时根据 rank 偏移种子以确保不同进程使用不同的数据顺序
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2.配置目录、模型参数、检查检查点 ==========
    # 创建保存目录 (如果不存在)
    os.makedirs(args.save_dir, exist_ok=True)
    # 创建 MiniMind 模型配置
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # 如果启用续训模式, 尝试从检查点恢复训练状态
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3.设置混合精度训练 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    # 根据配置选择数据类型 (bfloat16 或 float16)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # https://docs.pytorch.ac.cn/docs/stable/amp
    # 新版废弃: autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=dtype)
    
    # ========== 4.配置 wandb 日志 ==========
    wandb = None
    # 仅在主进程中初始化 wandb, 避免重复记录
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        # 从检查点恢复 wandb 运行 ID (如果存在)
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        # 构建运行名称, 包含关键训练参数
        wandb_run_name = f"MiniMind-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5.初始化模型和参考模型 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    # 如果启用 torch.compile, 使用 PyTorch 2.0 的编译加速功能
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    Logger(f'策略模型总参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
    # 初始化参考模型 (ref_model 冻结, 不参与训练, 仅用于计算参考 logits)
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model.eval()
    ref_model.requires_grad_(False)
    Logger(f'参考模型总参数量: {sum(p.numel() for p in ref_model.parameters()) / 1e6:.3f} M')
    
    # 创建 DPO 数据集和数据采样器
    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 分布式训练时使用 DistributedSampler 确保数据分配均匀
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # 创建梯度缩放器, 仅在 float16 模式下启用 (用于混合精度训练)
    # https://docs.pytorch.ac.cn/docs/stable/amp.html#gradient-scaling
    # 新版废弃: scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    scaler = torch.amp.GradScaler("cuda", enabled=(args.dtype == 'float16'))
    # 创建 AdamW 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6.从检查点恢复训练状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        # 从检查点加载模型权重、优化器状态、梯度缩放器状态
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        # 恢复训练进度
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7.使用 DDP 包装模型 (分布式训练) ==========
    if dist.is_initialized():
        # 忽略位置编码相关的参数 (不需要梯度同步)
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        # 使用 DistributedDataParallel 包装模型, 启用分布式训练
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8.开始训练循环 ==========
    for epoch in range(start_epoch, args.epochs):
        # 分布式训练时设置 epoch 以确保不同进程使用不同的数据顺序
        train_sampler and train_sampler.set_epoch(epoch)
        # 设置当前 epoch 的随机种子, 生成随机索引序列
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # 如果是恢复训练的第一个 epoch, 计算需要跳过的步数
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # 创建支持跳过功能的批次采样器
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        # 创建数据加载器
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        # 执行当前 epoch 的训练
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个 step, 从 step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, ref_model, lm_config, start_step, wandb, args.beta)
        else:
            train_epoch(epoch, loader, len(loader), ref_model, lm_config, 0, wandb, args.beta)
    
    # ========== 9.清理分布式进程组 ==========
    if dist.is_initialized(): dist.destroy_process_group()