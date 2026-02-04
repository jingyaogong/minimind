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
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

# 忽略所有警告信息, 避免训练过程中输出无关的警告日志
import warnings
warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """
    单个 epoch 的训练函数

    执行一个完整的训练周期, 包括前向传播、反向传播、梯度更新、日志打印和模型保存

    Args:
    - epoch:        当前训练轮次, 从 0 开始计数
    - loader:       数据加载器, 提供训练数据批次
    - iters:        当前 epoch 的总迭代次数
    - start_step:   起始步数, 用于恢复训练时跳过已训练的批次, 默认为 0
    - wandb:        Weights & Biases 日志对象, 为 None 时不记录日志
    """
    # 记录 epoch 开始时间, 用于计算 ETA (预计剩余时间)
    start_time = time.time()

    # 遍历数据加载器中的所有批次
    # enumerate 从 start_step + 1 开始, 保证恢复训练时 step 编号连续
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # 将输入数据和标签移动到训练设备 (GPU 或 CPU)
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)

        # 根据余弦退火策略计算当前步的学习率
        # 公式参数: 当前全局步数 (epoch * iters + step), 总步数 (epochs * iters), 基础学习率
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        # 更新优化器中所有参数组的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 使用混合精度上下文进行前向传播
        # 在 CUDA 设备上使用 bfloat16/float16, CPU 设备上不使用混合精度
        with autocast_ctx:
            # 模型前向传播, 输入 input_ids 和 labels, 返回 loss 和 aux_loss
            res = model(input_ids, labels=labels)
            # 总损失 = 主损失 (交叉熵) + 辅助损失 (MoE 负载均衡损失)
            loss = res.loss + res.aux_loss
            # 梯度累积: 将损失除以累积步数, 模拟更大 batch size 的训练效果
            loss = loss / args.accumulation_steps

        # 使用 GradScaler 进行反向传播, 支持混合精度训练的梯度缩放
        scaler.scale(loss).backward()

        # 每 accumulation_steps 步执行一次梯度更新
        if (step + 1) % args.accumulation_steps == 0:
            # 反缩放梯度, 将缩放后的梯度还原为真实梯度值
            scaler.unscale_(optimizer)
            # 梯度裁剪, 防止梯度爆炸, 限制梯度的 L2 范数不超过 grad_clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 执行优化器更新步骤
            scaler.step(optimizer)
            # 更新 GradScaler 的缩放因子, 根据梯度是否溢出动态调整
            scaler.update()

            # 清空梯度, set_to_none=True 可以节省内存
            optimizer.zero_grad(set_to_none=True)

        # 每 log_interval 步打印一次日志, 或在该 epoch 最后一步打印
        if step % args.log_interval == 0 or step == iters - 1:
            # 计算已花费的时间
            spend_time = time.time() - start_time
            # 还原真实的损失值 (之前被除以 accumulation_steps)
            current_loss = loss.item() * args.accumulation_steps
            # 获取 MoE 辅助损失, 如果不存在则为 0
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            # 计算主损失 (总损失减去辅助损失)
            current_logits_loss = current_loss - current_aux_loss
            # 获取当前学习率
            current_lr = optimizer.param_groups[-1]['lr']
            # 计算 ETA (预计剩余时间), 单位为分钟
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            # 打印训练日志
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            # 如果启用了 wandb, 记录指标
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # 每 save_interval 步保存一次模型, 或在 epoch 最后一步保存, 仅在主进程中执行
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            # 切换到评估模式, 确保保存时模型处于稳定状态
            model.eval()
            # 根据是否使用 MoE 构建模型权重文件名后缀
            moe_suffix = '_moe' if lm_config.use_moe else ''
            # 构建权重保存路径
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # 如果模型被 DDP 包装, 获取原始模型
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            # 如果模型被 torch.compile 编译, 获取原始模型
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            # 获取模型状态字典
            state_dict = raw_model.state_dict()
            # 将权重转换为半精度 (FP16) 并保存到 CPU, 然后保存到磁盘
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # 保存完整的检查点 (包含优化器状态等), 用于恢复训练
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scaler=scaler)
            # 切换回训练模式
            model.train()
            # 删除状态字典以释放内存
            del state_dict

        # 删除张量以释放 GPU 内存
        del input_ids, labels, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='full_sft', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度 (中文 1 token ≈ 1.5~1.7 字符)")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构(0=否, 1=是)")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_mini_512.jsonl", help="训练数据路径")
    parser.add_argument('--from_weight', default='pretrain', type=str, help="基于哪个权重训练, 为 none 则不基于任何权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训(0=否, 1=是)")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速(0=否, 1=是)")
    args = parser.parse_args()

    # ========== 1.初始化环境和随机种子 ==========
    # 初始化分布式训练模式, 如果是分布式训练则返回 local_rank, 否则返回 0
    local_rank = init_distributed_mode()
    # 如果分布式环境已初始化, 根据 local_rank 设置当前进程使用的 GPU 设备
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 设置随机种子, 确保实验可复现, 不同 rank 使用不同种子避免数据同步问题
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2.配置目录、模型参数、检查ckp ==========
    # 创建模型保存目录, 如果目录已存在则不报错
    os.makedirs(args.save_dir, exist_ok=True)
    # 创建模型配置对象, 包含隐藏层维度、层数、是否使用 MoE 等参数
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # 如果启用自动恢复训练, 尝试从检查点加载数据, 否则返回 None
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None

    # ========== 3.设置混合精度 ==========
    # 根据设备类型确定是 cuda 还是 cpu
    device_type = "cuda" if "cuda" in args.device else "cpu"
    # 根据配置选择混合精度的数据类型: bfloat16 或 float16
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # https://docs.pytorch.ac.cn/docs/stable/amp
    # 新版废弃: autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=dtype)

    # ========== 4.配置 wandb ==========
    wandb = None
    # 仅在主进程中初始化 Weights & Biases 日志记录
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        # 如果有检查点数据, 获取之前的 wandb 运行 ID 用于恢复日志记录
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        # 如果有 wandb_id, 设置为必须恢复模式; 否则创建新运行
        resume = 'must' if wandb_id else None
        # 构建运行名称, 包含训练配置参数
        wandb_run_name = f"MiniMind-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        # 初始化 wandb 运行
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5.定义模型、数据、优化器 ==========
    # 初始化模型和分词器, 可以选择从预训练权重加载
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    # 如果启用 torch.compile, 编译模型以加速训练
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    # 创建 SFT 数据集, 用于监督微调
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 如果启用分布式训练, 创建 DistributedSampler; 否则使用 None (后续会用普通索引)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # 创建梯度缩放器, 仅在 float16 模式下启用 (用于混合精度训练)
    # https://docs.pytorch.ac.cn/docs/stable/amp.html#gradient-scaling
    # 新版废弃: scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    scaler = torch.amp.GradScaler("cuda", enabled=(args.dtype == 'float16'))
    # 创建 AdamW 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6.从 ckp 恢复状态 ==========
    # 初始化起始轮次和起始步数
    start_epoch, start_step = 0, 0
    # 如果有检查点数据, 恢复模型、优化器、scaler 的状态, 以及训练进度
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7.DDP 包装模型 ==========
    # 如果启用分布式训练, 使用 DistributedDataParallel 包装模型
    if dist.is_initialized():
        # 设置 DDP 忽略同步的参数: RoPE 的位置编码不需要梯度同步
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        # 使用 DDP 包装模型, 指定当前进程使用的 GPU 设备
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8.开始训练 ==========
    # 从 start_epoch 开始训练, 直到 args.epochs
    for epoch in range(start_epoch, args.epochs):
        # 设置 DistributedSampler 的 epoch, 确保每个 epoch 的数据打乱方式不同
        train_sampler and train_sampler.set_epoch(epoch)
        # 为每个 epoch 设置不同的随机种子, 确保数据打乱顺序不同
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # 如果是恢复训练的第一个 epoch, 需要跳过已经训练过的 step; 其他 epoch 从 0 开始
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # 创建 SkipBatchSampler, 用于跳过指定数量的批次
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        # 创建 DataLoader, 使用自定义的 batch_sampler
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        # 如果需要跳过批次, 打印提示信息并调用 train_epoch 时传入正确的总步数和起始步数
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step, 从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)

    # ========== 9. 清理分布进程 ==========
    # 训练结束后, 如果启用了分布式训练, 销毁进程组
    if dist.is_initialized(): dist.destroy_process_group()