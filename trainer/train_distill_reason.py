# 导入必要的系统库
import os
import sys

# 设置包名和添加项目根目录到系统路径
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import math
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import SFTDataset

warnings.filterwarnings('ignore')


def Logger(content):
    """日志打印函数，在分布式训练时只在主进程上打印
    Args:
        content: 需要打印的内容
    """
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    """使用余弦退火策略计算学习率
    Args:
        current_step: 当前训练步数
        total_steps: 总训练步数
        lr: 基础学习率
    Returns:
        当前步数对应的学习率
    """
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb):
    """训练一个epoch
    Args:
        epoch: 当前训练轮数
        wandb: wandb日志记录器
    """
    # 获取特殊标记的token ID
    # 用于标记模型思考和回答的开始结束位置
    start_of_think_ids = tokenizer('<think>').input_ids
    end_of_think_ids = tokenizer('</think>').input_ids
    start_of_answer_ids = tokenizer('<answer>').input_ids
    end_of_answer_ids = tokenizer('</answer>').input_ids
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            sp_ids = torch.isin(Y.view(-1),
                                torch.tensor(start_of_think_ids + end_of_think_ids
                                             + start_of_answer_ids + end_of_answer_ids
                                             ).to(args.device))
            # 在 sp_ids 对应的位置增加额外的惩罚
            loss_mask = loss_mask.view(-1)
            loss_mask_sum = loss_mask.sum()
            loss_mask[sp_ids] = 10
            loss_mask = loss_mask.view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask_sum
            loss += res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/reason_{lm_config.hidden_size}{moe_path}.pth'

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
            torch.save(state_dict, ckp)
            model.train()


def init_model(lm_config):
    """初始化模型和分词器
    Args:
        lm_config: 模型配置参数
    Returns:
        model: 初始化好的模型
        tokenizer: 分词器
    """
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained('../model')
    # 初始化模型
    model = MiniMindForCausalLM(lm_config)
    # 根据是否使用MoE设置检查点路径
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/rlhf_{lm_config.hidden_size}{moe_path}.pth'
    # 加载预训练权重
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    # 打印模型参数量
    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    # 将模型移动到指定设备
    model = model.to(args.device)
    return model, tokenizer


def init_distributed_mode():
    """初始化分布式训练环境
    设置分布式训练的各项参数，包括进程组、设备等
    """
    if not ddp: return
    global ddp_local_rank, DEVICE

    # 初始化分布式进程组，使用NCCL后端
    dist.init_process_group(backend="nccl")
    # 获取当前进程的全局排名
    ddp_rank = int(os.environ["RANK"])
    # 获取当前进程在本机的局部排名
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    # 获取总进程数
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    # 设置当前进程使用的GPU设备
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="MiniMind Distill Reasoning")
    # 输出目录，用于保存模型检查点和日志
    parser.add_argument("--out_dir", type=str, default="../out")
    # 训练轮数
    parser.add_argument("--epochs", type=int, default=1)
    # 训练批次大小
    parser.add_argument("--batch_size", type=int, default=8)
    # 基础学习率
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    # 训练设备，默认使用GPU
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    # 训练精度，支持float16和bfloat16
    parser.add_argument("--dtype", type=str, default="bfloat16")
    # 是否使用wandb记录训练日志
    parser.add_argument("--use_wandb", action="store_true")
    # wandb项目名称
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT")
    # 数据加载进程数
    parser.add_argument("--num_workers", type=int, default=1)
    # 是否使用分布式训练
    parser.add_argument("--ddp", action="store_true")
    # 梯度累积步数
    parser.add_argument("--accumulation_steps", type=int, default=1)
    # 梯度裁剪阈值
    parser.add_argument("--grad_clip", type=float, default=1.0)
    # 预热迭代次数
    parser.add_argument("--warmup_iters", type=int, default=0)
    # 日志打印间隔
    parser.add_argument("--log_interval", type=int, default=1)
    # 模型保存间隔
    parser.add_argument("--save_interval", type=int, default=50)
    # 分布式训练的局部进程号
    parser.add_argument('--local_rank', type=int, default=-1)
    # 模型隐藏层维度
    parser.add_argument('--hidden_size', default=512, type=int)
    # Transformer层数
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    # 最大序列长度
    parser.add_argument('--max_seq_len', default=1024, type=int)
    # 是否使用MoE(混合专家)架构
    parser.add_argument('--use_moe', default=False, type=bool)
    # 训练数据路径
    parser.add_argument("--data_path", type=str, default="../dataset/r1_mix_1024.jsonl")

    args = parser.parse_args()

    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                         use_moe=args.use_moe)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniMind-Distill-Reasoning-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"
    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        # 同时设置 CUDA 的随机种子
        torch.cuda.manual_seed(base_seed + rank)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    model, tokenizer = init_model(lm_config)

    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)
