import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import math
import warnings
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import PretrainDataset

warnings.filterwarnings('ignore')


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))

loss_fct = nn.CrossEntropyLoss(reduction='none')

def train_epoch(epoch, start_step, wandb):
    start_time = time.time()
    if epoch == start_epoch:
        train_loader_iter = iter(train_loader)
        for _ in range(start_step + 1):
            next(train_loader_iter)
        for step, (X, Y, loss_mask) in enumerate(train_loader_iter, start=start_step + 1):
            train_step(epoch, step, X, Y, loss_mask, wandb, start_time)
    else:
        for step, (X, Y, loss_mask) in enumerate(train_loader):
            train_step(epoch, step, X, Y, loss_mask, wandb, start_time)


def train_step(epoch, step, X, Y, loss_mask, wandb, start_time):
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
        loss = (loss * loss_mask).sum() / loss_mask.sum()
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
            wandb.log({"loss": loss.item() * args.accumulation_steps,
                       "lr": optimizer.param_groups[-1]['lr'],
                       "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

    if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
        model.eval()
        moe_path = '_moe' if lm_config.use_moe else ''
        ckp = f'{args.save_dir}/pretrain_{lm_config.hidden_size}{moe_path}.pth'

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
        torch.save(state_dict, ckp)

        # 保存训练状态
        training_state = {
            'epoch': epoch,
            'step': step,
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler is not None else None
        }
        training_state_path = f'{args.save_dir}/training_state.pth'
        torch.save(training_state, training_state_path)

        model.train()


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    model = MiniMindForCausalLM(lm_config).to(args.device)
    Logger(f'LLM可训练总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model, tokenizer


def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


# torchrun --nproc_per_node 2 1-pretrain.py
if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--out_dir", type=str, default="../out")  # 输出目录
    parser.add_argument("--epochs", type=int, default=8)          # 训练轮次
    parser.add_argument("--batch_size", type=int, default=8)     # 物理批次大小
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")  # 混合精度类型
    parser.add_argument("--use_wandb", action="store_true")       # 是否使用WandB
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")
    parser.add_argument("--num_workers", type=int, default=1)     # 数据加载线程数
    parser.add_argument("--ddp", action="store_true")             # 是否启用DDP
    parser.add_argument("--accumulation_steps", type=int, default=4)  # 梯度累积步数
    parser.add_argument("--grad_clip", type=float, default=1.0)   # 梯度裁剪阈值
    parser.add_argument("--log_interval", type=int, default=100)  # 日志间隔（步）
    parser.add_argument("--save_interval", type=int, default=100) # 保存间隔（步）
    parser.add_argument('--local_rank', type=int, default=-1)     # DDP自动传入参数
    # 模型架构参数
    parser.add_argument('--hidden_size', default=512, type=int)   # 隐藏层维度
    parser.add_argument('--num_hidden_layers', default=8, type=int) # 层数
    parser.add_argument('--max_seq_len', default=4096, type=int)   # 序列长度
    parser.add_argument('--use_moe', default=False, type=bool)    # 是否使用MoE
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl")
    args = parser.parse_args()

    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=args.use_moe)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

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
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
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

    # 检查是否存在保存的训练状态文件
    training_state_path = f'{args.save_dir}/training_state.pth'
    if os.path.exists(training_state_path):
        training_state = torch.load(training_state_path)
        start_epoch = training_state['epoch']
        start_step = training_state['step']
        optimizer.load_state_dict(training_state['optimizer_state_dict'])
        if scaler is not None and training_state['scaler_state_dict'] is not None:
            scaler.load_state_dict(training_state['scaler_state_dict'])
    else:
        start_epoch = 0
        start_step = 0

    for epoch in range(start_epoch, args.epochs):
        train_epoch(epoch, start_step if epoch == start_epoch else 0, wandb)