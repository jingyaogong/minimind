# 导入必要的库
import os
import sys

__package__ = "trainer"  # 设置包名
# 将项目根目录添加到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import time
import math
import warnings

import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import SFTDataset

warnings.filterwarnings('ignore')


# 日志打印函数
# 在分布式训练时只在主进程(rank=0)上打印日志
def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


# 余弦学习率调度器
# 在训练过程中逐渐降低学习率，最终降到初始值的1/10
def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


# 知识蒸馏损失函数
# 通过KL散度计算学生模型和教师模型输出分布之间的差异
def distillation_loss_fn(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    # 计算教师模型的概率分布（停止梯度传播）
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, hidden_size=-1).detach()

    student_log_probs = F.log_softmax(student_logits / temperature, hidden_size=-1)

    kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction=reduction
    )
    return (temperature ** 2) * kl


# 训练一个epoch
# alpha: 用于平衡蒸馏损失和交叉熵损失的权重
# temperature: 软目标的温度系数，用于控制知识蒸馏中概率分布的平滑程度
def train_epoch(epoch, wandb, alpha=0.0, temperature=1.0):
    start_time = time.time()

    # 确保教师模型处于评估模式且参数冻结
    if teacher_model is not None:
        teacher_model.eval()
        teacher_model.requires_grad_(False)

    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        lr = get_lr(epoch * iter_per_epoch + step,
                    args.epochs * iter_per_epoch,
                    args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 前向传播（学生模型）
        with ctx:
            res = model(X)
            student_logits = res.logits

        # 教师模型前向传播（只在eval & no_grad）
        if teacher_model is not None:
            with torch.no_grad():
                teacher_logits = teacher_model(X).logits
                vocab_size_student = student_logits.size(-1)  # N
                teacher_logits = teacher_logits[..., :vocab_size_student]

        # ========== 计算损失 ==========
        # 1) Ground-Truth CE Loss（可选）
        loss_mask_flat = loss_mask.view(-1)
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            Y.view(-1),
            ignore_index=0,
            reduction='none'
        )
        ce_loss = torch.sum(ce_loss * loss_mask_flat) / loss_mask_flat.sum()
        if lm_config_student.use_moe:
            ce_loss += res.aux_loss

        # 2) Distillation Loss（可选）
        if teacher_model is not None:
            # 只在有效token位置做蒸馏
            distill_loss = distillation_loss_fn(
                student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],
                teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask_flat == 1],
                temperature=temperature
            )
        else:
            distill_loss = torch.tensor(0.0, device=args.device)

        # 3) 总损失 = alpha * CE + (1-alpha) * Distill
        loss = (alpha * ce_loss + (1 - alpha) * distill_loss) / args.accumulation_steps

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
                'Epoch:[{}/{}]({}/{}) loss:{:.4f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch,
                    args.epochs - 1,
                    step,
                    iter_per_epoch,
                    loss.item(),
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60
                )
            )

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({
                    "loss": loss.item(),
                    "ce_loss": ce_loss.item(),
                    "distill_loss": distill_loss.item() if teacher_model is not None else 0.0,
                    "lr": optimizer.param_groups[-1]['lr'],
                    "last-time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60
                })

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config_student.use_moe else ''
            ckp = f'{args.save_dir}/full_dist_{lm_config_student.hidden_size}{moe_path}.pth'
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
            torch.save(state_dict, ckp)
            model.train()


# 初始化学生模型
# 学生模型通常比教师模型更小，但需要保持相同的任务能力
def init_student_model(lm_config):
    # 加载预训练的分词器
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    # 初始化学生模型
    model = MiniMindForCausalLM(lm_config)
    # 根据是否使用MoE设置权重路径
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/full_sft_{lm_config.hidden_size}{moe_path}.pth'
    # 加载预训练权重
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    # 打印模型可训练参数总量
    Logger(f'学生模型(LLM)总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    # 将模型移动到指定设备
    model = model.to(args.device)

    return model, tokenizer


# 初始化教师模型
# 教师模型通常是一个更大、性能更好的预训练模型
def init_teacher_model(lm_config):
    # 初始化教师模型
    model = MiniMindForCausalLM(lm_config)
    # 根据是否使用MoE设置权重路径
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/full_sft_{lm_config.hidden_size}{moe_path}.pth'
    # 加载预训练权重
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    # 打印模型可训练参数总量
    Logger(f'教师模型(LLM)总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    # 将模型移动到指定设备
    model = model.to(args.device)
    return model


def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")
    parser.add_argument("--out_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument("--data_path", type=str, default="../dataset/sft_xxx.jsonl")
    parser.add_argument("--wandb_api_key", type=str, default=None, help="WandB API Key，用于无交互环境自动登录")


    args = parser.parse_args()
    # 定义学生模型和教师模型
    lm_config_student = MiniMindConfig(hidden_size=512, num_hidden_layers=8)
    lm_config_teacher = MiniMindConfig(hidden_size=768, num_hidden_layers=16)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniMind-Dist-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

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
        # 优先使用命令行参数，其次环境变量
        api_key = args.wandb_api_key or os.environ.get("WANDB_API_KEY", None)
        if api_key is not None:
            # 自动登录，适用于无交互环境
            wandb.login(key=api_key)
        # 初始化 wandb 任务
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    # 初始化学生模型和教师模型
    model, tokenizer = init_student_model(lm_config_student)
    teacher_model = init_teacher_model(lm_config_teacher)

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
