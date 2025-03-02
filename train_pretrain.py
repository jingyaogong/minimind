import io
import os
import sys
import platform
import argparse
import threading
import time
import math
import warnings
import pandas as pd
import torch
import torch.distributed as dist
from torch import optim, nn
from torch._C._profiler import ActiveProfilerType
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from mem_utils import log_memory_snapshot, ActivationMemoryTracker
from model.model import MiniMindLM
from model.LMConfig import LMConfig
from model.dataset import PretrainDataset

warnings.filterwarnings('ignore')


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_dynamic_lr(cur_step, warmup_steps, decay_steps, lr):
    min_lr = lr / 10
    if cur_step < warmup_steps:
        return lr * (cur_step / warmup_steps)
    if cur_step > decay_steps:
        return min_lr

    step_ratio = (cur_step - warmup_steps)/(decay_steps-warmup_steps)
    cos_scope = 0.5 * (1 + math.cos(math.pi * step_ratio))
    return min_lr + (lr - min_lr) * cos_scope


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb):
    # 初始化跟踪器
    # tracker = ActivationMemoryTracker()
    # tracker.track_activations(model)

    loss_fct = nn.CrossEntropyLoss(reduction='none')
    step_base = epoch * iter_per_epoch
    total_steps = iter_per_epoch * args.epochs
    warmup_steps = int(total_steps * 0.005)
    decay_steps = int(total_steps * 0.9)

    if epoch == 0:
        for param_group in optimizer.param_groups:  # 在此处设置学习率
            param_group['lr'] = 5e-6

    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

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
            current_step = step_base + step  # 计算当前总步数
            lr = get_dynamic_lr(current_step, warmup_steps, decay_steps, args.learning_rate)
            for param_group in optimizer.param_groups:  # 在此处设置学习率
                param_group['lr'] = lr

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} spend:{}min remain:{}min'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time // 60,
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "remain": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/pretrain_{lm_config.dim}{moe_path}.pth'

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            buffer = io.BytesIO()
            torch.save(state_dict, buffer)
            threading.Thread(target=save_checkpoint, args=(buffer, ckp)).start()
            model.train()
            # 内存调试
            # log_memory_snapshot(model, optimizer, step)
            # # 输出激活值显存统计
            # print("中间激活值显存占用：")
            # total = 0
            # for name, info in tracker.activation_info.items():
            #     print(f"Layer: {name} | {info}")
            #     total += info["size"]
            # print(f"中间激活值显存总占用: {total:.2f} MB")
            # print(torch.cuda.memory_summary())
            # return


def save_checkpoint(buffer: io.BytesIO, path):
    with open(path, 'wb') as f:
        f.write(buffer.getvalue())


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
    # 启用动态显存段扩展
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="out")
    # 若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--n_heads', default=8, type=int)
    parser.add_argument('--n_kv_heads', default=2, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_mla', default=False, action='store_true')
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--vocab', default='minimind', type=str)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_paths", type=str, nargs="+", default=["./dataset/pretrain_hq.jsonl"])
    parser.add_argument("--ckpt_path", type=str, default=None)
    args = parser.parse_args()


    args.log_interval = args.save_interval
    os.makedirs(args.save_dir, exist_ok=True)

    torch.manual_seed(1337)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    if device_type == "cpu":
        ctx = nullcontext()
    else:
        # 映射dtype字符串到torch类型
        amp_dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
        ctx = torch.cuda.amp.autocast(dtype=amp_dtype)

    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"

    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    lm_config = LMConfig(
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        max_seq_len=args.max_seq_len,
        use_moe=args.use_moe,
        use_mla=args.use_mla,
        torch_dtype=args.dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(f'./model/{args.vocab}_tokenizer')
    lm_config.vocab_size = tokenizer.vocab_size
    model = MiniMindLM(lm_config)
    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')

    if args.ckpt_path is not None:
        # 接着训练
        state_dict = torch.load(args.ckpt_path, map_location=args.device)
        model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=True)

    model.to(args.device)

    train_ds = PretrainDataset(args.data_paths, tokenizer, max_length=args.max_seq_len)

    def collate_fn(batch):
        # 对 batch 中的每个样本进行 padding，动态padding到每个batch的最大长度
        batch = pad_sequence(batch, batch_first=True, padding_value=tokenizer.pad_token_id)
        # 以下是padding到最大长度，方便调试显存占用情况
        # max_len = args.max_seq_len
        # processed = []
        # for input_ids in batch:
        #     # 截断超过 max_len 的部分
        #     if len(input_ids) > max_len:
        #         processed.append(input_ids[:max_len])
        #     else:
        #         # 不足则填充到 max_len
        #         padded = torch.full(
        #             (max_len,),
        #             tokenizer.pad_token_id,
        #             dtype=input_ids.dtype
        #         )
        #         padded[:len(input_ids)] = input_ids
        #         processed.append(padded)
        # batch = torch.stack(processed)  # 直接堆叠已等长的样本
        # 直接通过切片生成输入和目标序列
        bx = batch[:, :-1]  # 输入序列（去掉最后一个token）
        by = batch[:, 1:]   # 目标序列（去掉第一个token）
        # 生成attention mask（去掉第一个token的位置）
        bl = (batch != tokenizer.pad_token_id)[:, 1:]
        return bx, by, bl

    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=True,
        num_workers=args.num_workers,
        sampler=train_sampler,
        collate_fn=collate_fn,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)
