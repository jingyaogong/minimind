import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from model.quant import prepare_qat, DEFAULT_SKIP_PATTERNS
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import (
    get_lr, Logger, is_main_process, lm_checkpoint,
    init_distributed_mode, setup_seed, init_model, SkipBatchSampler,
    detect_gpu_peak_tflops_bf16, compute_model_flops_per_token,
)

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    last_log_time, last_log_step = start_time, start_step
    last_step = start_step
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)
        last_step = step
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            now = time.time()
            spend_time = now - start_time
            window_dt = max(now - last_log_time, 1e-6)
            window_steps = step - last_log_step
            window_its = window_steps / window_dt
            # MFU 报的是 dense 6P 等价吞吐；QAT 会因 fake-quant 逐 Linear 加 round/clamp/dequant
            # 有额外内存搬运，MFU 通常比同 seq 下的 SFT 低 5-15%——这本身就是我们要观测的量。
            tokens_window = window_steps * args.batch_size * args.max_seq_len * world_size
            mfu = (flops_per_token * tokens_window / window_dt) / (gpu_peak_tflops * 1e12) * 100
            last_log_time, last_log_step = now, step
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            # observer_warmup 期内 act_fq 只统计不量化，日志上会看到一个"200 步后 loss 抖一下"的转折点。
            qat_phase = 'observe' if step <= args.quant_observer_steps else 'fake-quant'
            Logger(
                f'[{time.strftime("%H:%M:%S")}] Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                f'it/s: {window_its:.2f}, MFU: {mfu:.1f}%, phase: {qat_phase}, '
                f'loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, '
                f'aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, '
                f'epoch_time: {eta_min:.1f}min'
            )
            if wandb:
                wandb.log({
                    "loss": current_loss, "logits_loss": current_logits_loss,
                    "aux_loss": current_aux_loss, "learning_rate": current_lr,
                    "epoch_time": eta_min, "it_per_s": window_its, "mfu_pct": mfu,
                })

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(
                lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scaler=scaler,
            )
            model.train()
            del state_dict

        del input_ids, labels, res, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind QAT (W8A8 fake-quant fine-tune)")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument('--save_weight', default='qat', type=str)
    parser.add_argument("--epochs", type=int, default=1, help="QAT 通常 1~2 epoch 即可")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-6,
                        help="QAT 用比 SFT 更小的 lr（默认 1/5），防止打乱已收敛的权重分布")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=768, type=int)
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1])
    parser.add_argument("--data_path", type=str, default="../dataset/sft_t2t.jsonl")
    parser.add_argument("--qat_subset_size", type=int, default=0,
                        help="QAT 数据子集大小（0=用全量）；64M dry run 建议 100000，1B 正式建议 200000")
    parser.add_argument("--gpu_peak_tflops", type=float, default=0,
                        help="GPU bf16 理论峰值 TFLOPS，用于 MFU 计算；0=按 GPU 名自动检测")
    parser.add_argument('--from_weight', default='full_sft', type=str,
                        help="QAT 起点权重前缀（部署走 full_sft；1.12 dry run 走 pretrain）")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-QAT")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1])

    # ---- QAT-specific ----
    parser.add_argument("--w_bits", type=int, default=8, help="weight 量化位宽")
    parser.add_argument("--a_bits", type=int, default=8, help="activation 量化位宽")
    parser.add_argument("--quant_observer_steps", type=int, default=200,
                        help="activation observer 预热步数（仅采集 EMA，不做 fake-quant）")
    parser.add_argument("--quant_skip", type=str, default=",".join(DEFAULT_SKIP_PATTERNS),
                        help="名称中包含这些子串的 Linear 跳过量化（逗号分隔）"
                             "；默认跳过 lm_head 和 MoE 路由 mlp.gate")
    args = parser.parse_args()

    # 1. dist + seed
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # 2. config + checkpoint probe
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    ckp_data = lm_checkpoint(
        lm_config, weight=args.save_weight, save_dir='../checkpoints',
    ) if args.from_resume == 1 else None

    # 3. autocast
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # 4. wandb
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = (
            f"MiniMind-QAT-W{args.w_bits}A{args.a_bits}-Epoch-{args.epochs}-"
            f"BS-{args.batch_size}-LR-{args.learning_rate}"
        )
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # 5. model: load fp weights -> wrap with QATLinear -> dataset/optimizer
    if args.from_weight == 'none':
        raise SystemExit("--from_weight=none is not supported for QAT; load a pretrained/SFT ckpt.")
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)

    skip_patterns = tuple(s.strip() for s in args.quant_skip.split(',') if s.strip())
    replaced = prepare_qat(
        model, w_bits=args.w_bits, a_bits=args.a_bits,
        skip_patterns=skip_patterns, a_observer_steps=args.quant_observer_steps,
    )
    Logger(f'QAT: wrapped {replaced} Linear modules '
           f'(W{args.w_bits}A{args.a_bits}, skip={skip_patterns}, '
           f'observer_warmup={args.quant_observer_steps} steps)')

    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len,
                          subset_size=args.qat_subset_size)
    Logger(f'QAT dataset: {len(train_ds)} samples '
           f'(subset_size={args.qat_subset_size}, seq_len={args.max_seq_len})')
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, fused=True)

    # MFU 测量准备
    flops_per_token = compute_model_flops_per_token(model, lm_config)
    gpu_peak_tflops = detect_gpu_peak_tflops_bf16(override=args.gpu_peak_tflops)
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    Logger(f'[MFU] params_flops/token={flops_per_token/1e9:.3f} GFLOPS, '
           f'GPU peak={gpu_peak_tflops:.0f} TFLOPS bf16, world_size={world_size}')

    # 6. resume
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])  # state dict already contains QAT buffers
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # 7. compile + DDP
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # 8. train
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=args.num_workers > 0)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)

    # 9. cleanup
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
