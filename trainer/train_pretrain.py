import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import math
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import (
    get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode,
    setup_seed, init_model, SkipBatchSampler,
    detect_gpu_peak_tflops_bf16, compute_model_flops_per_token,
)

warnings.filterwarnings('ignore')


@torch.no_grad()
def eval_holdout(model, eval_loader, max_batches, autocast_ctx, device):
    """
    rank-0 only 跑 holdout PPL；调用方需保证只 rank-0 进入。
    返回 (avg_loss, ppl)。max_batches 控制 eval 时长（1B 时 20-50 batch 足够，~1s）。
    """
    was_training = model.training
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i, (input_ids, labels) in enumerate(eval_loader):
        if i >= max_batches: break
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast_ctx:
            res = model(input_ids, labels=labels)
        # loss 已经是 mean over non-pad tokens；乘 n_tokens 便于跨 batch weighted avg
        n_tokens = (labels != -100).sum().item()
        if n_tokens > 0:
            total_loss += res.loss.item() * n_tokens
            total_tokens += n_tokens
    if was_training: model.train()
    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss, math.exp(avg_loss)


def train_epoch(epoch, loader, iters, start_step=0, wandb=None, eval_loader=None, profiler=None):
    start_time = time.time()
    last_log_time, last_log_step = start_time, start_step
    last_step = start_step
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # torch.compile(mode="max-autotune") 会启用 CUDA graph capture 复用输出 buffer；
        # 训练循环里每步都是新 batch，必须显式 mark step 边界告诉 CUDAGraph "上一步的输出
        # 现在不能再复用了"。否则 backward 拿到被覆盖的 tensor 报 "accessing tensor output
        # of CUDAGraphs that has been overwritten" 错。torch 2.6+ 严格执行此检查。
        if args.use_compile == 1:
            torch.compiler.cudagraph_mark_step_begin()
        # non_blocking=True overlaps host->device copy with compute (pin_memory=True 已在 DataLoader)
        input_ids = input_ids.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)
        last_step = step
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate, args.warmup_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # DDP gradient accumulation optimization：
        # 默认 DDP 每次 .backward() 都做全参数 all-reduce（4GB grad × 85 bucket ≈ 30ms）。
        # 但 accum=N 时前 N-1 次 all-reduce 是浪费（grad 还要继续累积）。用 model.no_sync()
        # 让 DDP 在非最后一个 micro-step 里 skip all-reduce，只在最后一个 micro-step 才同步。
        # profile 显示 all-reduce 占 73% 时间 → 修完预期总时间省 ~55%，MFU 17%→~35%。
        is_last_accum_step = (step % args.accumulation_steps == 0)
        need_sync = is_last_accum_step or not isinstance(model, DistributedDataParallel)
        sync_ctx = nullcontext() if need_sync else model.no_sync()

        with sync_ctx:
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

        # profiler step 打点（每 step 一次；schedule 决定何时真的记录）
        if profiler is not None:
            profiler.step()

        if step % args.log_interval == 0 or step == iters:
            now = time.time()
            spend_time = now - start_time
            window_dt = max(now - last_log_time, 1e-6)
            window_steps = step - last_log_step
            window_its = window_steps / window_dt
            # MFU = per-card 实际 FLOPs/s / per-card 峰值。分子分母都要一致口径：
            # - 分子: 4 卡总 tokens × flops/token = 总 flops；÷ world_size 得 per-card flops
            # - 分母: gpu_peak_tflops (单卡峰值，312 TFLOPS on A100)
            # 早期版本 bug：分子用全部 tokens 但分母只除 1 卡峰值 → 报数是真实 MFU 的 world_size 倍。
            tokens_window = window_steps * args.batch_size * args.max_seq_len * world_size
            total_flops_per_s = flops_per_token * tokens_window / window_dt
            per_card_flops_per_s = total_flops_per_s / world_size
            mfu = per_card_flops_per_s / (gpu_peak_tflops * 1e12) * 100
            last_log_time, last_log_step = now, step
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'[{time.strftime("%H:%M:%S")}] Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), it/s: {window_its:.2f}, MFU: {mfu:.1f}%, loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # ---- eval loop（可选） ----
        if args.eval_interval > 0 and step % args.eval_interval == 0 and eval_loader is not None:
            # ⚠️ 全 rank 都跑 eval：单 rank eval (即使 unwrap raw model) 也会导致 DDP guard 失效 /
            # torch.compile recompile → rank 0 长 stall → 下一步 all-reduce seq 不同步 → NCCL 10min timeout。
            # 解决：所有 rank 用同一份 eval_loader (顺序一致) 跑同样的 batch，然后 all-reduce mean loss。
            # 每 rank 都工作，无 rank 掉队问题。max_batches=20 * 5 rank = 20 batch × ~0.3s ≈ 6s 开销。
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            val_loss_local, _ = eval_holdout(raw_model, eval_loader, args.eval_max_batches, autocast_ctx, args.device)
            if dist.is_initialized():
                loss_t = torch.tensor(val_loss_local, device=args.device)
                dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
                val_loss = loss_t.item()
            else:
                val_loss = val_loss_local
            val_ppl = math.exp(val_loss)
            if is_main_process():
                Logger(f'[eval] step={step} val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}')
                if wandb: wandb.log({"val_loss": val_loss, "val_ppl": val_ppl, "step": step})

        if step % args.save_interval == 0 or step == iters:
            if is_main_process():
                model.eval()
                moe_suffix = '_moe' if lm_config.use_moe else ''
                ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
                raw_model = model.module if isinstance(model, DistributedDataParallel) else model
                raw_model = getattr(raw_model, '_orig_mod', raw_model)
                state_dict = raw_model.state_dict()
                # atomic: tmp + rename 避免读者读到半写文件（当前的 torch.save 直接写不 atomic，
                # 极端情况 kill -9 会留 corrupt ckpt；lm_checkpoint 内部已 atomic，此处也补上）
                ckp_tmp = ckp + '.tmp'
                torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp_tmp)
                os.replace(ckp_tmp, ckp)
                lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
                model.train()
                del state_dict
            # 所有 rank 在此汇合，让非 rank-0 等 rank-0 写完再进下一 step
            # （防止 NCCL 隐式同步下的 timing race，也确保 ckpt 文件对 resume 一致）
            if dist.is_initialized():
                dist.barrier()

        del input_ids, labels, res, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=1, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是，A100 上推荐默认 1）")
    parser.add_argument("--gpu_peak_tflops", type=float, default=0,
                        help="GPU bf16 理论峰值 TFLOPS，用于 MFU 计算；默认 0 表示按 GPU 名自动检测（A100=312）")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="linear warmup 步数。1B from-scratch 建议 500-2000；64M 用现有 pretrain ckpt 微调设 0 即可")
    parser.add_argument("--grad_checkpointing", type=int, default=0, choices=[0, 1],
                        help="激活值 gradient checkpointing：省 30-50%% activation 显存，慢 ~30%%。"
                             "1B + seq=1024 在 80GB 用不上；seq=2048 或 8B+ 或大 batch 时需要打开")
    parser.add_argument("--eval_interval", type=int, default=0,
                        help="每 N step 跑一次 holdout eval（0=关闭）。1B 建议 500-1000")
    parser.add_argument("--eval_data_path", type=str, default=None,
                        help="holdout jsonl 路径；不设则关闭 eval")
    parser.add_argument("--eval_max_batches", type=int, default=20,
                        help="每次 eval 最多跑几个 batch（20*batch_size 通常够 ~1s 内出数）")
    parser.add_argument("--profile_steps", type=int, default=0,
                        help="torch.profiler 抓 N 步（0=关）；建议 20。会在 wait=10 warmup=5 active=N 后自动退出。"
                             "trace 存到 --profile_out 目录，chrome://tracing 打开或 tensorboard --logdir profile_out")
    parser.add_argument("--profile_out", type=str, default="../profile_out",
                        help="profile trace 输出目录")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    if args.grad_checkpointing == 1:
        model.model.set_grad_checkpointing(True)
        Logger('gradient checkpointing enabled (activation 省 30-50%%, 慢 ~30%%)')
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # fused=True 把所有 param 的 Adam 更新融成一个 CUDA kernel（A100/3090 + CUDA tensor 支持）
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, fused=True)

    # ========== 5b. MFU 测量准备 ==========
    flops_per_token = compute_model_flops_per_token(model, lm_config)
    gpu_peak_tflops = detect_gpu_peak_tflops_bf16(override=args.gpu_peak_tflops)
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    Logger(f'[MFU] params_flops/token={flops_per_token/1e9:.3f} GFLOPS, '
           f'GPU peak={gpu_peak_tflops:.0f} TFLOPS bf16, world_size={world_size}')

    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        # 新 CUDA 13 stack 上 max-autotune 会为每个 GEMM shape 试 21 种 triton kernel（分钟级
        # autotune 循环，且实测 cuBLAS aten mm 几乎每次都赢过 triton 变体，autotune 白花时间）。
        # 用 default 更实惠——inductor 融合仍在，只是不额外扫 triton kernel。
        # (max-autotune 还需要 model 无 graph break，否则 CUDA graph 跨 step 覆盖崩溃；
        # 我们已消除 RoPE graph break，若将来单独 GEMM shape 少可再试 max-autotune)
        model = torch.compile(model)
        Logger('torch.compile enabled (mode=default)')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 7b. eval loader（可选） ==========
    # 全 rank 都建 eval loader（不能只 rank 0，否则 all-reduce 时 rank 1/2/3 无 eval loss 参与聚合）
    eval_loader = None
    if args.eval_interval > 0 and args.eval_data_path:
        eval_ds = PretrainDataset(args.eval_data_path, tokenizer, max_length=args.max_seq_len)
        # 所有 rank 用同一份数据 + shuffle=False → 保证跑相同 batch → all-reduce 有意义
        eval_loader = DataLoader(eval_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=2, pin_memory=True)
        Logger(f'[eval] loader ready: {len(eval_ds)} samples from {args.eval_data_path}, '
               f'eval_interval={args.eval_interval}, max_batches={args.eval_max_batches}')

    # ========== 7c. profiler（可选） ==========
    profiler_ctx = None
    if args.profile_steps > 0 and is_main_process():
        # schedule: 前 10 步 warmup（skip compile 编译期），5 步观察，profile_steps 步真记录
        from torch.profiler import profile, schedule, ProfilerActivity, tensorboard_trace_handler
        os.makedirs(args.profile_out, exist_ok=True)
        profiler_ctx = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=10, warmup=5, active=args.profile_steps, repeat=1),
            on_trace_ready=tensorboard_trace_handler(args.profile_out),
            record_shapes=True, with_stack=False, with_flops=False,
        )
        Logger(f'[profile] enabled: warmup 15 step → capture {args.profile_steps} step → trace to {args.profile_out}')
        Logger('[profile] 训练完 profile 步后不自动退出——手动 Ctrl+C 停即可。trace 已 flush 到磁盘。')

    # ========== 8. 开始训练 ==========
    if profiler_ctx is not None:
        profiler_ctx.__enter__()
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers,
                            pin_memory=True, persistent_workers=args.num_workers > 0)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb, eval_loader, profiler_ctx)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb, eval_loader, profiler_ctx)
    if profiler_ctx is not None:
        profiler_ctx.__exit__(None, None, None)

    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
