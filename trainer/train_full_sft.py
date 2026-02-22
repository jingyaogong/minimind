import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import (
    get_lr,
    Logger,
    is_main_process,
    lm_checkpoint,
    init_distributed_mode,
    setup_seed,
    init_model,
    SkipBatchSampler,
    init_neuron_mask,
    set_neuron_tracking,
    grow_neurons,
    save_run_config,
    update_run_config,
    get_active_ratio_by_layer,
    get_active_ratio_stats
)

warnings.filterwarnings('ignore')


def get_neuron_active_ratio(model):
    total = 0
    active = 0
    for m in model.modules():
        if hasattr(m, "mask"):
            total += m.mask.numel()
            active += int(m.mask.sum().item())
    return (active / total) if total > 0 else 1.0


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    tokens_seen = 0
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        tokens_seen += input_ids.numel()
        global_step = epoch * iters + step
        should_update = ((step + 1) % args.accumulation_steps == 0)
        update_step = global_step // args.accumulation_steps
        should_grow = bool(args.neuron_growth) and should_update and (update_step > 0) and (update_step % args.grow_interval == 0)

        if args.neuron_growth:
            track_activity = (args.grow_method != "random")
            track_grad = (args.grow_method != "random") and should_grow
            set_neuron_tracking(model, track_activity=track_activity, track_mask_grad=track_grad, ema_beta=args.neuron_ema_beta)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            if should_grow:
                grow_neurons(
                    model,
                    method=args.grow_method,
                    grow_ratio=args.grow_ratio,
                    max_active_ratio=args.max_active_ratio,
                    score_alpha=args.grow_score_alpha,
                    score_beta=args.grow_score_beta,
                    seed=global_step
                )
                set_neuron_tracking(model, track_activity=(args.grow_method != "random"), track_mask_grad=False, ema_beta=args.neuron_ema_beta)

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            tokens_per_sec = tokens_seen / max(spend_time, 1e-6)
            active_ratio = get_neuron_active_ratio(model) if args.neuron_growth else None
            active_stats = get_active_ratio_stats(model) if args.neuron_growth else None
            active_msg = f', active_ratio: {active_ratio:.3f}' if active_ratio is not None else ''
            if active_stats:
                active_msg += f", active_mean: {active_stats['mean']:.3f}, active_min: {active_stats['min']:.3f}, active_max: {active_stats['max']:.3f}"
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min, tok/s: {tokens_per_sec:.1f}{active_msg}')
            if wandb:
                log_data = {"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min, "tokens_per_sec": tokens_per_sec}
                if active_ratio is not None:
                    log_data["active_ratio"] = active_ratio
                if active_stats:
                    log_data["active_ratio_mean"] = active_stats["mean"]
                    log_data["active_ratio_min"] = active_stats["min"]
                    log_data["active_ratio_max"] = active_stats["max"]
                if args.neuron_growth:
                    layer_ratios = get_active_ratio_by_layer(model)
                    for name, ratio in layer_ratios.items():
                        key = name.replace(".", "_")
                        log_data[f"active_{key}"] = ratio
                wandb.log(log_data)

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scaler=scaler)
            model.train()
            del state_dict

        del input_ids, labels, res, loss

    return tokens_seen


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
    parser.add_argument("--seed", type=int, default=42, help="随机种子（复现实验用）")
    # 动态神经元生长相关参数（可选）
    parser.add_argument("--neuron_growth", default=0, type=int, choices=[0, 1], help="是否启用动态神经元生长")
    parser.add_argument("--init_active_ratio", type=float, default=0.8, help="初始激活神经元比例")
    parser.add_argument("--grow_method", type=str, default="random", choices=["random", "act_grad"], help="神经元生长方式")
    parser.add_argument("--grow_interval", type=int, default=100, help="每隔多少次优化器更新触发生长")
    parser.add_argument("--grow_ratio", type=float, default=0.02, help="每次生长激活比例")
    parser.add_argument("--max_active_ratio", type=float, default=0.99, help="最多激活到多少比例")
    parser.add_argument("--grow_score_alpha", type=float, default=1.0, help="活动分数权重(EMA)")
    parser.add_argument("--grow_score_beta", type=float, default=1.0, help="梯度分数权重")
    parser.add_argument("--neuron_ema_beta", type=float, default=0.1, help="活动EMA系数")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_mini_512.jsonl", help="训练数据路径")
    parser.add_argument('--from_weight', default='pretrain', type=str, help="基于哪个权重训练，为none则不基于任何权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    run_config_path = None
    if is_main_process():
        run_name = f"{args.save_weight}_{args.hidden_size}_{time.strftime('%Y%m%d_%H%M%S')}"
        run_config_path = save_run_config(args, args.save_dir, run_name=run_name, extra={"resume": bool(ckp_data)})
    
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
        wandb_run_name = f"MiniMind-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    if args.neuron_growth and not ckp_data:
        init_neuron_mask(model, init_active_ratio=args.init_active_ratio, seed=42)
    if args.neuron_growth:
        set_neuron_tracking(
            model,
            track_activity=(args.grow_method != "random"),
            track_mask_grad=False,
            ema_beta=args.neuron_ema_beta
        )
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. DDP包模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    train_start = time.time()
    total_tokens = 0
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(args.seed + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            epoch_tokens = train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            epoch_tokens = train_epoch(epoch, loader, len(loader), 0, wandb)
        total_tokens += epoch_tokens

    if is_main_process() and run_config_path:
        update_run_config(run_config_path, {
            "train_time_sec": time.time() - train_start,
            "total_tokens": total_tokens,
            "final_active_ratio": get_neuron_active_ratio(model) if args.neuron_growth else None
        })
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()
