"""
DeepSpeed ZeRO-2 全参 SFT 入口。

推荐在 trainer/ 目录启动：
    deepspeed --num_gpus=7 train_full_sft_deepspeed.py --model_profile SearchLM-300M
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import math
import time
import warnings

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import (
    Logger,
    SkipBatchSampler,
    add_deepspeed_args,
    add_model_profile_args,
    apply_model_profile,
    build_lm_config,
    build_training_signature,
    assert_no_training_output_collision,
    init_model,
    is_main_process,
    get_cuda_peak_memory_gb,
    load_deepspeed_checkpoint,
    load_deepspeed_config,
    log_training_setup,
    normalize_resume_position,
    reduce_metrics,
    resolve_attention_type,
    save_deepspeed_checkpoint,
    save_model_weights,
    setup_seed,
    validate_training_signature,
)

warnings.filterwarnings("ignore")


def validate_sft_supervision(train_ds, max_samples=32):
    checked = min(len(train_ds), max_samples)
    supervised_tokens = 0
    total_tokens = 0
    valid_samples = 0
    for index in range(checked):
        _, labels = train_ds[index]
        current_supervised = int(labels.ne(-100).sum().item())
        supervised_tokens += current_supervised
        total_tokens += int(labels.numel())
        valid_samples += int(current_supervised > 0)
    if valid_samples == 0:
        raise RuntimeError(
            "SFT data validation failed: sampled labels are all -100. "
            "Check conversations roles and tokenizer chat-template markers."
        )
    if is_main_process():
        ratio = supervised_tokens / max(total_tokens, 1)
        Logger(
            f"SFT supervision check: samples={checked}, valid_samples={valid_samples}, "
            f"supervised_tokens={supervised_tokens}, supervised_ratio={ratio:.4f}"
        )


def validate_chat_template(tokenizer):
    try:
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": "template check"},
                {"role": "assistant", "content": "ok"},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "Tokenizer chat template is invalid. Check model/chat_template.jinja and "
            "model/tokenizer_config.json before loading the SFT dataset."
        ) from exc
    required_markers = ("<|im_start|>user", "<|im_start|>assistant", "<|im_end|>")
    missing = [marker for marker in required_markers if marker not in rendered]
    if missing:
        raise RuntimeError(f"Tokenizer chat template is missing required markers: {missing}")
    if is_main_process():
        Logger("Tokenizer chat template validation passed")


def init_deepspeed_runtime(args, deepspeed):
    deepspeed.init_distributed(dist_backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank >= 0 else 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"
    else:
        args.device = "cpu"
    args.training_backend = "DeepSpeed-ZeRO"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else local_rank))
    return local_rank


def train_epoch(args, lm_config, model_engine, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)

        res = model_engine(input_ids, labels=labels)
        aux_loss = res.aux_loss if res.aux_loss is not None else torch.tensor(0.0, device=args.device)
        loss = res.loss + aux_loss

        model_engine.backward(loss)
        model_engine.step()

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item()
            current_aux_loss = aux_loss.item()
            current_logits_loss = current_loss - current_aux_loss
            lr_list = model_engine.get_lr()
            current_lr = lr_list[0] if lr_list else args.learning_rate
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            metrics = reduce_metrics({
                "loss": current_loss,
                "logits_loss": current_logits_loss,
                "aux_loss": current_aux_loss,
            })
            peak_memory_gb = get_cuda_peak_memory_gb()
            memory_text = f", peak_memory: {peak_memory_gb:.2f} GB" if peak_memory_gb is not None else ""
            Logger(
                f"Epoch:[{args.current_epoch + 1}/{args.epochs}]({step}/{iters}), "
                f"loss: {metrics['loss']:.4f}, logits_loss: {metrics['logits_loss']:.4f}, "
                f"aux_loss: {metrics['aux_loss']:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min"
                f"{memory_text}"
            )
            if wandb and is_main_process():
                wandb.log(
                    {
                        "loss": metrics["loss"],
                        "logits_loss": metrics["logits_loss"],
                        "aux_loss": metrics["aux_loss"],
                        "learning_rate": current_lr,
                        "epoch_time": eta_min,
                    }
                )

        should_save = step % args.save_interval == 0 or step == iters
        del input_ids, labels, res, loss, aux_loss

        if should_save:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model_engine.eval()
            save_deepspeed_checkpoint(
                model_engine,
                lm_config,
                weight=args.save_weight,
                epoch=args.current_epoch,
                step=step,
                wandb=wandb,
                save_dir="../checkpoints",
                training_signature=args.training_signature,
            )
            if is_main_process():
                ckp = save_model_weights(lm_config, model_engine, save_dir=args.save_dir, weight=args.save_weight)
                Logger(f"Lightweight model saved: {ckp}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # rank0 导出轻量权重时，其他 rank 必须等待，避免提前销毁 NCCL process group。
            if dist.is_initialized():
                dist.barrier()
            model_engine.train()


def parse_args():
    parser = argparse.ArgumentParser(description="Full SFT with DeepSpeed ZeRO")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument("--save_weight", default="full_sft", type=str)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--num_attention_heads", default=8, type=int)
    parser.add_argument("--num_key_value_heads", default=4, type=int)
    parser.add_argument("--intermediate_size", default=None, type=int)
    parser.add_argument("--max_seq_len", default=768, type=int)
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1])
    parser.add_argument("--attention_type", default="gqa", choices=["gqa", "mha", "mqa", "mla"])
    parser.add_argument("--use_mla", default=0, type=int, choices=[0, 1])
    parser.add_argument("--kv_lora_rank", default=128, type=int)
    parser.add_argument("--q_lora_rank", default=256, type=int)
    parser.add_argument("--rope_dim", default=None, type=int)
    parser.add_argument("--data_path", type=str, default="../dataset/sft_t2t_mini.jsonl")
    parser.add_argument("--from_weight", default="pretrain", type=str)
    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1])
    parser.add_argument("--allow_resume_mismatch", default=0, type=int, choices=[0, 1])
    parser.add_argument("--overwrite_output", default=0, type=int, choices=[0, 1])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="SearchLM-Full-SFT-DS")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1])
    add_model_profile_args(parser)
    add_deepspeed_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_profile = apply_model_profile(args)

    try:
        import deepspeed
    except ImportError as exc:
        raise RuntimeError("DeepSpeed未安装。服务器环境执行: pip install deepspeed") from exc

    local_rank = init_deepspeed_runtime(args, deepspeed)
    os.makedirs(args.save_dir, exist_ok=True)

    lm_config = build_lm_config(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        attention_type=resolve_attention_type(args),
        kv_lora_rank=args.kv_lora_rank,
        q_lora_rank=args.q_lora_rank,
        rope_dim=args.rope_dim,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        intermediate_size=args.intermediate_size,
    )
    assert_no_training_output_collision(
        lm_config,
        weight=args.save_weight,
        save_dir=args.save_dir,
        checkpoint_dir="../checkpoints",
        from_resume=args.from_resume,
        overwrite=args.overwrite_output,
    )

    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    validate_chat_template(tokenizer)
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger("torch.compile enabled before DeepSpeed initialize")

    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    validate_sft_supervision(train_ds)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    args.training_signature = build_training_signature(args, args.data_path, len(train_ds), iters)
    total_optimizer_steps = max(1, math.ceil(iters / args.accumulation_steps) * args.epochs)
    ds_config = load_deepspeed_config(args, total_optimizer_steps)

    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )

    ckp_data = None
    if args.from_resume == 1:
        ckp_data = load_deepspeed_checkpoint(
            model_engine,
            lm_config,
            weight=args.save_weight,
            save_dir="../checkpoints",
            required=True,
        )
        validate_training_signature(
            ckp_data.get("training_signature"),
            args.training_signature,
            allow_mismatch=bool(args.allow_resume_mismatch),
        )

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        wandb.init(
            project=args.wandb_project,
            name=f"SearchLM-Full-SFT-DS-{args.attention_type}-{args.hidden_size}",
            id=wandb_id,
            resume="must" if wandb_id else None,
        )

    extra = {
        "model_profile": args.model_profile if model_profile else "",
        "deepspeed_config": args.deepspeed_config,
        "zero_stage": ds_config.get("zero_optimization", {}).get("stage", "none"),
        "total_optimizer_steps": total_optimizer_steps,
    }
    log_training_setup(
        args,
        lm_config,
        stage="full_sft_deepspeed",
        dataset_len=len(train_ds),
        iters=iters,
        tokens_per_sample=args.max_seq_len,
        extra=extra,
    )

    start_epoch = ckp_data.get("epoch", 0) if ckp_data else 0
    start_step = ckp_data.get("step", 0) if ckp_data else 0
    start_epoch, start_step = normalize_resume_position(start_epoch, start_step, iters)
    if args.from_resume == 1 and is_main_process():
        Logger(f"SFT resume position: epoch={start_epoch}, step={start_step}, iters_per_epoch={iters}")
    if start_epoch >= args.epochs and is_main_process():
        Logger(f"Training already complete: resume epoch={start_epoch}, target epochs={args.epochs}")

    for epoch in range(start_epoch, args.epochs):
        args.current_epoch = epoch
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch + local_rank)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        if skip > 0:
            Logger(f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始")
        train_epoch(args, lm_config, model_engine, loader, len(loader) + skip, skip, wandb)

    if dist.is_initialized():
        dist.destroy_process_group()
