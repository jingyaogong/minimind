"""
MiniMind Pretrain — DeepSpeed 版本
使用 DeepSpeed ZeRO-2 替代 DDP，自动管理混合精度、梯度累积、梯度裁剪和学习率调度
启动方式:
    deepspeed --num_gpus=6 train_pretrain_deepspeed.py --use_wandb
    torchrun --nproc_per_node=6 train_pretrain_deepspeed.py --use_wandb
"""
import os, sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse, time, json, warnings
import torch, torch.distributed as dist, deepspeed
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from model.model_minimind_mla import MiniMindMLAConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import (
    Logger, is_main_process, setup_seed, init_model, SkipBatchSampler, get_model_suffix,
    init_distributed_mode
)

warnings.filterwarnings('ignore')

# DeepSpeed 默认配置路径
DEFAULT_DS_CONFIG = os.path.join(os.path.dirname(__file__), "ds_config_pretrain.json")


def _get_ds_config(args, total_steps):
    """加载 DeepSpeed 配置，并填入训练总步数"""
    ds_path = getattr(args, 'deepspeed_config', None) or DEFAULT_DS_CONFIG
    with open(ds_path) as f:
        cfg = json.load(f)

    # 自动推算 batch size / accumulation
    if cfg.get("train_micro_batch_size_per_gpu") == "auto":
        cfg["train_micro_batch_size_per_gpu"] = args.batch_size
    if cfg.get("gradient_accumulation_steps") == "auto":
        cfg["gradient_accumulation_steps"] = args.accumulation_steps
    if cfg.get("train_batch_size") == "auto":
        world = dist.get_world_size() if dist.is_initialized() else 1
        cfg["train_batch_size"] = args.batch_size * args.accumulation_steps * world

    # 填入学习率和总步数
    cfg["optimizer"]["params"]["lr"] = args.learning_rate
    sched = cfg.get("scheduler", {})
    if sched.get("type") == "WarmupDecayLR":
        p = sched.setdefault("params", {})
        if p.get("warmup_max_lr") == "auto":
            p["warmup_max_lr"] = args.learning_rate
        if p.get("warmup_num_steps") == "auto":
            p["warmup_num_steps"] = max(1, total_steps // 20)
        if p.get("total_num_steps") == "auto":
            p["total_num_steps"] = total_steps
    return cfg


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    last_step = start_step
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step

        # DeepSpeed 自动管理混合精度，不需要 autocast_ctx
        res = model_engine(input_ids, labels=labels)
        loss = res.loss + res.aux_loss
        # DeepSpeed 自动处理梯度累积和 loss scaling
        model_engine.backward(loss)
        # step：梯度裁剪 → 更新参数 → lr scheduler step → zero_grad
        model_engine.step()

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item()
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = model_engine.get_lr()[0] if model_engine.get_lr() else args.learning_rate
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb:
                wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model_engine.eval()
            moe_suffix = get_model_suffix(lm_config)
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # 保存权重（轻量）
            raw_model = model_engine.module
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            torch.save({k: v.half().cpu() for k, v in raw_model.state_dict().items()}, ckp)
            # 保存完整训练状态（含 optimizer / lr scheduler）
            ds_tag = f'{args.save_weight}_{lm_config.hidden_size}{moe_suffix}'
            model_engine.save_checkpoint('../checkpoints', tag=ds_tag)
            Logger(f'DeepSpeed checkpoint saved: ../checkpoints/{ds_tag}')
            model_engine.train()
        del input_ids, labels, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining (DeepSpeed)")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument('--save_weight', default='pretrain_deepspeed', type=str)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=340, type=int)
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1])
    parser.add_argument('--use_mla', default=0, type=int, choices=[0, 1])
    parser.add_argument('--kv_lora_rank', default=128, type=int)
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t.jsonl")
    parser.add_argument('--from_weight', default='none', type=str)
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain-DS")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1])
    parser.add_argument("--deepspeed_config", type=str, default=DEFAULT_DS_CONFIG, help="DeepSpeed 配置 JSON")
    parser.add_argument("--local_rank", type=int, default=-1, help="DeepSpeed 自动注入")
    args = parser.parse_args()

    # ========== 1. 初始化环境（DeepSpeed 全权负责）==========
    # deepspeed.initialize() 内部会自动 init_process_group + set_device
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    args.device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    setup_seed(42 + local_rank)

    # ========== 2. 模型配置 ==========
    os.makedirs(args.save_dir, exist_ok=True)
    if args.use_mla:
        lm_config = MiniMindMLAConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                                      use_moe=bool(args.use_moe), kv_lora_rank=args.kv_lora_rank)
    else:
        lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                                   use_moe=bool(args.use_moe))

    # ========== 3. 初始化模型和 tokenizer ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)

    # ========== 4. 数据集 ==========
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    # ========== 5. 算总步数，填充 DeepSpeed 配置 ==========
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    steps_per_epoch = len(train_ds) // (args.batch_size * args.accumulation_steps * world_size)
    total_steps = steps_per_epoch * args.epochs
    ds_config = _get_ds_config(args, total_steps)

    # ========== 6. DeepSpeed 初始化 ==========
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
        model_parameters=model.parameters(),
    )

    # ========== 7. wandb ==========
    wandb = None
    ckp_data = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_run_name = f"MiniMind-Pretrain-DS-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name)

    # ========== 8. 从 DeepSpeed checkpoint 恢复 ==========
    start_epoch, start_step = 0, 0
    ds_tag = f'{args.save_weight}_{lm_config.hidden_size}{get_model_suffix(lm_config)}'
    if args.from_resume == 1:
        try:
            _, client_state = model_engine.load_checkpoint('../checkpoints', tag=ds_tag)
            start_epoch = client_state.get('epoch', 0)
            start_step = client_state.get('step', 0)
            Logger(f'从 DeepSpeed checkpoint 恢复: epoch={start_epoch}, step={start_step}')
        except Exception as e:
            Logger(f'未找到 DeepSpeed checkpoint: {e}，从头开始训练')

    # 预热：DeepSpeed 在 bf16 下不需要 loss scaling
    Logger(f'DeepSpeed 配置: world_size={world_size}, '
           f'micro_batch={ds_config["train_micro_batch_size_per_gpu"]}, '
           f'accumulation={ds_config["gradient_accumulation_steps"]}, '
           f'total_steps={total_steps}')

    # ========== 9. torch.compile（可选）==========
    if args.use_compile == 1:
        Logger('[WARN] torch.compile 与 DeepSpeed 可能会冲突，建议关闭')
        # 如果坚持要用，只编译内部 module
        if hasattr(model_engine, 'module'):
            model_engine.module = torch.compile(model_engine.module)

    # ========== 10. 训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)

        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)

    if dist.is_initialized():
        dist.destroy_process_group()
