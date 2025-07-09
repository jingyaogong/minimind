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
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM, check_config_compatibility
from dataset.lm_dataset import PretrainDataset
import time

warnings.filterwarnings('ignore')


def Logger(content):
    # 检查是否在分布式环境中
    try:
        if not ddp or dist.get_rank() == 0:
            print(content)
    except NameError:
        # 如果ddp未定义，直接打印（单机模式）
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # 添加10ms睡眠，让显卡休息
        time.sleep(0.01)
        
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
            model.train()


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    model = MiniMindForCausalLM(lm_config)
    
    # 如果启用继续预训练，则加载已有的预训练模型
    if args.continue_pretrain:
        moe_path = '_moe' if lm_config.use_moe else ''
        ckp = f'{args.save_dir}/pretrain_{lm_config.hidden_size}{moe_path}.pth'
        
        if os.path.exists(ckp):
            Logger(f'继续预训练模式：加载已有预训练模型 {ckp}')
            state_dict = torch.load(ckp, map_location=args.device)
            model.load_state_dict(state_dict, strict=False)
            Logger('✅ 已有预训练模型加载成功，将在此基础上继续预训练')
        else:
            Logger(f'⚠️  警告：未找到预训练模型文件 {ckp}，将从随机权重开始训练')
    else:
        Logger('从随机权重开始预训练')
    
    model = model.to(args.device)
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
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--out_dir", type=str, default="../out")
    # 若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--hidden_size', default=512, type=int)#隐藏层维度大小
    parser.add_argument('--num_hidden_layers', default=8, type=int)#隐藏层数量
    parser.add_argument('--max_seq_len', default=512, type=int, help='训练时的序列长度，但模型支持动态扩展到更长')#最大序列长度
    parser.add_argument('--use_moe', default=False, type=bool)#是否使用moe
    parser.add_argument('--dynamic_rope', default=True, type=bool, help='是否启用动态RoPE扩展')
    parser.add_argument('--rope_scaling_factor', default=1.0, type=float, help='RoPE缩放因子')
    parser.add_argument('--rope_scaling_type', default='linear', type=str, help='RoPE缩放类型：linear或dynamic')
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl")
    # 继续预训练相关参数
    parser.add_argument("--continue_pretrain", action="store_true", 
                        help="是否在已有预训练模型基础上继续预训练")
    parser.add_argument("--continue_data_path", type=str, default="../dataset/pretrain_hq_add.jsonl",
                        help="继续预训练时使用的数据路径")
    parser.add_argument("--continue_lr_scale", type=float, default=0.1,
                        help="继续预训练时的学习率缩放因子，默认为原学习率的0.1倍")
    args = parser.parse_args()

    # 配置RoPE缩放
    rope_scaling = None
    if args.rope_scaling_factor != 1.0:
        rope_scaling = {
            "type": args.rope_scaling_type,
            "factor": args.rope_scaling_factor
        }
    
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers, 
        use_moe=args.use_moe,
        max_position_embeddings=None,  # 设为None以支持动态长度
        dynamic_rope=args.dynamic_rope,
        rope_scaling=rope_scaling
    )
    
    # 检查配置兼容性
    Logger("🔍 检查模型配置兼容性...")
    check_config_compatibility(lm_config, warn=True)
    
    if lm_config.dynamic_rope:
        Logger("✅ 启用动态RoPE：模型支持处理任意长度序列（内存允许范围内）")
    else:
        Logger(f"📏 使用固定长度：模型最大支持 {lm_config.max_position_embeddings} tokens")
    
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # 根据是否继续预训练调整wandb运行名称、数据路径和学习率
    if args.continue_pretrain:
        # 继续预训练时使用较小的学习率
        actual_learning_rate = args.learning_rate * args.continue_lr_scale
        args.wandb_run_name = f"MiniMind-ContinuePretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{actual_learning_rate}"
        actual_data_path = args.continue_data_path
        Logger(f"继续预训练模式：使用数据集 {actual_data_path}")
        Logger(f"继续预训练学习率：{actual_learning_rate} (原学习率 {args.learning_rate} × 缩放因子 {args.continue_lr_scale})")
    else:
        actual_learning_rate = args.learning_rate
        args.wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        actual_data_path = args.data_path
        Logger(f"从零预训练模式：使用数据集 {actual_data_path}")
        Logger(f"初始预训练学习率：{actual_learning_rate}")

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
    train_ds = PretrainDataset(actual_data_path, tokenizer, max_length=args.max_seq_len)
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
    optimizer = optim.AdamW(model.parameters(), lr=actual_learning_rate)

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)
