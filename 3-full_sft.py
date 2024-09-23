import os
import platform
import time
import math
import warnings

import pandas as pd
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext

from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModel
from model.model import Transformer
from model.LMConfig import LMConfig
from model.dataset import SFTDataset

warnings.filterwarnings('ignore')


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(it, all):
    warmup_iters = 0
    lr_decay_iters = all
    min_lr = learning_rate / epochs

    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# ------------------------------------------------------------------------------
def train_epoch(epoch, wandb):
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(device)
        Y = Y.to(device)
        loss_mask = loss_mask.to(device)
        lr = get_lr(epoch * iter_per_epoch + step, epochs * iter_per_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            logits = model(X, Y).logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=0, reduction='none')
            loss_mask = loss_mask.view(-1)
            loss = torch.sum(loss * loss_mask) / loss_mask.sum()

        # Backward pass
        scaler.scale(loss).backward()

        # Unscale gradients and clip them
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters
        scaler.step(optimizer)
        scaler.update()

        # Zero the gradients
        optimizer.zero_grad(set_to_none=True)

        # 打印日志
        if step % 100 == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.8f} epoch_Time:{}min:'.format(
                    epoch,
                    epochs,
                    step,
                    iter_per_epoch,
                    loss,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))
        if use_wandb != None:
            wandb.log({"loss": loss, "lr": optimizer.param_groups[-1]['lr'],
                       "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        if (step + 1) % 1000 == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            # torch.save(model.state_dict(), '{}/sft_iter_{}.pth'.format(save_dir, int(step + epoch * iter_per_epoch)))
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{save_dir}/full_sft_{lm_config.dim}{moe_path}.pth'
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            torch.save(state_dict, ckp)
            model.train()


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    model_from = 1  # 1从权重，2用transformers

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    if model_from == 1:
        model = Transformer(lm_config)
        moe_path = '_moe' if lm_config.use_moe else ''
        ckp = f'./out/pretrain_{lm_config.dim}{moe_path}.pth'
        state_dict = torch.load(ckp, map_location=device)
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict, strict=False)
    else:
        model = AutoModel.from_pretrained('./minimind', trust_remote_code=True)

    Logger(f'LLM总参数量：{count_parameters(model) / 1e6:.3f} 百万')
    model = model.to(device)

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


# I/O
if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    lm_config = LMConfig()
    max_seq_len = lm_config.max_seq_len
    out_dir = 'out'
    epochs = 19
    gradient_accumulation_steps = 1
    batch_size = 40
    learning_rate = 1e-4
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dtype = 'bfloat16'
    # dtype = 'float16'
    save_dir = os.path.join(out_dir)
    os.makedirs(save_dir, exist_ok=True)
    tokens_per_iter = gradient_accumulation_steps * batch_size * max_seq_len
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337)
    device_type = device if "cuda" in device else "cpu"

    use_wandb = True #是否使用wandb
    wandb_project = "MiniMind-Full-SFT"
    wandb_run_name = f"MiniMind-Full-SFT-Epoch-{epochs}-BatchSize-{batch_size}-LearningRate-{learning_rate}"
    if use_wandb:
        import wandb
        wandb.init(project=wandb_project, name=wandb_run_name)
    else:
        wandb = None

    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.cuda.amp.autocast()
    )

    ### ddp config
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"
    if ddp:
        init_distributed_mode()
        device = torch.device(DEVICE)
    # -----------------------------------------------------------------------------

    model, tokenizer = init_model(lm_config)
    # -----init dataloader------
    df = pd.read_csv('./dataset/sft_data_single.csv')
    df = df.sample(frac=1.0)
    train_ds = SFTDataset(df, tokenizer, max_length=max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        pin_memory=False,
        drop_last=False,
        shuffle=False,
        num_workers=8,
        sampler=train_sampler
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == dtype))
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    iter_per_epoch = len(train_loader)
    # compile the model
    if False and not lm_config.use_moe and platform.system() != 'Windows' and float(
            torch.__version__.split('.')[0]) >= 2:
        Logger("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0

    if ddp:
        # Ignore the pos_cis buffer so that DDP does not broadcast it at
        # construction time since NCCL does not support ComplexFloat
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    # training loop
    for epoch in range(epochs,wandb):
        train_epoch(epoch)
