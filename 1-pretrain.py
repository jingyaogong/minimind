import os
import platform
import time
import math
import warnings
import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext
from model.model import Transformer
from model.LMConfig import LMConfig
from model.dataset import PretrainDataset

warnings.filterwarnings('ignore')


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(it, all):
    warmup_iters = 0
    lr_decay_iters = all
    min_lr = learning_rate / 10

    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def train_epoch(epoch, wandb, accumulation_steps=8):
    start_time = time.time()
    for step, (X, Y) in enumerate(train_loader):
        X = X.to(device)
        Y = Y.to(device)

        lr = get_lr(epoch * iter_per_epoch + step, epochs * iter_per_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            out = model(X, Y)
            loss = out.last_loss / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % 100 == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                    epoch,
                    epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))
            if wandb != None:
                wandb.log({"loss": loss.item() * accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        if (step + 1) % 1000 == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            # torch.save(model.state_dict(), '{}/iter_{}.pth'.format(save_dir, int(step + epoch * iter_per_epoch)))
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{save_dir}/pretrain_{lm_config.dim}{moe_path}.pth'

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            torch.save(state_dict, ckp)
            model.train()


def init_model():
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # model init
    model = Transformer(lm_config).to(device)
    moe_path = '_moe' if lm_config.use_moe else ''
    # ckp = f'{save_dir}/pretrain_{lm_config.dim}{moe_path}.pth'
    #
    # state_dict = torch.load(ckp, map_location=device)
    # unwanted_prefix = '_orig_mod.'
    # for k, v in list(state_dict.items()):
    #     if k.startswith(unwanted_prefix):
    #         state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    # model.load_state_dict(state_dict, strict=False)

    Logger(f'LLM总参数量：{count_parameters(model) / 1e6:.3f} 百万')
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

# torchrun --nproc_per_node 2 1-pretrain.py
# I/O
if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    lm_config = LMConfig()
    max_seq_len = lm_config.max_seq_len
    out_dir = 'out'
    epochs = 20
    batch_size = 64
    learning_rate = 2e-4
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dtype = 'bfloat16'
    save_dir = os.path.join(out_dir)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    tokens_per_iter = batch_size * max_seq_len
    torch.manual_seed(1337)
    device_type = device if "cuda" in device else "cpu"

    use_wandb = True #是否使用wandb
    wandb_project = "MiniMind-Pretrain"
    wandb_run_name = f"MiniMind-Pretrain-Epoch-{epochs}-BatchSize-{batch_size}-LearningRate-{learning_rate}"
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
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"
    if ddp:
        init_distributed_mode()
        device = torch.device(DEVICE)
    # -----------------------------------------------------------------------------

    # -----init dataloader------
    data_path_list = ['./dataset/pretrain_data.bin']
    train_ds = PretrainDataset(data_path_list, max_length=max_seq_len, memmap=True)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    num_workers = 8  # 可以根据系统的 CPU 核心数来调整
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=num_workers,
        sampler=train_sampler
    )

    # init model
    model = init_model()

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == dtype))
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # compile the model
    if False and platform.system() != 'Windows' and float(torch.__version__.split('.')[0]) >= 2:
        Logger("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)

    if ddp:
        # Ignore the freqs_cis buffer so that DDP does not broadcast it at
        # construction time since NCCL does not support ComplexFloat
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    # training loop
    iter_per_epoch = len(train_loader)
    for epoch in range(epochs):
        train_epoch(epoch, wandb)
