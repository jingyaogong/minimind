import os
import platform
import time
import math
import warnings
import torch
import pandas as pd
import torch.nn.functional as F
from contextlib import nullcontext

from torch import optim
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import DataLoader
from model.LMConfig import LMConfig
from model.dataset import SFTDataset

warnings.filterwarnings('ignore', category=UserWarning)


def get_lr(it):
    warmup_iters = 1000
    lr_decay_iters = 80000
    min_lr = 1e-5

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
        lr = get_lr(epoch * iter_per_epoch + step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            logits = model(X, Y).logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=0, reduction='none')
            loss_mask = loss_mask.view(-1)
            loss = torch.sum(loss * loss_mask) / loss_mask.sum()

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad(set_to_none=True)

        if step % 100 == 0:
            spend_time = time.time() - start_time
            print(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                    epoch,
                    epochs,
                    step,
                    iter_per_epoch,
                    loss.item(),
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if wandb is not None:
                wandb.log({"loss": loss.item(), "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def init_model():
    model_name_or_path = "./minimind-v1-small"
    tokenizer_name_or_path = "./minimind-v1-small"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True).to(device)

    target_modules = find_all_linear_names(model)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        inference_mode=False,
        target_modules=target_modules
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model = model.to(device)
    return model, tokenizer


# I/O
if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    lm_config = LMConfig()
    max_seq_len = lm_config.max_seq_len
    out_dir = 'out'
    epochs = 20
    gradient_accumulation_steps = 1
    batch_size = 16
    learning_rate = 1e-4
    weight_decay = 1e-1
    device = 'cuda:0'
    dtype = 'bfloat16'
    save_dir = os.path.join(out_dir)
    os.makedirs(save_dir, exist_ok=True)
    tokens_per_iter = gradient_accumulation_steps * batch_size * max_seq_len
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337)
    device_type = device if "cuda" in device else "cpu"

    use_wandb = False  # 是否使用wandb
    wandb_project = "MiniMind-LoRA-SFT"
    wandb_run_name = f"MiniMind-LoRA-SFT-Epoch-{epochs}-BatchSize-{batch_size}-LearningRate-{learning_rate}"
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
    # -----------------------------------------------------------------------------

    model, tokenizer = init_model()

    # -----init dataloader------
    df = pd.read_csv('./dataset/sft_data_single.csv')
    df = df.sample(frac=1.0)
    train_ds = SFTDataset(df, tokenizer, max_length=max_seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        pin_memory=False,
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    iter_per_epoch = len(train_loader)
    # compile the model
    if False and platform.system() != 'Windows' and float(torch.__version__.split('.')[0]) >= 2:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)

    raw_model = model
    # training loop
    for epoch in range(epochs):
        train_epoch(epoch, wandb)
        model.save_pretrained('minimind')
