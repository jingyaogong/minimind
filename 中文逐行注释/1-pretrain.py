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

# 忽略警告信息
warnings.filterwarnings('ignore')

# 定义日志打印函数，仅在主进程（rank 0）打印日志信息
def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)

# 定义学习率调度函数，根据当前迭代次数计算学习率，采用余弦退火策略
def get_lr(it, all):
    warmup_iters = 0  # 预热迭代次数
    lr_decay_iters = all  # 学习率衰减的总迭代次数
    min_lr = learning_rate / 10  # 最小学习率

    # 如果当前迭代次数小于预热迭代次数，使用线性预热策略
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 如果当前迭代次数大于衰减迭代次数，返回最小学习率
    if it > lr_decay_iters:
        return min_lr
    # 计算衰减系数，使用余弦退火策略
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# 定义训练 epoch 的函数
def train_epoch(epoch, accumulation_steps=8):
    start_time = time.time()  # 记录开始时间
    for step, (X, Y) in enumerate(train_loader):  # 遍历数据加载器
        X = X.to(device)  # 将输入数据移动到设备上
        Y = Y.to(device)  # 将目标数据移动到设备上

        lr = get_lr(epoch * iter_per_epoch + step, epochs * iter_per_epoch)  # 计算当前学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr  # 设置优化器的学习率

        with ctx:  # 使用混合精度训练（如果设备是 GPU）
            out = model(X, Y)  # 前向传播，计算输出
            loss = out.last_loss / accumulation_steps  # 计算损失，并进行梯度累积

        scaler.scale(loss).backward()  # 反向传播，计算梯度

        # 每 accumulation_steps 步进行一次梯度更新
        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)  # 反缩放梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪

            scaler.step(optimizer)  # 更新模型参数
            scaler.update()  # 更新缩放器

            optimizer.zero_grad(set_to_none=True)  # 清空梯度

        # 每 100 步打印一次训练信息
        if step % 100 == 0:
            spend_time = time.time() - start_time  # 计算已用时间
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                    epoch,
                    epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

        # 每 1000 步保存一次模型
        if (step + 1) % 1000 == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()  # 切换到评估模式
            # torch.save(model.state_dict(), '{}/iter_{}.pth'.format(save_dir, int(step + epoch * iter_per_epoch)))
            moe_path = '_moe' if lm_config.use_moe else ''  # 根据是否使用 MoE 设置保存路径
            ckp = f'{save_dir}/pretrain_{lm_config.dim}{moe_path}.pth'

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()  # 获取模型状态字典
            else:
                state_dict = model.state_dict()

            torch.save(state_dict, ckp)  # 保存模型
            model.train()  # 切换回训练模式

# 定义初始化模型的函数
def init_model():
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)  # 计算模型可训练参数的数量

    # 初始化模型
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

    Logger(f'LLM总参数量：{count_parameters(model) / 1e6:.3f} 百万')  # 打印模型总参数量
    return model

# 定义初始化分布式训练环境的函数
def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")  # 初始化分布式进程组，使用 NCCL 后端
    ddp_rank = int(os.environ["RANK"])  # 获取当前进程的 rank
    ddp_local_rank = int(os.environ["LOCAL_RANK"])  # 获取当前进程的本地 rank
    ddp_world_size = int(os.environ["WORLD_SIZE"])  # 获取分布式训练的总进程数
    DEVICE = f"cuda:{ddp_local_rank}"  # 设置当前设备的 CUDA 设备
    torch.cuda.set_device(DEVICE)  # 设置当前设备的 CUDA 设备


# torchrun --nproc_per_node 2 1-pretrain.py
# I/O
if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    lm_config = LMConfig()  # 加载配置文件
    max_seq_len = lm_config.max_seq_len  # 获取最大序列长度
    out_dir = 'out'  # 设置输出目录
    epochs = 20  # 设置训练 epoch 数
    batch_size = 64  # 设置批量大小
    learning_rate = 2e-4  # 设置初始学习率
    device = 'cuda:0'  # 设置设备为 CUDA:0
    dtype = 'bfloat16'  # 设置数据类型为 bfloat16
    save_dir = os.path.join(out_dir)  # 设置模型保存目录
    os.makedirs(save_dir, exist_ok=True)  # 创建模型保存目录
    os.makedirs(out_dir, exist_ok=True)  # 创建输出目录
    tokens_per_iter = batch_size * max_seq_len  # 计算每个迭代处理的 token 数量
    torch.manual_seed(1337)  # 设置随机种子
    device_type = device if "cuda" in device else "cpu"  # 设置设备类型
    ctx = (
        nullcontext()  # 如果设备是 CPU，使用 nullcontext
        if device_type == "cpu"
        else torch.cuda.amp.autocast()  # 如果设备是 GPU，使用混合精度训练
    )
    ddp = int(os.environ.get("RANK", -1)) != -1  # 判断是否为分布式训练
    ddp_local_rank, DEVICE = 0, "cuda:0"  # 初始化分布式训练的本地 rank 和设备
    if ddp:
        init_distributed_mode()  # 初始化分布式训练环境
        device = torch.device(DEVICE)  # 设置设备
    # -----------------------------------------------------------------------------

    # -----init dataloader------
    data_path_list = ['./dataset/pretrain_data.bin']  # 设置数据路径
    train_ds = PretrainDataset(data_path_list, max_length=max_seq_len, memmap=True)  # 初始化数据集
    train_sampler = DistributedSampler(train_ds) if ddp else None  # 如果是分布式训练，使用分布式采样器
    num_workers = 8  # 设置数据加载器的 num_workers
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=num_workers,
        sampler=train_sampler
    )  # 初始化数据加载器

    # init model
    model = init_model()  # 初始化模型

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == dtype))  # 初始化梯度缩放器
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 初始化优化器
    # compile the model
    if False and platform.system() != 'Windows' and float(torch.__version__.split('.')[0]) >= 2:
        Logger("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)  # 编译模型（如果条件满足）

    if ddp:
        # Ignore the freqs_cis buffer so that DDP does not broadcast it at
        # construction time since NCCL does not support ComplexFloat
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}  # 设置 DDP 忽略的参数和缓冲区
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])  # 使用 DDP 包装模型

    # training loop
    iter_per_epoch = len(train_loader)  # 计算每个 epoch 的迭代次数
    for epoch in range(epochs):  # 遍历每个 epoch
        train_epoch(epoch)  # 训练一个 epoch