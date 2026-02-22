import os  # 导入 os，用于路径/文件操作
import sys  # 导入 sys，用于修改模块搜索路径

__package__ = "trainer"  # 指定当前脚本所属包，便于相对导入
# 把项目根目录加入 Python 路径，方便直接运行本脚本时能找到上层模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # 将上级目录加入 sys.path

import argparse  # 命令行参数解析
import time  # 计时
import warnings  # 控制警告输出
import torch  # PyTorch 主库
import torch.distributed as dist  # 分布式训练工具
from contextlib import nullcontext  # 空上下文管理器（CPU 下不用 autocast）
from torch import optim, nn  # 优化器与神经网络模块
from torch.nn.parallel import DistributedDataParallel  # DDP 封装
from torch.utils.data import DataLoader, DistributedSampler  # 数据加载与分布式采样
from model.model_minimind import MiniMindConfig  # 模型配置类
from dataset.lm_dataset import PretrainDataset  # 预训练数据集类
from trainer.trainer_utils import (  # 训练工具函数集合
    get_lr,  # 计算学习率
    Logger,  # 日志打印（主进程）
    is_main_process,  # 判断是否主进程
    lm_checkpoint,  # 保存/读取断点
    init_distributed_mode,  # 初始化分布式环境
    setup_seed,  # 设置随机种子
    init_model,  # 初始化模型与分词器
    SkipBatchSampler,  # 可跳过 batch 的采样器
    init_neuron_mask,  # 初始化神经元 mask
    set_neuron_tracking,  # 控制活动/梯度统计
    grow_neurons,  # 动态激活神经元
    save_run_config,  # 保存实验配置
    update_run_config,  # 更新实验配置
    get_active_ratio_by_layer,  # 按层统计激活比例
    get_active_ratio_stats  # 激活比例统计
)  # 结束导入列表

warnings.filterwarnings('ignore')  # 忽略警告信息

# =============================================================================
# 新手必读（最重要的几个角度）：
# 1) 训练是否能跑通，最常见的问题是 data_path 指向的文件不存在。
# 2) OOM（显存不够）优先降低：batch_size -> max_seq_len -> hidden_size/num_hidden_layers。
# 3) loss 不下降/变成 NaN：先把学习率降 10 倍试试，再检查数据质量。
# 4) 单卡/多卡差别：多卡时用 DistributedSampler + set_epoch，且只主进程保存。
# 5) 混合精度：bfloat16 更稳，float16 更快但更容易数值不稳。
# =============================================================================

# =============================================================================
# 本脚本的训练主线（给新手看的超简版）：
# 1) 读取 jsonl 数据 -> token -> input_ids/labels
# 2) 模型前向得到 loss
# 3) loss.backward() 反向传播
# 4) optimizer.step() 更新参数
# 5) 周期性打印日志、保存权重、保存断点
# =============================================================================


def get_neuron_active_ratio(model):  # 统计当前激活神经元比例
    total = 0  # 总神经元数
    active = 0  # 已激活神经元数
    for m in model.modules():  # 遍历所有模块
        if hasattr(m, "mask"):  # 只统计带 mask 的 FFN
            total += m.mask.numel()
            active += int(m.mask.sum().item())
    return (active / total) if total > 0 else 1.0  # 返回比例（若无 mask 则视为 1）


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):  # 训练一个 epoch 的函数
    """
    训练一个 epoch
    - epoch: 当前第几个 epoch（从 0 开始）
    - loader: DataLoader（会产出 batch）
    - iters: 这个 epoch 里总的 step 数
    - start_step: 断点续训时，跳过前面已经训练过的 step 数
    - wandb: 记录日志用（可选）
    """
    # 记录本 epoch 开始时间，用于估算剩余时间
    start_time = time.time()  # 当前时间戳
    tokens_seen = 0  # 统计已处理 token 数，用于吞吐率
    # 这里的 step 从 start_step + 1 开始计数，日志更直观（人类习惯从 1 开始）
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):  # 遍历 DataLoader 的每个 batch
        # 1) 把数据搬到指定设备（GPU/CPU）
        # input_ids/labels 形状通常是 (batch_size, seq_len)
        # 例：batch_size=2, seq_len=128 -> shape=(2, 128)
        input_ids = input_ids.to(args.device)  # 把输入 token 放到 GPU/CPU
        labels = labels.to(args.device)  # 把监督标签放到 GPU/CPU
        tokens_seen += input_ids.numel()  # 统计 tokens（含 padding，用于吞吐率）

        # 计算全局步数与是否需要“生长”神经元
        global_step = epoch * iters + step  # 全局 step（从 1 开始）
        should_update = ((step + 1) % args.accumulation_steps == 0)  # 本步是否会更新参数
        update_step = global_step // args.accumulation_steps  # 优化器更新步数（从 0 开始）
        should_grow = bool(args.neuron_growth) and should_update and (update_step > 0) and (update_step % args.grow_interval == 0)

        # 根据需要开启/关闭活动与 mask 梯度追踪
        if args.neuron_growth:
            track_activity = (args.grow_method != "random")  # 随机增长不需要活动统计
            track_grad = (args.grow_method != "random") and should_grow  # 只有在增长步才追踪 mask 梯度
            set_neuron_tracking(model, track_activity=track_activity, track_mask_grad=track_grad, ema_beta=args.neuron_ema_beta)

        # 2) 计算当前 step 的学习率（这里用余弦衰减）
        #    说明：学习率不是固定的，会随训练进度变化
        #    current_step = epoch * iters + step 是“全局步数”
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)  # 计算当前学习率
        for param_group in optimizer.param_groups:  # 遍历优化器的参数组
            param_group['lr'] = lr  # 设置该参数组的学习率

        # 3) 前向 + 计算 loss（混合精度可选）
        with autocast_ctx:  # GPU 下启用混合精度，CPU 下为空上下文
            # res 是模型输出对象，包含 loss / logits / aux_loss 等
            res = model(input_ids, labels=labels)  # 前向计算（内部会算 loss）
            # res.logits 形状通常是 (batch_size, seq_len, vocab_size)
            # 例：batch_size=2, seq_len=128, vocab=6400 -> shape=(2, 128, 6400)
            # loss 在模型内部计算，核心是“预测下一个 token”
            # shift_logits: (batch_size, seq_len-1, vocab_size)
            # shift_labels: (batch_size, seq_len-1)
            # res.loss 是语言模型的交叉熵损失（预测下一个 token）
            # res.aux_loss 仅在 MoE 模型中存在，用于专家负载均衡
            # 主损失 + MoE 的辅助损失（如果有）
            loss = res.loss + res.aux_loss  # 总损失
            # 梯度累积：把 loss 平均分摊到多次小步
            # 这样累积 N 次的梯度，等价于“大 batch”的效果
            # 等效总 batch_size = batch_size * accumulation_steps * world_size
            loss = loss / args.accumulation_steps  # 按累积步数缩放 loss

        # 4) 反向传播：把 loss 的梯度累积到参数上
        # 注意：这里不会立刻更新参数，只是把梯度累积在参数上
        scaler.scale(loss).backward()  # 反向传播（混合精度下用 scaler）

        # 5) 每积累一定步数才更新一次参数
        #    accumulation_steps=8 表示每 8 个小步更新一次参数
        if (step + 1) % args.accumulation_steps == 0:  # 到达累积步数时更新参数
            # 反缩放后再裁剪梯度，避免梯度爆炸
            # clip_grad_norm_ 会限制梯度范数不超过 args.grad_clip
            scaler.unscale_(optimizer)  # 先取消缩放，得到真实梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 裁剪梯度

            # 更新参数
            scaler.step(optimizer)  # 执行一步优化器更新
            scaler.update()  # 更新缩放比例

            # 动态激活更多神经元（仅在指定步数触发）
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
                # 生长完成后关闭 mask 梯度追踪，避免额外开销
                set_neuron_tracking(model, track_activity=(args.grow_method != "random"), track_mask_grad=False, ema_beta=args.neuron_ema_beta)

            # 清空梯度，进入下一轮
            optimizer.zero_grad(set_to_none=True)  # 清零梯度，节省显存

        # 6) 日志打印
        if step % args.log_interval == 0 or step == iters - 1:  # 按间隔或最后一步打印
            spend_time = time.time() - start_time  # 已花时间（秒）
            # 注意：这里把 loss 乘回来，恢复到“真实的单步损失”
            # 因为前面为了梯度累积把 loss 除过
            current_loss = loss.item() * args.accumulation_steps  # 真实损失值
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0  # MoE 辅助损失
            current_logits_loss = current_loss - current_aux_loss  # 语言模型主损失
            current_lr = optimizer.param_groups[-1]['lr']  # 当前学习率
            # 估算当前 epoch 剩余时间（分钟）
            # eta = 已花时间 / 已完成步数 * 总步数 - 已花时间
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60  # 估算剩余分钟数
            tokens_per_sec = tokens_seen / max(spend_time, 1e-6)  # 吞吐率（tokens/s）
            # 如果启用了动态生长，额外记录当前激活比例
            active_ratio = get_neuron_active_ratio(model) if args.neuron_growth else None
            active_stats = get_active_ratio_stats(model) if args.neuron_growth else None
            active_msg = f', active_ratio: {active_ratio:.3f}' if active_ratio is not None else ''
            if active_stats:
                active_msg += f", active_mean: {active_stats['mean']:.3f}, active_min: {active_stats['min']:.3f}, active_max: {active_stats['max']:.3f}"
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min, tok/s: {tokens_per_sec:.1f}{active_msg}')  # 打印日志
            if wandb:  # 如果启用 wandb
                log_data = {
                    "loss": current_loss,
                    "logits_loss": current_logits_loss,
                    "aux_loss": current_aux_loss,
                    "learning_rate": current_lr,
                    "epoch_time": eta_min,
                    "tokens_per_sec": tokens_per_sec
                }
                if active_ratio is not None:
                    log_data["active_ratio"] = active_ratio
                if active_stats:
                    log_data["active_ratio_mean"] = active_stats["mean"]
                    log_data["active_ratio_min"] = active_stats["min"]
                    log_data["active_ratio_max"] = active_stats["max"]
                # 逐层激活比例（便于画曲线）
                if args.neuron_growth:
                    layer_ratios = get_active_ratio_by_layer(model)
                    for name, ratio in layer_ratios.items():
                        key = name.replace(".", "_")
                        log_data[f"active_{key}"] = ratio
                wandb.log(log_data)  # 记录到 wandb

        # 7) 保存模型（只在主进程保存，避免多卡重复写文件）
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():  # 满足保存条件且是主进程
            # eval() 主要是关闭 dropout / 避免不一致
            model.eval()  # 切换到评估模式
            moe_suffix = '_moe' if lm_config.use_moe else ''  # MoE 模型在文件名加后缀
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'  # 权重保存路径
            # raw_model 兼容两种情况：
            # - DDP 包裹时，真实模型在 model.module 里
            # - torch.compile 包裹时，真实模型在 _orig_mod 里
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model  # 取出真实模型
            raw_model = getattr(raw_model, '_orig_mod', raw_model)  # 兼容 torch.compile
            state_dict = raw_model.state_dict()  # 获取模型参数字典
            # 只保存半精度权重，节省空间（推理时再转回）
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)  # 保存权重到硬盘
            # 保存断点信息（可用于恢复训练）
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')  # 保存断点
            model.train()  # 切回训练模式
            del state_dict  # 释放权重变量

        # 释放中间变量，减少显存占用
        # 对新手来说：这不是必须，但有助于长时间训练稳定
        del input_ids, labels, res, loss  # 删除临时变量

    return tokens_seen  # 返回本 epoch 处理的 token 数


if __name__ == "__main__":  # 仅在直接运行本脚本时执行以下代码
    # -------------------------
    # 0. 解析训练参数
    # -------------------------
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")  # 创建参数解析器
    # 保存相关参数
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")  # 输出权重目录
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")  # 权重前缀
    # 训练相关参数
    # batch_size：越大越快但越吃显存
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数（建议1轮zero或2-6轮充分训练）")  # 训练轮数
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")  # 每步样本数
    # learning_rate：影响收敛速度与稳定性，过大会震荡/发散
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")  # 初始学习率
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")  # 训练设备
    # dtype：bfloat16 更稳，float16 更快但风险更高
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")  # 混合精度类型
    # num_workers：数据加载并行数，太大可能反而拖慢
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")  # 数据加载线程数
    # accumulation_steps：梯度累积，等效扩大 batch_size（但会更慢）
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")  # 梯度累积步数
    # grad_clip：限制梯度范数，避免梯度爆炸
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")  # 梯度裁剪阈值
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")  # 日志间隔
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")  # 保存间隔
    parser.add_argument("--seed", type=int, default=42, help="随机种子（复现实验用）")  # 随机种子
    # 动态神经元生长相关参数（可选）
    parser.add_argument("--neuron_growth", default=0, type=int, choices=[0, 1], help="是否启用动态神经元生长")  # 是否开启
    parser.add_argument("--init_active_ratio", type=float, default=0.8, help="初始激活神经元比例")  # 初始激活比例
    parser.add_argument("--grow_method", type=str, default="random", choices=["random", "act_grad"], help="神经元生长方式")  # 生长方法
    parser.add_argument("--grow_interval", type=int, default=100, help="每隔多少次优化器更新触发生长")  # 生长间隔
    parser.add_argument("--grow_ratio", type=float, default=0.02, help="每次生长激活比例")  # 每次激活比例
    parser.add_argument("--max_active_ratio", type=float, default=0.99, help="最多激活到多少比例")  # 最大激活比例
    parser.add_argument("--grow_score_alpha", type=float, default=1.0, help="活动分数权重(EMA)")  # 活动权重
    parser.add_argument("--grow_score_beta", type=float, default=1.0, help="梯度分数权重")  # 梯度权重
    parser.add_argument("--neuron_ema_beta", type=float, default=0.1, help="活动EMA系数")  # EMA 衰减系数
    # 模型结构参数
    # hidden_size/num_hidden_layers 越大：参数越多、训练越慢、显存越高
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")  # 隐藏层维度
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")  # 层数
    # max_seq_len 越大：上下文更长，但显存/算力增长很快
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")  # 最大序列长度
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")  # 是否启用 MoE
    # 数据与权重加载
    # data_path 必须是 jsonl，且每行包含 {"text": "..."}
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl", help="预训练数据路径")  # 数据文件路径
    # from_weight：从已有权重继续训练；none 表示从头开始
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")  # 加载已有权重
    # from_resume：是否自动检测断点续训（保存/恢复 optimizer 状态）
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")  # 是否断点续训
    # 日志与加速
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")  # 是否启用 wandb
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")  # wandb 项目名
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")  # 是否启用 torch.compile
    args = parser.parse_args()  # 解析命令行参数

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()  # 初始化分布式环境并获取本地 rank
    if dist.is_initialized():  # 如果启用了分布式
        args.device = f"cuda:{local_rank}"  # 让当前进程绑定到对应 GPU
    # 每个进程使用不同随机种子，保证多卡可复现且不重复
    # 这样多卡不会完全“学到相同的 batch”
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))  # 设置随机种子
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)  # 创建模型保存目录
    # 构建模型配置（这里是最小的 MiniMind 配置）
    # hidden_size / num_hidden_layers 决定模型规模（大=更慢=更耗显存）
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))  # 创建模型配置
    # 若启用断点续训，就尝试从 checkpoints 里读取
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume == 1 else None  # 读取断点
    # 保存本次实验配置（仅主进程），便于论文复现实验
    run_config_path = None
    if is_main_process():
        run_name = f"{args.save_weight}_{args.hidden_size}_{time.strftime('%Y%m%d_%H%M%S')}"
        run_config_path = save_run_config(args, args.save_dir, run_name=run_name, extra={"resume": bool(ckp_data)})  # 写入配置文件
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"  # 判断设备类型
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16  # 选择混合精度类型
    # CPU 不使用混合精度；GPU 才启用 autocast
    # bfloat16 相对更稳定；float16 更快但可能不稳定
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)  # 设置 autocast 上下文
    
    # ========== 4. 配wandb ==========
    wandb = None  # 默认不启用 wandb
    if args.use_wandb and is_main_process():  # 仅主进程初始化 wandb
        import swanlab as wandb  # swanlab 与 wandb API 兼容
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None  # 续训时复用 run id
        resume = 'must' if wandb_id else None  # 若有 id 则强制恢复
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"  # 生成 run 名
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)  # 初始化 wandb
    
    # ========== 5. 定义模型、数据、优化器 ==========
    # 载入模型与分词器（若指定 from_weight 则加载权重）
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)  # 创建模型与 tokenizer
    # 若启用动态神经元生长，先初始化 mask（仅在非断点恢复时）
    if args.neuron_growth and not ckp_data:
        init_neuron_mask(model, init_active_ratio=args.init_active_ratio, seed=42)
    # 设置活动/梯度统计开关（随机增长不需要活动统计）
    if args.neuron_growth:
        set_neuron_tracking(
            model,
            track_activity=(args.grow_method != "random"),
            track_mask_grad=False,
            ema_beta=args.neuron_ema_beta
        )
    if args.use_compile == 1:  # 是否启用 torch.compile
        model = torch.compile(model)  # 编译模型以加速
        Logger('torch.compile enabled')  # 打印提示
    # 读取预训练数据（jsonl）
    # 每一行都是 {"text": "..."} 的格式
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)  # 创建数据集
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None  # 多卡时使用分布式采样
    # 仅 float16 时启用 GradScaler；bfloat16 不需要
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))  # 混合精度缩放器
    # AdamW 是语言模型最常用的优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)  # 创建优化器
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0  # 默认从头开始
    if ckp_data:  # 如果找到断点
        # 恢复模型、优化器、混合精度状态
        model.load_state_dict(ckp_data['model'])  # 恢复模型权重
        optimizer.load_state_dict(ckp_data['optimizer'])  # 恢复优化器状态
        scaler.load_state_dict(ckp_data['scaler'])  # 恢复 scaler 状态
        start_epoch = ckp_data['epoch']  # 继续的 epoch
        start_step = ckp_data.get('step', 0)  # 继续的 step
    
    # ========== 7. DDP包模型 ==========
    if dist.is_initialized():  # 如果启用了分布式
        # 这两个 buffer 在 DDP 下不需要同步
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}  # 忽略这些 buffer
        # DDP 会自动帮我们做多卡梯度同步
        model = DistributedDataParallel(model, device_ids=[local_rank])  # 包装成 DDP 模型
    
    # ========== 8. 开始训练 ==========
    train_start = time.time()
    total_tokens = 0
    for epoch in range(start_epoch, args.epochs):  # 遍历 epoch
        # 多卡时每个 epoch 都要 set_epoch，保证采样不同
        train_sampler and train_sampler.set_epoch(epoch)  # 设置采样器的 epoch
        # 打乱索引（单卡）
        setup_seed(args.seed + epoch)  # 每个 epoch 设置不同随机种子
        indices = torch.randperm(len(train_ds)).tolist()  # 生成随机索引列表
        # 断点续训时跳过已经训练过的 step
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0  # 需要跳过的 step 数
        # 自定义 batch_sampler，用于跳过前面若干 batch
        # 注意：这里用 batch_sampler 时，DataLoader 不再传 batch_size
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)  # 创建可跳过的采样器
        # pin_memory=True 会让 CPU->GPU 拷贝更快
        # loader 每次返回的 input_ids/labels 形状是 (batch_size, max_seq_len)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)  # 构建 DataLoader
        if skip > 0:  # 如果需要跳过
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')  # 打印提示
            epoch_tokens = train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)  # 从指定 step 继续训练
        else:  # 正常从头开始
            epoch_tokens = train_epoch(epoch, loader, len(loader), 0, wandb)  # 正常训练
        total_tokens += epoch_tokens

    # 训练结束后更新配置文件（记录总 tokens 和耗时）
    if is_main_process() and run_config_path:
        update_run_config(run_config_path, {
            "train_time_sec": time.time() - train_start,
            "total_tokens": total_tokens,
            "final_active_ratio": get_neuron_active_ratio(model) if args.neuron_growth else None
        })
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():  # 如果启用了分布式
        dist.destroy_process_group()  # 关闭分布式进程组
