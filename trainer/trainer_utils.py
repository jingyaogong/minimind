"""
训练工具函数集合
"""
import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import math
import json
import time
import subprocess
import platform
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer
from model.model_minimind import MiniMindForCausalLM

def get_model_params(model, config):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_git_commit():
    """尽量获取当前 git commit（失败则返回 None）"""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            text=True
        )
        commit = res.stdout.strip()
        return commit if commit else None
    except Exception:
        return None


def save_run_config(args, save_dir, run_name="run", extra=None):
    """
    保存训练配置，便于复现实验
    - args: argparse.Namespace
    - save_dir: 保存目录
    - run_name: 文件名前缀
    - extra: 额外信息 dict
    """
    os.makedirs(save_dir, exist_ok=True)
    config = {
        "run_name": run_name,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "args": vars(args) if hasattr(args, "__dict__") else args,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "git_commit": _get_git_commit()
    }
    if extra:
        config.update(extra)
    path = os.path.join(save_dir, f"{run_name}_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    return path


def update_run_config(path, extra):
    """更新 run_config.json（追加训练完成后的统计信息）"""
    if not path or not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception:
        config = {}
    config.update(extra or {})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def get_active_ratio_by_layer(model):
    """
    返回每个带 mask 的 FFN 模块激活比例
    - key: 模块名称（如 model.layers.0.mlp）
    - value: 激活比例（0~1）
    """
    ratios = {}
    for name, m in model.named_modules():
        if hasattr(m, "mask"):
            total = m.mask.numel()
            if total > 0:
                ratios[name] = float(m.mask.sum().item() / total)
    return ratios


def get_active_ratio_stats(model):
    """返回激活比例的统计值（均值/最小/最大）"""
    ratios = list(get_active_ratio_by_layer(model).values())
    if not ratios:
        return None
    return {
        "mean": float(sum(ratios) / len(ratios)),
        "min": float(min(ratios)),
        "max": float(max(ratios))
    }

def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data
        torch.cuda.empty_cache()
    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model', save_dir='../out', device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)

    if from_weight!= 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer


# ===================== 动态神经元生长相关工具函数 =====================
def _iter_ffn_modules(model):
    """遍历所有带 mask 的 FFN 模块（Dense + MoE Experts 都会被包含）"""
    for m in model.modules():
        if hasattr(m, "mask") and hasattr(m, "ema_act"):
            yield m


def set_neuron_tracking(model, track_activity=False, track_mask_grad=False, ema_beta=0.1):
    """开启/关闭神经元活动与 mask 梯度的统计"""
    for m in _iter_ffn_modules(model):
        m.track_activity = track_activity
        m.track_mask_grad = track_mask_grad
        m.ema_beta = ema_beta


def init_neuron_mask(model, init_active_ratio=0.8, seed=42):
    """初始化神经元 mask（随机激活部分神经元）"""
    # 多卡时只在主进程生成，再广播
    if dist.is_initialized():
        if is_main_process():
            _init_neuron_mask_impl(model, init_active_ratio, seed)
        # 广播到所有进程，确保 mask 一致
        for m in _iter_ffn_modules(model):
            dist.broadcast(m.mask, src=0)
    else:
        _init_neuron_mask_impl(model, init_active_ratio, seed)


def _init_neuron_mask_impl(model, init_active_ratio, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    for m in _iter_ffn_modules(model):
        total = m.mask.numel()
        n_active = max(1, int(total * init_active_ratio))
        # 先清零，再随机选一部分置 1
        m.mask.zero_()
        idx = torch.randperm(total, generator=g)[:n_active].to(m.mask.device)
        m.mask[idx] = 1.0


def grow_neurons(
    model,
    method="random",
    grow_ratio=0.02,
    max_active_ratio=0.99,
    score_alpha=1.0,
    score_beta=1.0,
    seed=None
):
    """
    动态激活更多神经元
    - method: "random" 或 "act_grad"
    - grow_ratio: 每次激活的比例（相对于总神经元数）
    - max_active_ratio: 最多激活到多少比例
    - score_alpha/score_beta: 活动/梯度的权重
    - seed: 随机种子（用于 random）
    """
    # 多卡：所有进程同时计算（内部会 all_reduce），再统一广播以保证一致
    if dist.is_initialized():
        _grow_neurons_impl(model, method, grow_ratio, max_active_ratio, score_alpha, score_beta, seed)
        for m in _iter_ffn_modules(model):
            dist.broadcast(m.mask, src=0)
    else:
        _grow_neurons_impl(model, method, grow_ratio, max_active_ratio, score_alpha, score_beta, seed)


def _grow_neurons_impl(model, method, grow_ratio, max_active_ratio, score_alpha, score_beta, seed):
    # 用于随机选择的 generator
    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)

    for m in _iter_ffn_modules(model):
        mask = m.mask
        total = mask.numel()
        active = int(mask.sum().item())
        max_active = int(total * max_active_ratio)
        if active >= max_active:
            continue

        n_add = max(1, int(total * grow_ratio))
        n_add = min(n_add, max_active - active)
        if n_add <= 0:
            continue

        if method == "random":
            inactive_idx = (mask == 0).nonzero(as_tuple=False).flatten()
            if inactive_idx.numel() == 0:
                continue
            # 从未激活的神经元中随机选
            perm = torch.randperm(inactive_idx.numel(), generator=g)[:n_add].to(inactive_idx.device)
            chosen = inactive_idx[perm]
        else:
            # 活动 + 梯度加权得分
            score = torch.zeros_like(mask)
            if score_alpha > 0:
                score += score_alpha * m.ema_act
            if score_beta > 0 and m._mask_proxy is not None and m._mask_proxy.grad is not None:
                grad = m._mask_proxy.grad.detach().abs()
                # 多卡时梯度求平均（主进程决策）
                if dist.is_initialized():
                    dist.all_reduce(grad, op=dist.ReduceOp.SUM)
                    grad /= dist.get_world_size()
                score += score_beta * grad

            # 已激活的神经元设为极小，保证不会被选中
            score = score.clone()
            score[mask > 0] = -1e9
            chosen = torch.topk(score, k=n_add).indices

        mask[chosen] = 1.0


class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)
