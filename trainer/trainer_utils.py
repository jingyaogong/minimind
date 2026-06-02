"""
训练工具函数集合
"""
import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_minimind_mla import MiniMindMLAConfig, MiniMindMLAForCausalLM

DEFAULT_DEEPSPEED_CONFIG = os.path.join(os.path.dirname(__file__), "ds_config_zero2.json")


def _resolve_repo_path(path):
    if os.path.isabs(path) and os.path.exists(path):
        return path
    if os.path.exists(path):
        return path
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    candidate = os.path.join(repo_root, path)
    return candidate


def add_model_profile_args(parser):
    parser.add_argument("--model_profile", type=str, default="", help="SearchLM模型规模配置名，如 SearchLM-100M/SearchLM-300M")
    parser.add_argument("--model_profiles_path", type=str, default="configs/searchlm_profiles.json", help="模型规模配置文件")


def add_deepspeed_args(parser, default_config=DEFAULT_DEEPSPEED_CONFIG):
    parser.add_argument("--deepspeed_config", type=str, default=default_config, help="DeepSpeed ZeRO配置JSON")
    parser.add_argument("--local_rank", type=int, default=-1, help="DeepSpeed launcher自动注入")


def apply_model_profile(args):
    """将 SearchLM profile 写入 argparse args；显式传空则不修改。"""
    profile_name = getattr(args, "model_profile", "")
    if not profile_name:
        return None
    profiles_path = _resolve_repo_path(getattr(args, "model_profiles_path", "configs/searchlm_profiles.json"))
    with open(profiles_path, "r", encoding="utf-8") as f:
        profiles = json.load(f)
    if profile_name not in profiles:
        raise ValueError(f"Unknown model_profile={profile_name}. Available: {', '.join(profiles)}")
    profile = profiles[profile_name]
    if "hidden_size" not in profile:
        raise ValueError(f"Profile {profile_name} is not a trainable from-scratch profile")
    fields = [
        "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "attention_type", "intermediate_size", "kv_lora_rank", "q_lora_rank", "rope_dim", "max_seq_len"
    ]
    for field in fields:
        if field in profile and hasattr(args, field):
            setattr(args, field, profile[field])
    return profile

def get_model_suffix(lm_config):
    """根据模型配置返回 checkpoint 文件名后缀"""
    suffix = ''
    if getattr(lm_config, 'use_moe', False): suffix += '_moe'
    attention_type = getattr(lm_config, "attention_type", "gqa")
    if isinstance(lm_config, MiniMindMLAConfig) or attention_type == "mla":
        suffix += '_mla'
    elif attention_type in {"mha", "mqa"}:
        suffix += f'_{attention_type}'
    return suffix


def unwrap_model(model):
    """兼容 DDP / DeepSpeed / torch.compile 的模型解包。"""
    raw_model = getattr(model, "module", model)
    return getattr(raw_model, "_orig_mod", raw_model)


def save_model_weights(lm_config, model, save_dir="../out", weight="model", dtype=torch.float16):
    """保存一份轻量推理权重；DeepSpeed完整训练状态由 save_checkpoint 单独保存。"""
    os.makedirs(save_dir, exist_ok=True)
    model_suffix = get_model_suffix(lm_config)
    ckp_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{model_suffix}.pth"
    raw_model = unwrap_model(model)
    state_dict = raw_model.state_dict()
    state_dict = {k: v.detach().to(dtype).cpu() for k, v in state_dict.items()}
    tmp_path = ckp_path + ".tmp"
    torch.save(state_dict, tmp_path)
    os.replace(tmp_path, ckp_path)
    del state_dict
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ckp_path


def load_deepspeed_config(args, total_steps):
    """加载并补全 DeepSpeed 配置，保持 batch / lr / precision 与 CLI 参数一致。"""
    ds_path = _resolve_repo_path(getattr(args, "deepspeed_config", DEFAULT_DEEPSPEED_CONFIG))
    with open(ds_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    world_size = dist.get_world_size() if dist.is_initialized() else int(os.environ.get("WORLD_SIZE", "1"))
    if cfg.get("train_micro_batch_size_per_gpu") == "auto":
        cfg["train_micro_batch_size_per_gpu"] = args.batch_size
    if cfg.get("gradient_accumulation_steps") == "auto":
        cfg["gradient_accumulation_steps"] = args.accumulation_steps
    if cfg.get("train_batch_size") == "auto":
        cfg["train_batch_size"] = args.batch_size * args.accumulation_steps * world_size
    if "gradient_clipping" in cfg:
        cfg["gradient_clipping"] = args.grad_clip

    optimizer_cfg = cfg.setdefault("optimizer", {"type": "AdamW", "params": {}})
    optimizer_cfg.setdefault("params", {})["lr"] = args.learning_rate

    scheduler_cfg = cfg.get("scheduler")
    if scheduler_cfg and scheduler_cfg.get("type") == "WarmupDecayLR":
        sched_params = scheduler_cfg.setdefault("params", {})
        if sched_params.get("warmup_max_lr") == "auto":
            sched_params["warmup_max_lr"] = args.learning_rate
        if sched_params.get("warmup_num_steps") == "auto":
            sched_params["warmup_num_steps"] = max(1, total_steps // 20)
        if sched_params.get("total_num_steps") == "auto":
            sched_params["total_num_steps"] = max(1, total_steps)

    dtype = getattr(args, "dtype", "float16")
    cfg.setdefault("fp16", {})["enabled"] = dtype == "float16"
    cfg.setdefault("bf16", {})["enabled"] = dtype == "bfloat16"
    return cfg


def get_wandb_id(wandb=None):
    if not wandb:
        return None
    if hasattr(wandb, "get_run"):
        run = wandb.get_run()
        return getattr(run, "id", None) if run else None
    return getattr(wandb, "id", None)


def get_cuda_peak_memory_gb():
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated(torch.cuda.current_device()) / (1024 ** 3)


def reduce_metrics(metrics, average=True):
    """聚合所有 rank 的标量指标，避免日志只反映 rank0 局部 batch。"""
    if not dist.is_initialized():
        return metrics
    numeric_items = [(key, value) for key, value in metrics.items() if isinstance(value, (int, float))]
    if not numeric_items:
        return metrics
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    values = torch.tensor([float(value) for _, value in numeric_items], device=device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    if average:
        values /= dist.get_world_size()
    reduced = dict(metrics)
    for (key, _), value in zip(numeric_items, values.tolist()):
        reduced[key] = value
    return reduced


def get_deepspeed_tag(lm_config, weight):
    return f"{weight}_{lm_config.hidden_size}{get_model_suffix(lm_config)}"


def get_deepspeed_step_tag(lm_config, weight, epoch, step):
    return f"{get_deepspeed_tag(lm_config, weight)}_epoch{epoch}_step{step}"


def get_deepspeed_latest_file(lm_config, weight, save_dir="../checkpoints"):
    return os.path.join(save_dir, f"{get_deepspeed_tag(lm_config, weight)}_latest")


def _read_deepspeed_latest_tag(lm_config, weight, save_dir="../checkpoints"):
    latest_file = get_deepspeed_latest_file(lm_config, weight, save_dir)
    if not os.path.exists(latest_file):
        return None
    with open(latest_file, "r", encoding="utf-8") as f:
        tag = f.read().strip()
    return tag or None


def _write_deepspeed_latest_tag(lm_config, weight, tag, save_dir="../checkpoints"):
    latest_file = get_deepspeed_latest_file(lm_config, weight, save_dir)
    tmp_file = latest_file + ".tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        f.write(tag)
    os.replace(tmp_file, latest_file)


def _candidate_deepspeed_tags(lm_config, weight, save_dir="../checkpoints"):
    base_tag = get_deepspeed_tag(lm_config, weight)
    candidates = []
    latest_tag = _read_deepspeed_latest_tag(lm_config, weight, save_dir)
    if latest_tag:
        candidates.append(latest_tag)
    candidates.append(base_tag)
    if os.path.exists(save_dir):
        step_tags = []
        prefix = f"{base_tag}_epoch"
        for name in os.listdir(save_dir):
            path = os.path.join(save_dir, name)
            if name.startswith(prefix) and os.path.isdir(path):
                step_tags.append((os.path.getmtime(path), name))
        candidates.extend(name for _, name in sorted(step_tags, reverse=True))
    dedup = []
    seen = set()
    for tag in candidates:
        if tag not in seen:
            seen.add(tag)
            dedup.append(tag)
    return dedup


def save_deepspeed_checkpoint(model_engine, lm_config, weight, epoch=0, step=0, wandb=None, save_dir="../checkpoints", **kwargs):
    """DeepSpeed checkpoint 必须所有 rank 都调用，不能只在 rank0 保存。"""
    os.makedirs(save_dir, exist_ok=True)
    wandb_id = get_wandb_id(wandb) if is_main_process() else None
    if dist.is_initialized():
        wandb_box = [wandb_id]
        dist.broadcast_object_list(wandb_box, src=0)
        wandb_id = wandb_box[0]
    client_state = {
        "epoch": epoch,
        "step": step,
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "wandb_id": wandb_id,
    }
    client_state.update({k: v for k, v in kwargs.items() if v is not None})
    tag = get_deepspeed_step_tag(lm_config, weight, epoch, step)
    model_engine.save_checkpoint(save_dir, tag=tag, client_state=client_state)
    if is_main_process():
        _write_deepspeed_latest_tag(lm_config, weight, tag, save_dir)
        Logger(f"DeepSpeed checkpoint saved: {save_dir}/{tag}")
    return tag


def load_deepspeed_checkpoint(model_engine, lm_config, weight, save_dir="../checkpoints"):
    errors = []
    for tag in _candidate_deepspeed_tags(lm_config, weight, save_dir):
        try:
            load_path, client_state = model_engine.load_checkpoint(save_dir, tag=tag)
        except Exception as exc:
            errors.append(f"{tag}: {exc}")
            continue
        if load_path is None:
            errors.append(f"{tag}: load_path=None")
            continue
        break
    else:
        if is_main_process():
            Logger("DeepSpeed checkpoint not loaded" + (f": {'; '.join(errors)}" if errors else ""))
        return None
    saved_ws = client_state.get("world_size", 1) if client_state else 1
    current_ws = dist.get_world_size() if dist.is_initialized() else 1
    if saved_ws != current_ws and is_main_process():
        Logger(f"DeepSpeed checkpoint world_size changed: {saved_ws} -> {current_ws}")
    if is_main_process():
        Logger(f"DeepSpeed checkpoint loaded: {load_path}")
    return client_state or {}


def build_lm_config(hidden_size=768, num_hidden_layers=8, use_moe=False, attention_type="gqa", kv_lora_rank=128, **kwargs):
    """统一构建不同注意力结构的 MiniMind 配置。"""
    if attention_type == "mla":
        return MiniMindMLAConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            use_moe=use_moe,
            kv_lora_rank=kv_lora_rank,
            **kwargs
        )
    if attention_type not in {"gqa", "mha", "mqa"}:
        raise ValueError(f"Unsupported attention_type: {attention_type}")
    return MiniMindConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        use_moe=use_moe,
        attention_type=attention_type,
        **kwargs
    )


def resolve_attention_type(args):
    """兼容旧参数 --use_mla，同时支持新的 --attention_type。"""
    if getattr(args, "use_mla", 0):
        return "mla"
    return getattr(args, "attention_type", "gqa")


def get_model_params(model, config):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    #getattr用于动态获取对象的属性或方法
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')


def log_training_setup(args, lm_config, stage="train", dataset_len=None, iters=None, tokens_per_sample=None, extra=None):
    """打印可直接沉淀到实验报告里的 DDP/训练配置信息。"""
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    per_gpu_batch = getattr(args, "batch_size", 1)
    accumulation = getattr(args, "accumulation_steps", 1)
    effective_batch = per_gpu_batch * world_size * accumulation
    max_seq_len = tokens_per_sample or getattr(args, "max_seq_len", None)
    tokens_per_step = effective_batch * max_seq_len if max_seq_len else None
    attention_type = getattr(lm_config, "attention_type", "gqa")
    device = getattr(args, "device", "unknown")
    gpu_name = None
    if torch.cuda.is_available() and "cuda" in str(device):
        try:
            gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        except Exception:
            gpu_name = None

    if rank == 0:
        Logger("=" * 80)
        Logger(f"[Training Setup] stage={stage}")
        backend = getattr(args, "training_backend", None)
        if backend is None:
            backend = "DDP" if world_size > 1 else "single-process"
        Logger(
            f"  distributed={backend}, world_size={world_size}, "
            f"device={device}" + (f", gpu={gpu_name}" if gpu_name else "")
        )
        Logger(
            f"  model: attention_type={attention_type}, hidden_size={lm_config.hidden_size}, "
            f"num_layers={lm_config.num_hidden_layers}, use_moe={getattr(lm_config, 'use_moe', False)}"
        )
        if attention_type == "mla" or isinstance(lm_config, MiniMindMLAConfig):
            Logger(
                f"  mla: kv_lora_rank={lm_config.kv_lora_rank}, q_lora_rank={lm_config.q_lora_rank}, "
                f"rope_dim={lm_config.rope_dim}"
            )
        Logger(
            f"  batch: per_gpu={per_gpu_batch}, accumulation_steps={accumulation}, "
            f"effective_batch={effective_batch}"
        )
        if tokens_per_step:
            Logger(f"  sequence: max_seq_len={max_seq_len}, tokens_per_optimizer_step~{tokens_per_step:,}")
        if dataset_len is not None:
            Logger(f"  data: samples={dataset_len:,}" + (f", batches_per_epoch={iters:,}" if iters is not None else ""))
        Logger(
            f"  precision={getattr(args, 'dtype', 'unknown')}, lr={getattr(args, 'learning_rate', 'unknown')}, "
            f"epochs={getattr(args, 'epochs', 'unknown')}, compile={getattr(args, 'use_compile', 0)}"
        )
        if extra:
            for key, value in extra.items():
                Logger(f"  {key}: {value}")
        Logger("=" * 80)


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
    random.seed(seed)#设置python的随机种子
    np.random.seed(seed)#设置numpy的随机种子
    torch.manual_seed(seed)#设置cpu随机种子
    torch.cuda.manual_seed(seed)#设置pytorch在当前GPU上的随机种子
    torch.cuda.manual_seed_all(seed)#设置pytorch在所有GPU上的随机种子
    torch.backends.cudnn.deterministic = True#使用确定性算法
    torch.backends.cudnn.benchmark = False#禁用cudnn的自动优化

def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    moe_path = get_model_suffix(lm_config)
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:#存储模型模式
        #如果模型被DDP包了，真正的模型在model.module里
        raw_model = unwrap_model(model)
        state_dict = raw_model.state_dict()
        #half表示转成float16，然后移到cpu上
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        #os.replace用于将一个文件替换成另一个文件 os.replace(src, dst) 将 src 文件替换为 dst 文件，如果 dst 已经存在，则会被替换掉。这是一个原子操作，确保在替换过程中不会出现文件损坏或丢失的情况。
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
            #训练时用了几张GPU
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = unwrap_model(value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        #删除临时变量，然后清理cuda缓存
        del state_dict, resume_data
        torch.cuda.empty_cache()
    else:  # 加载模式
        if os.path.exists(resume_path):
            #加载到gpu
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
    if isinstance(lm_config, MiniMindMLAConfig):
        model = MiniMindMLAForCausalLM(lm_config)
    else:
        model = MiniMindForCausalLM(lm_config)

    if from_weight!= 'none':
        model_suffix = get_model_suffix(lm_config)
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{model_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer

#对sampler的重写，主要重写iter方法和len方法，sampler决定取哪些数据，给dataloaer拿到索引后，通过getitem方法取数据
class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        #self.sampler每次吐出一个样本的索引
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


class LMForRewardModel:
    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def get_score(self, messages, response):
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])
        last_query = messages[-1]['content'] if messages else ""
        message_context = f"{history_text}\n以上是对话历史。我的新问题是：\n{last_query}" if history_text else last_query
        eval_messages = [
            {"role": "user", "content": message_context},
            {"role": "assistant", "content": response}
        ]
        score = self.model.get_score(self.tokenizer, eval_messages)
        #clip到[-3,3]
        return max(min(score, 3.0), -3.0)
