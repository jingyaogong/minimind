import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import gc
import json
import math
import random
import warnings
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoTokenizer

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.opd_utils import (
    build_completion_mask,
    generalized_jsd_loss,
    use_on_policy_batch,
)
from trainer.rollout_engine import create_rollout_engine
from trainer.trainer_utils import (
    Logger,
    SkipBatchSampler,
    init_distributed_mode,
    is_main_process,
    lm_checkpoint,
    setup_seed,
)

warnings.filterwarnings("ignore")
REPO_ROOT = Path(__file__).resolve().parents[1]


class OPDDataset(Dataset):
    """Raw conversational samples used by both GKD data branches."""

    def __init__(self, data_path):
        self.samples = load_dataset("json", data_files=str(data_path), split="train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = sample.get("conversations", sample.get("messages"))
        if not isinstance(conversations, list) or not conversations:
            raise ValueError(
                "OPD data requires a non-empty 'conversations' or 'messages' list"
            )
        return {"conversations": [dict(message) for message in conversations]}


def collate_conversations(batch):
    return {"conversations": [item["conversations"] for item in batch]}


def resolve_device(device):
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_path(value):
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def validate_args(parsed_args):
    if not 0.0 <= parsed_args.lmbda <= 1.0:
        raise ValueError("--lmbda must be in [0, 1]")
    if not 0.0 <= parsed_args.beta <= 1.0:
        raise ValueError("--beta must be in [0, 1]")
    if parsed_args.temperature <= 0 or parsed_args.distill_temperature <= 0:
        raise ValueError("sampling and distillation temperatures must be positive")
    if not 0.0 < parsed_args.rollout_top_p <= 1.0:
        raise ValueError("--rollout_top_p must be in (0, 1]")
    if parsed_args.rollout_top_k < 0:
        raise ValueError("--rollout_top_k must be zero (disabled) or positive")
    if parsed_args.num_generations < 1 or parsed_args.max_gen_len < 1:
        raise ValueError("--num_generations and --max_gen_len must be positive")
    if parsed_args.max_seq_len < 1 or parsed_args.accumulation_steps < 1:
        raise ValueError("--max_seq_len and --accumulation_steps must be positive")
    if parsed_args.max_train_steps < 0:
        raise ValueError("--max_train_steps must be zero or positive")
    if parsed_args.log_interval < 1 or parsed_args.save_interval < 1:
        raise ValueError("log/save intervals must be positive")
    if parsed_args.rollout_top_p < 1.0 or parsed_args.rollout_top_k > 0:
        Logger(
            "[OPD WARNING] top-p/top-k truncation changes the behavior policy; "
            "top_p=1 and top_k=0 match the GKD baseline"
        )


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_transformers_state_dict(model_dir):
    single_safetensors = model_dir / "model.safetensors"
    safetensors_index = model_dir / "model.safetensors.index.json"
    pytorch_bin = model_dir / "pytorch_model.bin"
    if single_safetensors.exists() or safetensors_index.exists():
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "Loading a Transformers safetensors model requires safetensors"
            ) from exc
    if single_safetensors.exists():
        return load_file(str(single_safetensors), device="cpu")
    if safetensors_index.exists():
        index = json.loads(safetensors_index.read_text(encoding="utf-8"))
        state_dict = {}
        for shard_name in sorted(set(index["weight_map"].values())):
            state_dict.update(load_file(str(model_dir / shard_name), device="cpu"))
        return state_dict
    if pytorch_bin.exists():
        return _torch_load(pytorch_bin)
    raise FileNotFoundError(
        f"No model.safetensors, model.safetensors.index.json, or pytorch_model.bin in {model_dir}"
    )


def resolve_model_source(source, config):
    direct = Path(source).expanduser()
    if not direct.is_absolute():
        direct = REPO_ROOT / direct
    if direct.exists():
        return direct.resolve()
    moe_suffix = "_moe" if config.use_moe else ""
    weight_path = REPO_ROOT / "out" / f"{source}_{config.hidden_size}{moe_suffix}.pth"
    if weight_path.exists():
        return weight_path.resolve()
    raise FileNotFoundError(
        f"Model source {source!r} was not found directly or as {weight_path}"
    )


def load_state_dict(source, config):
    source = resolve_model_source(source, config)
    state_dict = (
        _load_transformers_state_dict(source) if source.is_dir() else _torch_load(source)
    )
    if isinstance(state_dict, dict) and isinstance(state_dict.get("model"), dict):
        state_dict = state_dict["model"]
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(f"Invalid or empty state dict from {source}")
    for prefix in ("module.", "_orig_mod."):
        if all(key.startswith(prefix) for key in state_dict):
            state_dict = {key[len(prefix):]: value for key, value in state_dict.items()}
    if "lm_head.weight" not in state_dict and "model.embed_tokens.weight" in state_dict:
        state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]
    if "model.embed_tokens.weight" not in state_dict and "lm_head.weight" in state_dict:
        state_dict["model.embed_tokens.weight"] = state_dict["lm_head.weight"]
    return state_dict, source


def validate_directory_config(source, config, label):
    source_path = Path(source).expanduser()
    if not source_path.is_absolute():
        source_path = REPO_ROOT / source_path
    config_path = source_path / "config.json" if source_path.is_dir() else None
    if not config_path or not config_path.exists():
        return
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "vocab_size": config.vocab_size,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
    }
    mismatches = {
        key: (saved.get(key), value)
        for key, value in expected.items()
        if saved.get(key) is not None and saved.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{label} config does not match CLI architecture: {mismatches}")


def load_native_model(config, source, device, label, model_dtype=None):
    validate_directory_config(source, config, label)
    state_dict, resolved_source = load_state_dict(source, config)
    loaded_model = MiniMindForCausalLM(config)
    loaded_model.load_state_dict(state_dict, strict=True)
    del state_dict
    gc.collect()
    if model_dtype is None:
        loaded_model = loaded_model.to(device)
    else:
        loaded_model = loaded_model.to(device=device, dtype=model_dtype)
    Logger(f"Loaded {label} strictly from {resolved_source}")
    return loaded_model


def load_tokenizer(tokenizer_path, vocab_size):
    tokenizer = AutoTokenizer.from_pretrained(
        str(resolve_path(tokenizer_path)), trust_remote_code=True
    )
    if len(tokenizer) != vocab_size:
        raise ValueError(
            f"Tokenizer size {len(tokenizer)} does not match model vocab {vocab_size}"
        )
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("OPD requires tokenizer.pad_token_id and tokenizer.eos_token_id")
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    return tokenizer


def _parse_json_field(value):
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def render_sample(tokenizer, conversations):
    """Render one prompt and its optional fixed completion."""
    messages = []
    tools = None
    for raw_message in conversations:
        message = {key: value for key, value in dict(raw_message).items() if value is not None}
        if message.get("tools"):
            parsed_tools = _parse_json_field(message["tools"])
            if isinstance(parsed_tools, list):
                tools = parsed_tools
        if message.get("tool_calls"):
            message["tool_calls"] = _parse_json_field(message["tool_calls"])
        messages.append(message)

    has_completion = messages[-1].get("role") == "assistant" and any(
        messages[-1].get(field) for field in ("content", "reasoning_content", "tool_calls")
    )
    prompt_messages = messages[:-1] if messages[-1].get("role") == "assistant" else messages
    template_kwargs = {"tools": tools} if tools else {}
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    full_text = None
    if has_completion:
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            **template_kwargs,
        )
    return prompt_text, full_text


def tokenize_prompts(tokenizer, prompt_texts, max_seq_len, device):
    encoded = tokenizer(
        prompt_texts,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_len,
        return_token_type_ids=False,
    )
    return {key: value.to(device) for key, value in encoded.items()}


def build_off_policy_batch(
    tokenizer, prompt_texts, full_texts, max_seq_len, max_gen_len, device
):
    """Tokenize fixed prompt-completion pairs and mask all prompt tokens."""
    sequences = []
    label_masks = []
    for prompt_text, full_text in zip(prompt_texts, full_texts):
        if full_text is None:
            raise ValueError(
                "Off-policy GKD selected a sample without a fixed assistant completion; "
                "use --lmbda 1 for prompt-only data"
            )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
        if full_ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError(
                "The rendered prompt is not a token prefix of the fixed conversation; "
                "the tokenizer/chat template is incompatible with prompt-completion GKD"
            )
        completion_ids = full_ids[len(prompt_ids):]
        prompt_ids = prompt_ids[-max_seq_len:]
        completion_ids = completion_ids[:max_gen_len]
        if not completion_ids:
            raise ValueError("A fixed completion became empty after tokenization/truncation")
        sequences.append(prompt_ids + completion_ids)
        label_masks.append([False] * len(prompt_ids) + [True] * len(completion_ids))

    max_length = max(len(sequence) for sequence in sequences)
    input_ids = []
    attention_mask = []
    labels = []
    for sequence, label_mask in zip(sequences, label_masks):
        padding = max_length - len(sequence)
        input_ids.append(sequence + [tokenizer.pad_token_id] * padding)
        attention_mask.append([1] * len(sequence) + [0] * padding)
        labels.append(
            [token if keep else -100 for token, keep in zip(sequence, label_mask)]
            + [-100] * padding
        )
    return (
        torch.tensor(input_ids, dtype=torch.long, device=device),
        torch.tensor(attention_mask, dtype=torch.long, device=device),
        torch.tensor(labels, dtype=torch.long, device=device),
    )


def distributed_mean(value):
    result = value.detach().float().clone()
    if dist.is_initialized():
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        result /= dist.get_world_size()
    return result.item()


def choose_on_policy(lmbda, device):
    if dist.is_initialized():
        draw = torch.zeros((), device=device)
        if dist.get_rank() == 0:
            draw.fill_(random.random())
        dist.broadcast(draw, src=0)
        random_value = draw.item()
    else:
        random_value = random.random()
    return use_on_policy_batch(lmbda, random_value)


def save_student(lm_config, epoch, step, wandb):
    if not is_main_process():
        return
    model.eval()
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model = getattr(raw_model, "_orig_mod", raw_model)
    moe_suffix = "_moe" if lm_config.use_moe else ""
    output_path = resolve_path(args.save_dir) / (
        f"{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
    )
    state_dict = raw_model.state_dict()
    torch.save({key: value.half().cpu() for key, value in state_dict.items()}, output_path)
    lm_checkpoint(
        lm_config,
        weight=args.save_weight,
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        step=step,
        wandb=wandb,
        save_dir=str(resolve_path(args.checkpoint_dir)),
        scheduler=scheduler,
        scaler=scaler,
    )
    model.train()
    del state_dict


def gkd_train_epoch(
    epoch, loader, iters, rollout_engine, teacher_model, lm_config,
    start_step=0, wandb=None
):
    teacher_model.eval().requires_grad_(False)
    model.train()
    on_policy_batches = 0
    processed_batches = 0

    for step, batch in enumerate(loader, start=start_step + 1):
        if args.max_train_steps > 0 and processed_batches >= args.max_train_steps:
            break
        processed_batches += 1
        on_policy = choose_on_policy(args.lmbda, args.device)
        on_policy_batches += int(on_policy)
        rendered = [render_sample(tokenizer, item) for item in batch["conversations"]]
        prompt_texts = [item[0] for item in rendered]
        full_texts = [item[1] for item in rendered]

        if on_policy:
            prompt_inputs = tokenize_prompts(
                tokenizer, prompt_texts, args.max_seq_len, args.device
            )
            model.eval()
            rollout_result = rollout_engine.rollout(
                prompt_ids=prompt_inputs["input_ids"],
                attention_mask=prompt_inputs["attention_mask"],
                num_generations=args.num_generations,
                max_new_tokens=args.max_gen_len,
                temperature=args.temperature,
                top_p=args.rollout_top_p,
                top_k=args.rollout_top_k,
                calculate_logps=False,
            )
            model.train()
            input_ids = rollout_result.output_ids.to(args.device)
            completion_ids = rollout_result.completion_ids.to(args.device)
            completion_mask = build_completion_mask(
                completion_ids,
                rollout_result.completion_mask.to(args.device),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            prompt_mask = prompt_inputs["attention_mask"].repeat_interleave(
                args.num_generations, dim=0
            )
            attention_mask = torch.cat(
                (prompt_mask, completion_mask.to(prompt_mask.dtype)), dim=1
            )
            with autocast_ctx:
                student_output = model(
                    input_ids,
                    attention_mask=attention_mask,
                    logits_to_keep=completion_ids.size(1) + 1,
                )
                student_logits = student_output.logits[:, :-1, :]
            with torch.no_grad(), autocast_ctx:
                teacher_logits = teacher_model(
                    input_ids,
                    attention_mask=attention_mask,
                    logits_to_keep=completion_ids.size(1) + 1,
                ).logits[:, :-1, :]
            avg_completion_length = completion_mask.sum(dim=1).float().mean()
            debug_completion = rollout_result.completions[0]
        else:
            input_ids, attention_mask, labels = build_off_policy_batch(
                tokenizer,
                prompt_texts,
                full_texts,
                args.max_seq_len,
                args.max_gen_len,
                args.device,
            )
            completion_mask = labels[:, 1:].ne(-100)
            with autocast_ctx:
                student_output = model(input_ids, attention_mask=attention_mask)
                student_logits = student_output.logits[:, :-1, :]
            with torch.no_grad(), autocast_ctx:
                teacher_logits = teacher_model(
                    input_ids, attention_mask=attention_mask
                ).logits[:, :-1, :]
            avg_completion_length = completion_mask.sum(dim=1).float().mean()
            debug_completion = full_texts[0]

        gkd_loss, gkd_metrics = generalized_jsd_loss(
            student_logits,
            teacher_logits,
            completion_mask,
            beta=args.beta,
            temperature=args.distill_temperature,
        )
        aux_loss = student_output.aux_loss if lm_config.use_moe else gkd_loss.new_zeros(())
        total_loss = gkd_loss + aux_loss
        scaler.scale(total_loss / args.accumulation_steps).backward()

        should_step = step % args.accumulation_steps == 0 or processed_batches == iters
        if should_step:
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            Logger(f"[OPD DEBUG] on_policy={on_policy}, prompt={prompt_texts[0]!r}")
            Logger(f"[OPD DEBUG] completion={debug_completion!r}")

        if step % args.log_interval == 0 or processed_batches == iters:
            values = {
                "loss/gkd": distributed_mean(gkd_loss),
                "loss/aux": distributed_mean(aux_loss),
                "loss/total": distributed_mean(total_loss),
                "distill/forward_kl": distributed_mean(gkd_metrics["forward_kl"]),
                "distill/reverse_kl": distributed_mean(gkd_metrics["reverse_kl"]),
                "distill/top1_agreement": distributed_mean(gkd_metrics["top1_agreement"]),
                "distill/valid_tokens": distributed_mean(gkd_metrics["valid_tokens"]),
                "distill/on_policy": float(on_policy),
                "distill/on_policy_fraction": on_policy_batches / processed_batches,
                "rollout/avg_completion_length": distributed_mean(avg_completion_length),
                "train/learning_rate": optimizer.param_groups[0]["lr"],
            }
            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{start_step + iters}), "
                f"GKD Loss: {values['loss/gkd']:.4f}, "
                f"Forward KL: {values['distill/forward_kl']:.4f}, "
                f"Reverse KL: {values['distill/reverse_kl']:.4f}, "
                f"Top1 Agree: {values['distill/top1_agreement']:.4f}, "
                f"On Policy: {int(on_policy)}, LR: {values['train/learning_rate']:.8f}"
            )
            if wandb and is_main_process():
                wandb.log(values, step=epoch * iters + step)

        if should_step and (step % args.save_interval == 0 or processed_batches == iters):
            save_student(lm_config, epoch, step, wandb)

        del input_ids, attention_mask, completion_mask
        del student_output, student_logits, teacher_logits
        del gkd_loss, gkd_metrics, aux_loss, total_loss
        if on_policy:
            del prompt_inputs, completion_ids, prompt_mask, rollout_result
        else:
            del labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MiniMind Generalized/On-Policy Knowledge Distillation"
    )
    parser.add_argument("--data_path", type=str, default="dataset/rlaif.jsonl")
    parser.add_argument("--student_model", type=str, default="full_sft")
    parser.add_argument("--teacher_model", type=str, default="full_sft")
    parser.add_argument("--tokenizer_path", type=str, default="model")
    parser.add_argument("--save_dir", type=str, default="out")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--save_weight", type=str, default="opd")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["float32", "float16", "bfloat16"]
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--max_seq_len", type=int, default=768)
    parser.add_argument("--max_gen_len", type=int, default=256)
    parser.add_argument("--max_train_steps", type=int, default=0)
    parser.add_argument(
        "--lmbda", type=float, default=1.0,
        help="fraction of student-generated batches; 1.0 is pure OPD"
    )
    parser.add_argument(
        "--beta", type=float, default=0.5,
        help="generalized JSD coefficient; 0=forward KL, 1=reverse KL"
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--distill_temperature", type=float, default=1.0)
    parser.add_argument("--rollout_top_p", type=float, default=1.0)
    parser.add_argument("--rollout_top_k", type=int, default=0)
    parser.add_argument("--num_generations", type=int, default=1)
    parser.add_argument("--student_hidden_size", type=int, default=768)
    parser.add_argument("--student_num_layers", type=int, default=8)
    parser.add_argument("--student_use_moe", type=int, default=0, choices=[0, 1])
    parser.add_argument("--teacher_hidden_size", type=int, default=768)
    parser.add_argument("--teacher_num_layers", type=int, default=8)
    parser.add_argument("--teacher_use_moe", type=int, default=1, choices=[0, 1])
    parser.add_argument("--from_resume", type=int, default=0, choices=[0, 1])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-OPD")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument(
        "--wandb_mode", type=str, default="cloud",
        choices=["cloud", "local", "disabled"]
    )
    parser.add_argument("--wandb_logdir", type=str, default="swanlog")
    parser.add_argument("--use_compile", type=int, default=0, choices=[0, 1])
    parser.add_argument("--debug_mode", action="store_true")
    parser.add_argument("--debug_interval", type=int, default=20)
    args = parser.parse_args()
    validate_args(args)

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    else:
        args.device = resolve_device(args.device)
    rank = dist.get_rank() if dist.is_initialized() else 0
    setup_seed(args.seed + rank)
    resolve_path(args.save_dir).mkdir(parents=True, exist_ok=True)
    resolve_path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    max_position_embeddings = max(32768, args.max_seq_len + args.max_gen_len)
    student_config = MiniMindConfig(
        hidden_size=args.student_hidden_size,
        num_hidden_layers=args.student_num_layers,
        use_moe=bool(args.student_use_moe),
        max_position_embeddings=max_position_embeddings,
    )
    teacher_config = MiniMindConfig(
        hidden_size=args.teacher_hidden_size,
        num_hidden_layers=args.teacher_num_layers,
        use_moe=bool(args.teacher_use_moe),
        max_position_embeddings=max_position_embeddings,
    )
    if student_config.vocab_size != teacher_config.vocab_size:
        raise ValueError("GKD requires student and teacher to share the same vocabulary")
    tokenizer = load_tokenizer(args.tokenizer_path, student_config.vocab_size)

    device_type = args.device.split(":", 1)[0]
    mixed_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    if device_type == "mps" and mixed_dtype == torch.bfloat16:
        Logger("[OPD WARNING] MPS uses float16 for the frozen teacher")
        mixed_dtype = torch.float16
    teacher_dtype = mixed_dtype if device_type in {"cuda", "mps"} else torch.float32
    autocast_ctx = (
        torch.cuda.amp.autocast(dtype=mixed_dtype)
        if device_type == "cuda" and mixed_dtype != torch.float32
        else nullcontext()
    )

    model = load_native_model(
        student_config, args.student_model, args.device, "student"
    )
    teacher_model = load_native_model(
        teacher_config,
        args.teacher_model,
        args.device,
        "teacher",
        model_dtype=teacher_dtype,
    ).eval().requires_grad_(False)
    Logger(f"Student params: {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M")
    Logger(f"Teacher params: {sum(p.numel() for p in teacher_model.parameters()) / 1e6:.3f}M")

    checkpoint = (
        lm_checkpoint(
            student_config,
            weight=args.save_weight,
            save_dir=str(resolve_path(args.checkpoint_dir)),
        )
        if args.from_resume == 1 else None
    )
    train_ds = OPDDataset(resolve_path(args.data_path))
    train_sampler = (
        DistributedSampler(train_ds, shuffle=True, seed=args.seed)
        if dist.is_initialized() else None
    )
    count_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=collate_conversations,
    )
    full_iters = len(count_loader)
    iters = min(full_iters, args.max_train_steps) if args.max_train_steps > 0 else full_iters
    optimizer_steps_per_epoch = math.ceil(iters / args.accumulation_steps)
    total_optimizer_steps = max(1, optimizer_steps_per_epoch * args.epochs)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=(device_type == "cuda" and mixed_dtype == torch.float16)
    )

    start_epoch, start_step = 0, 0
    if checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"]
        start_step = checkpoint.get("step", 0)
        Logger(f"Resuming GKD from epoch={start_epoch}, step={start_step}")

    if args.use_compile == 1:
        model = torch.compile(model)
        Logger("torch.compile enabled")
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    rollout_engine = create_rollout_engine(
        engine_type="torch",
        policy_model=model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
    )

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = checkpoint.get("wandb_id") if checkpoint else None
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or (
                f"GKD-L{args.lmbda}-B{args.beta}-BS{args.batch_size}x"
                f"{args.accumulation_steps}-LR{args.learning_rate}"
            ),
            config=vars(args),
            mode=args.wandb_mode,
            logdir=str(resolve_path(args.wandb_logdir)),
            id=wandb_id if args.wandb_mode == "cloud" else None,
            resume="must" if wandb_id and args.wandb_mode == "cloud" else None,
        )

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        setup_seed(args.seed + epoch + rank)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if epoch == start_epoch and start_step > 0 else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=(device_type == "cuda"),
            collate_fn=collate_conversations,
        )
        if skip > 0:
            Logger(f"Epoch [{epoch + 1}/{args.epochs}]: skipping {skip} completed batches")
        gkd_train_epoch(
            epoch,
            loader,
            iters,
            rollout_engine,
            teacher_model,
            student_config,
            start_step=skip,
            wandb=wandb,
        )
        start_step = 0

    if wandb and is_main_process() and hasattr(wandb, "finish"):
        wandb.finish()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
