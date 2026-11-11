import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import ast
import gc
import json
import math
import queue
import random
import re
import signal
import threading
import time
import warnings
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset

from model.model_minimind import MiniMindConfig
from trainer.rollout_engine import create_rollout_engine, RolloutResult
from trainer.trainer_utils import (
    LMForRewardModel,
    Logger,
    init_model,
    is_main_process,
    lm_checkpoint,
    setup_seed,
)

warnings.filterwarnings("ignore")


MATH_AGENT_SYSTEM_INSTRUCTION = (
    "你是一个简洁、严格的工具调用助手。遇到算术表达式时，必须先调用 calculate_math 工具；"
    "不要用其它工具解算术题，不要自己心算。工具调用格式必须严格为："
    '<tool_call>{"name":"calculate_math","arguments":{"expression":"1+1"}}</tool_call>'
    "。每个独立算式调用一次工具，按用户问题顺序调用。拿到全部工具结果后，最终只输出 "
    "<answer>结果1, 结果2</answer>；多个结果按问题顺序用英文逗号分隔，不要输出额外解释。"
)

CALCULATE_MATH_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "calculate_math",
        "description": "Evaluate an arithmetic expression. 用于计算加减乘除、幂、sqrt/log/sin/cos 等简单数学表达式。",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Arithmetic expression, e.g. 123+456 or 2**10."}
            },
            "required": ["expression"],
        },
    },
}

UNIT_CONVERTER_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "unit_converter",
        "description": "Convert common units. 仅在用户明确要求单位换算时使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "value": {"type": "number"},
                "from_unit": {"type": "string"},
                "to_unit": {"type": "string"},
            },
            "required": ["value", "from_unit", "to_unit"],
        },
    },
}

TIME_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Return a fixed or live current time for a timezone. 仅在用户明确询问时间时使用。",
        "parameters": {
            "type": "object",
            "properties": {"timezone": {"type": "string"}},
            "required": [],
        },
    },
}

CALCULATOR_ALIAS_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Backward-compatible alias for calculate_math. Prefer calculate_math.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}


# -------------------------- generic helpers --------------------------

def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("true", "1", "yes", "y", "on")


def unwrap_model(model):
    raw = model.module if isinstance(model, DistributedDataParallel) else model
    return getattr(raw, "_orig_mod", raw)


def finite_float(x: Any, default: float = 0.0) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def safe_mean(xs: Iterable[float], default: float = 0.0) -> float:
    vals = [finite_float(x, default=float("nan")) for x in xs]
    vals = [x for x in vals if math.isfinite(x)]
    return sum(vals) / max(len(vals), 1) if vals else default


def percentile(xs: Iterable[float], q: float, default: float = 0.0) -> float:
    vals = sorted(finite_float(x, default=float("nan")) for x in xs)
    vals = [x for x in vals if math.isfinite(x)]
    if not vals:
        return default
    if len(vals) == 1:
        return float(vals[0])
    q = min(max(float(q), 0.0), 1.0)
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(vals[lo])
    frac = pos - lo
    return float(vals[lo] * (1.0 - frac) + vals[hi] * frac)


def ddp_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def trainer_rank() -> int:
    return 0


def is_trainer_rollout_mode(args) -> bool:
    return bool(getattr(args, "multi_gpu_mode", "trainer_rollout") == "trainer_rollout")


def is_trainer_process(args) -> bool:
    rank = dist.get_rank() if dist.is_initialized() else 0
    return rank == trainer_rank() if is_trainer_rollout_mode(args) else True


def ddp_all_true(flag: bool, device: str) -> bool:
    if not dist.is_initialized():
        return bool(flag)
    x = torch.tensor([1 if flag else 0], device=device, dtype=torch.int64)
    dist.all_reduce(x, op=dist.ReduceOp.MIN)
    return bool(x.item() > 0)


def ddp_mean_scalar(value: float, device: str) -> float:
    if not dist.is_initialized():
        return float(value)
    x = torch.tensor([finite_float(value)], device=device, dtype=torch.float32)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    x /= max(ddp_world_size(), 1)
    return float(x.item())


def ddp_sum_scalar(value: float, device: str) -> float:
    if not dist.is_initialized():
        return float(value)
    x = torch.tensor([finite_float(value)], device=device, dtype=torch.float32)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return float(x.item())


def ddp_average_metrics(metrics: Dict[str, Any], device: str) -> Dict[str, Any]:
    if not dist.is_initialized():
        return metrics
    keys = [k for k, v in metrics.items() if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not keys:
        return metrics
    values = torch.tensor([float(metrics[k]) for k in keys], device=device, dtype=torch.float32)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= max(ddp_world_size(), 1)
    out = dict(metrics)
    for key, value in zip(keys, values.tolist()):
        out[key] = float(value)
    return out


def assert_rollout_alignment(response_ids, response_mask, response_old_logps, response_versions):
    n = len(response_ids)
    if not (n == len(response_mask) == len(response_old_logps) == len(response_versions)):
        raise AssertionError(
            f"rollout alignment mismatch: ids={len(response_ids)}, mask={len(response_mask)}, "
            f"logps={len(response_old_logps)}, versions={len(response_versions)}"
        )


def init_distributed_mode_compat(backend: str = "auto"):
    if int(os.environ.get("RANK", -1)) == -1:
        return 0

    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    def _init_pg(pg_backend: str):
        dist.init_process_group(backend=pg_backend, timeout=timedelta(minutes=30))
        if pg_backend == "nccl":
            probe = torch.ones(1, device=f"cuda:{local_rank}")
            dist.all_reduce(probe)
            torch.cuda.synchronize(local_rank)

    backend = str(backend or "auto").lower()
    if backend == "auto":
        preferred = "nccl" if torch.cuda.is_available() else "gloo"
        candidates = [preferred] if preferred == "gloo" else ["nccl", "gloo"]
    else:
        candidates = [backend]

    for idx, candidate in enumerate(candidates):
        try:
            _init_pg(candidate)
            if candidate != "nccl" and int(os.environ.get("RANK", "0")) == 0:
                Logger(f"[WARN] distributed backend fallback to {candidate}")
            return local_rank
        except Exception as exc:
            if dist.is_initialized():
                try:
                    dist.destroy_process_group()
                except Exception:
                    pass
            if idx + 1 >= len(candidates):
                raise
            if int(os.environ.get("RANK", "0")) == 0:
                Logger(f"[WARN] distributed backend {candidate} unavailable: {repr(exc)}; fallback to {candidates[idx + 1]}")
    return local_rank


def ensure_tokenizer_ids(tokenizer, lm_config=None):
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = getattr(lm_config, "eos_token_id", 2)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer.pad_token_id, tokenizer.eos_token_id


@torch.no_grad()
def model_has_nonfinite(model: torch.nn.Module, max_checks: int = 32) -> bool:
    raw = unwrap_model(model)
    checked = 0
    for p in raw.parameters():
        if p is None:
            continue
        if not torch.isfinite(p.detach()).all():
            return True
        checked += 1
        if checked >= max_checks:
            break
    return False


@torch.no_grad()
def copy_policy_weights(dst_model: torch.nn.Module, src_model: torch.nn.Module) -> bool:
    if dst_model is None:
        return False
    if model_has_nonfinite(src_model):
        Logger("[WARN] train model has non-finite weights; skip rollout policy sync")
        return False
    src = unwrap_model(src_model)
    dst = unwrap_model(dst_model)
    dst.load_state_dict(src.state_dict(), strict=False)
    dst.to(next(src.parameters()).device)
    dst.eval().requires_grad_(False)
    return True


def save_checkpoint(args, lm_config, model, optimizer, scheduler, epoch: int, step: int, opt_step: int, version: int, wandb=None):
    if not is_main_process():
        return
    if model_has_nonfinite(model):
        Logger("[WARN] checkpoint skipped because model has non-finite weights")
        return
    os.makedirs(args.save_dir, exist_ok=True)
    moe_suffix = "_moe" if lm_config.use_moe else ""
    ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
    raw_model = unwrap_model(model)
    state_dict = raw_model.state_dict()
    tmp = ckp + ".tmp"
    torch.save({k: v.half().cpu() for k, v in state_dict.items()}, tmp)
    os.replace(tmp, ckp)
    lm_checkpoint(
        lm_config,
        weight=args.save_weight,
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        step=step,
        wandb=wandb,
        save_dir="../checkpoints",
        scheduler=scheduler,
        opt_step=opt_step,
        policy_version=version,
        reward_baseline=float(getattr(args, "reward_baseline", 0.0)),
    )
    Logger(f"checkpoint saved: {ckp}")
    del state_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def apply_chat_template_compat(tokenizer, messages, tools=None, add_generation_prompt=True, open_thinking=False) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        call_orders = [
            dict(tokenize=False, add_generation_prompt=add_generation_prompt, tools=tools, open_thinking=open_thinking),
            dict(tokenize=False, add_generation_prompt=add_generation_prompt, tools=tools),
            dict(tokenize=False, add_generation_prompt=add_generation_prompt),
        ]
        for kwargs in call_orders:
            try:
                return tokenizer.apply_chat_template(messages, **kwargs)
            except TypeError:
                continue
            except Exception:
                continue
    parts = []
    if tools:
        parts.append("<|tools|>" + json.dumps(tools, ensure_ascii=False, separators=(",", ":")) + "<|/tools|>")
    for msg in messages:
        parts.append(f"<|im_start|>{msg.get('role', 'user')}\n{msg.get('content', '')}<|im_end|>")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def tokenize_context(tokenizer, text: str, device: str, max_context_len: int = 0):
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    if max_context_len and input_ids.size(1) > max_context_len:
        input_ids = input_ids[:, -max_context_len:]
        attention_mask = attention_mask[:, -max_context_len:]
    return {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device)}


# -------------------------- rollout engine --------------------------

class StableTorchRolloutEngine:
    """Safe torch rollout.

    PPO fix: old_logps are computed under the raw policy distribution. Sampling may use
    temperature/top-p/top-k/repetition penalty, but PPO behavior logprobs are taken from
    the unmodified policy distribution.
    """

    def __init__(
        self,
        policy_model,
        tokenizer,
        device="cuda",
        autocast_factory=None,
        top_p=0.90,
        top_k=50,
        repetition_penalty=1.03,
        use_cache=True,
    ):
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.device = device
        self.autocast_factory = autocast_factory or (lambda: nullcontext())
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.repetition_penalty = float(repetition_penalty)
        self.use_cache = bool(use_cache)
        self.nan_batches = 0
        self.bad_prob_batches = 0
        self.cache_prefill_tokens = 0
        self.cache_decode_tokens = 0
        self.cache_fallbacks = 0

    def update_policy(self, model):
        self.policy_model = model
        return True

    @staticmethod
    def _sanitize_logits(logits: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(logits.float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(-80.0, 80.0)

    def _apply_repetition_penalty_for_sampling(self, sample_logits, input_ids):
        if not self.repetition_penalty or self.repetition_penalty == 1.0:
            return sample_logits
        sample_logits = sample_logits.clone()
        for i in range(input_ids.size(0)):
            uniq = torch.unique(input_ids[i])
            uniq = uniq[(uniq >= 0) & (uniq < sample_logits.size(-1))]
            if uniq.numel() > 0:
                positive = sample_logits[i, uniq] > 0
                sample_logits[i, uniq] = torch.where(
                    positive,
                    sample_logits[i, uniq] / self.repetition_penalty,
                    sample_logits[i, uniq] * self.repetition_penalty,
                )
        return sample_logits

    def _sample_next(self, logits, input_ids, temperature, do_sample, eos_token_id):
        had_bad = (~torch.isfinite(logits)).any().item()
        raw_logits = self._sanitize_logits(logits)
        raw_log_probs = F.log_softmax(raw_logits, dim=-1)
        sample_logits = self._apply_repetition_penalty_for_sampling(raw_logits, input_ids)

        if (not do_sample) or temperature <= 1e-6:
            next_token = torch.argmax(sample_logits, dim=-1, keepdim=True)
            next_logp = raw_log_probs.gather(1, next_token).squeeze(1)
            return next_token, next_logp, bool(had_bad), False

        sample_logits = sample_logits / max(float(temperature), 1e-6)
        vocab_size = sample_logits.size(-1)
        top_k = min(max(self.top_k, 0), vocab_size)
        if 0 < top_k < vocab_size:
            threshold = torch.topk(sample_logits, top_k, dim=-1).values[..., -1, None]
            sample_logits = sample_logits.masked_fill(sample_logits < threshold, -1e9)

        if 0.0 < self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(sample_logits, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            sorted_probs = torch.nan_to_num(sorted_probs, nan=0.0, posinf=0.0, neginf=0.0)
            sorted_remove = torch.cumsum(sorted_probs, dim=-1) > self.top_p
            sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
            sorted_remove[..., 0] = False
            remove = torch.zeros_like(sample_logits, dtype=torch.bool)
            remove.scatter_(1, sorted_indices, sorted_remove)
            sample_logits = sample_logits.masked_fill(remove, -1e9)

        probs = torch.softmax(sample_logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
        denom = probs.sum(dim=-1, keepdim=True)
        bad_rows = (~torch.isfinite(denom)) | (denom <= 0)
        if bad_rows.any():
            probs = torch.where(bad_rows, torch.zeros_like(probs), probs)
            fallback_id = int(eos_token_id) if eos_token_id is not None else 0
            fallback_id = min(max(fallback_id, 0), probs.size(-1) - 1)
            probs[bad_rows.squeeze(1), fallback_id] = 1.0
            denom = probs.sum(dim=-1, keepdim=True)
        probs = probs / denom.clamp(min=1e-12)
        next_token = torch.multinomial(probs, num_samples=1)
        next_logp = raw_log_probs.gather(1, next_token).squeeze(1)
        next_logp = torch.nan_to_num(next_logp, nan=0.0, posinf=0.0, neginf=0.0)
        return next_token, next_logp, bool(had_bad), bool(bad_rows.any().item())

    @torch.no_grad()
    def rollout(self, prompt_ids, attention_mask, num_generations, max_new_tokens, temperature=0.8):
        model = unwrap_model(self.policy_model).eval()
        eos_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 2
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_id
        input_ids = prompt_ids.repeat_interleave(num_generations, dim=0).to(self.device)
        attn = attention_mask.repeat_interleave(num_generations, dim=0).to(self.device) if attention_mask is not None else None
        self.cache_prefill_tokens += int(input_ids.numel())
        vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
        if vocab_size is not None:
            bad_ids = (input_ids < 0) | (input_ids >= int(vocab_size))
            if bad_ids.any():
                input_ids = input_ids.clone()
                input_ids[bad_ids] = eos_id

        generated_tokens, generated_logps = [], []
        past_key_values = None
        finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)
        do_sample = temperature > 1e-6
        use_cache = bool(self.use_cache)

        for _ in range(int(max_new_tokens)):
            step_input = input_ids[:, -1:] if (use_cache and past_key_values is not None) else input_ids
            try:
                with self.autocast_factory():
                    outputs = model(
                        input_ids=step_input,
                        attention_mask=attn,
                        past_key_values=past_key_values if use_cache else None,
                        use_cache=use_cache,
                    )
            except RuntimeError as exc:
                msg = str(exc)
                if use_cache and ("attention_mask" in msg or "size of tensor" in msg or "must match" in msg):
                    self.cache_fallbacks += 1
                    use_cache = False
                    past_key_values = None
                    with self.autocast_factory():
                        outputs = model(input_ids=input_ids, attention_mask=attn, past_key_values=None, use_cache=False)
                else:
                    raise

            if use_cache and past_key_values is not None:
                self.cache_decode_tokens += int(step_input.numel())

            next_token, next_logp, had_bad, bad_prob = self._sample_next(
                outputs.logits[:, -1, :], input_ids, temperature, do_sample, eos_id
            )
            self.nan_batches += int(had_bad)
            self.bad_prob_batches += int(bad_prob)
            forced_eos = torch.full_like(next_token, int(eos_id))
            next_token = torch.where(finished.unsqueeze(-1), forced_eos, next_token)
            next_logp = torch.where(finished, torch.zeros_like(next_logp), next_logp)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            generated_tokens.append(next_token)
            generated_logps.append(next_logp.unsqueeze(1))
            if attn is not None:
                attn = torch.cat([attn, attn.new_ones(attn.size(0), 1)], dim=-1)
            past_key_values = getattr(outputs, "past_key_values", None) if use_cache else None
            finished |= next_token.squeeze(-1).eq(int(eos_id))
            if finished.all():
                break

        if generated_tokens:
            completion_ids = torch.cat(generated_tokens, dim=1)
            per_token_logps = torch.cat(generated_logps, dim=1)
        else:
            completion_ids = input_ids.new_full((input_ids.size(0), 0), int(pad_id))
            per_token_logps = input_ids.new_empty((input_ids.size(0), 0), dtype=torch.float32)
        output_ids = torch.cat([prompt_ids.repeat_interleave(num_generations, dim=0).to(self.device), completion_ids], dim=1)
        completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        return RolloutResult(output_ids=output_ids, completion_ids=completion_ids, per_token_logps=per_token_logps, completions=completions)


# -------------------------- logprob / entropy helpers --------------------------


def compute_per_token_logps_safe(model, input_ids, attention_mask, autocast_factory=None):
    raw = unwrap_model(model)
    ctx_factory = autocast_factory or (lambda: nullcontext())
    with ctx_factory():
        res = raw(input_ids=input_ids, attention_mask=attention_mask)
        logits = res.logits[:, :-1, :]
    logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(-80.0, 80.0)
    token_ids = input_ids[:, 1:].clamp(min=0, max=logits.size(-1) - 1)
    return F.log_softmax(logits, dim=-1).gather(2, token_ids.unsqueeze(-1)).squeeze(-1)


def logits_entropy_safe(logits, mask):
    with torch.no_grad():
        logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(-80.0, 80.0)
        logp = F.log_softmax(logits, dim=-1)
        ent = -(torch.exp(logp) * logp).sum(dim=-1)
        return float(((ent * mask).sum() / mask.sum().clamp(min=1.0)).item())


# -------------------------- safe math and robust extraction --------------------------

class SafeMathEvaluator(ast.NodeVisitor):
    ALLOWED_BINOPS = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a ** b,
    }
    ALLOWED_UNARY = {ast.UAdd: lambda a: +a, ast.USub: lambda a: -a}
    ALLOWED_NAMES = {"pi": math.pi, "e": math.e}
    ALLOWED_FUNCS = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "exp": math.exp,
        "abs": abs,
        "round": round,
    }

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_Constant(self, node):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("invalid constant")

    def visit_Num(self, node):
        return float(node.n)

    def visit_Name(self, node):
        if node.id in self.ALLOWED_NAMES:
            return float(self.ALLOWED_NAMES[node.id])
        raise ValueError(f"invalid name: {node.id}")

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name) or node.func.id not in self.ALLOWED_FUNCS:
            raise ValueError("invalid function")
        if node.keywords:
            raise ValueError("keyword args not allowed")
        if len(node.args) > 3:
            raise ValueError("too many args")
        return float(self.ALLOWED_FUNCS[node.func.id](*[self.visit(arg) for arg in node.args]))

    def visit_BinOp(self, node):
        op = self.ALLOWED_BINOPS.get(type(node.op))
        if op is None:
            raise ValueError("invalid operator")
        return float(op(self.visit(node.left), self.visit(node.right)))

    def visit_UnaryOp(self, node):
        op = self.ALLOWED_UNARY.get(type(node.op))
        if op is None:
            raise ValueError("invalid unary operator")
        return float(op(self.visit(node.operand)))

    def generic_visit(self, node):
        raise ValueError(f"unsupported node: {type(node).__name__}")


def _build_fullwidth_translation() -> Dict[int, str]:
    mapping = {ord(src): dst for src, dst in {
        "＋": "+",
        "－": "-",
        "−": "-",
        "–": "-",
        "—": "-",
        "﹣": "-",
        "×": "*",
        "✕": "*",
        "＊": "*",
        "÷": "/",
        "／": "/",
        "（": "(",
        "）": ")",
        "［": "[",
        "］": "]",
        "｛": "{",
        "｝": "}",
        "，": ",",
        "。": ".",
        "．": ".",
        "：": ":",
        "；": ";",
        "％": "%",
        "＾": "^",
        "　": " ",
    }.items()}
    for src, dst in zip("０１２３４５６７８９", "0123456789"):
        mapping[ord(src)] = dst
    return mapping


_FULLWIDTH_TRANS = _build_fullwidth_translation()


def canonicalize_expression(expr: str) -> str:
    s = str(expr).translate(_FULLWIDTH_TRANS)
    s = s.replace("^", "**")
    s = re.sub(r"\s+", "", s)
    return s


def _strip_expr_edges(expr: str) -> str:
    s = str(expr).translate(_FULLWIDTH_TRANS).strip()
    s = s.strip(" \t\r\n:;,.?？!！、，。；：")
    # Trim unmatched right parens that appear from casual text while preserving valid groups.
    while s and s.count(")") > s.count("("):
        s = s[:-1].rstrip()
    return s


def safe_eval_math(expr: str) -> float:
    expr = canonicalize_expression(expr)
    if len(expr) > 256:
        raise ValueError("expression too long")
    # Avoid interpreting dates as subtraction chains.
    if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", expr):
        raise ValueError("date-like string is not an arithmetic expression")
    return SafeMathEvaluator().visit(ast.parse(expr, mode="eval"))


def format_number(value: float) -> str:
    value = float(value)
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.10f}".rstrip("0").rstrip(".")


_NUM = r"[-+]?\d+(?:\.\d+)?"
_FUNC = r"(?:sqrt|sin|cos|tan|log|exp|abs|round)"
# Non-nested parentheses are enough for the synthetic data and most simple tool-use data.
_ATOM = rf"(?:{_FUNC}\s*\([^()]+\)|\([^()]+\)|{_NUM})"
_OP = r"(?:\*\*|//|[+\-*/×÷^%])"
# ASCII boundary only. Do not use \w boundary: in Python, Chinese characters are \w and would hide expressions after “计算”.
ARITH_EXPR_RE = re.compile(rf"(?<![A-Za-z0-9_]){_ATOM}(?:\s*{_OP}\s*{_ATOM})+(?![A-Za-z0-9_])", re.IGNORECASE)


def _candidate_has_real_operator(expr: str) -> bool:
    s = canonicalize_expression(expr)
    # Remove leading unary sign before checking for a binary operator.
    s2 = re.sub(r"(?<=^)[+-]", "", s)
    return any(op in s2 for op in ("**", "//", "+", "-", "*", "/", "%"))


def extract_arithmetic_expressions(text: str, max_exprs: int = 8) -> List[str]:
    """Extract simple arithmetic expressions from mixed Chinese/English text.

    v3 used a boundary based on \\w and therefore missed expressions immediately after
    Chinese words, e.g. “请计算884+645*18” -> only “645*18”. It also truncated
    parenthesized expressions, e.g. “(289+532)*11” -> “(289+532)”. Both errors poison
    BC targets and reward diagnostics. This extractor keeps complete arithmetic spans
    and validates every candidate with SafeMathEvaluator.
    """
    raw = str(text).translate(_FULLWIDTH_TRANS)
    candidates: List[Tuple[int, int, str]] = []
    for m in ARITH_EXPR_RE.finditer(raw):
        expr = _strip_expr_edges(m.group(0))
        if not expr or not _candidate_has_real_operator(expr):
            continue
        try:
            safe_eval_math(expr)
        except Exception:
            continue
        candidates.append((m.start(), m.end(), expr))

    # Prefer longer overlapping spans to avoid keeping “289+532” when “(289+532)*11” exists.
    candidates.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    selected: List[Tuple[int, int, str]] = []
    for start, end, expr in candidates:
        overlaps = [i for i, (s, e, _) in enumerate(selected) if not (end <= s or start >= e)]
        if overlaps:
            # Keep the candidate covering more characters.
            keep = True
            for i in overlaps:
                s, e, _ = selected[i]
                if (e - s) >= (end - start):
                    keep = False
                    break
            if not keep:
                continue
            selected = [x for j, x in enumerate(selected) if j not in overlaps]
        selected.append((start, end, expr))
        selected.sort(key=lambda x: x[0])
        if len(selected) >= max_exprs:
            break

    out: List[str] = []
    seen = set()
    for _, _, expr in selected:
        key = canonicalize_expression(expr)
        if key not in seen:
            out.append(expr)
            seen.add(key)
        if len(out) >= max_exprs:
            break
    return out


# -------------------------- tool env --------------------------

class RealToolEnv:
    def __init__(self, time_mode="fixed", active_tools: Optional[List[str]] = None, allow_calculator_alias: bool = False):
        self.time_mode = time_mode
        self.active_tools = active_tools or ["calculate_math", "unit_converter", "get_current_time"]
        self.allow_calculator_alias = bool(allow_calculator_alias)
        self.time_data = {
            "Asia/Shanghai": "2025-03-07 14:30:00",
            "America/New_York": "2025-03-07 01:30:00",
            "Europe/London": "2025-03-07 06:30:00",
            "Asia/Tokyo": "2025-03-07 15:30:00",
            "Europe/Paris": "2025-03-07 07:30:00",
            "Australia/Sydney": "2025-03-07 17:30:00",
        }
        self.unit_data = {
            "km_miles": 0.621371,
            "miles_km": 1.60934,
            "kg_pounds": 2.20462,
            "pounds_kg": 0.453592,
            "meters_feet": 3.28084,
            "feet_meters": 0.3048,
        }

    def tool_specs(self):
        spec_map = {
            "calculate_math": CALCULATE_MATH_TOOL_SPEC,
            "unit_converter": UNIT_CONVERTER_TOOL_SPEC,
            "get_current_time": TIME_TOOL_SPEC,
        }
        out = []
        for name in self.active_tools:
            if name in spec_map:
                out.append(clone_tool_spec(spec_map[name]))
        if self.allow_calculator_alias and "calculate_math" in self.active_tools:
            out.append(clone_tool_spec(CALCULATOR_ALIAS_TOOL_SPEC))
        return out

    def supported_names(self):
        names = set(supported_tool_names(self.tool_specs())) | {"python_math"}
        if self.allow_calculator_alias:
            names.add("calculator")
        return names

    def execute(self, name, args):
        if name in ("calculate_math", "calculator", "python_math"):
            expr = str(args.get("expression", args.get("query", ""))).strip()
            if not expr:
                raise ValueError("missing expression")
            value = safe_eval_math(expr)
            return {"value": value, "result": format_number(value)}
        if name == "unit_converter":
            value = float(args["value"])
            from_unit = str(args["from_unit"]).lower().strip()
            to_unit = str(args["to_unit"]).lower().strip()
            key = f"{from_unit}_{to_unit}"
            if key == "celsius_fahrenheit":
                result = value * 1.8 + 32.0
            elif key == "fahrenheit_celsius":
                result = (value - 32.0) / 1.8
            elif key in self.unit_data:
                result = value * self.unit_data[key]
            else:
                raise ValueError(f"unsupported conversion: {key}")
            return {"value": value, "from_unit": from_unit, "to_unit": to_unit, "result": format_number(round(result, 6))}
        if name == "get_current_time":
            timezone = str(args.get("timezone", "Asia/Shanghai") or "Asia/Shanghai")
            if self.time_mode == "live":
                return {"timezone": timezone, "datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "mode": "live_utc"}
            return {"timezone": timezone, "datetime": self.time_data.get(timezone, self.time_data["Asia/Shanghai"]), "mode": "fixed"}
        raise ValueError(f"unknown tool: {name}")


# -------------------------- schema/data helpers --------------------------

def normalize_tool_schema(raw_tools):
    if raw_tools is None or raw_tools == "":
        return None
    if isinstance(raw_tools, str):
        try:
            raw_tools = json.loads(raw_tools)
        except Exception:
            return None
    if isinstance(raw_tools, dict):
        raw_tools = [raw_tools]
    if not isinstance(raw_tools, list):
        return None
    tools = []
    for item in raw_tools:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function" and isinstance(item.get("function"), dict):
            fn = dict(item["function"])
        elif isinstance(item.get("function"), dict):
            fn = dict(item["function"])
        elif "name" in item:
            fn = {
                "name": item.get("name"),
                "description": item.get("description", ""),
                "parameters": item.get("parameters", {"type": "object", "properties": {}}),
            }
        else:
            continue
        if isinstance(fn.get("parameters"), str):
            try:
                fn["parameters"] = json.loads(fn["parameters"])
            except Exception:
                fn["parameters"] = {"type": "object", "properties": {}}
        if fn.get("name"):
            tools.append({"type": "function", "function": fn})
    return tools or None


def supported_tool_names(tool_specs):
    if not tool_specs:
        return []
    out = []
    for tool in tool_specs:
        try:
            name = tool.get("function", {}).get("name")
            if name:
                out.append(name)
        except Exception:
            pass
    return out


def clone_tool_spec(tool):
    return json.loads(json.dumps(tool, ensure_ascii=False))


def tool_name_set(tool_specs):
    return set(supported_tool_names(tool_specs))


def merge_tool_specs(primary, extra, keep_alias=False, allowed_names: Optional[set] = None):
    merged = []
    seen = set()
    for src in (primary or []), (extra or []):
        for tool in src:
            name = tool.get("function", {}).get("name") if isinstance(tool, dict) else None
            if not name or name in seen:
                continue
            if name == "calculator" and not keep_alias:
                continue
            if allowed_names is not None and name not in allowed_names:
                continue
            merged.append(clone_tool_spec(tool))
            seen.add(name)
    return merged or None


def resolve_tool_specs(sample_tools, default_tools, force_all_tools=False, keep_alias=False, allowed_names: Optional[set] = None):
    sample_only = merge_tool_specs(sample_tools, None, keep_alias=keep_alias, allowed_names=allowed_names)
    default_only = merge_tool_specs(default_tools, None, keep_alias=keep_alias, allowed_names=allowed_names)
    if force_all_tools:
        return merge_tool_specs(sample_only, default_only, keep_alias=keep_alias, allowed_names=allowed_names)
    return sample_only or default_only


def append_calculator_alias(tools):
    if not tools:
        return tools
    names = tool_name_set(tools)
    if "calculate_math" in names and "calculator" not in names:
        tools = list(tools) + [clone_tool_spec(CALCULATOR_ALIAS_TOOL_SPEC)]
    return tools


def prompt_text_from_messages(messages: List[Dict[str, Any]], roles: Tuple[str, ...] = ("system", "user")) -> str:
    role_set = set(roles)
    return "\n".join(str(m.get("content", "")) for m in messages if m.get("role") in role_set)


def task_text_from_messages(messages: Optional[List[Dict[str, Any]]]) -> str:
    if not messages:
        return ""
    user_text = prompt_text_from_messages(messages, roles=("user",))
    if user_text.strip():
        return user_text
    return prompt_text_from_messages(messages, roles=("system", "user"))


def task_text_from_prompt(prompt: str) -> str:
    raw = re.sub(r"<\|tools\|>.*?<\|/tools\|>", "", str(prompt), flags=re.DOTALL)
    matches = re.findall(r"<\|im_start\|>(system|user|assistant|tool)\n(.*?)<\|im_end\|>", raw, flags=re.DOTALL)
    if matches:
        user_parts = [content.strip() for role, content in matches if role == "user" and content.strip()]
        if user_parts:
            return "\n".join(user_parts)
        su_parts = [content.strip() for role, content in matches if role in ("system", "user") and content.strip()]
        if su_parts:
            return "\n".join(su_parts)
    return raw.strip()


def parse_tool_calls(text):
    calls = []
    raw_text = str(text)

    for match in re.finditer(r"<tool_call>(.*?)</tool_call>", raw_text, flags=re.DOTALL):
        raw = match.group(1).strip()
        try:
            obj = json.loads(raw)
            name = obj.get("name") or obj.get("tool_name") or obj.get("function", {}).get("name")
            args = obj.get("arguments", obj.get("args", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"expression": args}
            calls.append({"name": name or "", "arguments": args if isinstance(args, dict) else {}, "canonical": True})
        except Exception:
            calls.append({"name": "", "arguments": {}, "parse_error": raw[:512], "canonical": False})

    if calls:
        return calls

    # Tolerant exploration parser: calculate_math("1+1") or calculator(expression="(1+2)*3").
    for name, body in extract_function_style_tool_calls(raw_text, ("calculate_math", "calculator", "python_math")):
        expr = ""
        q = re.search(r"""expression\s*=\s*(['"])(.*?)\1""", body, flags=re.DOTALL)
        if q:
            expr = q.group(2)
        else:
            q = re.search(r"""(['"])(.*?)\1""", body, flags=re.DOTALL)
            if q:
                expr = q.group(2)
        expr = str(expr).strip()
        if expr:
            calls.append({"name": name, "arguments": {"expression": expr}, "canonical": False})
    return calls


def extract_function_style_tool_calls(text: str, names: Tuple[str, ...]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    raw = str(text)
    cursor = 0
    while cursor < len(raw):
        match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", raw[cursor:])
        if not match:
            break
        name = match.group(1)
        if name not in names:
            cursor += match.end()
            continue
        open_idx = cursor + match.end() - 1
        close_idx = find_matching_paren(raw, open_idx)
        if close_idx < 0:
            cursor = open_idx + 1
            continue
        out.append((name, raw[open_idx + 1:close_idx]))
        cursor = close_idx + 1
    return out


def find_matching_paren(text: str, open_idx: int) -> int:
    if open_idx < 0 or open_idx >= len(text) or text[open_idx] != "(":
        return -1
    depth = 0
    quote = ""
    escaped = False
    for idx in range(open_idx, len(text)):
        ch = text[idx]
        if quote:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == quote:
                quote = ""
            continue
        if ch in ("'", '"'):
            quote = ch
            continue
        if ch == "(":
            depth += 1
            continue
        if ch == ")":
            depth -= 1
            if depth == 0:
                return idx
    return -1


class PromptPoolDataset(Dataset):
    def __init__(
        self,
        jsonl_path,
        max_samples=0,
        filter_mode="all",
        require_gt=True,
        allowed_tools=None,
        inject_answer_instruction=True,
        augment_calculator_alias=False,
        default_tools=None,
        force_all_tools=True,
    ):
        self.jsonl_path = jsonl_path
        self.max_samples = int(max_samples)
        self.filter_mode = filter_mode
        self.require_gt = bool(require_gt)
        self.allowed_tools = set(allowed_tools or [])
        self.inject_answer_instruction = bool(inject_answer_instruction)
        self.augment_calculator_alias = bool(augment_calculator_alias)
        self.default_tools = normalize_tool_schema(default_tools) or []
        self.force_all_tools = bool(force_all_tools)
        self.samples = []
        self.stats = {
            "loaded": 0,
            "kept": 0,
            "bad_json": 0,
            "dropped": 0,
            "skip_no_gt": 0,
            "skip_unsupported_tools": 0,
            "skip_filter": 0,
            "augmented_calculator_alias": 0,
            "forced_default_tools": 0,
            "defaulted_tools": 0,
        }
        self._load()

    def _maybe_inject_instruction(self, messages, tool_names):
        if not self.inject_answer_instruction or "calculate_math" not in tool_names:
            return messages
        messages = [dict(m) for m in messages]
        for msg in messages:
            if msg.get("role") == "system":
                content = str(msg.get("content", "")).strip()
                if "calculate_math" not in content or "<answer>" not in content:
                    msg["content"] = (content + "\n" + MATH_AGENT_SYSTEM_INSTRUCTION).strip()
                return messages
        return [{"role": "system", "content": MATH_AGENT_SYSTEM_INSTRUCTION}] + messages

    def _normalize(self, raw):
        tools = normalize_tool_schema(raw.get("tools")) or normalize_tool_schema(raw.get("functions"))
        gt = raw.get("gt", raw.get("answer", raw.get("answers", raw.get("label", []))))
        if gt is None:
            gt = []
        if not isinstance(gt, list):
            gt = [gt]
        gt = [x for x in gt if str(x).strip()]

        messages = raw.get("messages")
        if not messages and isinstance(raw.get("conversations"), list):
            messages = []
            for msg in raw["conversations"]:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role", msg.get("from", ""))
                role = {"human": "user", "gpt": "assistant"}.get(role, role)
                content = msg.get("content", msg.get("value", ""))
                if msg.get("tools") is not None:
                    tools = normalize_tool_schema(msg.get("tools")) or tools
                if msg.get("functions") is not None:
                    tools = normalize_tool_schema(msg.get("functions")) or tools
                messages.append({"role": role, "content": str(content)})
            if messages and messages[-1].get("role") == "assistant" and not str(messages[-1].get("content", "")).strip():
                messages = messages[:-1]

        if not messages:
            prompt = raw.get("prompt") or raw.get("question") or raw.get("query")
            if prompt:
                messages = [{"role": "user", "content": str(prompt)}]
        if not isinstance(messages, list) or not messages:
            return None

        norm_messages = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "user")
            role = {"human": "user", "gpt": "assistant"}.get(role, role)
            if role not in ("system", "user", "assistant", "tool"):
                role = "user"
            norm_messages.append({"role": role, "content": str(msg.get("content", msg.get("value", "")))})
        if not norm_messages:
            return None

        allowed_names = set(self.allowed_tools) if self.allowed_tools else None
        sample_tool_count = len(tool_name_set(merge_tool_specs(tools, None, keep_alias=self.augment_calculator_alias, allowed_names=allowed_names)))
        default_tool_count = len(tool_name_set(merge_tool_specs(self.default_tools, None, keep_alias=self.augment_calculator_alias, allowed_names=allowed_names)))
        tools = resolve_tool_specs(
            tools,
            self.default_tools,
            force_all_tools=self.force_all_tools,
            keep_alias=self.augment_calculator_alias,
            allowed_names=allowed_names,
        )
        after_n = len(tool_name_set(tools))
        if sample_tool_count <= 0 and default_tool_count > 0 and after_n > 0:
            self.stats["defaulted_tools"] += 1
        if self.force_all_tools and after_n > sample_tool_count:
            self.stats["forced_default_tools"] += 1

        if self.augment_calculator_alias:
            before_names = tool_name_set(tools)
            tools = append_calculator_alias(tools)
            after_names = tool_name_set(tools)
            if "calculator" not in before_names and "calculator" in after_names:
                self.stats["augmented_calculator_alias"] += 1

        if self.require_gt and not gt:
            self.stats["skip_no_gt"] += 1
            return None
        tool_names = supported_tool_names(tools)
        if self.allowed_tools and any(name not in self.allowed_tools for name in tool_names):
            self.stats["skip_unsupported_tools"] += 1
            return None
        if self.filter_mode == "math_only":
            text = task_text_from_messages(norm_messages)
            has_math_expr = bool(extract_arithmetic_expressions(text, max_exprs=1))
            if not has_math_expr:
                self.stats["skip_filter"] += 1
                return None
        norm_messages = self._maybe_inject_instruction(norm_messages, tool_names)
        return {"messages": norm_messages, "tools": tools, "gt": gt, "raw": raw}

    def _load(self):
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if self.max_samples > 0 and len(self.samples) >= self.max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                self.stats["loaded"] += 1
                try:
                    raw = json.loads(line)
                except Exception:
                    self.stats["bad_json"] += 1
                    continue
                sample = self._normalize(raw)
                if sample is None:
                    self.stats["dropped"] += 1
                    continue
                self.samples.append(sample)
                self.stats["kept"] += 1
        if not self.samples:
            raise ValueError(f"No valid samples found in {self.jsonl_path}; stats={json.dumps(self.stats, ensure_ascii=False)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class PromptPoolSubset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.stats = {"kept": len(samples)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def split_prompt_pool(dataset, eval_ratio, eval_max_samples, seed=42):
    if eval_ratio <= 0 or len(dataset) < 10:
        return dataset, PromptPoolSubset([])
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    eval_n = min(max(1, int(len(dataset) * eval_ratio)), int(eval_max_samples), len(dataset) - 1)
    eval_ids = set(indices[:eval_n])
    train_samples = [s for i, s in enumerate(dataset.samples) if i not in eval_ids]
    eval_samples = [s for i, s in enumerate(dataset.samples) if i in eval_ids]
    dataset.samples = train_samples
    dataset.stats["kept"] = len(train_samples)
    return dataset, PromptPoolSubset(eval_samples)


def build_fixed_eval_indices(eval_ds, eval_samples, seed=1234):
    n = min(int(eval_samples), len(eval_ds))
    if n <= 0:
        return []
    indices = list(range(len(eval_ds)))
    random.Random(seed).shuffle(indices)
    return indices[:n]


def shard_prompt_pool(dataset, world_size, rank):
    if world_size <= 1 or dataset is None or len(dataset) == 0:
        return dataset
    samples = [sample for idx, sample in enumerate(dataset.samples) if idx % world_size == rank]
    if not samples:
        samples = [dataset.samples[rank % len(dataset.samples)]]
    return PromptPoolSubset(samples)


# -------------------------- answer and reward helpers --------------------------

def strip_answer(text):
    text = str(text)
    text = re.sub(r"<\|im_start\|>[^\n]*\n?", "", text)
    text = re.sub(r"<\|im_end\|>", "", text)
    if "</think>" in text:
        text = text.split("</think>", 1)[-1]
    return text.strip()


def remove_tool_blocks(text):
    cleaned = re.sub(r"<tool_call>.*?</tool_call>", "", str(text), flags=re.DOTALL)
    cleaned = re.sub(r"<tool_response>.*?</tool_response>", "", cleaned, flags=re.DOTALL)
    return strip_answer(cleaned).strip()


def answer_tag_stats(text):
    clean = remove_tool_blocks(text)
    open_n = len(re.findall(r"<answer>", clean, flags=re.IGNORECASE))
    close_n = len(re.findall(r"</answer>", clean, flags=re.IGNORECASE))
    matches = re.findall(r"<answer>(.*?)</answer>", clean, flags=re.DOTALL | re.IGNORECASE)
    has = 1.0 if matches else 0.0
    exactly_one = 1.0 if len(matches) == 1 and open_n == 1 and close_n == 1 else 0.0
    content = strip_answer(matches[-1]) if matches else clean
    nonempty = 1.0 if content.strip() else 0.0
    malformed = abs(open_n - close_n) + max(0, open_n - len(matches)) + max(0, close_n - len(matches))
    return {
        "has_answer_tag": has,
        "exactly_one_answer_tag": exactly_one,
        "answer_tag_nonempty": nonempty,
        "malformed_answer_tags": float(malformed),
        "answer_content": content,
        "clean_text": clean,
    }


def extract_answer_region(text):
    return answer_tag_stats(text)["answer_content"]


def numeric_candidates(text):
    nums = []
    for match in re.finditer(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?", str(text)):
        try:
            nums.append(float(match.group(0).replace(",", "")))
        except Exception:
            pass
    return nums


def normalize_text(x):
    return re.sub(r"\s+", " ", str(x).lower().strip().replace(",", ""))


def _as_float_or_none(x):
    try:
        s = str(x).strip().replace(",", "")
        if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", s):
            return None
        return float(s)
    except Exception:
        return None


def numeric_gt_items(gt_items):
    vals = []
    for gt in gt_items or []:
        f = _as_float_or_none(gt)
        if f is None:
            return []
        vals.append(f)
    return vals


def num_close(a, b):
    return abs(float(a) - float(b)) <= max(1e-6, abs(float(b)) * 1e-6)


def ordered_numeric_metrics(answer_region, gt_items):
    pred_nums = numeric_candidates(answer_region)
    gt_nums = numeric_gt_items(gt_items)
    if not gt_nums:
        return {
            "pred_nums": pred_nums,
            "gt_nums": gt_nums,
            "ordered_match_count": 0.0,
            "ordered_cov": 0.0,
            "strict_exact_match": 0.0,
            "order_ok": 0.0,
            "extra_numbers": float(len(pred_nums)),
            "missing_numbers": 0.0,
        }
    ordered_match = 0
    for i, gt in enumerate(gt_nums):
        if i < len(pred_nums) and num_close(pred_nums[i], gt):
            ordered_match += 1
    strict_exact = 1.0 if len(pred_nums) == len(gt_nums) and ordered_match == len(gt_nums) else 0.0
    order_ok = 1.0 if ordered_match == min(len(pred_nums), len(gt_nums)) and len(pred_nums) >= len(gt_nums) else 0.0
    return {
        "pred_nums": pred_nums,
        "gt_nums": gt_nums,
        "ordered_match_count": float(ordered_match),
        "ordered_cov": float(ordered_match / max(len(gt_nums), 1)),
        "strict_exact_match": float(strict_exact),
        "order_ok": float(order_ok),
        "extra_numbers": float(max(0, len(pred_nums) - len(gt_nums))),
        "missing_numbers": float(max(0, len(gt_nums) - len(pred_nums))),
    }


def gt_matches_text(text, gt_items):
    answer_text = extract_answer_region(text)
    pred_nums = numeric_candidates(answer_text)
    used = [False] * len(pred_nums)
    norm = normalize_text(answer_text)
    matched = []
    for gt in gt_items:
        gt_s = normalize_text(gt)
        if not gt_s:
            continue
        gt_f = _as_float_or_none(gt)
        ok = False
        if gt_f is not None:
            for i, pred in enumerate(pred_nums):
                if used[i]:
                    continue
                if num_close(pred, gt_f):
                    used[i] = True
                    ok = True
                    break
        else:
            ok = gt_s in norm
        if ok:
            matched.append(gt)
    return len(matched), matched


def rep_penalty(text):
    toks = re.findall(r"\w+|[^\w\s]", str(text).lower())
    if len(toks) < 16:
        return 0.0
    tri = [tuple(toks[i:i + 3]) for i in range(len(toks) - 2)]
    repeat_rate = 1.0 - len(set(tri)) / max(len(tri), 1)
    return min(0.8, max(0.0, repeat_rate - 0.25) * 2.0)


def flatten_json_scalars(obj):
    out = []
    if isinstance(obj, dict):
        for v in obj.values():
            out.extend(flatten_json_scalars(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(flatten_json_scalars(v))
    elif obj is not None:
        s = str(obj).strip()
        if s:
            out.append(s)
    return out


def final_refs_tool_result(final_answer, tool_events):
    answer_region = extract_answer_region(final_answer)
    lowered = answer_region.lower()
    nums = numeric_candidates(answer_region)
    for event in tool_events:
        if not event.get("ok"):
            continue
        for value in flatten_json_scalars(event.get("result", {})):
            if value.lower() in lowered:
                return True
            v = value.replace(",", "")
            if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", v):
                try:
                    target = float(v)
                    if any(num_close(n, target) for n in nums):
                        return True
                except Exception:
                    pass
    return False


def repeated_tool_call_penalty(tool_events):
    keys = [(e.get("name", ""), json.dumps(e.get("args", {}), sort_keys=True, ensure_ascii=False)) for e in tool_events]
    return min(0.8, 0.2 * (len(keys) - len(set(keys)))) if keys else 0.0


def math_task_like(task_text, tools=None):
    return bool(extract_arithmetic_expressions(task_text, max_exprs=1))


def expected_math_call_count(task_text, gt_items):
    expr_n = len(extract_arithmetic_expressions(task_text, max_exprs=16))
    gt_n = len(gt_items or [])
    return max(expr_n, gt_n, 1)


def tool_result_float(event: Dict[str, Any]) -> Optional[float]:
    result = event.get("result", {}) or {}
    for key in ("value", "result"):
        if isinstance(result, dict) and key in result:
            f = _as_float_or_none(result.get(key))
            if f is not None:
                return f
    args = event.get("args", {}) or {}
    expr = args.get("expression", args.get("query", "")) if isinstance(args, dict) else ""
    if expr:
        try:
            return float(safe_eval_math(expr))
        except Exception:
            return None
    return None


def math_tool_value_metrics(task_text: str, gt_items: List[Any], tool_events: List[Dict[str, Any]]):
    exprs = extract_arithmetic_expressions(task_text, max_exprs=16)
    expr_vals: List[float] = []
    for expr in exprs:
        try:
            expr_vals.append(float(safe_eval_math(expr)))
        except Exception:
            pass
    gt_nums = numeric_gt_items(gt_items)
    # If GT length matches expression length, use GT for comparison to tolerate formatting/rounding policies.
    expected_vals = gt_nums if gt_nums and len(gt_nums) == len(expr_vals) else expr_vals
    expected_n = len(expected_vals)
    math_events = [e for e in tool_events if e.get("name") in ("calculate_math", "calculator", "python_math")]
    ok_events = [e for e in math_events if e.get("ok")]
    call_vals = [tool_result_float(e) for e in ok_events]
    call_vals = [v for v in call_vals if v is not None]

    ordered_match = 0
    for i, exp_v in enumerate(expected_vals):
        if i < len(call_vals) and num_close(call_vals[i], exp_v):
            ordered_match += 1
    used = [False] * len(call_vals)
    set_match = 0
    for exp_v in expected_vals:
        for j, got_v in enumerate(call_vals):
            if used[j]:
                continue
            if num_close(got_v, exp_v):
                used[j] = True
                set_match += 1
                break
    over_calls = max(0, len(math_events) - expected_n) if expected_n else len(math_events)
    under_calls = max(0, expected_n - len(math_events))
    return {
        "expected_exprs": exprs,
        "expected_values": expected_vals,
        "tool_values": call_vals,
        "tool_value_order_match_count": float(ordered_match),
        "tool_value_order_cov": float(ordered_match / max(expected_n, 1)) if expected_n else 0.0,
        "tool_value_set_cov": float(set_match / max(expected_n, 1)) if expected_n else 0.0,
        "tool_value_exact": float(expected_n > 0 and len(call_vals) == expected_n and ordered_match == expected_n),
        "tool_over_calls": float(over_calls),
        "tool_under_calls": float(under_calls),
    }


def math_tool_expr_metrics(task_text: str, tool_events: List[Dict[str, Any]]):
    expected = [canonicalize_expression(x) for x in extract_arithmetic_expressions(task_text, max_exprs=16)]
    called = []
    for event in tool_events:
        if event.get("name") not in ("calculate_math", "calculator", "python_math"):
            continue
        args = event.get("args", {}) or {}
        expr = args.get("expression", args.get("query", "")) if isinstance(args, dict) else ""
        expr = canonicalize_expression(expr) if expr else ""
        if expr:
            called.append(expr)

    ordered_match = 0
    for i, exp in enumerate(expected):
        if i < len(called) and called[i] == exp:
            ordered_match += 1
    used = [False] * len(called)
    set_match = 0
    for exp in expected:
        for j, got in enumerate(called):
            if used[j]:
                continue
            if got == exp:
                used[j] = True
                set_match += 1
                break
    expected_n = len(expected)
    return {
        "expected_expr_canon": expected,
        "tool_exprs": called,
        "tool_expr_order_match_count": float(ordered_match),
        "tool_expr_order_cov": float(ordered_match / max(expected_n, 1)) if expected_n else 0.0,
        "tool_expr_set_cov": float(set_match / max(expected_n, 1)) if expected_n else 0.0,
        "tool_expr_exact": float(expected_n > 0 and len(called) == expected_n and ordered_match == expected_n),
    }


def format_strength(args):
    warmup = max(int(getattr(args, "format_warmup_steps", 0)), 0)
    step = max(int(getattr(args, "current_step", 0)), 0)
    if warmup <= 0:
        return 1.0
    return min(1.0, step / max(warmup, 1))


def scheduled_reward_cap(args, initial_attr: str, final_attr: str) -> float:
    final_value = float(getattr(args, final_attr, 0.0))
    initial_value = float(getattr(args, initial_attr, final_value))
    warmup = max(int(getattr(args, "reward_cap_warmup_steps", 0)), 0)
    step = max(int(getattr(args, "current_step", 0)), 0)
    if warmup <= 0:
        return final_value
    strength = min(1.0, step / max(warmup, 1))
    return initial_value + strength * (final_value - initial_value)


def score_one_trajectory(args, prompt, completion, gt_items, tools, tool_events, unfinished, reward_model=None, messages=None):
    task_text = task_text_from_messages(messages) if messages else task_text_from_prompt(prompt)
    tag = answer_tag_stats(completion)
    final_answer = remove_tool_blocks(completion)
    answer_region = tag["answer_content"]
    matched_count, _ = gt_matches_text(completion, gt_items)
    gt_total = max(len(gt_items), 1)
    unordered_cov = matched_count / gt_total if gt_items else 0.0

    nmet = ordered_numeric_metrics(answer_region, gt_items)
    has_numeric_gt = len(nmet["gt_nums"]) > 0
    answer_cov = nmet["ordered_cov"] if has_numeric_gt and bool(getattr(args, "strict_order_reward", 1)) else unordered_cov
    exact_match = nmet["strict_exact_match"] if has_numeric_gt and bool(getattr(args, "strict_exact_reward", 1)) else (1.0 if gt_items and unordered_cov >= 1.0 else 0.0)

    total_calls = len(tool_events)
    success_calls = sum(1 for x in tool_events if x.get("ok"))
    failed_calls = total_calls - success_calls
    tool_success_rate = success_calls / max(total_calls, 1) if total_calls else 0.0
    refs_tool = 1.0 if final_refs_tool_result(completion, tool_events) else 0.0
    answer_tag = tag["has_answer_tag"]
    exact_answer_tag = tag["exactly_one_answer_tag"]
    malformed_tags = float(abs(str(completion).count("<tool_call>") - str(completion).count("</tool_call>"))) + tag["malformed_answer_tags"]
    is_math = math_task_like(task_text, tools)
    expected_math_calls = expected_math_call_count(task_text, gt_items) if is_math else 0

    preferred_math_tool_calls = sum(1 for x in tool_events if x.get("name") == "calculate_math")
    alias_math_tool_calls = sum(1 for x in tool_events if x.get("name") in ("calculator", "python_math"))
    math_tool_calls = preferred_math_tool_calls + alias_math_tool_calls
    non_math_tool_calls = max(0, total_calls - math_tool_calls) if is_math else 0
    successful_math_calls = sum(1 for x in tool_events if x.get("ok") and x.get("name") in ("calculate_math", "calculator", "python_math"))
    successful_preferred_math_calls = sum(1 for x in tool_events if x.get("ok") and x.get("name") == "calculate_math")
    math_call_coverage = min(1.0, successful_math_calls / max(expected_math_calls, 1)) if is_math else 0.0
    preferred_math_rate = successful_preferred_math_calls / max(successful_math_calls, 1) if is_math and successful_math_calls else 0.0
    correct_tool_choice = 1.0 if (is_math and total_calls > 0 and non_math_tool_calls == 0 and preferred_math_tool_calls > 0) else (1.0 if not is_math else 0.0)
    tool_count_exact = 1.0 if (is_math and math_tool_calls == expected_math_calls and non_math_tool_calls == 0) else (1.0 if not is_math else 0.0)
    canonical_tool_rate = safe_mean([1.0 if e.get("canonical", False) else 0.0 for e in tool_events], 0.0) if tool_events else 0.0
    tvm = math_tool_value_metrics(task_text, gt_items, tool_events) if is_math else {
        "tool_value_order_cov": 0.0,
        "tool_value_set_cov": 0.0,
        "tool_value_exact": 0.0,
        "tool_over_calls": 0.0,
        "tool_under_calls": 0.0,
        "tool_value_order_match_count": 0.0,
    }
    tem = math_tool_expr_metrics(task_text, tool_events) if is_math else {
        "tool_expr_order_cov": 0.0,
        "tool_expr_set_cov": 0.0,
        "tool_expr_exact": 0.0,
        "tool_expr_order_match_count": 0.0,
    }
    answer_len = len(answer_region)
    rep = rep_penalty(final_answer)
    repeat_tool_pen = repeated_tool_call_penalty(tool_events)

    reward = float(getattr(args, "reward_base", -0.10))
    reward += float(getattr(args, "reward_answer_cov_weight", 8.0)) * answer_cov
    reward += float(getattr(args, "reward_exact_match_weight", 8.0)) * exact_match

    if is_math:
        if total_calls > 0:
            reward += float(getattr(args, "reward_tool_success_weight", 0.60)) * tool_success_rate
            reward += float(getattr(args, "reward_math_call_cov_weight", 0.60)) * math_call_coverage
            reward += float(getattr(args, "reward_tool_expr_order_weight", 1.40)) * float(tem["tool_expr_order_cov"])
            reward += float(getattr(args, "reward_tool_expr_set_weight", 0.50)) * float(tem["tool_expr_set_cov"])
            reward += float(getattr(args, "reward_tool_expr_exact_weight", 0.60)) * float(tem["tool_expr_exact"])
            reward += float(getattr(args, "reward_tool_value_order_weight", 2.40)) * float(tvm["tool_value_order_cov"])
            reward += float(getattr(args, "reward_tool_value_set_weight", 0.80)) * float(tvm["tool_value_set_cov"])
            reward += float(getattr(args, "reward_tool_value_exact_weight", 0.80)) * float(tvm["tool_value_exact"])
            reward += float(getattr(args, "reward_preferred_math_tool_weight", 0.70)) * preferred_math_rate
            reward += float(getattr(args, "reward_correct_tool_choice_weight", 0.70)) * correct_tool_choice
            reward += float(getattr(args, "reward_tool_count_exact_weight", 0.30)) * tool_count_exact
            reward += float(getattr(args, "reward_canonical_tool_weight", 0.25)) * canonical_tool_rate
            if answer_cov > 0.0:
                reward += float(getattr(args, "reward_final_refs_tool_weight", 0.45)) * refs_tool
            reward -= float(getattr(args, "reward_tool_fail_penalty", 0.55)) * min(failed_calls, 4)
            reward -= float(getattr(args, "reward_wrong_tool_penalty", 0.55)) * min(non_math_tool_calls, 4)
            reward -= float(getattr(args, "reward_tool_over_call_penalty", 0.25)) * min(float(tvm["tool_over_calls"]), 6.0)
            reward -= float(getattr(args, "reward_tool_under_call_penalty", 0.35)) * min(float(tvm["tool_under_calls"]), 6.0)
        else:
            reward -= float(getattr(args, "reward_no_tool_penalty", 1.50))
        if alias_math_tool_calls > 0 and preferred_math_tool_calls <= 0:
            reward -= float(getattr(args, "reward_alias_only_penalty", 0.35))

    strength = format_strength(args)
    missing_format_penalty = float(getattr(args, "format_missing_penalty_initial", 0.30)) + strength * (
        float(getattr(args, "format_missing_penalty_final", 1.50)) - float(getattr(args, "format_missing_penalty_initial", 0.30))
    )
    reward += float(getattr(args, "reward_answer_tag_weight", 0.90)) * answer_tag
    reward += float(getattr(args, "reward_exact_answer_tag_weight", 0.80)) * exact_answer_tag
    if answer_tag <= 0.0 and is_math:
        reward -= missing_format_penalty
        if getattr(args, "format_enforce_mode", "semi_hard") in ("semi_hard", "hard"):
            reward = min(reward, float(getattr(args, "no_answer_tag_reward_cap", 2.0)))
        if getattr(args, "format_enforce_mode", "semi_hard") == "hard" and strength >= 1.0:
            answer_cov = 0.0
            exact_match = 0.0
            reward = min(reward, 0.0)
    reward += 0.10 if malformed_tags == 0 else -0.50 * malformed_tags
    reward += 0.10 if 1 <= answer_len <= int(getattr(args, "max_reward_answer_len", 96)) else -0.30

    extra_numbers = nmet["extra_numbers"] if has_numeric_gt else 0.0
    missing_numbers = nmet["missing_numbers"] if has_numeric_gt else 0.0
    order_ok = nmet["order_ok"] if has_numeric_gt else exact_match
    if has_numeric_gt:
        reward -= float(getattr(args, "extra_number_penalty", 0.55)) * min(extra_numbers, 6.0)
        reward -= float(getattr(args, "missing_number_penalty", 0.70)) * min(missing_numbers, 6.0)
        if order_ok <= 0.0 and unordered_cov > 0.0:
            reward -= float(getattr(args, "order_mismatch_penalty", 1.20))

    reward -= repeat_tool_pen
    reward -= 0.80 if unfinished else 0.0
    reward -= rep

    if gt_items and exact_match < 1.0:
        reward = min(reward, scheduled_reward_cap(args, "non_exact_reward_cap_initial", "non_exact_reward_cap"))
    if gt_items and answer_cov <= 0.0:
        reward = min(reward, scheduled_reward_cap(args, "zero_answer_cov_reward_cap_initial", "zero_answer_cov_reward_cap"))
    if reward_model is not None:
        try:
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            messages = [{"role": role, "content": content.strip()} for role, content in re.findall(pattern, prompt, re.DOTALL)]
            reward += max(min(float(reward_model.get_score(messages, final_answer)), 1.0), -1.0)
        except Exception:
            pass
    raw_reward = float(reward)
    reward = max(min(raw_reward, 12.0), -12.0)

    return reward, {
        "reward": reward,
        "raw_reward": raw_reward,
        "clipped": float(abs(raw_reward - reward) > 1e-6),
        "answer_cov": float(answer_cov),
        "unordered_answer_cov": float(unordered_cov),
        "exact_match": float(exact_match),
        "strict_exact_match": float(nmet["strict_exact_match"] if has_numeric_gt else exact_match),
        "matched_count": float(matched_count),
        "ordered_match_count": float(nmet["ordered_match_count"]),
        "extra_numbers": float(extra_numbers),
        "missing_numbers": float(missing_numbers),
        "order_ok": float(order_ok),
        "tool_calls": float(total_calls),
        "expected_math_calls": float(expected_math_calls),
        "math_tool_calls": float(math_tool_calls),
        "preferred_math_tool_calls": float(preferred_math_tool_calls),
        "alias_math_tool_calls": float(alias_math_tool_calls),
        "non_math_tool_calls": float(non_math_tool_calls),
        "tool_success_calls": float(success_calls),
        "tool_failed_calls": float(failed_calls),
        "tool_success_rate": float(tool_success_rate),
        "math_call_coverage": float(math_call_coverage),
        "tool_expr_order_cov": float(tem["tool_expr_order_cov"]),
        "tool_expr_set_cov": float(tem["tool_expr_set_cov"]),
        "tool_expr_exact": float(tem["tool_expr_exact"]),
        "tool_expr_order_match_count": float(tem["tool_expr_order_match_count"]),
        "tool_value_order_cov": float(tvm["tool_value_order_cov"]),
        "tool_value_set_cov": float(tvm["tool_value_set_cov"]),
        "tool_value_exact": float(tvm["tool_value_exact"]),
        "tool_value_order_match_count": float(tvm["tool_value_order_match_count"]),
        "tool_over_calls": float(tvm["tool_over_calls"]),
        "tool_under_calls": float(tvm["tool_under_calls"]),
        "preferred_math_rate": float(preferred_math_rate),
        "correct_tool_choice": float(correct_tool_choice),
        "tool_count_exact": float(tool_count_exact),
        "canonical_tool_rate": float(canonical_tool_rate),
        "final_refs_tool": float(refs_tool),
        "answer_tag": float(answer_tag),
        "exact_answer_tag": float(exact_answer_tag),
        "answer_tag_nonempty": float(tag["answer_tag_nonempty"]),
        "malformed_tags": float(malformed_tags),
        "answer_len": float(answer_len),
        "unfinished": float(bool(unfinished)),
        "rep_penalty": float(rep),
        "repeat_tool_penalty": float(repeat_tool_pen),
        "format_strength": float(strength),
    }


def calculate_rewards(args, prompts, completions, gt_batch, tools_batch, tool_events_batch, unfinished_batch, num_generations, device, reward_model=None, return_details=False, messages_batch=None):
    rewards_list, details = [], []
    for idx, completion in enumerate(completions):
        sample_idx = idx // max(num_generations, 1)
        reward, detail = score_one_trajectory(
            args,
            prompts[sample_idx],
            completion,
            gt_batch[sample_idx],
            tools_batch[sample_idx],
            tool_events_batch[idx],
            unfinished_batch[idx],
            reward_model,
            messages_batch[sample_idx] if messages_batch is not None else None,
        )
        rewards_list.append(reward)
        details.append(detail)
    rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)
    return (rewards, details) if return_details else rewards


# -------------------------- rollout data structures --------------------------

@dataclass
class Trajectory:
    prompt: str
    messages: List[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]]
    gt: List[Any]
    completion: str
    context: str
    prompt_ids: List[int]
    response_ids: List[int]
    response_mask: List[int]
    response_old_logps: List[float]
    response_versions: List[int]
    turn_outputs: List[str]
    tool_events: List[Dict[str, Any]]
    unfinished: bool
    policy_version: int
    data_index: int
    rollout_time: float


@dataclass
class RolloutGroup:
    prompt: str
    data_index: int
    policy_version: int
    trajectories: List[Trajectory]
    created_at: float


class PolicyVersion:
    def __init__(self, initial=0):
        self._version = int(initial)
        self._lock = threading.Lock()

    def get(self):
        with self._lock:
            return self._version

    def bump(self):
        with self._lock:
            self._version += 1
            return self._version

    def set(self, value):
        with self._lock:
            self._version = int(value)


def execute_tool_call(env, valid_names, tool_call):
    name = tool_call.get("name", "") or ""
    raw_args = tool_call.get("arguments", tool_call.get("args", {}))
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except Exception:
            raw_args = {"expression": raw_args}
    if not isinstance(raw_args, dict):
        raw_args = {}
    if not name:
        return {"ok": False, "name": name, "args": raw_args, "result": {"error": "bad tool call"}, "canonical": bool(tool_call.get("canonical", False))}
    if valid_names and name not in valid_names:
        return {
            "ok": False,
            "name": name,
            "args": raw_args,
            "result": {"error": f"tool not allowed: {name}"},
            "canonical": bool(tool_call.get("canonical", False)),
        }
    try:
        return {"ok": True, "name": name, "args": raw_args, "result": env.execute(name, raw_args), "canonical": bool(tool_call.get("canonical", False))}
    except Exception as exc:
        return {"ok": False, "name": name, "args": raw_args, "result": {"error": str(exc)[:200]}, "canonical": bool(tool_call.get("canonical", False))}


def suffix_delta(old_ids, new_ids):
    if len(new_ids) >= len(old_ids) and new_ids[:len(old_ids)] == old_ids:
        return new_ids[len(old_ids):]
    common = 0
    max_common = min(len(old_ids), len(new_ids))
    while common < max_common and old_ids[common] == new_ids[common]:
        common += 1
    return new_ids[common:]


def rollout_single_online(
    rollout_engine,
    tokenizer,
    env,
    sample,
    max_turns,
    max_new_tokens,
    max_context_len,
    thinking_ratio,
    device,
    policy_version,
    temperature,
):
    messages = [dict(m) for m in sample["messages"]]
    tools = sample["tools"] or env.tool_specs()
    valid_names = set(supported_tool_names(tools)) or env.supported_names()
    prompt = apply_chat_template_compat(tokenizer, messages, tools=tools, add_generation_prompt=True)
    all_outputs, tool_events = [], []
    prompt_ids = None
    response_ids, response_mask, response_old_logps, response_versions = [], [], [], []

    # Full untruncated rendered context used only for observation-delta tracking.
    # The packed training sequence still uses prompt_ids + response_ids as before.
    current_ids_full = None
    final_context = prompt
    unfinished = False
    open_thinking = random.random() < thinking_ratio
    rollout_start = time.time()

    for turn in range(max_turns):
        context = apply_chat_template_compat(tokenizer, messages, tools=tools, add_generation_prompt=True, open_thinking=open_thinking)
        inputs = tokenize_context(tokenizer, context, device=device, max_context_len=max_context_len)
        context_ids = inputs["input_ids"][0].tolist()
        context_ids_full = tokenize_context(tokenizer, context, device="cpu", max_context_len=0)["input_ids"][0].tolist()
        if prompt_ids is None:
            prompt_ids = context_ids
        # Source of truth for full rendered dialogue before the current assistant generation.
        if current_ids_full is None:
            current_ids_full = context_ids_full
        elif current_ids_full != context_ids_full:
            # This should rarely happen. If the chat template re-renders differently,
            # use the freshly rendered context to avoid a broken suffix_delta.
            current_ids_full = context_ids_full
        rr = rollout_engine.rollout(
            inputs["input_ids"],
            inputs["attention_mask"],
            num_generations=1,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        gen_ids_raw = rr.completion_ids[0].tolist()
        gen_logps_raw = rr.per_token_logps[0].tolist()
        pairs = [(int(tok), finite_float(logp)) for tok, logp in zip(gen_ids_raw, gen_logps_raw) if tok != tokenizer.pad_token_id and tok != tokenizer.eos_token_id]
        gen_ids = [x for x, _ in pairs]
        gen_logps = [x for _, x in pairs]
        gen_text = rr.completions[0]
        all_outputs.append(gen_text)
        response_ids.extend(gen_ids)
        response_mask.extend([1] * len(gen_ids))
        response_old_logps.extend(gen_logps)
        response_versions.extend([policy_version] * len(gen_ids))
        assert_rollout_alignment(response_ids, response_mask, response_old_logps, response_versions)

        # Critical fix:
        # current_ids_full must include the assistant generation before computing
        # the observation delta. Otherwise gen_ids will be duplicated as mask=0 obs.
        current_ids_after_gen_full = list(current_ids_full) + list(gen_ids)

        final_context = context + gen_text
        calls = parse_tool_calls(gen_text)
        if not calls:
            break

        unfinished = turn == max_turns - 1
        messages.append({"role": "assistant", "content": gen_text})
        for tool_call in calls:
            event = execute_tool_call(env, valid_names, tool_call)
            tool_events.append(event)
            messages.append({"role": "tool", "content": json.dumps(event["result"], ensure_ascii=False, separators=(",", ":"))[:1024]})
        observe_context = apply_chat_template_compat(tokenizer, messages, tools=tools, add_generation_prompt=not unfinished, open_thinking=open_thinking)
        observe_ids_full = tokenize_context(tokenizer, observe_context, device="cpu", max_context_len=0)["input_ids"][0].tolist()

        obs_delta = suffix_delta(current_ids_after_gen_full, observe_ids_full)

        # Extra safety: if suffix tracking ever fails and the generated text appears
        # again at the front of obs_delta, strip it instead of corrupting PPO context.
        if gen_ids and len(obs_delta) >= len(gen_ids) and obs_delta[:len(gen_ids)] == gen_ids:
            obs_delta = obs_delta[len(gen_ids):]

        current_ids_full = observe_ids_full
        if obs_delta:
            response_ids.extend(obs_delta)
            response_mask.extend([0] * len(obs_delta))
            response_old_logps.extend([0.0] * len(obs_delta))
            response_versions.extend([-1] * len(obs_delta))
            assert_rollout_alignment(response_ids, response_mask, response_old_logps, response_versions)
        final_context = observe_context
        if unfinished:
            break

    # Score the whole assistant trajectory. v3 only scored all_outputs[-1], which can lose
    # an answer or format signal in multi-turn/tool-call trajectories.
    completion = "\n".join(x for x in all_outputs if x is not None)
    assert_rollout_alignment(response_ids, response_mask, response_old_logps, response_versions)
    return Trajectory(
        prompt,
        sample["messages"],
        tools,
        sample["gt"],
        completion,
        final_context,
        prompt_ids or [],
        response_ids,
        response_mask,
        response_old_logps,
        response_versions,
        list(all_outputs),
        tool_events,
        unfinished,
        policy_version,
        -1,
        time.time() - rollout_start,
    )


def rollout_group_online(
    rollout_engine,
    tokenizer,
    env,
    sample,
    data_index,
    num_generations,
    max_turns,
    max_new_tokens,
    max_context_len,
    thinking_ratio,
    device,
    policy_version,
    temperature,
):
    trajectories = []
    for _ in range(num_generations):
        traj = rollout_single_online(
            rollout_engine,
            tokenizer,
            env,
            sample,
            max_turns,
            max_new_tokens,
            max_context_len,
            thinking_ratio,
            device,
            policy_version,
            temperature,
        )
        traj.data_index = data_index
        trajectories.append(traj)
    return RolloutGroup(trajectories[0].prompt if trajectories else "", data_index, policy_version, trajectories, time.time())


class AsyncRolloutManager:
    def __init__(
        self,
        dataset,
        tokenizer,
        rollout_engine,
        env,
        version,
        batch_size,
        num_generations,
        max_turns,
        max_new_tokens,
        max_context_len,
        thinking_ratio,
        temperature,
        device,
        max_version_gap,
        max_queue_groups,
        num_workers,
        engine_lock,
        seed,
    ):
        self.dataset, self.tokenizer, self.rollout_engine, self.env, self.version = dataset, tokenizer, rollout_engine, env, version
        self.batch_size, self.num_generations, self.max_turns, self.max_new_tokens = int(batch_size), int(num_generations), int(max_turns), int(max_new_tokens)
        self.max_context_len, self.thinking_ratio, self.temperature, self.device = int(max_context_len), float(thinking_ratio), float(temperature), device
        self.max_version_gap = max(0, int(max_version_gap))
        self.max_pending_groups = max(1, int(max_queue_groups))
        self.num_workers = max(0, int(num_workers))
        self.engine_lock, self.seed = engine_lock, int(seed)
        self.buffer = queue.Queue(maxsize=self.max_pending_groups)
        self.stop_event = threading.Event()
        self.threads = []
        self.index_lock = threading.Lock()
        self.next_index = 0
        self.stats_lock = threading.Lock()
        self.rollout_time_hist = deque(maxlen=512)
        self.train_wait_hist = deque(maxlen=512)
        self.rollout_lock_wait_hist = deque(maxlen=512)
        self.queue_hist = deque(maxlen=512)
        self.stats = {
            "produced": 0.0,
            "accepted": 0.0,
            "generated": 0.0,
            "stale_drop": 0.0,
            "errors": 0.0,
            "queue_full": 0.0,
            "rollout_time_sum": 0.0,
            "rollout_lock_wait_sum": 0.0,
            "train_wait_sum": 0.0,
            "train_batches": 0.0,
            "rollout_turns": 0.0,
            "rollout_tool_calls": 0.0,
            "rollout_tokens": 0.0,
            "last_train_wait": 0.0,
            "last_rollout_time": 0.0,
            "last_rollout_lock_wait": 0.0,
        }

    def _next_sample(self, rng):
        with self.index_lock:
            if rng.random() < 0.50:
                idx = self.next_index % len(self.dataset)
                self.next_index += 1
            else:
                idx = rng.randrange(len(self.dataset))
        return idx, self.dataset[idx]

    def _inc(self, key, val=1.0):
        with self.stats_lock:
            self.stats[key] = self.stats.get(key, 0.0) + float(val)

    def _set(self, key, val):
        with self.stats_lock:
            self.stats[key] = float(val)

    def _push_hist(self, hist: deque, value: float):
        with self.stats_lock:
            hist.append(float(value))

    def _producer(self, worker_id):
        rank = dist.get_rank() if dist.is_initialized() else 0
        rng = random.Random(self.seed + 1009 * (worker_id + 1) + 9173 * rank)
        while not self.stop_event.is_set():
            try:
                if self.buffer.qsize() >= self.max_pending_groups:
                    self._inc("queue_full")
                    time.sleep(0.05)
                    continue
                data_index, sample = self._next_sample(rng)
                policy_version = self.version.get()
                lock_wait_start = time.time()
                with self.engine_lock:
                    lock_wait = time.time() - lock_wait_start
                    t0 = time.time()
                    group = rollout_group_online(
                        self.rollout_engine,
                        self.tokenizer,
                        self.env,
                        sample,
                        data_index,
                        self.num_generations,
                        self.max_turns,
                        self.max_new_tokens,
                        self.max_context_len,
                        self.thinking_ratio,
                        self.device,
                        policy_version,
                        self.temperature,
                    )
                    rollout_sec = time.time() - t0
                self._inc("produced")
                self._inc("generated", len(group.trajectories))
                self._inc("rollout_time_sum", rollout_sec)
                self._inc("rollout_lock_wait_sum", lock_wait)
                self._inc("rollout_turns", sum(len(t.turn_outputs) for t in group.trajectories))
                self._inc("rollout_tool_calls", sum(len(t.tool_events) for t in group.trajectories))
                self._inc("rollout_tokens", sum(sum(t.response_mask) for t in group.trajectories))
                self._set("last_rollout_time", rollout_sec)
                self._set("last_rollout_lock_wait", lock_wait)
                self._push_hist(self.rollout_time_hist, rollout_sec)
                self._push_hist(self.rollout_lock_wait_hist, lock_wait)
                self._push_hist(self.queue_hist, self.buffer.qsize())
                if self.version.get() - group.policy_version > self.max_version_gap:
                    self._inc("stale_drop")
                    continue
                self.buffer.put(group, timeout=1.0)
                self._inc("accepted")
            except queue.Full:
                self._inc("queue_full")
            except Exception as exc:
                self._inc("errors")
                if is_main_process():
                    Logger(f"[WARN] rollout worker {worker_id} error: {repr(exc)}")
                time.sleep(0.2)

    def start(self):
        for worker_id in range(self.num_workers):
            thread = threading.Thread(target=self._producer, args=(worker_id,), daemon=False)
            thread.start()
            self.threads.append(thread)

    def get_batch(self, timeout=None):
        groups = []
        start = time.time()
        while len(groups) < self.batch_size and not self.stop_event.is_set():
            try:
                group = self.buffer.get(timeout=timeout if timeout is not None else 60.0)
            except queue.Empty:
                break
            if self.version.get() - group.policy_version > self.max_version_gap:
                self._inc("stale_drop")
                continue
            groups.append(group)
        wait_sec = time.time() - start
        self._set("last_train_wait", wait_sec)
        self._inc("train_wait_sum", wait_sec)
        self._inc("train_batches")
        self._push_hist(self.train_wait_hist, wait_sec)
        self._push_hist(self.queue_hist, self.buffer.qsize())
        return groups

    def get_one(self, timeout=None):
        start = time.time()
        while not self.stop_event.is_set():
            try:
                group = self.buffer.get(timeout=timeout if timeout is not None else 60.0)
            except queue.Empty:
                return None
            if self.version.get() - group.policy_version > self.max_version_gap:
                self._inc("stale_drop")
                continue
            wait_sec = time.time() - start
            self._set("last_train_wait", wait_sec)
            self._inc("train_wait_sum", wait_sec)
            self._inc("train_batches")
            self._push_hist(self.train_wait_hist, wait_sec)
            self._push_hist(self.queue_hist, self.buffer.qsize())
            return group
        return None

    def wait_until_ready(self, min_groups: int, timeout_sec: float = 180.0):
        target = min(max(int(min_groups), 0), self.max_pending_groups)
        if target <= 0:
            return True
        start = time.time()
        while not self.stop_event.is_set():
            if self.buffer.qsize() >= target:
                return True
            if (time.time() - start) >= max(float(timeout_sec), 0.0):
                return False
            time.sleep(0.05)
        return False

    def snapshot_stats(self):
        with self.stats_lock:
            out = dict(self.stats)
            rollout_hist = list(self.rollout_time_hist)
            train_hist = list(self.train_wait_hist)
            lock_hist = list(self.rollout_lock_wait_hist)
            queue_hist = list(self.queue_hist)
        produced = max(out.get("produced", 0.0), 1.0)
        generated = max(out.get("generated", 0.0), 1.0)
        train_batches = max(out.get("train_batches", 0.0), 1.0)
        out["queued"] = float(self.buffer.qsize())
        out["queue_capacity"] = float(self.max_pending_groups)
        out["queue_utilization"] = out["queued"] / max(out["queue_capacity"], 1.0)
        out["online_group_accept_rate"] = out.get("accepted", 0.0) / produced
        out["stale_drop_rate"] = out.get("stale_drop", 0.0) / produced
        out["avg_rollout_sec"] = out.get("rollout_time_sum", 0.0) / produced
        out["avg_train_wait_sec"] = out.get("train_wait_sum", 0.0) / train_batches
        out["avg_tokens_per_traj"] = out.get("rollout_tokens", 0.0) / generated
        out["avg_turns_per_traj"] = out.get("rollout_turns", 0.0) / generated
        out["avg_tool_calls_per_traj"] = out.get("rollout_tool_calls", 0.0) / generated
        out["rollout_sec_p50"] = percentile(rollout_hist, 0.50)
        out["rollout_sec_p90"] = percentile(rollout_hist, 0.90)
        out["rollout_sec_p99"] = percentile(rollout_hist, 0.99)
        out["train_wait_sec_p50"] = percentile(train_hist, 0.50)
        out["train_wait_sec_p90"] = percentile(train_hist, 0.90)
        out["train_wait_sec_p99"] = percentile(train_hist, 0.99)
        out["rollout_lock_wait_sec_p90"] = percentile(lock_hist, 0.90)
        out["queue_size_p50"] = percentile(queue_hist, 0.50)
        out["queue_size_p90"] = percentile(queue_hist, 0.90)
        return out

    def stop(self):
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=30.0)


def sync_rollout_policy(args, rollout_engine, rollout_model, model, engine_lock, version):
    with engine_lock:
        if args.rollout_engine == "torch":
            ok = copy_policy_weights(rollout_model, model)
            if ok:
                rollout_engine.update_policy(rollout_model)
        else:
            ok = bool(rollout_engine.update_policy(model))
    return version.bump() if ok else version.get()


def manager_pop_one_group(manager, timeout=None):
    return manager.get_one(timeout=timeout)


def sync_rollout_policy_distributed(args, rollout_engine, rollout_model, model, engine_lock, version, src_rank=0):
    raw_model = unwrap_model(model)
    state_tensors = [tensor for tensor in raw_model.state_dict().values() if torch.is_tensor(tensor)]
    ok = True
    if dist.get_rank() == src_rank and model_has_nonfinite(model):
        ok = False
    ok_tensor = torch.tensor([1 if ok else 0], device=args.device, dtype=torch.int64)
    dist.broadcast(ok_tensor, src=src_rank)
    ok = bool(ok_tensor.item() > 0)
    if ok:
        for tensor in state_tensors:
            dist.broadcast(tensor.data, src=src_rank)
    with engine_lock:
        if args.rollout_engine == "torch":
            if rollout_model is not raw_model:
                ok = ok and copy_policy_weights(rollout_model, raw_model)
            if ok:
                rollout_engine.update_policy(rollout_model)
        else:
            ok = ok and bool(rollout_engine.update_policy(raw_model))
    version_value = version.bump() if (dist.get_rank() == src_rank and ok) else version.get()
    version_tensor = torch.tensor([int(version_value)], device=args.device, dtype=torch.int64)
    dist.broadcast(version_tensor, src=src_rank)
    version.set(int(version_tensor.item()))
    return version.get()


def distributed_fetch_groups(rank, world_size, manager, pending_groups, target_groups, fetch_timeout_sec=180.0, producer_ranks=None, src_rank=0, min_return_groups=None):
    start = time.time()
    target_groups = max(0, int(target_groups))
    if min_return_groups is None:
        min_return_groups = target_groups
    min_return_groups = max(1, int(min_return_groups)) if target_groups > 0 else 0
    if target_groups > 0:
        min_return_groups = min(min_return_groups, target_groups)
    producers = max(1, int(producer_ranks) if producer_ranks is not None else int(world_size))
    while True:
        if rank == src_rank:
            available = len(pending_groups)
            need = max(0, target_groups - available)
            if available >= min_return_groups:
                rounds = 0
            else:
                rounds = 0 if need <= 0 else math.ceil(need / producers)
            if need > 0 and available < min_return_groups and (time.time() - start) >= max(float(fetch_timeout_sec), 0.0):
                rounds = -1
            ctrl = [int(rounds)]
        else:
            ctrl = [0]
        dist.broadcast_object_list(ctrl, src=src_rank)
        rounds = int(ctrl[0])
        if rounds < 0:
            return None if rank == src_rank else False
        if rounds == 0:
            break

        for _ in range(rounds):
            remain = max(0.0, float(fetch_timeout_sec) - (time.time() - start))
            local_group = manager_pop_one_group(manager, timeout=min(max(remain, 0.2), 30.0)) if (remain > 0.0 and rank != src_rank) else None
            if rank == src_rank:
                gathered = [None for _ in range(world_size)]
                dist.gather_object(local_group, object_gather_list=gathered, dst=src_rank)
                for item in gathered:
                    if item is not None:
                        pending_groups.append(item)
            else:
                dist.gather_object(local_group, dst=src_rank)

    if rank == src_rank:
        if len(pending_groups) < min_return_groups:
            return None
        out = [pending_groups.pop() for _ in range(min(target_groups, len(pending_groups)))]
        return out if out else None
    return True


# -------------------------- packing / PPO loss --------------------------

def pack_groups(groups, tokenizer, max_total_len, device, num_generations, strict_group_packing=True):
    prompts, messages_batch, gt_batch, tools_batch, versions = [], [], [], [], []
    trajectory_versions = []
    completions, tool_events_batch, unfinished_batch, packed = [], [], [], []
    dropped_groups = 0
    for group in groups:
        if len(group.trajectories) != num_generations:
            dropped_groups += 1
            continue
        group_packed, group_completions, group_tool_events, group_unfinished = [], [], [], []
        for traj in group.trajectories:
            ids = traj.prompt_ids + traj.response_ids
            mask = [0] * len(traj.prompt_ids) + traj.response_mask
            old_logps = [0.0] * max(len(traj.prompt_ids) - 1, 0) + traj.response_old_logps
            token_versions = [-1] * max(len(traj.prompt_ids) - 1, 0) + traj.response_versions
            if len(ids) > max_total_len:
                ids = ids[-max_total_len:]
                mask = mask[-max_total_len:]
                old_logps = old_logps[-(len(ids) - 1):]
                token_versions = token_versions[-(len(ids) - 1):]
            if len(ids) < 2 or sum(mask) <= 0 or len(old_logps) != len(ids) - 1 or len(token_versions) != len(ids) - 1:
                group_packed = []
                break
            group_packed.append((ids, mask, old_logps, token_versions))
            group_completions.append(traj.completion)
            group_tool_events.append(traj.tool_events)
            group_unfinished.append(traj.unfinished)
        if strict_group_packing and len(group_packed) != num_generations:
            dropped_groups += 1
            continue
        if not group_packed:
            dropped_groups += 1
            continue
        prompts.append(group.prompt)
        messages_batch.append(group.trajectories[0].messages)
        gt_batch.append(group.trajectories[0].gt)
        tools_batch.append(group.trajectories[0].tools)
        versions.append(group.policy_version)
        trajectory_versions.extend([group.policy_version] * len(group_packed))
        packed.extend(group_packed)
        completions.extend(group_completions)
        tool_events_batch.extend(group_tool_events)
        unfinished_batch.extend(group_unfinished)
    if not packed:
        raise RuntimeError("empty packed batch")
    if len(packed) % num_generations != 0:
        raise RuntimeError(f"packed trajectories not divisible by num_generations: {len(packed)} vs {num_generations}")
    seq_lens = torch.tensor([len(ids) for ids, _, _, _ in packed], device=device, dtype=torch.long)
    max_len = int(seq_lens.max().item())
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = torch.tensor([ids + [pad_id] * (max_len - len(ids)) for ids, _, _, _ in packed], device=device, dtype=torch.long)
    full_response_masks = torch.tensor([mask + [0] * (max_len - len(mask)) for _, mask, _, _ in packed], device=device, dtype=torch.float32)
    old_per_token_logps = torch.tensor([old_lp + [0.0] * ((max_len - 1) - len(old_lp)) for _, _, old_lp, _ in packed], device=device, dtype=torch.float32)
    per_token_versions = torch.tensor([v + [-1] * ((max_len - 1) - len(v)) for _, _, _, v in packed], device=device, dtype=torch.long)
    return {
        "input_ids": input_ids,
        "seq_lens": seq_lens,
        "full_response_masks": full_response_masks,
        "old_per_token_logps": old_per_token_logps,
        "per_token_versions": per_token_versions,
        "prompts": prompts,
        "messages_batch": messages_batch,
        "gt_batch": gt_batch,
        "tools_batch": tools_batch,
        "versions": versions,
        "trajectory_versions": trajectory_versions,
        "completions": completions,
        "tool_events_batch": tool_events_batch,
        "unfinished_batch": unfinished_batch,
        "dropped_groups": dropped_groups,
    }


def resolve_prox_logps(old_logps, current_logps_detached, token_versions, current_version, method="loglinear"):
    old_logps = torch.nan_to_num(old_logps, nan=0.0, posinf=0.0, neginf=0.0)
    current_logps_detached = torch.nan_to_num(current_logps_detached, nan=0.0, posinf=0.0, neginf=0.0)
    if method == "behavior":
        return old_logps
    valid = token_versions >= 0
    v_behav = token_versions.float()
    v_theta = torch.full_like(v_behav, float(current_version))
    v_prox = torch.full_like(v_behav, float(max(current_version - 1, 0)))
    alpha = ((v_prox - v_behav) / (v_theta - v_behav).clamp(min=1.0)).clamp(min=0.0, max=1.0)
    prox = old_logps + alpha * (current_logps_detached - old_logps)
    return torch.where(valid, prox, old_logps)


def masked_stats(values, mask, prefix):
    with torch.no_grad():
        selected = values[mask.bool()]
        if selected.numel() == 0:
            return {f"{prefix}_mean": 0.0, f"{prefix}_p05": 0.0, f"{prefix}_p50": 0.0, f"{prefix}_p95": 0.0}
        selected = selected.float().detach()
        return {
            f"{prefix}_mean": float(selected.mean().item()),
            f"{prefix}_p05": float(torch.quantile(selected, 0.05).item()),
            f"{prefix}_p50": float(torch.quantile(selected, 0.50).item()),
            f"{prefix}_p95": float(torch.quantile(selected, 0.95).item()),
        }


def build_advantages(args, rewards, num_generations):
    grouped_rewards = rewards.view(-1, num_generations)
    group_mean = grouped_rewards.mean(dim=1).repeat_interleave(num_generations)
    group_std_raw = grouped_rewards.std(dim=1, unbiased=False)
    group_std = group_std_raw.repeat_interleave(num_generations)
    group_adv = (rewards - group_mean) / group_std.clamp(min=max(float(args.adv_min_std), 1e-4))
    baseline = float(getattr(args, "reward_baseline", 0.0))
    scale = max(float(getattr(args, "reward_scale", 3.0)), 1e-3)
    baseline_adv = (rewards - baseline) / scale
    zero_group = (group_std_raw < args.adv_min_std).repeat_interleave(num_generations)
    zero_std_mode = str(getattr(args, "zero_std_advantage_mode", "baseline"))
    if zero_std_mode == "zero":
        zero_std_adv = torch.zeros_like(baseline_adv)
    elif zero_std_mode == "positive_baseline":
        zero_std_adv = baseline_adv.clamp(min=0.0)
    else:
        zero_std_adv = baseline_adv
    if args.advantage_mode == "group":
        advantages = group_adv
    elif args.advantage_mode == "baseline":
        advantages = baseline_adv
    else:
        advantages = torch.where(zero_group, zero_std_adv, group_adv)
    if args.adv_clip > 0:
        advantages = advantages.clamp(-args.adv_clip, args.adv_clip)
    return advantages, {
        "reward/group_std": float(group_std_raw.mean().item()),
        "reward/zero_std_group_ratio": float((group_std_raw < args.adv_min_std).float().mean().item()),
        "reward/nonzero_adv_ratio": float((advantages.abs() > 1e-6).float().mean().item()),
        "adv/negative_ratio": float((advantages < -1e-6).float().mean().item()),
        "adv/positive_ratio": float((advantages > 1e-6).float().mean().item()),
        "adv/mean": float(advantages.mean().item()),
        "adv/std": float(advantages.std(unbiased=False).item()),
    }


# -------------------------- BC auxiliary loss --------------------------

def canonical_tool_call(name: str, args: Dict[str, Any]) -> str:
    return "<tool_call>" + json.dumps({"name": name, "arguments": args}, ensure_ascii=False, separators=(",", ":")) + "</tool_call>"


def build_math_bc_examples(args, tokenizer, env, messages_batch, tools_batch, gt_batch, device):
    examples = []
    max_examples = int(getattr(args, "bc_max_examples", 16))
    for messages, tools, gt_items in zip(messages_batch, tools_batch, gt_batch):
        if len(examples) >= max_examples:
            break
        tools = tools or env.tool_specs()
        text = task_text_from_messages(messages)
        exprs = extract_arithmetic_expressions(text, max_exprs=max(1, int(getattr(args, "bc_max_exprs", 8))))
        if not exprs:
            continue
        call_texts, tool_results = [], []
        for expr in exprs:
            try:
                result = env.execute("calculate_math", {"expression": expr})
            except Exception:
                continue
            call_texts.append(canonical_tool_call("calculate_math", {"expression": expr}))
            tool_results.append(result)
        if not call_texts:
            continue

        gt_clean = [str(x).strip() for x in (gt_items or []) if str(x).strip()]
        # If GT count agrees with extracted expression count, final-answer BC uses GT; otherwise use tool results.
        if gt_clean and len(gt_clean) == len(tool_results):
            answer_values = gt_clean
        else:
            answer_values = [str(x.get("result", "")) for x in tool_results]
        answer_text = "<answer>" + ", ".join(answer_values) + "</answer>"

        prefix1 = apply_chat_template_compat(tokenizer, messages, tools=tools, add_generation_prompt=True)
        examples.append((prefix1, "\n".join(call_texts)))

        hist = [dict(m) for m in messages] + [{"role": "assistant", "content": "\n".join(call_texts)}]
        for result in tool_results:
            hist.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False, separators=(",", ":"))})
        prefix2 = apply_chat_template_compat(tokenizer, hist, tools=tools, add_generation_prompt=True)
        examples.append((prefix2, answer_text))
    return examples[:max_examples]


def bc_coef_at_step(args) -> float:
    base = float(getattr(args, "bc_coef", 0.15))
    if base <= 0:
        return 0.0
    step = int(getattr(args, "current_step", 0))
    hold = int(getattr(args, "bc_hold_steps", 1200))
    decay = int(getattr(args, "bc_decay_steps", 4000))
    min_coef = max(0.0, float(getattr(args, "bc_min_coef", 0.03)))
    stop = int(getattr(args, "bc_stop_steps", 0))
    if stop > 0 and step > stop:
        return 0.0
    if step <= hold:
        return base
    if decay <= 0:
        return max(min_coef, 0.0)
    frac = max(0.0, 1.0 - (step - hold) / max(decay, 1))
    return max(base * frac, min_coef)


def compute_bc_aux_loss(args, batch, model, tokenizer, env, autocast_factory):
    coef = bc_coef_at_step(args)
    if coef <= 0:
        zero = torch.tensor(0.0, device=args.device)
        return zero, {"bc/loss": 0.0, "bc/coef": 0.0, "bc/examples": 0.0}
    examples = build_math_bc_examples(args, tokenizer, env, batch["messages_batch"], batch["tools_batch"], batch["gt_batch"], args.device)
    if not examples:
        zero = torch.tensor(0.0, device=args.device)
        return zero, {"bc/loss": 0.0, "bc/coef": float(coef), "bc/examples": 0.0}

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id
    max_len = int(getattr(args, "bc_max_len", min(args.max_total_len, 1536)))
    all_ids, all_labels = [], []
    for prefix, target in examples:
        prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
        if eos_id is not None:
            target_ids = target_ids + [int(eos_id)]
        if len(target_ids) >= max_len:
            ids = target_ids[-max_len:]
            labels = ids[:]
        else:
            keep_prefix = max(0, max_len - len(target_ids))
            kept_prefix_ids = prefix_ids[-keep_prefix:] if keep_prefix > 0 else []
            ids = kept_prefix_ids + target_ids
            labels = [-100] * len(kept_prefix_ids) + target_ids
        if len(ids) < 2 or all(x == -100 for x in labels[1:]):
            continue
        all_ids.append(ids)
        all_labels.append(labels)

    if not all_ids:
        zero = torch.tensor(0.0, device=args.device)
        return zero, {"bc/loss": 0.0, "bc/coef": float(coef), "bc/examples": 0.0}

    ml = max(len(x) for x in all_ids)
    input_ids = torch.tensor([x + [pad_id] * (ml - len(x)) for x in all_ids], device=args.device, dtype=torch.long)
    labels = torch.tensor([x + [-100] * (ml - len(x)) for x in all_labels], device=args.device, dtype=torch.long)
    attn = (input_ids != pad_id).long()
    with autocast_factory():
        res = unwrap_model(model)(input_ids=input_ids, attention_mask=attn)
        logits = res.logits[:, :-1, :]
    labels_shift = labels[:, 1:]
    loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), labels_shift.reshape(-1), ignore_index=-100)
    return loss, {"bc/loss": float(loss.detach().item()), "bc/coef": float(coef), "bc/examples": float(len(all_ids))}


def build_dashboard_metrics(detail, prefix="train"):
    return {
        f"{prefix}/accuracy_exact_match": float(detail.get("exact_match", 0.0)),
        f"{prefix}/accuracy_answer_cov": float(detail.get("answer_cov", 0.0)),
        f"{prefix}/format_answer_tag_rate": float(detail.get("answer_tag", 0.0)),
        f"{prefix}/format_exact_answer_tag_rate": float(detail.get("exact_answer_tag", 0.0)),
        f"{prefix}/tool_success_rate": float(detail.get("tool_success_rate", 0.0)),
        f"{prefix}/tool_correct_choice_rate": float(detail.get("correct_tool_choice", 0.0)),
        f"{prefix}/tool_math_call_coverage": float(detail.get("math_call_coverage", 0.0)),
        f"{prefix}/tool_expr_order_cov": float(detail.get("tool_expr_order_cov", 0.0)),
        f"{prefix}/tool_expr_set_cov": float(detail.get("tool_expr_set_cov", 0.0)),
        f"{prefix}/tool_expr_exact": float(detail.get("tool_expr_exact", 0.0)),
        f"{prefix}/tool_value_order_cov": float(detail.get("tool_value_order_cov", 0.0)),
        f"{prefix}/tool_preferred_math_rate": float(detail.get("preferred_math_rate", 0.0)),
        f"{prefix}/tool_canonical_rate": float(detail.get("canonical_tool_rate", 0.0)),
    }


def compute_async_loss(args, batch, model, ref_model, reward_model, tokenizer, autocast_factory, lm_config, env):
    input_ids = batch["input_ids"]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    full_mask = (input_ids != pad_id).long()
    full_response_masks = batch["full_response_masks"]
    old_per_token_logps = torch.nan_to_num(batch["old_per_token_logps"], nan=0.0, posinf=0.0, neginf=0.0)
    per_token_versions = batch["per_token_versions"]

    with autocast_factory():
        res = unwrap_model(model)(input_ids=input_ids, attention_mask=full_mask)
        aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
        logits = res.logits[:, :-1, :]
    logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(-80.0, 80.0)
    target_ids = input_ids[:, 1:].clamp(min=0, max=logits.size(-1) - 1)
    per_token_logps = F.log_softmax(logits, dim=-1).gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        ref_per_token_logps = compute_per_token_logps_safe(ref_model, input_ids, full_mask, autocast_factory=None)

    completion_mask = full_response_masks[:, 1:]
    is_eos = (input_ids[:, 1:] == tokenizer.eos_token_id) & completion_mask.bool()
    eos_idx = torch.full((completion_mask.size(0),), completion_mask.size(1) - 1, device=args.device, dtype=torch.long)
    has_eos = is_eos.any(dim=1)
    eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
    pos = torch.arange(completion_mask.size(1), device=args.device).unsqueeze(0)
    completion_mask = completion_mask * (pos <= eos_idx.unsqueeze(1)).float()
    trajectory_versions = batch.get("trajectory_versions", [])
    if trajectory_versions and len(trajectory_versions) == completion_mask.size(0):
        row_versions = torch.tensor(trajectory_versions, device=args.device, dtype=torch.long)
        row_staleness = (torch.full_like(row_versions, int(args.current_version)) - row_versions).clamp(min=0)
    else:
        row_staleness = torch.zeros(completion_mask.size(0), device=args.device, dtype=torch.long)
    stale_discount = min(max(float(getattr(args, "async_stale_discount", 0.75)), 0.0), 1.0)
    row_weights = torch.pow(torch.full_like(row_staleness, stale_discount, dtype=torch.float32), row_staleness.float()) if stale_discount < 1.0 else torch.ones_like(row_staleness, dtype=torch.float32)

    rewards, reward_details = calculate_rewards(
        args,
        batch["prompts"],
        batch["completions"],
        batch["gt_batch"],
        batch["tools_batch"],
        batch["tool_events_batch"],
        batch["unfinished_batch"],
        args.num_generations,
        device=args.device,
        reward_model=reward_model,
        return_details=True,
        messages_batch=batch["messages_batch"],
    )
    advantages, adv_metrics = build_advantages(args, rewards, args.num_generations)
    ref_delta = (ref_per_token_logps - per_token_logps).clamp(-20, 20)
    per_token_kl = torch.exp(ref_delta) - ref_delta - 1

    if args.use_decoupled_ppo:
        prox_logps = resolve_prox_logps(old_per_token_logps, per_token_logps.detach(), per_token_versions, args.current_version, args.prox_logp_method)
        ratio = torch.exp((per_token_logps - prox_logps).clamp(-10, 10))
        behav_weight = torch.exp((prox_logps - old_per_token_logps).clamp(-10, 10)).detach()
        if args.behav_imp_weight_cap > 0:
            behav_cap_frac = ((behav_weight > args.behav_imp_weight_cap).float() * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            behav_weight = behav_weight.clamp(max=args.behav_imp_weight_cap)
        else:
            behav_cap_frac = torch.tensor(0.0, device=args.device)
        clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
        surrogate = behav_weight * torch.min(ratio * advantages.unsqueeze(1), clipped_ratio * advantages.unsqueeze(1))
        m2_mask_frac = torch.tensor(0.0, device=args.device)
        if args.m2_threshold > 0:
            m2 = (old_per_token_logps - prox_logps).pow(2)
            keep = (m2 <= args.m2_threshold).float()
            m2_mask_frac = ((1.0 - keep) * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            completion_mask = completion_mask * keep
    else:
        ratio = torch.exp((per_token_logps - old_per_token_logps).clamp(-10, 10))
        clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
        surrogate = torch.min(ratio * advantages.unsqueeze(1), clipped_ratio * advantages.unsqueeze(1))
        behav_weight = torch.ones_like(ratio)
        behav_cap_frac = torch.tensor(0.0, device=args.device)
        m2_mask_frac = torch.tensor(0.0, device=args.device)

    per_token_loss = -(surrogate - args.beta * per_token_kl)
    token_counts = completion_mask.sum(dim=1)
    valid_rows = token_counts > 0
    if getattr(args, "loss_reduction", "token") == "token":
        weighted_mask = completion_mask * row_weights.unsqueeze(1)
        policy_loss = (per_token_loss * weighted_mask).sum() / weighted_mask.sum().clamp(min=1.0)
    else:
        row_loss = (per_token_loss * completion_mask).sum(dim=1) / token_counts.clamp(min=1.0)
        policy_loss = (row_loss[valid_rows] * row_weights[valid_rows]).sum() / row_weights[valid_rows].sum().clamp(min=1.0) if valid_rows.any() else per_token_loss.sum() * 0.0

    bc_loss, bc_metrics = compute_bc_aux_loss(args, batch, model, tokenizer, env, autocast_factory)
    bc_coef = float(bc_metrics.get("bc/coef", 0.0))
    loss = policy_loss + aux_loss + bc_coef * bc_loss

    with torch.no_grad():
        mask_denom = completion_mask.sum().clamp(min=1.0)
        ratio_clip_frac = ((((ratio < 1 - args.epsilon) | (ratio > 1 + args.epsilon)).float() * completion_mask).sum() / mask_denom).item()
        approx_kl = (((per_token_logps - old_per_token_logps) * completion_mask).sum() / mask_denom).item()
        old_current_abs_delta = (((per_token_logps - old_per_token_logps).abs() * completion_mask).sum() / mask_denom).item()
        version_gaps = torch.tensor([args.current_version - v for v in batch["versions"]], device=args.device, dtype=torch.float32) if batch["versions"] else torch.zeros(1, device=args.device)
        valid_version_mask = (per_token_versions >= 0) & completion_mask.bool()
        token_stale = torch.where(valid_version_mask, torch.full_like(per_token_versions, int(args.current_version)) - per_token_versions, torch.zeros_like(per_token_versions)).float()
        reward_scalar = {k: safe_mean([d[k] for d in reward_details]) for k in reward_details[0].keys()} if reward_details else {}
        metrics = {
            "loss/total": float(loss.detach().item()),
            "loss/policy": float(policy_loss.detach().item()),
            "loss/aux": float(aux_loss.detach().item()),
            "reward/mean": float(rewards.mean().item()),
            "reward/min": float(rewards.min().item()),
            "reward/max": float(rewards.max().item()),
            "reward/baseline": float(getattr(args, "reward_baseline", 0.0)),
            "length/avg_completion_tokens": float(token_counts.float().mean().item()),
            "policy/kl_ref_k3": float(((per_token_kl * completion_mask).sum() / mask_denom).item()),
            "policy/approx_kl_old": float(approx_kl),
            "policy/approx_kl_old_abs": float(abs(approx_kl)),
            "policy/old_current_abs_delta": float(old_current_abs_delta),
            "policy/entropy": float(logits_entropy_safe(logits, completion_mask)),
            "policy/ratio_clip_frac": float(ratio_clip_frac),
            "policy/behav_imp_weight_cap_frac": float(behav_cap_frac.item()),
            "policy/m2_mask_frac": float(m2_mask_frac.item()),
            "policy/valid_completion_rows": float(valid_rows.float().mean().item()),
            "async/row_staleness_mean": float(row_staleness.float().mean().item()),
            "async/row_weight_mean": float(row_weights.mean().item()),
            "async/staleness_group_mean": float(version_gaps.mean().item()),
            "async/staleness_group_max": float(version_gaps.max().item()),
            "pack/dropped_groups": float(batch.get("dropped_groups", 0)),
            **adv_metrics,
            **masked_stats(ratio.detach(), completion_mask, "policy/ratio"),
            **masked_stats(behav_weight.detach(), completion_mask, "policy/behav_weight"),
            **masked_stats(token_stale.detach(), valid_version_mask.float(), "async/token_staleness"),
            **bc_metrics,
        }
        for k, v in reward_scalar.items():
            metrics[f"reward_detail/{k}"] = float(v)
        metrics.update(build_dashboard_metrics(reward_scalar, prefix="train"))
    return loss, metrics, {"rewards": [float(x) for x in rewards.detach().cpu().tolist()], "reward_details": reward_details}


@torch.no_grad()
def run_agent_eval(args, eval_ds, rollout_engine, tokenizer, env, engine_lock, policy_version, reward_model=None):
    if eval_ds is None or len(eval_ds) == 0 or args.eval_samples <= 0:
        return {}
    indices = list(getattr(args, "eval_fixed_indices", []))
    if not indices:
        indices = build_fixed_eval_indices(eval_ds, args.eval_samples, seed=getattr(args, "eval_seed", 1234))
    indices = indices[: min(args.eval_samples, len(indices), len(eval_ds))]
    prompts, completions, gt_batch, tools_batch, events_batch, unfinished_batch, messages_batch = [], [], [], [], [], [], []
    with engine_lock:
        for idx in indices:
            sample = eval_ds[idx]
            traj = rollout_single_online(
                rollout_engine,
                tokenizer,
                env,
                sample,
                args.max_turns,
                args.eval_max_gen_len if args.eval_max_gen_len > 0 else args.max_gen_len,
                args.max_context_len,
                0.0,
                args.device,
                policy_version,
                args.eval_temperature,
            )
            prompts.append(traj.prompt)
            messages_batch.append(traj.messages)
            completions.append(traj.completion)
            gt_batch.append(traj.gt)
            tools_batch.append(traj.tools)
            events_batch.append(traj.tool_events)
            unfinished_batch.append(traj.unfinished)
    rewards, details = calculate_rewards(
        args,
        prompts,
        completions,
        gt_batch,
        tools_batch,
        events_batch,
        unfinished_batch,
        1,
        args.device,
        reward_model,
        True,
        messages_batch=messages_batch,
    )
    scalar = {k: safe_mean([d[k] for d in details]) for k in details[0].keys()} if details else {}
    out = {
        "eval/reward": float(rewards.mean().item()),
        "eval/exact_match": float(scalar.get("exact_match", 0.0)),
        "eval/answer_cov": float(scalar.get("answer_cov", 0.0)),
        "eval/strict_exact_match": float(scalar.get("strict_exact_match", 0.0)),
        "eval/tool_success_rate": float(scalar.get("tool_success_rate", 0.0)),
        "eval/tool_correct_choice_rate": float(scalar.get("correct_tool_choice", 0.0)),
        "eval/math_call_coverage": float(scalar.get("math_call_coverage", 0.0)),
        "eval/tool_expr_order_cov": float(scalar.get("tool_expr_order_cov", 0.0)),
        "eval/tool_expr_set_cov": float(scalar.get("tool_expr_set_cov", 0.0)),
        "eval/tool_expr_exact": float(scalar.get("tool_expr_exact", 0.0)),
        "eval/tool_value_order_cov": float(scalar.get("tool_value_order_cov", 0.0)),
        "eval/tool_value_set_cov": float(scalar.get("tool_value_set_cov", 0.0)),
        "eval/tool_value_exact": float(scalar.get("tool_value_exact", 0.0)),
        "eval/preferred_math_rate": float(scalar.get("preferred_math_rate", 0.0)),
        "eval/tool_canonical_rate": float(scalar.get("canonical_tool_rate", 0.0)),
        "eval/non_math_tool_calls": float(scalar.get("non_math_tool_calls", 0.0)),
        "eval/final_refs_tool": float(scalar.get("final_refs_tool", 0.0)),
        "eval/answer_tag": float(scalar.get("answer_tag", 0.0)),
        "eval/exact_answer_tag": float(scalar.get("exact_answer_tag", 0.0)),
        "eval/extra_numbers": float(scalar.get("extra_numbers", 0.0)),
        "eval/missing_numbers": float(scalar.get("missing_numbers", 0.0)),
        "eval/order_ok": float(scalar.get("order_ok", 0.0)),
        "eval/avg_answer_len": float(scalar.get("answer_len", 0.0)),
        "eval/unfinished_rate": float(scalar.get("unfinished", 0.0)),
        "eval/reward_clipped_rate": float(scalar.get("clipped", 0.0)),
    }
    out.update(build_dashboard_metrics(scalar, prefix="eval_dashboard"))
    return out


def maybe_log_debug_sample(step, groups, diagnostics, debug_interval):
    if not is_main_process() or debug_interval <= 0 or step % debug_interval != 0 or not groups:
        return
    rewards = diagnostics.get("rewards", [])
    details = diagnostics.get("reward_details", [])
    group = groups[0]
    Logger(f"[DEBUG] step={step}, data_index={group.data_index}, policy_version={group.policy_version}")
    Logger("-" * 100)
    Logger(group.prompt[:1200])
    for i, traj in enumerate(group.trajectories[:4]):
        reward = rewards[i] if i < len(rewards) else None
        detail = details[i] if i < len(details) else {}
        Logger(
            f"[DEBUG] traj={i}, reward={reward}, exact={detail.get('exact_match')}, cov={detail.get('answer_cov')}, "
            f"tag={detail.get('answer_tag')}, tool_val={detail.get('tool_value_order_cov')}, "
            f"tool_choice={detail.get('correct_tool_choice')}, tool_success={detail.get('tool_success_rate')}, "
            f"unfinished={traj.unfinished}, events={len(traj.tool_events)}"
        )
        for turn_id, turn_text in enumerate(traj.turn_outputs):
            Logger(f"[DEBUG] assistant turn={turn_id}\n{turn_text[:1200]}")
        if traj.tool_events:
            Logger(f"[DEBUG] tool_events={json.dumps(traj.tool_events[:8], ensure_ascii=False)}")
        Logger("-" * 100)


# -------------------------- CLI / main --------------------------

def build_args():
    p = argparse.ArgumentParser(description="MiniMind stable async agent RL trainer - tool-use optimized v5 for 4-GPU torch")
    p.add_argument("--save_dir", type=str, default="../out")
    p.add_argument("--save_weight", type=str, default="agentic_async_rl_tooluse_v5_fix")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--total_steps", type=int, default=0)
    p.add_argument("--scale_total_steps_by_world_size", type=int, default=1, choices=[0, 1])
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=2e-7)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--accumulation_steps", type=int, default=2)
    p.add_argument("--grad_clip", type=float, default=0.5)
    p.add_argument("--log_interval", type=int, default=1)
    p.add_argument("--save_interval", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--num_hidden_layers", type=int, default=8)
    p.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--max_context_len", type=int, default=1024)
    p.add_argument("--max_gen_len", type=int, default=192)
    p.add_argument("--eval_max_gen_len", type=int, default=192)
    p.add_argument("--max_total_len", type=int, default=1536)
    p.add_argument("--data_path", type=str, default="/workspace/minimind/dataset/agent_rl_math.jsonl")
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--from_weight", type=str, default="full_sft")
    p.add_argument("--from_resume", type=int, default=0, choices=[0, 1])
    p.add_argument("--ref_from_rl_checkpoint", type=int, default=0, choices=[0, 1])
    p.add_argument("--use_compile", type=int, default=0, choices=[0, 1])
    p.add_argument("--num_generations", type=int, default=4)
    p.add_argument("--max_turns", type=int, default=3)
    p.add_argument("--thinking_ratio", type=float, default=0.0)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.88)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--repetition_penalty", type=float, default=1.03)
    p.add_argument("--torch_rollout_use_cache", type=int, default=1, choices=[0, 1])
    p.add_argument("--filter_mode", type=str, default="math_only", choices=["all", "math_only"])
    p.add_argument("--require_gt", type=int, default=1, choices=[0, 1])
    p.add_argument("--time_mode", type=str, default="fixed", choices=["fixed", "live"])
    p.add_argument("--tool_subset", type=str, default="calculate_math,unit_converter,get_current_time")
    p.add_argument("--inject_answer_instruction", type=int, default=1, choices=[0, 1])
    p.add_argument("--augment_calculator_alias", type=int, default=0, choices=[0, 1])
    p.add_argument("--force_all_tools", type=int, default=0, choices=[0, 1])

    p.add_argument("--epsilon", type=float, default=0.20)
    p.add_argument("--beta", type=float, default=0.03)
    p.add_argument("--use_decoupled_ppo", type=int, default=1, choices=[0, 1])
    p.add_argument("--prox_logp_method", type=str, default="loglinear", choices=["loglinear", "behavior"])
    p.add_argument("--behav_imp_weight_cap", type=float, default=1.5)
    p.add_argument("--m2_threshold", type=float, default=0.0)
    p.add_argument("--advantage_mode", type=str, default="mixed", choices=["group", "baseline", "mixed"])
    p.add_argument("--adv_min_std", type=float, default=0.10)
    p.add_argument("--zero_std_advantage_mode", type=str, default="baseline", choices=["zero", "baseline", "positive_baseline"])
    p.add_argument("--adv_clip", type=float, default=3.0)
    p.add_argument("--reward_ema_decay", type=float, default=0.98)
    p.add_argument("--reward_scale", type=float, default=3.0)
    p.add_argument("--beta_auto_tune", type=int, default=1, choices=[0, 1])
    p.add_argument("--beta_control_metric", type=str, default="old_current_abs_delta", choices=["old_current_abs_delta", "approx_kl_old", "kl_ref"])
    p.add_argument("--beta_update_interval", type=int, default=4)
    p.add_argument("--beta_ema_decay", type=float, default=0.90)
    p.add_argument("--kl_target", type=float, default=0.06)
    p.add_argument("--beta_min", type=float, default=0.01)
    p.add_argument("--beta_max", type=float, default=0.08)
    p.add_argument("--hard_kl_stop_mult", type=float, default=8.0)
    p.add_argument("--loss_reduction", type=str, default="sequence", choices=["token", "sequence"])

    p.add_argument("--bc_coef", type=float, default=0.16)
    p.add_argument("--bc_min_coef", type=float, default=0.03)
    p.add_argument("--bc_stop_steps", type=int, default=0)
    p.add_argument("--bc_hold_steps", type=int, default=1200)
    p.add_argument("--bc_decay_steps", type=int, default=4000)
    p.add_argument("--bc_max_examples", type=int, default=24)
    p.add_argument("--bc_max_exprs", type=int, default=8)
    p.add_argument("--bc_max_len", type=int, default=1536)

    p.add_argument("--format_enforce_mode", type=str, default="semi_hard", choices=["soft", "semi_hard", "hard"])
    p.add_argument("--format_warmup_steps", type=int, default=100)
    p.add_argument("--format_missing_penalty_initial", type=float, default=0.30)
    p.add_argument("--format_missing_penalty_final", type=float, default=1.50)
    p.add_argument("--no_answer_tag_reward_cap", type=float, default=2.0)
    p.add_argument("--strict_order_reward", type=int, default=1, choices=[0, 1])
    p.add_argument("--strict_exact_reward", type=int, default=1, choices=[0, 1])
    p.add_argument("--reward_base", type=float, default=-0.10)
    p.add_argument("--reward_answer_cov_weight", type=float, default=9.0)
    p.add_argument("--reward_exact_match_weight", type=float, default=9.0)
    p.add_argument("--reward_answer_tag_weight", type=float, default=0.90)
    p.add_argument("--reward_exact_answer_tag_weight", type=float, default=0.80)
    p.add_argument("--reward_tool_success_weight", type=float, default=0.60)
    p.add_argument("--reward_final_refs_tool_weight", type=float, default=0.45)
    p.add_argument("--reward_tool_fail_penalty", type=float, default=0.55)
    p.add_argument("--reward_no_tool_penalty", type=float, default=1.50)
    p.add_argument("--reward_math_call_cov_weight", type=float, default=0.60)
    p.add_argument("--reward_tool_expr_order_weight", type=float, default=0.60)
    p.add_argument("--reward_tool_expr_set_weight", type=float, default=0.20)
    p.add_argument("--reward_tool_expr_exact_weight", type=float, default=0.20)
    p.add_argument("--reward_tool_value_order_weight", type=float, default=2.40)
    p.add_argument("--reward_tool_value_set_weight", type=float, default=0.80)
    p.add_argument("--reward_tool_value_exact_weight", type=float, default=0.80)
    p.add_argument("--reward_tool_over_call_penalty", type=float, default=0.25)
    p.add_argument("--reward_tool_under_call_penalty", type=float, default=0.35)
    p.add_argument("--reward_preferred_math_tool_weight", type=float, default=0.70)
    p.add_argument("--reward_correct_tool_choice_weight", type=float, default=0.70)
    p.add_argument("--reward_tool_count_exact_weight", type=float, default=0.30)
    p.add_argument("--reward_canonical_tool_weight", type=float, default=0.25)
    p.add_argument("--reward_wrong_tool_penalty", type=float, default=0.55)
    p.add_argument("--reward_alias_only_penalty", type=float, default=0.35)
    p.add_argument("--extra_number_penalty", type=float, default=0.55)
    p.add_argument("--missing_number_penalty", type=float, default=0.70)
    p.add_argument("--order_mismatch_penalty", type=float, default=1.20)
    p.add_argument("--reward_cap_warmup_steps", type=int, default=300)
    p.add_argument("--non_exact_reward_cap_initial", type=float, default=4.0)
    p.add_argument("--non_exact_reward_cap", type=float, default=3.5)
    p.add_argument("--zero_answer_cov_reward_cap_initial", type=float, default=0.5)
    p.add_argument("--zero_answer_cov_reward_cap", type=float, default=0.0)
    p.add_argument("--max_reward_answer_len", type=int, default=96)

    p.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"])
    p.add_argument("--multi_gpu_mode", type=str, default="trainer_rollout", choices=["trainer_rollout", "ddp"])
    p.add_argument("--sglang_base_url", type=str, default="http://localhost:8998")
    p.add_argument("--sglang_model_path", type=str, default="../model")
    p.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_agentic_async")
    p.add_argument("--async_workers", type=int, default=1)
    p.add_argument("--max_version_gap", type=int, default=-1)
    p.add_argument("--max_head_offpolicyness", type=int, default=1)
    p.add_argument("--max_queue_groups", type=int, default=0)
    p.add_argument("--global_prefetch_groups", type=int, default=0)
    p.add_argument("--prefill_queue_groups", type=int, default=0)
    p.add_argument("--prefill_timeout_sec", type=float, default=180.0)
    p.add_argument("--fetch_timeout_sec", type=float, default=180.0)
    p.add_argument("--policy_sync_interval", type=int, default=1)
    p.add_argument("--async_stale_discount", type=float, default=0.65)
    p.add_argument("--strict_group_packing", type=int, default=1, choices=[0, 1])
    p.add_argument("--eval_ratio", type=float, default=0.02)
    p.add_argument("--eval_max_pool", type=int, default=256)
    p.add_argument("--eval_interval", type=int, default=100)
    p.add_argument("--eval_samples", type=int, default=32)
    p.add_argument("--eval_seed", type=int, default=1234)
    p.add_argument("--eval_temperature", type=float, default=0.0)
    p.add_argument("--eval_before_train", type=int, default=1, choices=[0, 1])
    p.add_argument("--use_reward_model", type=int, default=0, choices=[0, 1])
    p.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward")
    p.add_argument("--use_wandb", default=True, type=str2bool)
    p.add_argument("--wandb_project", type=str, default="MiniMind-Agentic-Async-RL")
    p.add_argument("--debug_interval", type=int, default=0)
    return p.parse_args()


def build_autocast_factory(device_type, dtype):
    if device_type == "cpu" or dtype == torch.float32:
        return lambda: nullcontext()
    return lambda: torch.cuda.amp.autocast(dtype=dtype)


def maybe_step_optimizer(args, model, optimizer, scheduler):
    grad_norm = 0.0
    if args.grad_clip > 0:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        grad_norm = float(norm.detach().item()) if torch.is_tensor(norm) else float(norm)
    bad_grad = any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters())
    if bad_grad or not math.isfinite(grad_norm):
        Logger("[WARN] non-finite gradient; skip optimizer step")
        optimizer.zero_grad(set_to_none=True)
        return False, grad_norm
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return True, grad_norm


def beta_control_metric_key(args) -> str:
    metric = getattr(args, "beta_control_metric", "old_current_abs_delta")
    if metric == "kl_ref":
        return "policy/kl_ref_k3"
    if metric == "approx_kl_old":
        return "policy/approx_kl_old_abs"
    return "policy/old_current_abs_delta"


def maybe_update_beta(args, metrics: Dict[str, Any], opt_step: int) -> float:
    metric_key = beta_control_metric_key(args)
    observed = finite_float(metrics.get(metric_key, 0.0), 0.0)
    prev_ema = getattr(args, "beta_ema", observed)
    decay = min(max(float(getattr(args, "beta_ema_decay", 0.90)), 0.0), 0.999)
    ema = observed if not math.isfinite(prev_ema) else decay * prev_ema + (1.0 - decay) * observed
    args.beta_ema = float(ema)
    if not getattr(args, "beta_auto_tune", False):
        return float(ema)

    interval = max(1, int(getattr(args, "beta_update_interval", 4)))
    if opt_step % interval != 0:
        return float(ema)

    target = max(finite_float(getattr(args, "kl_target", 0.08), 0.08), 1e-6)
    upper = target * 1.25
    lower = target * 0.80
    max_mult = 1.12
    min_mult = 1.0 / max_mult
    if ema > upper:
        ratio = min(max(ema / target, 1.0), max_mult * max_mult)
        args.beta = min(args.beta * max(min(math.sqrt(ratio), max_mult), 1.0), args.beta_max)
    elif ema < lower:
        ratio = min(max(target / max(ema, 1e-6), 1.0), max_mult * max_mult)
        args.beta = max(args.beta * min(max(1.0 / math.sqrt(ratio), min_mult), 1.0), args.beta_min)
    return float(ema)


def split_prefetched_groups(fetched_groups, pending_groups, batch_size: int, current_version: int, max_version_gap: int):
    if not fetched_groups:
        return None, 0
    batch_size = int(batch_size)
    current_version = int(current_version)
    max_version_gap = max(0, int(max_version_gap))
    stale_drops = 0
    batch = []
    remainder = []

    def keep_group(group) -> bool:
        nonlocal stale_drops
        if group is None:
            return False
        if current_version - int(group.policy_version) > max_version_gap:
            stale_drops += 1
            return False
        return True

    for group in fetched_groups:
        if not keep_group(group):
            continue
        if len(batch) < batch_size:
            batch.append(group)
        else:
            remainder.append(group)

    while len(batch) < batch_size and pending_groups:
        group = pending_groups.pop()
        if keep_group(group):
            batch.append(group)

    if remainder:
        pending_groups.extend(reversed(remainder))
    return (batch if len(batch) >= batch_size else None), stale_drops


def parse_tool_subset(text: str) -> List[str]:
    allowed = {"calculate_math", "unit_converter", "get_current_time"}
    names = [x.strip() for x in str(text).split(",") if x.strip()]
    names = [x for x in names if x in allowed]
    return names or ["calculate_math", "unit_converter", "get_current_time"]


def main():
    args = build_args()
    local_rank = init_distributed_mode_compat()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    rank = dist.get_rank() if dist.is_initialized() else 0
    setup_seed(args.seed + rank)
    use_default_max_version_gap = args.max_version_gap < 0
    args.use_decoupled_ppo = bool(args.use_decoupled_ppo)
    args.reward_baseline = 0.0
    args.beta_ema = finite_float(getattr(args, "kl_target", 0.08), 0.08)
    args.current_step = 0
    os.makedirs(args.save_dir, exist_ok=True)

    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_seq_len=max(args.max_seq_len + args.max_gen_len, args.max_total_len, args.max_context_len + args.max_gen_len),
        use_moe=bool(args.use_moe),
    )
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir="../checkpoints") if args.from_resume == 1 else None
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16 if args.dtype == "float16" else torch.float32
    autocast_factory = build_autocast_factory(device_type, dtype)

    wandb = None
    if args.use_wandb and is_main_process():
        try:
            import swanlab as wandb
            wandb_id = ckp_data.get("wandb_id") if ckp_data else None
            wandb.init(project=args.wandb_project, name=f"ToolUseV4-E{args.epochs}-B{args.batch_size}-G{args.num_generations}", id=wandb_id, resume="must" if wandb_id else None)
        except Exception as exc:
            Logger(f"[WARN] swanlab disabled: {repr(exc)}")
            wandb = None

    train_on_this_rank = is_trainer_process(args)
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    ensure_tokenizer_ids(tokenizer, lm_config)
    ref_model = None
    if train_on_this_rank:
        ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
        ref_model = ref_model.eval().requires_grad_(False)
    rollout_model = None
    if args.rollout_engine == "torch":
        rollout_model, _ = init_model(lm_config, args.from_weight, device=args.device)
        rollout_model = rollout_model.eval().requires_grad_(False)

    if ckp_data:
        model.load_state_dict(ckp_data["model"], strict=False)
        if ref_model is not None and args.ref_from_rl_checkpoint:
            ref_model.load_state_dict(ckp_data["model"], strict=False)
            Logger("[WARN] ref_model loaded from RL checkpoint because --ref_from_rl_checkpoint=1")
        if rollout_model is not None:
            rollout_model.load_state_dict(ckp_data["model"], strict=False)
        args.reward_baseline = float(ckp_data.get("reward_baseline", 0.0))

    if args.use_compile == 1 and train_on_this_rank:
        model = torch.compile(model)
        Logger("torch.compile enabled")
    if dist.is_initialized() and not is_trainer_rollout_mode(args):
        model = DistributedDataParallel(model, device_ids=[local_rank])
    elif not train_on_this_rank:
        model = model.eval().requires_grad_(False)

    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16) if args.use_reward_model and train_on_this_rank else None
    if reward_model is not None:
        Logger(f"Loaded reward model from {args.reward_model_path}")

    active_tools = parse_tool_subset(args.tool_subset)
    env = RealToolEnv(time_mode=args.time_mode, active_tools=active_tools, allow_calculator_alias=bool(args.augment_calculator_alias))
    train_ds = PromptPoolDataset(
        args.data_path,
        max_samples=args.max_samples,
        filter_mode=args.filter_mode,
        require_gt=bool(args.require_gt),
        allowed_tools=list(env.supported_names()),
        inject_answer_instruction=bool(args.inject_answer_instruction),
        augment_calculator_alias=bool(args.augment_calculator_alias),
        default_tools=env.tool_specs(),
        force_all_tools=bool(args.force_all_tools),
    )
    train_ds, eval_ds = split_prompt_pool(train_ds, args.eval_ratio, args.eval_max_pool, seed=args.seed)
    global_train_samples = len(train_ds)
    world_size = ddp_world_size()
    trainer_rollout_dist = dist.is_initialized() and is_trainer_rollout_mode(args) and world_size > 1
    rollout_ranks = max(1, world_size - 1) if trainer_rollout_dist else max(world_size, 1)
    trainer_produces_rollout = not (trainer_rollout_dist and rank == trainer_rank())
    if use_default_max_version_gap:
        if trainer_rollout_dist:
            args.max_version_gap = max(int(args.max_head_offpolicyness), min(max(2, int(args.policy_sync_interval) * 2), 4))
        else:
            args.max_version_gap = int(args.max_head_offpolicyness)
    if dist.is_initialized() and not trainer_rollout_dist:
        train_ds = shard_prompt_pool(train_ds, world_size, rank)
    args.eval_fixed_indices = build_fixed_eval_indices(eval_ds, args.eval_samples, seed=args.eval_seed)
    if args.rollout_engine == "torch" and world_size > 1 and args.async_workers > 1 and not trainer_rollout_dist:
        if is_main_process():
            Logger(f"[WARN] torch rollout + DDP: async_workers {args.async_workers} -> 1 to avoid per-rank lock contention")
        args.async_workers = 1
    if is_main_process():
        Logger(f"Train samples: global={global_train_samples}, local_rank0={len(train_ds)}, Eval samples: {len(eval_ds)}")
        Logger(f"Dataset stats: {json.dumps(train_ds.stats, ensure_ascii=False)}")
        Logger(f"Active tools: {','.join(supported_tool_names(env.tool_specs()))}")

    if args.rollout_engine == "torch":
        rollout_engine = StableTorchRolloutEngine(
            rollout_model,
            tokenizer,
            device=args.device,
            autocast_factory=autocast_factory,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            use_cache=bool(args.torch_rollout_use_cache),
        )
    else:
        rollout_engine = create_rollout_engine(
            engine_type=args.rollout_engine,
            policy_model=model,
            tokenizer=tokenizer,
            device=args.device,
            autocast_ctx=autocast_factory(),
            sglang_base_url=args.sglang_base_url,
            sglang_model_path=args.sglang_model_path,
            sglang_shared_path=args.sglang_shared_path,
        )

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay) if train_on_this_rank else None
    if args.total_steps > 0:
        if dist.is_initialized() and bool(args.scale_total_steps_by_world_size) and not trainer_rollout_dist:
            total_steps = max(1, math.ceil(args.total_steps / max(ddp_world_size(), 1)))
            if is_main_process():
                Logger(f"[WARN] total_steps scaled by world_size: requested={args.total_steps}, effective={total_steps}")
        else:
            total_steps = args.total_steps
    else:
        total_steps = math.ceil(len(train_ds) / args.batch_size) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, math.ceil(total_steps / args.accumulation_steps)), eta_min=args.learning_rate / 10) if train_on_this_rank else None
    start_step = int(ckp_data.get("step", 0)) if ckp_data else 0
    opt_step = int(ckp_data.get("opt_step", 0)) if ckp_data else 0
    if ckp_data and optimizer is not None:
        optimizer.load_state_dict(ckp_data["optimizer"])
        if scheduler is not None and ckp_data.get("scheduler") is not None:
            scheduler.load_state_dict(ckp_data["scheduler"])

    version = PolicyVersion(int(ckp_data.get("policy_version", 0)) if ckp_data else 0)
    args.current_version = version.get()
    engine_lock = threading.Lock()
    if trainer_rollout_dist:
        sync_rollout_policy_distributed(args, rollout_engine, rollout_model, model, engine_lock, version, src_rank=trainer_rank())
    else:
        sync_rollout_policy(args, rollout_engine, rollout_model, model, engine_lock, version)
    args.current_version = version.get()

    if args.eval_before_train:
        if dist.is_initialized():
            dist.barrier()
        if is_main_process():
            eval_metrics = run_agent_eval(args, eval_ds, rollout_engine, tokenizer, env, engine_lock, version.get(), reward_model)
            if eval_metrics:
                Logger(
                    f"[EVAL_INIT] R={eval_metrics['eval/reward']:.4f}, "
                    f"Exact={eval_metrics['eval/exact_match']:.3f}, Cov={eval_metrics['eval/answer_cov']:.3f}, "
                    f"Tag={eval_metrics['eval/answer_tag']:.3f}, ToolSucc={eval_metrics['eval/tool_success_rate']:.3f}, "
                    f"ToolChoice={eval_metrics['eval/tool_correct_choice_rate']:.3f}, ToolExpr={eval_metrics['eval/tool_expr_order_cov']:.3f}, ToolVal={eval_metrics['eval/tool_value_order_cov']:.3f}, "
                    f"Canon={eval_metrics['eval/tool_canonical_rate']:.3f}"
                )
                if wandb:
                    wandb.log(eval_metrics, step=start_step)
        if dist.is_initialized():
            dist.barrier()

    if args.max_queue_groups > 0:
        max_queue_groups = int(args.max_queue_groups)
    elif trainer_rollout_dist:
        max_queue_groups = max((args.max_version_gap + 1) * args.batch_size, args.batch_size * max(8, rollout_ranks * 4))
    else:
        max_queue_groups = (args.max_version_gap + 1) * args.batch_size
    manager_workers = int(args.async_workers) if trainer_produces_rollout else 0
    manager = AsyncRolloutManager(
        train_ds,
        tokenizer,
        rollout_engine,
        env,
        version,
        args.batch_size,
        args.num_generations,
        args.max_turns,
        args.max_gen_len,
        args.max_context_len,
        args.thinking_ratio,
        args.temperature,
        args.device,
        args.max_version_gap,
        max_queue_groups,
        manager_workers,
        engine_lock,
        args.seed,
    )
    stop_requested = {"flag": False}

    def _sig_handler(signum, frame):
        stop_requested["flag"] = True
        Logger("received stop signal, will save and exit safely")

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)
    manager.start()
    global_queue_groups = max_queue_groups * (rollout_ranks if trainer_rollout_dist else 1)
    pending_groups = deque()
    prefill_capacity = global_queue_groups if trainer_rollout_dist else max_queue_groups
    default_global_prefetch_groups = (
        max(args.batch_size, min(global_queue_groups, args.batch_size * max(int(args.max_version_gap) + 1, rollout_ranks)))
        if trainer_rollout_dist
        else args.batch_size
    )
    prefill_default = max(args.batch_size, min(prefill_capacity, default_global_prefetch_groups if trainer_rollout_dist else args.batch_size * 2))
    prefill_groups = min(prefill_capacity, args.prefill_queue_groups if args.prefill_queue_groups > 0 else prefill_default)
    if trainer_rollout_dist:
        prefetched = distributed_fetch_groups(
            rank,
            world_size,
            manager,
            pending_groups,
            prefill_groups,
            fetch_timeout_sec=args.prefill_timeout_sec,
            producer_ranks=rollout_ranks,
            src_rank=trainer_rank(),
            min_return_groups=prefill_groups,
        )
        ready = prefetched is not None if rank == trainer_rank() else bool(prefetched)
        if rank == trainer_rank() and prefetched is not None:
            pending_groups.extend(prefetched)
    else:
        ready = manager.wait_until_ready(prefill_groups, timeout_sec=args.prefill_timeout_sec)
    if is_main_process():
        Logger(f"Queue prefill: target={prefill_groups}, ready={int(bool(ready))}")
    Logger(
        f"ToolUseV4 async rollout started: engine={args.rollout_engine}, mode={args.multi_gpu_mode}, workers={manager_workers}, rollout_ranks={rollout_ranks}, "
        f"max_version_gap={args.max_version_gap}, queue(local/global)={max_queue_groups}/{global_queue_groups}, decoupled={int(args.use_decoupled_ppo)}, "
        f"advantage={args.advantage_mode}, beta={args.beta}, beta_max={args.beta_max}, kl_target={args.kl_target}, "
        f"bc_coef={args.bc_coef}, bc_min={args.bc_min_coef}, loss_reduction={args.loss_reduction}, cache={args.torch_rollout_use_cache}, "
        f"tools={','.join(supported_tool_names(env.tool_specs()))}"
    )

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    if train_on_this_rank:
        model.train()
    else:
        model.eval().requires_grad_(False)
    last_metrics = {}
    step = start_step
    try:
        for step in range(start_step + 1, total_steps + 1):
            if trainer_rollout_dist:
                loop_flag = torch.tensor([1 if (rank == trainer_rank() and not stop_requested["flag"]) else 0], device=args.device, dtype=torch.int64)
                dist.broadcast(loop_flag, src=trainer_rank())
                if loop_flag.item() <= 0:
                    break
            elif stop_requested["flag"]:
                break
            args.current_step = step
            args.current_version = version.get()
            diagnostics, metrics, groups = {}, {}, None
            sync_needed = 0

            if trainer_rollout_dist:
                fetch_target = min(global_queue_groups, int(args.global_prefetch_groups) if args.global_prefetch_groups > 0 else default_global_prefetch_groups)
                stale_prefetch_drops = 0
                enough_groups = False
                groups = None
                fetch_wait_rounds = 0
                stop_fetch = False
                while True:
                    fetched = distributed_fetch_groups(
                        rank,
                        world_size,
                        manager,
                        pending_groups,
                        fetch_target,
                        fetch_timeout_sec=args.fetch_timeout_sec,
                        producer_ranks=rollout_ranks,
                        src_rank=trainer_rank(),
                        min_return_groups=args.batch_size,
                    )
                    if rank == trainer_rank():
                        groups, stale_drops_now = split_prefetched_groups(fetched, pending_groups, args.batch_size, version.get(), args.max_version_gap)
                        stale_prefetch_drops += stale_drops_now
                        enough_groups = groups is not None and len(groups) >= args.batch_size
                    else:
                        enough_groups = False
                        stale_drops_now = 0
                    enough_tensor = torch.tensor([1 if (rank == trainer_rank() and enough_groups) else 0], device=args.device, dtype=torch.int64)
                    dist.broadcast(enough_tensor, src=trainer_rank())
                    enough_groups = bool(enough_tensor.item() > 0)
                    if enough_groups:
                        break
                    fetch_ctrl = torch.tensor([1 if (rank == trainer_rank() and stop_requested["flag"]) else 0], device=args.device, dtype=torch.int64)
                    dist.broadcast(fetch_ctrl, src=trainer_rank())
                    stop_fetch = bool(fetch_ctrl.item() > 0)
                    if stop_fetch:
                        break
                    if rank == trainer_rank() and fetched is None:
                        fetch_wait_rounds += 1
                        if fetch_wait_rounds == 1 or fetch_wait_rounds % 10 == 0:
                            Logger("[WARN] rollout queue lagging behind trainer; continue waiting for enough fresh groups")
                if stop_fetch:
                    break

                if rank == trainer_rank():
                    pack_ok = True
                    try:
                        batch = pack_groups(groups, tokenizer, args.max_total_len, args.device, args.num_generations, bool(args.strict_group_packing))
                    except RuntimeError as exc:
                        pack_ok = False
                        Logger(f"[WARN] pack_groups failed at step={step}: {repr(exc)}")
                        batch = None

                    if pack_ok:
                        loss, metrics, diagnostics = compute_async_loss(args, batch, model, ref_model, reward_model, tokenizer, autocast_factory, lm_config, env)
                        metrics["train/world_size"] = float(world_size)
                        metrics["policy/kl_control"] = float(metrics.get(beta_control_metric_key(args), 0.0))
                        if not bool(torch.isfinite(loss).item()):
                            Logger(f"[WARN] non-finite loss at step={step}; skip update")
                            optimizer.zero_grad(set_to_none=True)
                            metrics = {}
                        else:
                            kl_now = float(metrics.get("policy/kl_ref_k3", 0.0))
                            kl_control_now = float(metrics.get("policy/kl_control", kl_now))
                            if args.hard_kl_stop_mult > 0 and kl_control_now > args.kl_target * args.hard_kl_stop_mult:
                                Logger(f"[WARN] KL too high at step={step}: control={kl_control_now:.4f}, ref={kl_now:.4f}; skip update")
                                optimizer.zero_grad(set_to_none=True)
                            else:
                                sync_now = (step % args.accumulation_steps == 0)
                                with nullcontext():
                                    (loss / args.accumulation_steps).backward()
                                last_grad_norm = 0.0
                                if sync_now:
                                    updated, last_grad_norm = maybe_step_optimizer(args, model, optimizer, scheduler)
                                    if updated:
                                        opt_step += 1
                                        reward_now = float(metrics.get("reward/mean", 0.0))
                                        args.reward_baseline = reward_now if opt_step == 1 and abs(args.reward_baseline) < 1e-12 else args.reward_ema_decay * args.reward_baseline + (1 - args.reward_ema_decay) * reward_now
                                        metrics["train/beta_ema"] = maybe_update_beta(args, metrics, opt_step)
                                        if opt_step % args.policy_sync_interval == 0:
                                            sync_needed = 1
                                metrics["train/grad_norm"] = float(last_grad_norm)
                                metrics["train/beta"] = float(args.beta)
                                metrics["train/beta_ema"] = float(getattr(args, "beta_ema", 0.0))
                                metrics["train/lr"] = float(optimizer.param_groups[0]["lr"])
                                metrics["policy/version"] = float(version.get())
                                metrics["policy/max_version_gap"] = float(args.max_version_gap)
                                metrics["async/prefetch_stale_drop"] = float(stale_prefetch_drops)
                                if args.rollout_engine == "torch":
                                    metrics["rollout/sanitized_logits_batches"] = float(getattr(rollout_engine, "nan_batches", 0))
                                    metrics["rollout/bad_prob_batches"] = float(getattr(rollout_engine, "bad_prob_batches", 0))
                                last_metrics = metrics

                sync_tensor = torch.tensor([int(sync_needed)], device=args.device, dtype=torch.int64)
                dist.broadcast(sync_tensor, src=trainer_rank())
                if int(sync_tensor.item()) > 0:
                    args.current_version = sync_rollout_policy_distributed(args, rollout_engine, rollout_model, model, engine_lock, version, src_rank=trainer_rank())
                else:
                    args.current_version = version.get()
            else:
                groups = manager.get_batch(timeout=None)
                enough_groups = ddp_all_true(len(groups) >= args.batch_size, args.device)
                if not enough_groups:
                    Logger("Not enough rollout groups; stopping early.")
                    break
                pack_ok = True
                try:
                    batch = pack_groups(groups, tokenizer, args.max_total_len, args.device, args.num_generations, bool(args.strict_group_packing))
                except RuntimeError as exc:
                    pack_ok = False
                    Logger(f"[WARN] pack_groups failed at step={step}: {repr(exc)}")
                    batch = None
                if not ddp_all_true(pack_ok, args.device):
                    optimizer.zero_grad(set_to_none=True)
                    continue
                loss, metrics, diagnostics = compute_async_loss(args, batch, model, ref_model, reward_model, tokenizer, autocast_factory, lm_config, env)
                metrics["train/world_size"] = float(ddp_world_size())
                metrics["policy/kl_control"] = float(metrics.get(beta_control_metric_key(args), 0.0))
                if not ddp_all_true(bool(torch.isfinite(loss).item()), args.device):
                    Logger(f"[WARN] non-finite loss at step={step}; skip update")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                kl_now = ddp_mean_scalar(float(metrics.get("policy/kl_ref_k3", 0.0)), args.device)
                kl_control_now = ddp_mean_scalar(float(metrics.get("policy/kl_control", kl_now)), args.device)
                if args.hard_kl_stop_mult > 0 and kl_control_now > args.kl_target * args.hard_kl_stop_mult:
                    if is_main_process():
                        Logger(f"[WARN] KL too high at step={step}: control={kl_control_now:.4f}, ref={kl_now:.4f}; skip update")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                sync_now = (step % args.accumulation_steps == 0)
                backward_ctx = model.no_sync() if isinstance(model, DistributedDataParallel) and not sync_now else nullcontext()
                with backward_ctx:
                    (loss / args.accumulation_steps).backward()
                last_grad_norm = 0.0
                if sync_now:
                    updated, last_grad_norm = maybe_step_optimizer(args, model, optimizer, scheduler)
                    if updated:
                        opt_step += 1
                        reward_now = ddp_mean_scalar(float(metrics.get("reward/mean", 0.0)), args.device)
                        args.reward_baseline = reward_now if opt_step == 1 and abs(args.reward_baseline) < 1e-12 else args.reward_ema_decay * args.reward_baseline + (1 - args.reward_ema_decay) * reward_now
                        metrics["train/beta_ema"] = maybe_update_beta(args, metrics, opt_step)
                        if opt_step % args.policy_sync_interval == 0:
                            args.current_version = sync_rollout_policy(args, rollout_engine, rollout_model, model, engine_lock, version)

                metrics["train/grad_norm"] = float(last_grad_norm)
                metrics["train/beta"] = float(args.beta)
                metrics["train/beta_ema"] = float(getattr(args, "beta_ema", 0.0))
                metrics["train/lr"] = float(optimizer.param_groups[0]["lr"])
                metrics["policy/version"] = float(version.get())
                metrics["policy/max_version_gap"] = float(args.max_version_gap)
                if args.rollout_engine == "torch":
                    metrics["rollout/sanitized_logits_batches"] = float(getattr(rollout_engine, "nan_batches", 0))
                    metrics["rollout/bad_prob_batches"] = float(getattr(rollout_engine, "bad_prob_batches", 0))
                last_metrics = metrics

            if step % args.log_interval == 0 or step == total_steps:
                if trainer_rollout_dist:
                    local_stats = manager.snapshot_stats()
                    stats = ddp_average_metrics(local_stats, args.device)
                    local_queued_total = ddp_sum_scalar(float(local_stats.get("queued", 0.0)), args.device)
                    metrics_to_log = dict(last_metrics) if rank == trainer_rank() and last_metrics else {}
                    if args.rollout_engine == "torch":
                        cache_prefill = ddp_mean_scalar(float(getattr(rollout_engine, "cache_prefill_tokens", 0.0)), args.device)
                        cache_decode = ddp_mean_scalar(float(getattr(rollout_engine, "cache_decode_tokens", 0.0)), args.device)
                        cache_fallbacks = ddp_mean_scalar(float(getattr(rollout_engine, "cache_fallbacks", 0.0)), args.device)
                        if metrics_to_log:
                            metrics_to_log["rollout/cache_prefill_tokens"] = cache_prefill
                            metrics_to_log["rollout/cache_decode_tokens"] = cache_decode
                            metrics_to_log["rollout/cache_decode_ratio"] = cache_decode / max(cache_prefill + cache_decode, 1.0)
                            metrics_to_log["rollout/cache_fallbacks"] = cache_fallbacks
                    if rank == trainer_rank() and metrics_to_log:
                        stats["local_queued_total"] = float(local_queued_total)
                        stats["global_pending_groups"] = float(len(pending_groups))
                        stats["global_queue_capacity"] = float(global_queue_groups)
                        stats["global_available_groups"] = stats["local_queued_total"] + stats["global_pending_groups"]
                        stats["global_queue_utilization"] = stats["global_available_groups"] / max(stats["global_queue_capacity"], 1.0)
                        stats["rollout_ranks"] = float(rollout_ranks)
                        last_metrics = metrics_to_log
                        Logger(
                            f"Step:[{step}/{total_steps}] V:{version.get()} "
                            f"R:{metrics_to_log['reward/mean']:.3f}({metrics_to_log['reward/min']:.2f}/{metrics_to_log['reward/max']:.2f}) "
                            f"Acc(E/C):{metrics_to_log.get('reward_detail/exact_match', 0.0):.2f}/{metrics_to_log.get('reward_detail/answer_cov', 0.0):.2f} "
                            f"Fmt(T/ET/Mal):{metrics_to_log.get('reward_detail/answer_tag', 0.0):.2f}/{metrics_to_log.get('reward_detail/exact_answer_tag', 0.0):.2f}/{metrics_to_log.get('reward_detail/malformed_tags', 0.0):.2f} "
                            f"Tool(S/Cov/Expr/Val/Choice/Pref/Canon):{metrics_to_log.get('reward_detail/tool_success_rate', 0.0):.2f}/{metrics_to_log.get('reward_detail/math_call_coverage', 0.0):.2f}/{metrics_to_log.get('reward_detail/tool_expr_order_cov', 0.0):.2f}/{metrics_to_log.get('reward_detail/tool_value_order_cov', 0.0):.2f}/{metrics_to_log.get('reward_detail/correct_tool_choice', 0.0):.2f}/{metrics_to_log.get('reward_detail/preferred_math_rate', 0.0):.2f}/{metrics_to_log.get('reward_detail/canonical_tool_rate', 0.0):.2f} "
                            f"Num(+/-/Ord):{metrics_to_log.get('reward_detail/extra_numbers', 0.0):.1f}/{metrics_to_log.get('reward_detail/missing_numbers', 0.0):.1f}/{metrics_to_log.get('reward_detail/order_ok', 0.0):.2f} "
                            f"PPO(KL/Ent/Clip):{metrics_to_log['policy/kl_ref_k3']:.3f}/{metrics_to_log['policy/entropy']:.2f}/{metrics_to_log['policy/ratio_clip_frac']:.3f} "
                            f"BC:{metrics_to_log.get('bc/loss', 0.0):.3f}*{metrics_to_log.get('bc/coef', 0.0):.3f} "
                            f"Q(local/global):{stats['local_queued_total']:.0f}/{stats['global_available_groups']:.0f} Roll:{stats['last_rollout_time']:.2f}s Wait:{stats['last_train_wait']:.2f}s P90:{stats.get('train_wait_sec_p90', 0.0):.2f}s "
                            f"Tok:{stats['avg_tokens_per_traj']:.1f} LR:{optimizer.param_groups[0]['lr']:.2e} Beta:{args.beta:.3f}"
                        )
                        if wandb:
                            payload = {
                                **{k: v for k, v in metrics_to_log.items() if isinstance(v, (int, float)) and math.isfinite(float(v))},
                                **{f"async/{k}": v for k, v in stats.items() if isinstance(v, (int, float)) and math.isfinite(float(v))},
                            }
                            wandb.log(payload, step=step)
                else:
                    metrics = ddp_average_metrics(metrics, args.device)
                    if args.rollout_engine == "torch":
                        cache_prefill = ddp_mean_scalar(float(getattr(rollout_engine, "cache_prefill_tokens", 0.0)), args.device)
                        cache_decode = ddp_mean_scalar(float(getattr(rollout_engine, "cache_decode_tokens", 0.0)), args.device)
                        cache_fallbacks = ddp_mean_scalar(float(getattr(rollout_engine, "cache_fallbacks", 0.0)), args.device)
                        metrics["rollout/cache_prefill_tokens"] = cache_prefill
                        metrics["rollout/cache_decode_tokens"] = cache_decode
                        metrics["rollout/cache_decode_ratio"] = cache_decode / max(cache_prefill + cache_decode, 1.0)
                        metrics["rollout/cache_fallbacks"] = cache_fallbacks
                    stats = ddp_average_metrics(manager.snapshot_stats(), args.device)
                    last_metrics = metrics
                    Logger(
                        f"Step:[{step}/{total_steps}] V:{version.get()} "
                        f"R:{metrics['reward/mean']:.3f}({metrics['reward/min']:.2f}/{metrics['reward/max']:.2f}) "
                        f"Acc(E/C):{metrics.get('reward_detail/exact_match', 0.0):.2f}/{metrics.get('reward_detail/answer_cov', 0.0):.2f} "
                        f"Fmt(T/ET/Mal):{metrics.get('reward_detail/answer_tag', 0.0):.2f}/{metrics.get('reward_detail/exact_answer_tag', 0.0):.2f}/{metrics.get('reward_detail/malformed_tags', 0.0):.2f} "
                        f"Tool(S/Cov/Expr/Val/Choice/Pref/Canon):{metrics.get('reward_detail/tool_success_rate', 0.0):.2f}/{metrics.get('reward_detail/math_call_coverage', 0.0):.2f}/{metrics.get('reward_detail/tool_expr_order_cov', 0.0):.2f}/{metrics.get('reward_detail/tool_value_order_cov', 0.0):.2f}/{metrics.get('reward_detail/correct_tool_choice', 0.0):.2f}/{metrics.get('reward_detail/preferred_math_rate', 0.0):.2f}/{metrics.get('reward_detail/canonical_tool_rate', 0.0):.2f} "
                        f"Num(+/-/Ord):{metrics.get('reward_detail/extra_numbers', 0.0):.1f}/{metrics.get('reward_detail/missing_numbers', 0.0):.1f}/{metrics.get('reward_detail/order_ok', 0.0):.2f} "
                        f"PPO(KL/Ent/Clip):{metrics['policy/kl_ref_k3']:.3f}/{metrics['policy/entropy']:.2f}/{metrics['policy/ratio_clip_frac']:.3f} "
                        f"BC:{metrics.get('bc/loss', 0.0):.3f}*{metrics.get('bc/coef', 0.0):.3f} "
                        f"Q:{stats['queued']:.0f}/{stats['queue_capacity']:.0f} Roll:{stats['last_rollout_time']:.2f}s Wait:{stats['last_train_wait']:.2f}s P90:{stats.get('train_wait_sec_p90', 0.0):.2f}s "
                        f"Tok:{stats['avg_tokens_per_traj']:.1f} LR:{optimizer.param_groups[0]['lr']:.2e} Beta:{args.beta:.3f}"
                    )
                    if wandb and is_main_process():
                        payload = {
                            **{k: v for k, v in metrics.items() if isinstance(v, (int, float)) and math.isfinite(float(v))},
                            **{f"async/{k}": v for k, v in stats.items() if isinstance(v, (int, float)) and math.isfinite(float(v))},
                        }
                        wandb.log(payload, step=step)

            if not trainer_rollout_dist or rank == trainer_rank():
                maybe_log_debug_sample(step, groups, diagnostics, args.debug_interval)
            if args.eval_interval > 0 and (step % args.eval_interval == 0 or step == total_steps):
                if dist.is_initialized():
                    dist.barrier()
                if is_main_process():
                    eval_metrics = run_agent_eval(args, eval_ds, rollout_engine, tokenizer, env, engine_lock, version.get(), reward_model)
                    if eval_metrics:
                        Logger(
                            f"[EVAL] step={step}, R={eval_metrics['eval/reward']:.4f}, "
                            f"Exact={eval_metrics['eval/exact_match']:.3f}, Cov={eval_metrics['eval/answer_cov']:.3f}, "
                            f"Tag={eval_metrics['eval/answer_tag']:.3f}, ToolSucc={eval_metrics['eval/tool_success_rate']:.3f}, "
                            f"ToolChoice={eval_metrics['eval/tool_correct_choice_rate']:.3f}, ToolExpr={eval_metrics['eval/tool_expr_order_cov']:.3f}, ToolVal={eval_metrics['eval/tool_value_order_cov']:.3f}, "
                            f"Canon={eval_metrics['eval/tool_canonical_rate']:.3f}"
                        )
                        if wandb:
                            wandb.log(eval_metrics, step=step)
                if dist.is_initialized():
                    dist.barrier()
            if args.save_interval > 0 and step % args.save_interval == 0:
                save_checkpoint(args, lm_config, model, optimizer, scheduler, epoch=0, step=step, opt_step=opt_step, version=version.get(), wandb=wandb)
            if step % 20 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        save_checkpoint(args, lm_config, model, optimizer, scheduler, epoch=0, step=step, opt_step=opt_step, version=version.get(), wandb=wandb)
        if last_metrics and is_main_process():
            Logger(f"Training finished. final_reward={last_metrics.get('reward/mean', 0.0):.4f}")
    finally:
        manager.stop()
        if wandb and is_main_process():
            try:
                wandb.finish()
            except Exception:
                pass
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
