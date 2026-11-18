"""Single-device on-policy distillation through MiniMind's local tool environment."""

import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import datasets  # noqa: F401  # Keep pyarrow before torch on Windows (issue #771).
import argparse
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path

import torch
from transformers import AutoTokenizer

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.agent_opd_utils import collect_trajectory, pack_trajectories, distillation_loss

ROOT = Path(__file__).resolve().parents[1]


def load_prompts(path):
    samples = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            sample = json.loads(line)
            messages = sample.get("conversations", sample.get("messages", []))
            if messages and messages[-1].get("role") == "assistant":
                messages = messages[:-1]
            if not messages or messages[-1].get("role") not in {"user", "tool"}:
                raise ValueError("Each sample must end in a user/tool prompt or a removable assistant answer")
            tools = sample.get("tools")
            for message in messages:
                if message.get("role") == "system" and message.get("tools"):
                    tools = message["tools"]
            if isinstance(tools, str):
                tools = json.loads(tools)
            samples.append((messages, tools))
    if not samples:
        raise ValueError("The prompt dataset is empty")
    return samples


def load_model(path, config, device):
    weights = torch.load(path, map_location="cpu", weights_only=True)
    weights = weights.get("model", weights)
    model = MiniMindForCausalLM(config)
    model.load_state_dict(weights, strict=True)
    return model.to(device)


def training_signature(args):
    # A resumed run must retain its teacher, data order, and optimization setup.
    adjustable = {"resume", "epochs", "max_steps", "save_interval", "output_dir"}
    return {key: value for key, value in vars(args).items() if key not in adjustable}


def save_checkpoint(args, model, optimizer, scaler, epoch, batch, step):
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    suffix = "_moe" if args.student_use_moe else ""
    prefix = directory / f"agent_opd_{args.student_hidden_size}{suffix}"
    weights = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    state = {
        "model": weights, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
        "epoch": epoch, "batch": batch, "step": step, "signature": training_signature(args),
        "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "mps_rng": torch.mps.get_rng_state() if next(model.parameters()).device.type == "mps" else None,
    }
    for filename, content in [(f"{prefix}_resume.pth", state),
                              (f"{prefix}.pth", {key: value.half() for key, value in weights.items()})]:
        torch.save(content, filename + ".tmp")
        os.replace(filename + ".tmp", filename)


def train(args):
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("Agent OPD currently supports one training process/device")
    for key in ("epochs", "batch_size", "accumulation_steps", "max_turns", "max_gen_len",
                "max_total_len", "save_interval"):
        if getattr(args, key) < 1:
            raise ValueError(f"--{key} must be positive")
    if args.max_steps < 0 or any(not math.isfinite(value) or value <= 0 for value in
                               (args.learning_rate, args.temperature, args.distill_temperature, args.grad_clip)):
        raise ValueError("Steps must be nonnegative; learning rate, temperatures and grad clip must be positive")
    device = torch.device(args.device)
    if device.type == "cpu" and args.dtype != "float32":
        raise ValueError("Use --dtype float32 for the CPU reference run")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("The shared tokenizer must define PAD and EOS")
    configs = {
        who: MiniMindConfig(hidden_size=getattr(args, f"{who}_hidden_size"),
                            num_hidden_layers=getattr(args, f"{who}_num_layers"),
                            use_moe=bool(getattr(args, f"{who}_use_moe")))
        for who in ("student", "teacher")
    }
    if len(tokenizer) != configs["student"].vocab_size:
        raise ValueError("Both MiniMind checkpoints must use the supplied shared tokenizer/vocabulary")
    if args.max_total_len > min(config.max_position_embeddings for config in configs.values()):
        raise ValueError("--max_total_len exceeds the model context length")
    student = load_model(args.student_checkpoint, configs["student"], device)
    teacher = load_model(args.teacher_checkpoint, configs["teacher"], device).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.dtype == "float16")
    dtype = getattr(torch, args.dtype)
    autocast_ctx = nullcontext() if dtype == torch.float32 else torch.autocast(device.type, dtype=dtype)
    samples = load_prompts(args.data_path)
    batches_per_epoch = math.ceil(len(samples) / args.batch_size)
    start_epoch, start_batch, step = 0, 0, 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        if state["signature"] != training_signature(args):
            raise ValueError("Resume requires the same data, teacher, sampling and optimizer arguments")
        student.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, start_batch, step = state["epoch"], state["batch"], state["step"]
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        if state["mps_rng"] is not None:
            torch.mps.set_rng_state(state["mps_rng"])
    if args.max_steps and step >= args.max_steps:
        return []
    student.train()
    optimizer.zero_grad(set_to_none=True)
    metrics = []
    cursor_epoch, cursor_batch = start_epoch, start_batch
    for epoch in range(start_epoch, args.epochs):
        order = list(range(len(samples)))
        random.Random(args.seed + epoch).shuffle(order)
        first_batch = start_batch if epoch == start_epoch else 0
        for window in range(first_batch, batches_per_epoch, args.accumulation_steps):
            count = min(args.accumulation_steps, batches_per_epoch - window)
            total_loss, turns, calls, tokens, truncated, trajectories_seen = 0.0, 0, 0, 0, 0, 0
            for batch in range(window, window + count):
                indices = order[batch * args.batch_size:(batch + 1) * args.batch_size]
                trajectories = [collect_trajectory(
                    student, tokenizer, *samples[index], max_turns=args.max_turns,
                    max_new_tokens=args.max_gen_len, max_total_len=args.max_total_len,
                    temperature=args.temperature, open_thinking=args.open_thinking,
                    autocast_ctx=autocast_ctx,
                ) for index in indices]
                input_ids, attention_mask, target_mask = pack_trajectories(
                    trajectories, tokenizer.pad_token_id, device,
                )
                if not target_mask.any():
                    raise ValueError("No assistant tokens were generated for this batch")
                with torch.no_grad(), autocast_ctx:
                    teacher_logits = teacher(input_ids, attention_mask=attention_mask).logits[:, :-1]
                with autocast_ctx:
                    output = student(input_ids, attention_mask=attention_mask)
                    kd_loss = distillation_loss(output.logits[:, :-1], teacher_logits, target_mask,
                                                args.distill_temperature, args.kl_direction)
                    loss = kd_loss + output.aux_loss if configs["student"].use_moe else kd_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite OPD loss")
                scaler.scale(loss / count).backward()
                total_loss += loss.item() / count
                tokens += int(target_mask.sum())
                turns += sum(t.turns for t in trajectories)
                calls += sum(t.tool_calls for t in trajectories)
                truncated += sum(t.truncated for t in trajectories)
                trajectories_seen += len(trajectories)
                del teacher_logits, output, loss, kd_loss
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            cursor_batch = window + count
            cursor_epoch = epoch
            if cursor_batch == batches_per_epoch:
                cursor_epoch, cursor_batch = epoch + 1, 0
            record = {"step": step, "loss": total_loss, "assistant_tokens": tokens,
                      "mean_turns": turns / trajectories_seen, "tool_calls": calls,
                      "truncated_fraction": truncated / trajectories_seen}
            metrics.append(record)
            print(json.dumps(record), flush=True)
            if step % args.save_interval == 0:
                save_checkpoint(args, student, optimizer, scaler, cursor_epoch, cursor_batch, step)
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break
    save_checkpoint(args, student, optimizer, scaler, cursor_epoch, cursor_batch, step)
    return metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student_checkpoint", required=True, help="Native MiniMind .pth weights")
    parser.add_argument("--teacher_checkpoint", required=True, help="Frozen teacher using the same tokenizer")
    parser.add_argument("--data_path", default=str(ROOT / "dataset/agent_rl.jsonl"))
    parser.add_argument("--tokenizer_path", default=str(ROOT / "model"))
    parser.add_argument("--output_dir", default=str(ROOT / "out/agent_opd"))
    parser.add_argument("--resume", default="", help="Full-precision *_resume.pth checkpoint")
    for who in ("student", "teacher"):
        parser.add_argument(f"--{who}_hidden_size", type=int, default=768)
        parser.add_argument(f"--{who}_num_layers", type=int, default=8)
        parser.add_argument(f"--{who}_use_moe", type=int, choices=[0, 1], default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=0, help="Optimizer steps; 0 runs all epochs")
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--max_turns", type=int, default=3)
    parser.add_argument("--max_gen_len", type=int, default=256)
    parser.add_argument("--max_total_len", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--distill_temperature", type=float, default=1.0)
    parser.add_argument("--kl_direction", choices=["forward_kl", "reverse_kl"], default="forward_kl")
    parser.add_argument("--open_thinking", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
