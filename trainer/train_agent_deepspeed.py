"""
DeepSpeed Agentic DataAnalysis GRPO/CISPO 入口。

推荐在 trainer/ 目录启动：
    deepspeed --num_gpus=7 train_agent_deepspeed.py --model_profile SearchLM-300M
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import math
import random
import warnings
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

from agentic.data_analysis_env import average_agentic_metrics, parse_tool_calls, score_agentic_trajectory
from dataset.agentic_dataset import AgenticRLDataset
from trainer.rollout_engine import compute_per_token_logps, create_rollout_engine
from trainer.trainer_utils import (
    Logger,
    SkipBatchSampler,
    add_deepspeed_args,
    add_model_profile_args,
    apply_model_profile,
    assert_no_training_output_collision,
    build_lm_config,
    get_cuda_peak_memory_gb,
    init_model,
    is_main_process,
    load_deepspeed_checkpoint,
    load_deepspeed_config,
    log_training_setup,
    reduce_metrics,
    resolve_attention_type,
    save_deepspeed_checkpoint,
    save_model_weights,
    setup_seed,
)

warnings.filterwarnings("ignore")


def init_deepspeed_runtime(args, deepspeed):
    deepspeed.init_distributed(dist_backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank >= 0 else 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"
    else:
        args.device = "cpu"
    args.training_backend = "DeepSpeed-ZeRO"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else local_rank))
    return local_rank


def collate_fn(batch):
    return {
        "id": [x["id"] for x in batch],
        "messages": [x["messages"] for x in batch],
        "tools": [x["tools"] for x in batch],
        "sample": [x["sample"] for x in batch],
    }


def rollout_single(args, rollout_engine, tokenizer, messages, tools, sample):
    all_outputs = []
    prompt_ids = None
    response_ids = []
    response_mask = []
    response_old_logps = []
    final_context = ""
    unfinished = False
    open_thinking = random.random() < args.thinking_ratio

    for turn in range(args.max_turns):
        context = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=tools,
            open_thinking=open_thinking,
        )
        inputs = tokenizer(context, return_tensors="pt", add_special_tokens=False).to(args.device)
        context_ids = inputs["input_ids"][0].tolist()
        if prompt_ids is None:
            prompt_ids = context_ids

        rollout_result = rollout_engine.rollout(
            prompt_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_generations=1,
            max_new_tokens=args.max_gen_len,
            temperature=args.temperature,
        )
        new_ids = rollout_result.completion_ids[0].tolist()
        new_logps = rollout_result.per_token_logps[0].tolist()
        pairs = [
            (token_id, logp)
            for token_id, logp in zip(new_ids, new_logps)
            if token_id != tokenizer.pad_token_id and token_id != tokenizer.eos_token_id
        ]
        new_ids = [token_id for token_id, _ in pairs]
        new_logps = [logp for _, logp in pairs]
        new_text = rollout_result.completions[0]
        all_outputs.append(new_text)
        response_ids.extend(new_ids)
        response_mask.extend([1] * len(new_ids))
        response_old_logps.extend(new_logps)
        final_context = context + new_text

        calls = parse_tool_calls(new_text)
        if not calls:
            break

        unfinished = turn == args.max_turns - 1
        messages.append({"role": "assistant", "content": new_text})
        from agentic.data_analysis_env import AgenticToolEnv

        env = AgenticToolEnv(sample, repo_root=args.repo_root, timeout=args.tool_timeout)
        for call in calls:
            result = env.execute(call.get("name", ""), call.get("arguments", {}))
            result_str = json.dumps(result, ensure_ascii=False)[: args.max_observation_chars]
            messages.append({"role": "tool", "content": result_str})

        observe_context = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=not unfinished,
            tools=tools,
            open_thinking=open_thinking,
        )
        observe_ids = tokenizer(observe_context, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        current_len = len(prompt_ids) + len(response_ids)
        obs_delta = observe_ids[current_len:]
        response_ids.extend(obs_delta)
        response_mask.extend([0] * len(obs_delta))
        response_old_logps.extend([0.0] * len(obs_delta))
        final_context = observe_context
        if unfinished:
            break

    return (
        final_context,
        prompt_ids or [],
        response_ids,
        response_mask,
        response_old_logps,
        all_outputs,
        unfinished,
    )


def rollout_batch(args, rollout_engine, tokenizer, messages_batch, tools_batch, samples):
    contexts = []
    prompt_ids_batch = []
    response_ids_batch = []
    response_masks_batch = []
    response_old_logps_batch = []
    turn_outputs_batch = []
    unfinished_batch = []
    for messages, tools, sample in zip(messages_batch, tools_batch, samples):
        for _ in range(args.num_generations):
            messages_copy = [dict(message) for message in messages]
            context, prompt_ids, response_ids, response_mask, old_logps, turn_outputs, unfinished = rollout_single(
                args,
                rollout_engine,
                tokenizer,
                messages_copy,
                tools,
                sample,
            )
            contexts.append(context)
            prompt_ids_batch.append(prompt_ids)
            response_ids_batch.append(response_ids)
            response_masks_batch.append(response_mask)
            response_old_logps_batch.append(old_logps)
            turn_outputs_batch.append(turn_outputs)
            unfinished_batch.append(unfinished)
    return contexts, prompt_ids_batch, response_ids_batch, response_masks_batch, response_old_logps_batch, turn_outputs_batch, unfinished_batch


def calculate_agentic_rewards(args, turn_outputs_batch, samples, unfinished_batch):
    rewards = []
    metrics = []
    for idx, turn_outputs in enumerate(turn_outputs_batch):
        sample_idx = idx // args.num_generations
        reward, parts = score_agentic_trajectory(
            turn_outputs,
            samples[sample_idx],
            repo_root=args.repo_root,
            unfinished=unfinished_batch[idx],
            execute_tools=True,
        )
        rewards.append(reward)
        metrics.append(parts)
    return torch.tensor(rewards, dtype=torch.float32, device=args.device), average_agentic_metrics(metrics)


def pack_rollout_samples(args, tokenizer, prompt_ids_batch, response_ids_batch, response_masks_batch, response_old_logps_batch):
    packed = []
    for prompt_ids, response_ids, response_mask, old_logps in zip(
        prompt_ids_batch,
        response_ids_batch,
        response_masks_batch,
        response_old_logps_batch,
    ):
        ids = prompt_ids + response_ids
        mask = [0] * len(prompt_ids) + response_mask
        old_per_token = [0.0] * max(len(prompt_ids) - 1, 0) + old_logps
        if len(ids) > args.max_total_len:
            ids = ids[-args.max_total_len:]
            mask = mask[-args.max_total_len:]
            old_per_token = old_per_token[-(len(ids) - 1):]
        target_len = max(len(ids) - 1, 0)
        if len(old_per_token) < target_len:
            old_per_token += [0.0] * (target_len - len(old_per_token))
        elif len(old_per_token) > target_len:
            old_per_token = old_per_token[-target_len:]
        packed.append((ids, mask, old_per_token))

    max_len = max(len(ids) for ids, _, _ in packed)
    input_ids = torch.tensor(
        [ids + [tokenizer.pad_token_id] * (max_len - len(ids)) for ids, _, _ in packed],
        device=args.device,
        dtype=torch.long,
    )
    full_response_masks = torch.tensor(
        [mask + [0] * (max_len - len(mask)) for _, mask, _ in packed],
        device=args.device,
        dtype=torch.float32,
    )
    old_per_token_logps = torch.tensor(
        [old + [0.0] * ((max_len - 1) - len(old)) for _, _, old in packed],
        device=args.device,
        dtype=torch.float32,
    )
    return input_ids, full_response_masks, old_per_token_logps


def train_epoch(args, lm_config, model_engine, tokenizer, loader, iters, rollout_engine, ref_model, start_step=0, wandb=None):
    for step, batch in enumerate(loader, start=start_step + 1):
        contexts, prompt_ids_batch, response_ids_batch, response_masks_batch, response_old_logps_batch, turn_outputs_batch, unfinished_batch = rollout_batch(
            args,
            rollout_engine,
            tokenizer,
            batch["messages"],
            batch["tools"],
            batch["sample"],
        )
        input_ids, full_response_masks, old_per_token_logps = pack_rollout_samples(
            args,
            tokenizer,
            prompt_ids_batch,
            response_ids_batch,
            response_masks_batch,
            response_old_logps_batch,
        )
        full_mask = (input_ids != tokenizer.pad_token_id).long()
        rewards, reward_stats = calculate_agentic_rewards(args, turn_outputs_batch, batch["sample"], unfinished_batch)

        res = model_engine(input_ids, attention_mask=full_mask)
        aux_loss = res.aux_loss if getattr(lm_config, "use_moe", False) and res.aux_loss is not None else torch.tensor(0.0, device=args.device)
        per_token_logps = F.log_softmax(res.logits[:, :-1, :], dim=-1).gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            ref_per_token_logps = compute_per_token_logps(
                ref_model,
                input_ids,
                input_ids.size(1) - 1,
                attention_mask=full_mask,
            )

        completion_mask = full_response_masks[:, 1:]
        token_counts = completion_mask.sum(dim=1)
        valid_rows = token_counts > 0

        grouped_rewards = rewards.view(-1, args.num_generations)
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
        std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
        advantages = (rewards - mean_r) / (std_r + 1e-4)

        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1
        ratio = torch.exp(per_token_logps - old_per_token_logps)
        if args.loss_type == "cispo":
            clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
            per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)
        else:
            clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
            per_token_loss1 = ratio * advantages.unsqueeze(1)
            per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
            per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)
        policy_loss = (
            ((per_token_loss * completion_mask).sum(dim=1)[valid_rows] / token_counts[valid_rows].clamp(min=1)).mean()
            if valid_rows.any()
            else per_token_loss.sum() * 0.0
        )
        loss = policy_loss + aux_loss

        model_engine.backward(loss)
        model_engine.step()

        if step % args.log_interval == 0 or step == iters:
            reward_val = rewards.mean().item()
            avg_len_val = token_counts.float().mean().item()
            kl_ref_val = ((ref_per_token_logps - per_token_logps) * completion_mask).sum().item() / max(token_counts.sum().item(), 1)
            lr_list = model_engine.get_lr()
            current_lr = lr_list[0] if lr_list else args.learning_rate
            metrics = reduce_metrics(
                {
                    "reward": reward_val,
                    "avg_len": avg_len_val,
                    "kl_ref": kl_ref_val,
                    "policy_loss": policy_loss.item(),
                    "aux_loss": aux_loss.item(),
                    **reward_stats,
                }
            )
            peak_memory_gb = get_cuda_peak_memory_gb()
            memory_text = f", peak_memory: {peak_memory_gb:.2f} GB" if peak_memory_gb is not None else ""
            Logger(
                f"Epoch:[{args.current_epoch + 1}/{args.epochs}]({step}/{iters}), "
                f"Reward:{metrics['reward']:.4f}, Success:{metrics.get('task_success', 0):.4f}, "
                f"ToolF1:{metrics.get('tool_selection_f1', 0):.4f}, Invalid:{metrics.get('invalid_action_rate', 0):.4f}, "
                f"Exec:{metrics.get('exec_success_rate', 0):.4f}, Turns:{metrics.get('avg_turns', 0):.2f}, "
                f"KL:{metrics['kl_ref']:.4f}, Loss:{metrics['policy_loss']:.4f}, LR:{current_lr:.8f}"
                f"{memory_text}"
            )
            if wandb and is_main_process():
                log_data = {
                    "reward": metrics["reward"],
                    "kl_ref": metrics["kl_ref"],
                    "policy_loss": metrics["policy_loss"],
                    "aux_loss": metrics["aux_loss"],
                    "avg_response_len": metrics["avg_len"],
                    "learning_rate": current_lr,
                }
                log_data.update({key: value for key, value in metrics.items() if key not in log_data})
                wandb.log(log_data)

        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            for i, sample_id in enumerate(batch["id"][:2]):
                Logger(f"[DEBUG] sample={sample_id}")
                for j in range(args.num_generations):
                    idx = i * args.num_generations + j
                    Logger(f"[DEBUG] gen={j}, reward={rewards[idx].item():.4f}, unfinished={unfinished_batch[idx]}")
                    Logger("\n---\n".join(turn_outputs_batch[idx]))
                    Logger("=" * 80)

        if step % args.save_interval == 0 or step == iters:
            model_engine.eval()
            save_deepspeed_checkpoint(
                model_engine,
                lm_config,
                weight=args.save_weight,
                epoch=args.current_epoch,
                step=step,
                wandb=wandb,
                save_dir="../checkpoints",
            )
            if is_main_process():
                ckp = save_model_weights(lm_config, model_engine, save_dir=args.save_dir, weight=args.save_weight)
                Logger(f"Lightweight model saved: {ckp}")
            model_engine.train()
            rollout_engine.update_policy(model_engine)

        del contexts, prompt_ids_batch, response_ids_batch, response_masks_batch, response_old_logps_batch
        del turn_outputs_batch, unfinished_batch, input_ids, full_response_masks, old_per_token_logps
        del per_token_logps, ref_per_token_logps, rewards, grouped_rewards, mean_r, std_r, advantages
        del completion_mask, token_counts, res, loss, policy_loss, aux_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Agentic DataAnalysis GRPO/CISPO with DeepSpeed ZeRO")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument("--save_weight", default="agent_grpo", type=str)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-7)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=25)
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--num_attention_heads", default=8, type=int)
    parser.add_argument("--num_key_value_heads", default=4, type=int)
    parser.add_argument("--intermediate_size", default=None, type=int)
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1])
    parser.add_argument("--attention_type", default="gqa", choices=["gqa", "mha", "mqa", "mla"])
    parser.add_argument("--use_mla", default=0, type=int, choices=[0, 1])
    parser.add_argument("--kv_lora_rank", default=128, type=int)
    parser.add_argument("--q_lora_rank", default=256, type=int)
    parser.add_argument("--rope_dim", default=None, type=int)
    parser.add_argument("--max_seq_len", default=1536, type=int)
    parser.add_argument("--max_gen_len", type=int, default=512)
    parser.add_argument("--max_total_len", type=int, default=2500)
    parser.add_argument("--max_turns", type=int, default=4)
    parser.add_argument("--max_observation_chars", type=int, default=2048)
    parser.add_argument("--tool_timeout", type=int, default=3)
    parser.add_argument("--repo_root", type=str, default="..")
    parser.add_argument("--data_path", type=str, default="../dataset/agentic_data/train.jsonl")
    parser.add_argument("--num_generations", type=int, default=6)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo"])
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--epsilon_high", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--thinking_ratio", type=float, default=0.1)
    parser.add_argument("--from_weight", default="agent_sft", type=str)
    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1])
    parser.add_argument("--overwrite_output", default=0, type=int, choices=[0, 1])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Agentic-DataAnalysis-RL-DS")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1])
    parser.add_argument("--debug_mode", action="store_true")
    parser.add_argument("--debug_interval", type=int, default=20)
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"])
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998")
    parser.add_argument("--sglang_model_path", type=str, default="../model")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_agentic")
    add_model_profile_args(parser)
    add_deepspeed_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_profile = apply_model_profile(args)

    try:
        import deepspeed
    except ImportError as exc:
        raise RuntimeError("DeepSpeed未安装。服务器环境执行: pip install deepspeed") from exc

    local_rank = init_deepspeed_runtime(args, deepspeed)
    os.makedirs(args.save_dir, exist_ok=True)

    lm_config = build_lm_config(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_seq_len=max(args.max_seq_len + args.max_gen_len, args.max_total_len),
        use_moe=bool(args.use_moe),
        attention_type=resolve_attention_type(args),
        kv_lora_rank=args.kv_lora_rank,
        q_lora_rank=args.q_lora_rank,
        rope_dim=args.rope_dim,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        intermediate_size=args.intermediate_size,
    )
    assert_no_training_output_collision(
        lm_config,
        weight=args.save_weight,
        save_dir=args.save_dir,
        checkpoint_dir="../checkpoints",
        from_resume=args.from_resume,
        overwrite=args.overwrite_output,
    )

    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger("torch.compile enabled before DeepSpeed initialize")

    train_ds = AgenticRLDataset(args.data_path, tokenizer, max_length=args.max_total_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collate_fn)
    iters = len(loader_for_count)
    total_optimizer_steps = max(1, math.ceil(iters / args.accumulation_steps) * args.epochs)
    ds_config = load_deepspeed_config(args, total_optimizer_steps)

    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )

    ckp_data = None
    if args.from_resume == 1:
        ckp_data = load_deepspeed_checkpoint(
            model_engine,
            lm_config,
            weight=args.save_weight,
            save_dir="../checkpoints",
        )

    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    if torch.cuda.is_available() and args.dtype in {"float16", "bfloat16"}:
        ref_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
        ref_model = ref_model.to(dtype=ref_dtype)

    device_type = "cuda" if "cuda" in args.device else "cpu"
    amp_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" or args.dtype == "float32" else torch.cuda.amp.autocast(dtype=amp_dtype)

    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=model_engine,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )
    rollout_engine.update_policy(model_engine)

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        wandb.init(
            project=args.wandb_project,
            name=f"Agentic-RL-DS-{args.attention_type}-{args.hidden_size}",
            id=wandb_id,
            resume="must" if wandb_id else None,
        )

    log_training_setup(
        args,
        lm_config,
        stage="agentic_rl_deepspeed",
        dataset_len=len(train_ds),
        iters=iters,
        tokens_per_sample=args.max_total_len,
        extra={
            "model_profile": args.model_profile if model_profile else "",
            "deepspeed_config": args.deepspeed_config,
            "zero_stage": ds_config.get("zero_optimization", {}).get("stage", "none"),
            "num_generations": args.num_generations,
            "rollout_engine": args.rollout_engine,
            "loss_type": args.loss_type,
            "beta": args.beta,
            "max_turns": args.max_turns,
            "total_optimizer_steps": total_optimizer_steps,
        },
    )

    start_epoch = ckp_data.get("epoch", 0) if ckp_data else 0
    start_step = ckp_data.get("step", 0) if ckp_data else 0

    for epoch in range(start_epoch, args.epochs):
        args.current_epoch = epoch
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch + local_rank)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        if skip > 0:
            Logger(f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始")
        train_epoch(args, lm_config, model_engine, tokenizer, loader, len(loader) + skip, rollout_engine, ref_model, skip, wandb)

    if dist.is_initialized():
        dist.destroy_process_group()
