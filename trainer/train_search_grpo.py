import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import math
import warnings
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler

from dataset.search_shortqa_dataset import SearchShortQARLDataset
from trainer.rollout_engine import create_rollout_engine
from trainer.search_shortqa_reward import average_metrics, score_search_shortqa_response
from trainer.trainer_utils import (
    Logger,
    SkipBatchSampler,
    build_lm_config,
    get_model_suffix,
    init_distributed_mode,
    init_model,
    is_main_process,
    lm_checkpoint,
    log_training_setup,
    resolve_attention_type,
    setup_seed,
    add_model_profile_args,
    apply_model_profile,
)

warnings.filterwarnings("ignore")


def collate_fn(batch):
    return {
        "prompt": [x["prompt"] for x in batch],
        "question": [x["question"] for x in batch],
        "answers": [x["answers"] for x in batch],
        "gold_doc_ids": [x["gold_doc_ids"] for x in batch],
        "contexts": [x["contexts"] for x in batch],
    }


def calculate_search_rewards(completions, batch):
    rewards = []
    metrics = []
    for i, response in enumerate(completions):
        sample_idx = i // args.num_generations
        reward, parts = score_search_shortqa_response(
            response=response,
            answers=batch["answers"][sample_idx],
            gold_doc_ids=batch["gold_doc_ids"][sample_idx],
            contexts=batch["contexts"][sample_idx],
        )
        rewards.append(reward)
        metrics.append(parts)
    return torch.tensor(rewards, dtype=torch.float32, device=args.device), average_metrics(metrics)


def train_epoch(epoch, loader, iters, rollout_engine, ref_model, start_step=0, wandb=None):
    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompt"]
        prompt_inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
            padding_side="left",
            add_special_tokens=False,
        ).to(args.device)
        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

        rollout_result = rollout_engine.rollout(
            prompt_ids=prompt_inputs["input_ids"],
            attention_mask=prompt_inputs["attention_mask"],
            num_generations=args.num_generations,
            max_new_tokens=args.max_gen_len,
            temperature=args.temperature,
        )
        outputs = rollout_result.output_ids
        completion_ids = rollout_result.completion_ids
        completions = rollout_result.completions
        old_per_token_logps = rollout_result.per_token_logps.to(args.device).detach()
        prompt_lens = rollout_result.prompt_lens.to(args.device)
        full_mask = (outputs != tokenizer.pad_token_id).long()
        logp_pos = prompt_lens.unsqueeze(1) - 1 + torch.arange(completion_ids.size(1), device=args.device).unsqueeze(0)

        rewards, reward_stats = calculate_search_rewards(completions, batch)

        model_unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        with autocast_ctx:
            res = model_unwrapped(outputs, attention_mask=full_mask)
            aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
            per_token_logps = (
                F.log_softmax(res.logits[:, :-1, :], dim=-1)
                .gather(2, outputs[:, 1:].unsqueeze(-1))
                .squeeze(-1)
                .gather(1, logp_pos)
            )

        with torch.no_grad():
            ref_per_token_logps = (
                F.log_softmax(ref_model(outputs, attention_mask=full_mask).logits[:, :-1, :], dim=-1)
                .gather(2, outputs[:, 1:].unsqueeze(-1))
                .squeeze(-1)
                .gather(1, logp_pos)
            )

        grouped_rewards = rewards.view(-1, args.num_generations)
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
        std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
        advantages = (rewards - mean_r) / (std_r + 1e-4)

        completion_pad_mask = rollout_result.completion_mask.to(args.device).bool()
        is_eos = (completion_ids == tokenizer.eos_token_id) & completion_pad_mask
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1) - 1, dtype=torch.long, device=args.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        completion_mask = (
            (torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1))
            & completion_pad_mask
        ).int()

        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1
        ratio = torch.exp(per_token_logps - old_per_token_logps)
        clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
        per_token_loss1 = ratio * advantages.unsqueeze(1)
        per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
        per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)
        policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)).mean()
        loss = (policy_loss + aux_loss) / args.accumulation_steps
        loss.backward()

        if step % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % args.log_interval == 0 or step == iters:
            reward_val = rewards.mean().item()
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            kl_ref_val = ((ref_per_token_logps - per_token_logps) * completion_mask).sum().item() / max(completion_mask.sum().item(), 1)
            current_lr = optimizer.param_groups[0]["lr"]
            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                f"Reward: {reward_val:.4f}, F1: {reward_stats.get('answer_f1', 0):.4f}, "
                f"EM: {reward_stats.get('exact_match', 0):.4f}, Cite: {reward_stats.get('citation_score', 0):.4f}, "
                f"Format: {reward_stats.get('format_score', 0):.4f}, KL_ref: {kl_ref_val:.4f}, "
                f"Avg Len: {avg_len_val:.2f}, LR: {current_lr:.8f}"
            )
            if wandb and is_main_process():
                log_data = {
                    "reward": reward_val,
                    "kl_ref": kl_ref_val,
                    "policy_loss": policy_loss.item(),
                    "avg_response_len": avg_len_val,
                    "learning_rate": current_lr,
                }
                log_data.update(reward_stats)
                wandb.log(log_data)

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            model_suffix = get_model_suffix(lm_config)
            ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{model_suffix}.pth"
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, "_orig_mod", raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
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
            )
            model.train()
            del state_dict

        if step % args.save_interval == 0 or step == iters:
            rollout_engine.update_policy(model)

        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask, completion_pad_mask, prompt_lens

    if step > start_step and step % args.accumulation_steps != 0:
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SearchShortQA GRPO")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument("--save_weight", default="search_grpo", type=str)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=3e-7)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=50)
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
    parser.add_argument("--max_seq_len", default=1024, type=int)
    parser.add_argument("--max_gen_len", type=int, default=256)
    parser.add_argument("--data_path", type=str, default="../dataset/search_shortqa_train.jsonl")
    parser.add_argument("--num_generations", type=int, default=6)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--from_weight", default="full_sft_search", type=str)
    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="SearchShortQA-GRPO")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1])
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"])
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998")
    parser.add_argument("--sglang_model_path", type=str, default="../model")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_search_grpo")
    add_model_profile_args(parser)
    args = parser.parse_args()
    model_profile = apply_model_profile(args)

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = build_lm_config(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_seq_len=args.max_seq_len + args.max_gen_len,
        use_moe=bool(args.use_moe),
        attention_type=resolve_attention_type(args),
        kv_lora_rank=args.kv_lora_rank,
        q_lora_rank=args.q_lora_rank,
        rope_dim=args.rope_dim,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        intermediate_size=args.intermediate_size,
    )
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir="../checkpoints") if args.from_resume else None

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        wandb.init(project=args.wandb_project, name=f"SearchShortQA-GRPO-{args.attention_type}", id=wandb_id, resume="must" if wandb_id else None)

    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )
    train_ds = SearchShortQARLDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collate_fn)
    iters = len(loader_for_count)
    total_optimizer_steps = math.ceil(iters / args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    log_training_setup(
        args,
        lm_config,
        stage="search_shortqa_grpo",
        dataset_len=len(train_ds),
        iters=iters,
        tokens_per_sample=args.max_seq_len + args.max_gen_len,
        extra={
            "num_generations": args.num_generations,
            "rollout_engine": args.rollout_engine,
            "beta": args.beta,
            "total_optimizer_steps": total_optimizer_steps,
            "model_profile": args.model_profile if model_profile else "",
        },
    )

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        scheduler.load_state_dict(ckp_data["scheduler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)

    if args.use_compile == 1:
        model = torch.compile(model)
        Logger("torch.compile enabled")
        rollout_engine.update_policy(model)
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    rollout_engine.update_policy(model)

    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
        train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, skip, wandb)

    if dist.is_initialized():
        dist.destroy_process_group()
