import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import csv
import json
import math
import re
import gc
import warnings
from datetime import datetime, timezone
import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from dataset.math_rl_dataset import MathRLDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model, LMForRewardModel
from trainer.rollout_engine import create_rollout_engine
from trainer.math_verifier import ANSWER_MARKER, reward_tier, verdict

warnings.filterwarnings('ignore')


def rep_penalty(text, n=3, cap=0.5):
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


def rule_rewards(completions, gt_batch, marker=ANSWER_MARKER, tier="binary"):
    """Ground-truth verifier reward for a B*G completion list.

    Returns ``(scores, stats)``.  ``gt_batch`` must already be expanded to one
    ground truth per rollout; the length assert is the guard against the classic
    off-by-G bug where B ground truths are zipped with B*G completions.
    """

    if len(completions) != len(gt_batch):
        raise ValueError(
            f"completion/gt length mismatch: {len(completions)} completions vs "
            f"{len(gt_batch)} ground truths (expected repeat_interleave(G))"
        )
    if not completions:
        raise ValueError("empty completion batch")

    scores, verdicts = [], []
    for completion, gt in zip(completions, gt_batch):
        result = verdict(completion, gt, marker=marker)
        scores.append(reward_tier(result, mode=tier))
        verdicts.append(result)

    total = len(verdicts)
    stats = {
        "n": total,
        "correct": sum(bool(v["correct"]) for v in verdicts),
        "incorrect": sum(v["status"] == "incorrect" for v in verdicts),
        "no_answer": sum(v["status"] == "no_answer" for v in verdicts),
        "invalid_gt": sum(v["status"] == "invalid_gt" for v in verdicts),
        "correct_flags": [bool(v["correct"]) for v in verdicts],
        "verdicts": verdicts,
    }
    stats["correct_rate"] = stats["correct"] / total
    stats["no_answer_rate"] = stats["no_answer"] / total
    stats["format_rate"] = sum(bool(v["format_ok"]) for v in verdicts) / total
    stats["mean_rule_reward"] = sum(scores) / total
    return scores, stats


def _parse_prompt_messages(prompt):
    """Recover the chat messages that the official RM history parser expects."""

    pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
    matches = re.findall(pattern, prompt, re.DOTALL)
    return [{"role": role, "content": content.strip()} for role, content in matches]


def official_rm_rewards(prompts, responses, reward_model):
    """Pre-change reward, byte-for-byte behaviour: heuristic + RM judge.

    Kept as the parity control (``--reward_mode rm`` on math data, and
    ``--reward_mode official`` on ``rlaif.jsonl``) so a refactor can be diffed
    against the original numbers instead of being asserted to be equivalent.
    """

    if reward_model is None:
        raise ValueError("reward_model is required for RM-based rewards")
    scores = []
    for i in range(len(prompts)):
        for j in range(args.num_generations):
            response = responses[i * args.num_generations + j]
            messages = _parse_prompt_messages(prompts[i])
            answer = response
            score = 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
            if '</think>' in response:
                thinking_content, answer_content = response.split('</think>', 1)
                score += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                score += 0.25 if response.count('</think>') == 1 else -0.25
                answer = answer_content.strip()
            score -= rep_penalty(answer)
            score += reward_model.get_score(messages, answer)
            scores.append(score)
    return scores, {"n": len(scores), "mean_official_reward": sum(scores) / max(len(scores), 1)}


def pure_rm_rewards(prompts, responses, reward_model):
    """RM judge score only, without the heuristic length/think/rep terms.

    The mix arm must not silently inherit three reward components when the
    hypothesis is about combining exactly two signals.
    """

    if reward_model is None:
        raise ValueError("reward_model is required for RM-based rewards")
    scores = []
    for i in range(len(prompts)):
        for j in range(args.num_generations):
            response = responses[i * args.num_generations + j]
            messages = _parse_prompt_messages(prompts[i])
            answer = response.split('</think>', 1)[1].strip() if '</think>' in response else response
            scores.append(reward_model.get_score(messages, answer))
    return scores, {"n": len(scores), "mean_rm_reward": sum(scores) / max(len(scores), 1)}


def normalize_group(x, eps=1e-6):
    """Group-normalize a ``[B, G]`` tensor.

    An all-equal group has std 0 and therefore maps to exactly zero instead of
    NaN, which is the intended "no relative signal" semantics.
    """

    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, unbiased=False, keepdim=True)
    return (x - mean) / (std + eps)


def mix_rewards(rule_score, rm_score, num_generations, alpha, beta, eps=1e-6):
    """Standardise both components per group, then apply the explicit weights.

    Raw scores live on different scales (rule in [-0.5, 1], RM in [-3, 3]), so
    adding them unnormalised would silently turn the mix arm into the RM arm.
    """

    if len(rule_score) != len(rm_score):
        raise ValueError(f"mix component length mismatch: {len(rule_score)} vs {len(rm_score)}")
    if len(rule_score) % num_generations:
        raise ValueError(f"{len(rule_score)} samples is not divisible by G={num_generations}")

    rule_t = torch.tensor(rule_score, dtype=torch.float32).view(-1, num_generations)
    rm_t = torch.tensor(rm_score, dtype=torch.float32).view(-1, num_generations)
    degenerate = (rule_t.std(dim=1, unbiased=False) < eps) | (rm_t.std(dim=1, unbiased=False) < eps)
    mixed = (alpha * normalize_group(rule_t, eps) + beta * normalize_group(rm_t, eps)).reshape(-1)
    return mixed.tolist(), {
        "n": int(mixed.numel()),
        "mix_alpha": alpha,
        "mix_beta": beta,
        "component_degenerate_groups": int(degenerate.sum().item()),
    }


def calculate_rewards(prompts, responses, gt_for_rollouts=None, reward_model=None):
    """Reward dispatcher.

    Returns ``(scores, stats)`` where ``scores`` is a plain list so the caller
    decides the device.  The verifier always runs when ground truth is present,
    even on the RM arms: the reward comes from the RM there, but correct/format
    rates are what make the arms comparable at all.
    """

    mode = args.reward_mode
    rule_scores, stats = None, {}
    if mode != "official":
        if gt_for_rollouts is None:
            raise ValueError(f"--reward_mode {mode} requires ground-truth answers per rollout")
        rule_scores, stats = rule_rewards(
            responses, gt_for_rollouts, marker=args.reward_marker, tier=args.verdict_tier
        )

    if mode == "rule":
        if reward_model is not None:
            raise AssertionError("rule mode must not load a reward model")
        return rule_scores, stats

    if mode == "rm":
        rm_scores, rm_stats = official_rm_rewards(prompts, responses, reward_model)
        return rm_scores, {**stats, **rm_stats}

    if mode == "official":
        return official_rm_rewards(prompts, responses, reward_model)

    if mode == "mix":
        rm_scores, rm_stats = pure_rm_rewards(prompts, responses, reward_model)
        mixed, mix_stats = mix_rewards(
            rule_scores, rm_scores, args.num_generations, args.mix_alpha, args.mix_beta
        )
        return mixed, {**stats, **rm_stats, **mix_stats}

    raise ValueError(f"unknown reward mode: {mode}")


METRIC_FIELDS = [
    "timestamp", "step", "seed", "avg_reward", "format_rate", "correct_rate",
    "no_answer_rate", "avg_completion_tokens", "median_completion_tokens",
    "kl_ref", "entropy", "adv_std", "policy_loss", "valid_groups",
    "all_correct_groups", "all_wrong_groups", "filtered_fraction",
]


class RunLogger:
    """Append-only metrics CSV plus raw-sample JSONL for one run.

    The samples file is the evidence behind the error analysis and the manual
    audit, so it stores the prompt, the raw completion, the ground truth and the
    verifier verdict together.
    """

    def __init__(self, log_dir, run_tag, enabled=True):
        self.enabled = enabled
        self.metrics_file = None
        self.writer = None
        if not enabled:
            return
        os.makedirs(log_dir, exist_ok=True)
        self.metrics_path = os.path.join(log_dir, f"{run_tag}_metrics.csv")
        self.samples_path = os.path.join(log_dir, f"{run_tag}_samples.jsonl")
        is_new = not os.path.exists(self.metrics_path) or os.path.getsize(self.metrics_path) == 0
        self.metrics_file = open(self.metrics_path, "a", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.metrics_file, fieldnames=METRIC_FIELDS)
        if is_new:
            self.writer.writeheader()
            self.metrics_file.flush()

    def samples_due(self, step):
        if not self.enabled:
            return False
        interval = args.debug_interval if args.debug_mode else args.save_interval
        return interval > 0 and step % interval == 0

    def log_metrics(self, row):
        if self.writer is None:
            return
        self.writer.writerow({field: row.get(field, "") for field in METRIC_FIELDS})
        self.metrics_file.flush()

    def log_samples(self, records):
        if not self.enabled or not records:
            return
        with open(self.samples_path, "a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self):
        if self.metrics_file is not None:
            self.metrics_file.close()
            self.metrics_file = None
            self.writer = None


def grpo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model, start_step=0, wandb=None, run_logger=None, use_sglang=False):
    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch['prompt']  # list[str], length B
        gt_batch = batch.get('gt')  # list[str] | None
        difficulty_batch = batch.get('difficulty') or ['unknown'] * len(prompts)
        sample_id_batch = batch.get('sample_id') or [''] * len(prompts)
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
                                  padding_side="left", add_special_tokens=False).to(args.device)
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

        # Completion mask is built before the reward: the length penalty and every
        # per-sample metric below must use real EOS-scanned token counts rather
        # than decoded string lengths.
        completion_pad_mask = rollout_result.completion_mask.to(args.device).bool()
        is_eos = (completion_ids == tokenizer.eos_token_id) & completion_pad_mask  # [B*num_gen, R]
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1) - 1, dtype=torch.long, device=args.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        completion_mask = ((torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)) & completion_pad_mask).int()  # [B*num_gen, R]
        completion_lengths = completion_mask.sum(dim=1).float()  # [B*num_gen]

        # Ground truth is expanded to one entry per rollout in prompt order.
        gt_for_rollouts = None
        if gt_batch is not None:
            assert len(gt_batch) == len(prompts), f"gt batch {len(gt_batch)} != prompts {len(prompts)}"
            gt_for_rollouts = [gt for gt in gt_batch for _ in range(args.num_generations)]
            assert len(gt_for_rollouts) == len(completions), (
                f"gt/rollout mismatch: {len(gt_for_rollouts)} gt vs {len(completions)} completions")

        reward_list, reward_stats = calculate_rewards(
            prompts=prompts,
            responses=completions,
            gt_for_rollouts=gt_for_rollouts,
            reward_model=reward_model,
        )
        rewards = torch.tensor(reward_list, dtype=torch.float32, device=args.device)  # [B*num_gen]

        if args.len_penalty_coef > 0:
            excess = (completion_lengths - args.length_cap).clamp(min=0)
            rewards = rewards - args.len_penalty_coef * excess / args.length_cap

        # Degenerate-group bookkeeping. `valid_group` is what DAPO-style dynamic
        # sampling filters on: a group with no correct/wrong contrast carries no
        # relative task signal.
        all_correct = all_wrong = valid_group = None
        correct_flags = reward_stats.get("correct_flags")
        if correct_flags is not None and len(correct_flags) == len(completions):
            grouped_correct = torch.tensor(correct_flags, dtype=torch.bool).view(-1, args.num_generations)
            all_correct = grouped_correct.all(dim=1)
            all_wrong = ~grouped_correct.any(dim=1)
            valid_group = ~all_correct & ~all_wrong
        elif args.dynamic_sampling:
            raise ValueError(
                "--dynamic_sampling needs per-sample correctness, which only the ground-truth "
                "verifier provides (--reward_mode rule/rm/mix)")

        model_unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        log_now = (step % args.log_interval == 0 or step == iters)
        with autocast_ctx:
            res = model_unwrapped(outputs, attention_mask=full_mask)
            aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
            log_probs = F.log_softmax(res.logits[:, :-1, :], dim=-1)
            per_token_logps = log_probs.gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)
        if log_now:
            with torch.no_grad():
                detached = log_probs.detach()
                token_entropy = (-(detached.exp() * detached)).sum(dim=-1).gather(1, logp_pos)

        with torch.no_grad():
            ref_per_token_logps = F.log_softmax(ref_model(outputs, attention_mask=full_mask).logits[:, :-1, :], dim=-1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)

        verdicts = reward_stats.get("verdicts") or []
        if (args.debug_mode and is_main_process() and step % args.debug_interval == 0) or \
                (run_logger is not None and run_logger.samples_due(step) and is_main_process()):
            for i in range(len(prompts)):
                Logger(f"[DEBUG] step={step}, sample[{i}], gt={gt_batch[i] if gt_batch else 'n/a'}, "
                       f"difficulty={difficulty_batch[i]}")
                Logger('-'*100)
                Logger(f"{'=' * 30} [DEBUG] sample[{i}] CONTEXT_BEGIN {'=' * 30}")
                Logger(prompts[i])
                Logger(f"{'=' * 31} [DEBUG] sample[{i}] CONTEXT_END {'=' * 31}")
                for j in range(args.num_generations):
                    idx = i * args.num_generations + j
                    status = verdicts[idx]['status'] if idx < len(verdicts) else 'n/a'
                    Logger(f"{'=' * 28} [DEBUG] gen[{j}] RESPONSE_BEGIN {'=' * 28}")
                    Logger(completions[idx])
                    Logger(f"{'=' * 29} [DEBUG] gen[{j}] RESPONSE_END {'=' * 29}")
                    Logger(f"[DEBUG] gen[{j}] reward={rewards[idx].item():.4f}, "
                           f"status={status}, tokens={int(completion_lengths[idx].item())}")
                Logger('='*100)

        if run_logger is not None and run_logger.samples_due(step):
            run_logger.log_samples([
                {
                    "step": step,
                    "seed": args.seed,
                    "prompt": prompts[idx // args.num_generations],
                    "completion": completions[idx],
                    "gt": gt_for_rollouts[idx] if gt_for_rollouts else None,
                    "reward": round(float(rewards[idx].item()), 6),
                    "status": verdicts[idx]["status"] if idx < len(verdicts) else "n/a",
                    "extracted": verdicts[idx]["extracted"] if idx < len(verdicts) else None,
                    "think_closed": verdicts[idx]["think_closed"] if idx < len(verdicts) else None,
                    "completion_tokens": int(completion_lengths[idx].item()),
                    "difficulty": difficulty_batch[idx // args.num_generations],
                    "sample_id": sample_id_batch[idx // args.num_generations],
                }
                for idx in range(len(completions))
            ])

        grouped_rewards = rewards.view(-1, args.num_generations)  # [B, num_gen]
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)  # [B*num_gen]
        std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)  # [B*num_gen]
        advantages = (rewards - mean_r) / (std_r + 1e-4)  # [B*num_gen]

        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1  # [B*num_gen, R]
        ratio = torch.exp(per_token_logps - old_per_token_logps)  # [B*num_gen, R]
        if args.loss_type == "cispo":
            clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
            pg_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps)
        else:
            clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
            per_token_loss1 = ratio * advantages.unsqueeze(1)
            per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
            pg_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        kl_token_loss = args.beta * per_token_kl

        token_count = completion_mask.sum(dim=1).clamp(min=1)
        per_sample_pg = (pg_token_loss * completion_mask).sum(dim=1) / token_count
        per_sample_kl = (kl_token_loss * completion_mask).sum(dim=1) / token_count
        if args.dynamic_sampling:
            valid_sample = valid_group.repeat_interleave(args.num_generations)
            pg_loss = per_sample_pg[valid_sample].mean() if bool(valid_sample.any()) else per_sample_pg.sum() * 0.0
        else:
            pg_loss = per_sample_pg.mean()
        policy_loss = pg_loss + per_sample_kl.mean()
        loss = (policy_loss + aux_loss) / args.accumulation_steps  # scalar
        loss.backward()

        if step % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if log_now:
            policy_loss_val = loss.item() * args.accumulation_steps
            current_aux_loss = aux_loss.item()
            avg_reward_val = rewards.mean().item()
            avg_len_val = completion_lengths.mean().item()
            kl_ref_val = ((ref_per_token_logps - per_token_logps) * completion_mask).sum().item() / max(completion_mask.sum().item(), 1)
            entropy_val = ((token_entropy * completion_mask).sum() / completion_mask.sum().clamp(min=1)).item()
            advantages_mean_val = advantages.mean().item()
            advantages_std_val = advantages.std().item()
            current_lr = optimizer.param_groups[0]['lr']

            total_groups = rewards.size(0) // args.num_generations
            if valid_group is not None:
                valid_groups = int(valid_group.sum().item())
                all_correct_groups = int(all_correct.sum().item())
                all_wrong_groups = int(all_wrong.sum().item())
                filtered_fraction = 1.0 - valid_groups / total_groups if total_groups else 0.0
            else:
                valid_groups = all_correct_groups = all_wrong_groups = -1
                filtered_fraction = ""

            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'Reward: {avg_reward_val:.4f}, KL_ref: {kl_ref_val:.4f}, '
                   f'Adv Std: {advantages_std_val:.4f}, Adv Mean: {advantages_mean_val:.4f}, '
                   f'Actor Loss: {policy_loss_val:.4f}, Avg Response Len: {avg_len_val:.2f}, Learning Rate: {current_lr:.8f}')
            if reward_stats:
                Logger(f'    correct_rate: {reward_stats.get("correct_rate", float("nan")):.4f}, '
                       f'format_rate: {reward_stats.get("format_rate", float("nan")):.4f}, '
                       f'no_answer_rate: {reward_stats.get("no_answer_rate", float("nan")):.4f}, '
                       f'valid_groups: {valid_groups}/{total_groups}, '
                       f'all_correct: {all_correct_groups}, all_wrong: {all_wrong_groups}')

            if run_logger is not None:
                run_logger.log_metrics({
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "step": step,
                    "seed": args.seed,
                    "avg_reward": avg_reward_val,
                    "format_rate": reward_stats.get("format_rate", ""),
                    "correct_rate": reward_stats.get("correct_rate", ""),
                    "no_answer_rate": reward_stats.get("no_answer_rate", ""),
                    "avg_completion_tokens": avg_len_val,
                    "median_completion_tokens": completion_lengths.median().item(),
                    "kl_ref": kl_ref_val,
                    "entropy": entropy_val,
                    "adv_std": advantages_std_val,
                    "policy_loss": policy_loss_val,
                    "valid_groups": valid_groups,
                    "all_correct_groups": all_correct_groups,
                    "all_wrong_groups": all_wrong_groups,
                    "filtered_fraction": filtered_fraction,
                })

            if wandb and is_main_process():
                wandb.log({
                    "reward": avg_reward_val,
                    "kl_ref": kl_ref_val,
                    "advantages_std": advantages_std_val,
                    "advantages_mean": advantages_mean_val,
                    "policy_loss": policy_loss_val,
                    "avg_response_len": avg_len_val,
                    "entropy": entropy_val,
                    "learning_rate": current_lr,
                    **{k: v for k, v in reward_stats.items()
                       if isinstance(v, (int, float)) and not isinstance(v, bool)},
                })

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scheduler=scheduler)
            model.train()
            del state_dict

        if step % args.save_interval == 0 or step == iters: rollout_engine.update_policy(model)

        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps, log_probs
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask, completion_pad_mask, prompt_lens, logp_pos
        del completion_lengths, per_sample_pg, per_sample_kl

    if step > start_step and step % args.accumulation_steps != 0:
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()


def _dump_run_config(args, length_audit=None):
    """Persist the exact configuration of this run next to its metrics."""

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "args": vars(args),
        "length_audit": length_audit,
    }
    try:
        os.makedirs(args.log_dir, exist_ok=True)
        path = os.path.join(args.log_dir, f"{args.run_tag}_config.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        Logger(f"Run config written to {path}")
    except OSError as exc:
        Logger(f"[WARN] could not write run config: {exc}")


def _worker_init_fn(epoch):
    """Give DataLoader workers a seed derived from the same base seed."""

    def init_worker(worker_id):
        setup_seed(args.seed + epoch * 1000 + worker_id)
    return init_worker


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind GRPO (Group Relative Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='grpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_seq_len', default=768, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1024, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif.jsonl", help="RLAIF数据路径")
    parser.add_argument("--num_generations", type=int, default=6, help="每个prompt生成的样本数")
    parser.add_argument("--beta", type=float, default=0.1, help="KL惩罚系数")
    parser.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo"], help="loss类型")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GRPO的PPO clip epsilon")
    parser.add_argument("--epsilon_high", type=float, default=5.0, help="epsilon上界")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-GRPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--debug_mode", action="store_true", help="是否打印训练调试采样")
    parser.add_argument("--debug_interval", type=int, default=20, help="debug模式下每隔多少step打印一次采样")
    parser.add_argument("--thinking_ratio", type=float, default=1.0,
                        help="按概率开启thinking（0.0~1.0）。主线固定1.0，避免prompt模板在训练中随机变化；"
                             "official 回归护栏需要显式传 0.9 才能对齐改造前的默认值")
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"], help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_grpo", help="SGLang共享存储路径")
    # ===== 项目参数：奖励来源、严格可复现、运行归档 =====
    parser.add_argument("--reward_mode", type=str, default="rule", choices=["rule", "rm", "mix", "official"],
                        help="rule=GT verifier（不加载RM）, rm=官方启发式+RM 对照, mix=rule与纯RM标准化加权, "
                             "official=旧行为回归护栏（配 rlaif.jsonl + RLAIFDataset）")
    parser.add_argument("--reward_marker", type=str, default="答案是", help="答案标记，必须与数据清洗和评估一致")
    parser.add_argument("--verdict_tier", type=str, default="binary", choices=["binary", "tiered"], help="verifier奖励分档")
    parser.add_argument("--dynamic_sampling", type=int, default=0, choices=[0, 1], help="过滤退化组（全对/全错）的policy-gradient项")
    parser.add_argument("--len_penalty_coef", type=float, default=0.0, help="超过length_cap的长度惩罚系数")
    parser.add_argument("--length_cap", type=int, default=300, help="长度惩罚的预注册上限（token）")
    parser.add_argument("--mix_alpha", type=float, default=0.5, help="mix模式下rule分量的权重（标准化后）")
    parser.add_argument("--mix_beta", type=float, default=0.5, help="mix模式下RM分量的权重（标准化后）")
    parser.add_argument("--run_tag", type=str, default="unnamed", help="本次run的唯一标识，用于日志和权重命名")
    parser.add_argument("--log_dir", type=str, default="../logs", help="metrics/samples/run config的输出目录")
    parser.add_argument("--rlvr_data_path", type=str, default="../dataset/math_rlvr_train.jsonl", help="数学RLVR训练数据")
    parser.add_argument("--seed", type=int, default=42, help="全局随机种子（初始化/数据顺序/shuffle）")
    parser.add_argument("--temperature", type=float, default=0.8, help="rollout采样温度")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    rank = dist.get_rank() if dist.is_initialized() else 0
    setup_seed(args.seed + rank)
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               max_seq_len=args.max_seq_len + args.max_gen_len, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-GRPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 初始化模型和数据 ==========
    base_weight = args.from_weight
    # Policy模型
    model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    # Reference模型
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    # Reward模型：rule模式不加载，避免无谓的显存和串行前向开销
    reward_model = None
    if args.reward_mode in ("rm", "mix", "official"):
        rm_path = os.path.abspath(args.reward_model_path)
        if not os.path.isdir(rm_path):
            raise FileNotFoundError(f"reward model not found: {rm_path} (required by --reward_mode {args.reward_mode})")
        reward_model = LMForRewardModel(rm_path, device=args.device, dtype=torch.float16)
    else:
        assert reward_model is None, "rule mode must not load a reward model"
        if is_main_process(): Logger("reward_mode=rule: skipped reward model loading")
    # Rollout引擎（可插拔替换，只负责 policy 推理）
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
    # 数据和优化器
    if args.reward_mode == "official":
        # 改造前回归护栏：官方 rlaif 数据 + 原始启发式/RM 奖励，用于和旧数字逐位对比。
        train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len,
                                thinking_ratio=args.thinking_ratio)
        if is_main_process(): Logger(f"official baseline data: {os.path.abspath(args.data_path)}")
    else:
        data_path = os.path.abspath(args.rlvr_data_path)
        train_ds = MathRLDataset(data_path, tokenizer,
                                 thinking_ratio=args.thinking_ratio,
                                 max_length=args.max_seq_len,
                                 marker=args.reward_marker)
        assert len(train_ds) > 0
        assert all("gt" in row for row in train_ds.samples)
        if is_main_process():
            audit = train_ds.audit_lengths()
            Logger(f"RLVR data: {data_path} rows={len(train_ds)} difficulty={train_ds.difficulty_counts()}")
            Logger(f"Prompt length audit (budget={args.max_seq_len}): "
                   f"thinking_on p50={audit['thinking_on']['p50_tokens']} max={audit['thinking_on']['max_tokens']} "
                   f"overlong={audit['thinking_on']['overlong']} | "
                   f"thinking_off p50={audit['thinking_off']['p50_tokens']} max={audit['thinking_off']['max_tokens']} "
                   f"overlong={audit['thinking_off']['overlong']}")
            _dump_run_config(args, audit)
    train_sampler = DistributedSampler(train_ds, seed=args.seed) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_optimizer_steps = math.ceil(iters / args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
        rollout_engine.update_policy(model)
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    rollout_engine.update_policy(model)
    
    # ========== 8. 开始训练 ==========
    run_logger = RunLogger(args.log_dir, args.run_tag, enabled=is_main_process())
    if is_main_process():
        Logger(f"Metrics: {os.path.abspath(args.log_dir)}/{args.run_tag}_metrics.csv, "
               f"samples: {os.path.abspath(args.log_dir)}/{args.run_tag}_samples.jsonl")
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None: train_sampler.set_epoch(epoch)
        setup_seed(args.seed + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True,
                            worker_init_fn=_worker_init_fn(epoch))
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            grpo_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, reward_model, start_step, wandb, run_logger, use_sglang = (args.rollout_engine == "sglang"))
        else:
            grpo_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, reward_model, 0, wandb, run_logger, use_sglang = (args.rollout_engine == "sglang"))
    run_logger.close()
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()