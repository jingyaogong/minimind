import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import re
import warnings
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model

warnings.filterwarnings('ignore')


def calculate_rewards(prompts, responses, reward_model, reward_tokenizer, args):
    """
    计算复杂奖励，包含格式奖励、模型打分以及 DAPO 特有的长度惩罚。
    输入的 prompts/responses 已经是展开后的（batch_size * group_size）条目。
    """
    def reasoning_model_reward(rewards):
        # 1. 结构化格式奖励
        pattern  = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"
        matches_pattern  = [re.match(pattern,  r, re.S) for r in responses]
        matches_pattern2 = [re.match(pattern2, r, re.S) for r in responses]
        format_rewards = [0.5 if (m1 or m2) else 0.0
                         for m1, m2 in zip(matches_pattern, matches_pattern2)]
        rewards += torch.tensor(format_rewards, device=args.device)

        # 2. 标签存在奖励
        def mark_num(text):
            reward = 0
            for tag in ["<think>", "</think>", "<answer>", "</answer>"]:
                if text.count(tag) == 1:
                    reward += 0.25
            return reward

        mark_rewards = [mark_num(r) for r in responses]
        rewards += torch.tensor(mark_rewards, device=args.device)
        return rewards

    rewards = torch.zeros(len(responses), device=args.device)

    if args.reasoning == 1:
        rewards = reasoning_model_reward(rewards)

    # 3. 超长惩罚 (Overlong Punishment - DAPO 特性)
    len_penalty = []
    for resp in responses:
        curr_len = len(resp)
        if curr_len > args.length_threshold:
            penalty = -args.length_penalty * ((curr_len - args.length_threshold) / 100)
            len_penalty.append(max(penalty, -1.0))
        else:
            len_penalty.append(0.0)
    rewards += torch.tensor(len_penalty, device=args.device)

    # 4. 奖励模型评分
    with torch.no_grad():
        reward_model_scores = []
        for prompt, response in zip(prompts, responses):
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": content.strip()} for role, content in matches]

            tmp_chat = messages + [{"role": "assistant", "content": response}]
            score = reward_model.get_score(reward_tokenizer, tmp_chat)
            scale = 3.0
            score = max(min(score, scale), -scale)

            if args.reasoning == 1:
                answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                if answer_match:
                    answer_content = answer_match.group(1).strip()
                    tmp_chat = messages + [{"role": "assistant", "content": answer_content}]
                    answer_score = reward_model.get_score(reward_tokenizer, tmp_chat)
                    answer_score = max(min(answer_score, scale), -scale)
                    score = score * 0.4 + answer_score * 0.6
            reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


def dapo_train_epoch(epoch, loader, iters, old_actor_model, ref_model,
                     actor_scheduler, reward_model, reward_tokenizer,
                     start_step=0, wandb=None):
    actor_model.train()

    for step, batch in enumerate(loader, start=start_step + 1):
        prompts    = batch["prompt"]
        batch_size = len(prompts)
        G          = args.group_size

        # 每个 prompt 重复 G 次，展开为 (batch_size * G) 条输入
        prompts_expanded = [p for p in prompts for _ in range(G)]

        enc = tokenizer(
            prompts_expanded,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_seq_len,
            padding_side="left"
        ).to(args.device)
        prompt_length = enc.input_ids.shape[1]

        with torch.no_grad():
            model_for_gen = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            gen_out = model_for_gen.generate(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                max_new_tokens=args.max_gen_len,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # gen_out: (batch_size * G, seq_len)
        responses_text = [
            tokenizer.decode(gen_out[i, prompt_length:], skip_special_tokens=True)
            for i in range(len(prompts_expanded))
        ]
        rewards_flat = calculate_rewards(prompts_expanded, responses_text, reward_model, reward_tokenizer, args)
        # rewards_flat: (batch_size * G,)

        # ------------------------------------------------------------------
        # advantage = (r - mean_group) / std_group
        # ------------------------------------------------------------------
        rewards_grouped = rewards_flat.view(batch_size, G)
        mean_r          = rewards_grouped.mean(dim=1, keepdim=True)
        std_r           = rewards_grouped.std(dim=1, keepdim=True).clamp(min=1e-8)
        advantages      = ((rewards_grouped - mean_r) / std_r).view(-1)  # (batch_size * G,)

        # ------------------------------------------------------------------
        # Dynamic Sampling
        # ------------------------------------------------------------------
        if args.use_dynamic_sampling == 1:
            if (std_r < 1e-6).all().item():
                if is_main_process():
                    Logger(f"Step {step}: All groups have zero reward variance, skipping...")
                continue

        # ------------------------------------------------------------------
        # Actor forward
        # ------------------------------------------------------------------
        full_mask = (gen_out != tokenizer.pad_token_id).long()

        with autocast_ctx:
            res      = actor_model(input_ids=gen_out, attention_mask=full_mask)
            logits   = res.logits
            aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)

        labels      = gen_out[:, 1:].clone()
        logp_tokens = F.log_softmax(logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)

        seq_len    = gen_out.size(1) - 1
        resp_mask  = torch.arange(seq_len, device=gen_out.device).unsqueeze(0) >= prompt_length - 1
        final_mask = resp_mask & (~labels.eq(tokenizer.pad_token_id))

        # Token-level 平均 log-prob（防止长回复主导梯度）
        token_count = final_mask.sum(dim=1).clamp(min=1).float()
        actor_logp  = (logp_tokens * final_mask).sum(dim=1) / token_count

        with torch.no_grad():
            old_logits      = old_actor_model(input_ids=gen_out, attention_mask=full_mask).logits
            old_logp_tokens = F.log_softmax(old_logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)
            old_logp        = (old_logp_tokens * final_mask).sum(dim=1) / token_count

            ref_logits      = ref_model(input_ids=gen_out, attention_mask=full_mask).logits
            ref_logp_tokens = F.log_softmax(ref_logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)
            ref_logp        = (ref_logp_tokens * final_mask).sum(dim=1) / token_count

        kl     = (actor_logp - old_logp).mean()
        kl_ref = (actor_logp - ref_logp).mean()
        ratio  = torch.exp(actor_logp - old_logp)

        # ------------------------------------------------------------------
        # Decoupled Clip
        #   Adv > 0：只限上界，允许正向大步更新prompts_expanded = [p for p in prompts for _ in range(G)]
        #   Adv ≤ 0：只限下界，防止负向更新反弹
        # ------------------------------------------------------------------
        surr1 = ratio * advantages
        clipped_ratio = torch.where(
            advantages > 0,
            torch.clamp(ratio, max=1.0 + args.clip_epsilon_high),
            torch.clamp(ratio, min=1.0 - args.clip_epsilon_low),
        )
        surr2       = clipped_ratio * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        loss = (policy_loss + args.kl_coef * kl_ref + aux_loss) / args.accumulation_steps
        loss.backward()

        if (step + 1) % args.accumulation_steps == 0:
            clip_grad_norm_(actor_model.parameters(), args.grad_clip)
            actor_optimizer.step()
            actor_scheduler.step()
            actor_optimizer.zero_grad()

        if is_main_process():
            response_ids = gen_out[:, enc.input_ids.shape[1]:]
            is_eos       = (response_ids == tokenizer.eos_token_id)
            eos_indices  = torch.argmax(is_eos.int(), dim=1)
            lengths = torch.where(
                is_eos.any(dim=1),
                eos_indices + 1,
                torch.tensor(response_ids.shape[1], device=is_eos.device)
            )
            avg_len    = lengths.float().mean()
            reward_std = rewards_flat.std().item()

            if wandb is not None:
                wandb.log({
                    "actor_loss":       policy_loss.item(),
                    "reward":           rewards_flat.mean().item(),
                    "reward_std":       reward_std,
                    "kl":               kl.item(),
                    "kl_ref":           kl_ref.item(),
                    "avg_response_len": avg_len.item(),
                    "actor_lr":         actor_optimizer.param_groups[0]['lr'],
                })

            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                f"Loss: {policy_loss.item():.4f}, Reward: {rewards_flat.mean().item():.4f}, "
                f"KL: {kl.item():.4f}, Len: {avg_len.item():.2f}"
            )

        if (step + 1) % args.update_old_actor_freq == 0:
            raw_actor = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            raw_actor = getattr(raw_actor, '_orig_mod', raw_actor)
            old_actor_model.load_state_dict({k: v.detach().cpu() for k, v in raw_actor.state_dict().items()})
            old_actor_model.to(args.device)

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            actor_model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_actor = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            raw_actor = getattr(raw_actor, '_orig_mod', raw_actor)
            torch.save({k: v.half().cpu() for k, v in raw_actor.state_dict().items()}, ckp)
            lm_checkpoint(
                lm_config, weight=args.save_weight, model=actor_model,
                optimizer=actor_optimizer, epoch=epoch, step=step,
                wandb=wandb, save_dir='../checkpoints', scheduler=actor_scheduler,
            )
            actor_model.train()

        del enc, gen_out, responses_text, rewards_flat, rewards_grouped
        del advantages, full_mask, logits, labels, logp_tokens, final_mask, token_count
        del actor_logp, old_logits, old_logp, ref_logits, ref_logp
        del kl, kl_ref, ratio, surr1, surr2, clipped_ratio, policy_loss, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MiniMind DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization)"
    )
    parser.add_argument("--save_dir",              type=str,   default="../out",                      help="模型保存目录")
    parser.add_argument('--save_weight',            type=str,   default='dapo',                  help="保存权重的前缀名")
    parser.add_argument("--epochs",                type=int,   default=1,                             help="训练轮数")
    parser.add_argument("--batch_size",            type=int,   default=2,                             help="prompt 级别的 batch size（实际前向 = batch_size * group_size）")
    parser.add_argument("--learning_rate",         type=float, default=8e-8,                          help="Actor 学习率")
    parser.add_argument("--device",                type=str,   default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype",                 type=str,   default="bfloat16",                    help="混合精度类型")
    parser.add_argument("--num_workers",           type=int,   default=8,                             help="数据加载线程数")
    parser.add_argument("--accumulation_steps",    type=int,   default=1,                             help="梯度累积步数")
    parser.add_argument("--grad_clip",             type=float, default=1.0,                           help="梯度裁剪阈值")
    parser.add_argument("--log_interval",          type=int,   default=1,                             help="日志打印间隔")
    parser.add_argument("--save_interval",         type=int,   default=10,                            help="模型保存间隔")
    parser.add_argument('--hidden_size',            type=int,   default=512,                           help="隐藏层维度")
    parser.add_argument('--num_hidden_layers',      type=int,   default=8,                             help="隐藏层数量")
    parser.add_argument('--use_moe',                type=int,   default=0,  choices=[0, 1],            help="是否使用 MoE 架构（0=否，1=是）")
    parser.add_argument('--max_seq_len',            type=int,   default=66,                            help="Prompt 最大长度")
    parser.add_argument("--max_gen_len",            type=int,   default=1024,                          help="生成的最大长度")
    parser.add_argument("--data_path",              type=str,   default="../dataset/rlaif-mini.jsonl", help="RLAIF 数据路径")

    # DAPO 核心参数
    parser.add_argument("--clip_epsilon_low",      type=float, default=0.1,  help="Decoupled Clip 下界")
    parser.add_argument("--clip_epsilon_high",     type=float, default=0.2,  help="Decoupled Clip 上界")
    parser.add_argument("--group_size",            type=int,   default=8,    help="每个 prompt 的采样回复数")
    parser.add_argument("--use_dynamic_sampling",  type=int,   default=1,   choices=[0, 1], help="是否开启动态采样过滤")
    parser.add_argument("--length_penalty",        type=float, default=0.1,  help="长度惩罚系数")
    parser.add_argument("--length_threshold",      type=int,   default=1200, help="触发长度惩罚的字符阈值")
    parser.add_argument("--kl_coef",               type=float, default=0.02, help="KL 散度惩罚系数（相对 ref_model）")
    parser.add_argument("--reasoning",             type=int,   default=1,   choices=[0, 1], help="推理模型类型（0=普通模型，1=推理模型）")
    parser.add_argument("--update_old_actor_freq", type=int,   default=4,   help="更新 old_actor_model 的频率（步数）")
    parser.add_argument("--reward_model_path",     type=str,   default="../../internlm2-1_8b-reward", help="Reward 模型路径")
    parser.add_argument('--from_resume',            type=int,   default=0,   choices=[0, 1], help="是否自动检测并续训（0=否，1=是）")
    parser.add_argument("--use_wandb",             action="store_true",                     help="是否使用 wandb/swanlab")
    parser.add_argument("--wandb_project",         type=str,   default="MiniMind-DAPO",     help="wandb 项目名")
    parser.add_argument("--use_compile",           type=int,   default=0,   choices=[0, 1], help="是否使用 torch.compile 加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查 ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe)
    )
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') \
               if args.from_resume == 1 else None

    # ========== 3. 设置混合精度 ==========
    device_type  = "cuda" if "cuda" in args.device else "cpu"
    dtype        = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 配 wandb (SwanLab) ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume   = 'must' if wandb_id else None
        wandb.init(
            project=args.wandb_project,
            name=f"MiniMind-DAPO-{args.learning_rate}",
            id=wandb_id,
            resume=resume,
        )

    # ========== 5. 初始化模型 ==========
    base_weight = "reason" if args.reasoning == 1 else "full_sft"

    actor_model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    if args.use_compile == 1:
        actor_model = torch.compile(actor_model)

    old_actor_model, _ = init_model(lm_config, base_weight, device=args.device)
    old_actor_model = old_actor_model.eval().requires_grad_(False)

    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)

    reward_model = AutoModel.from_pretrained(
        args.reward_model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    reward_model = reward_model.to(args.device).eval().requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)

    # ========== 6. 数据、优化器、调度器 ==========
    train_ds        = RLAIFDataset(args.data_path, tokenizer, max_length=(args.max_seq_len + args.max_gen_len))
    train_sampler   = DistributedSampler(train_ds) if dist.is_initialized() else None
    actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)

    iters           = len(DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler))
    total_steps     = (iters // args.accumulation_steps) * args.epochs
    actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=total_steps, eta_min=args.learning_rate / 10)

    if ckp_data:
        actor_model.load_state_dict(ckp_data['model'])
        actor_optimizer.load_state_dict(ckp_data['optimizer'])
        actor_scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step  = ckp_data.get('step', 0)
    else:
        start_epoch, start_step = 0, 0

    if dist.is_initialized():
        actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank])
        old_actor_model.to(args.device)

    # ========== 7. 打印配置摘要 ==========
    if is_main_process():
        Logger("=" * 60)
        Logger("DAPO Training Config (Critic-free):")
        Logger(f"  group_size           = {args.group_size}")
        Logger(f"  clip_epsilon_low     = {args.clip_epsilon_low}")
        Logger(f"  clip_epsilon_high    = {args.clip_epsilon_high}")
        Logger(f"  use_dynamic_sampling = {args.use_dynamic_sampling}")
        Logger(f"  kl_coef              = {args.kl_coef}")
        Logger(f"  effective_batch      = {args.batch_size} prompts × {args.group_size} = {args.batch_size * args.group_size}")
        Logger("=" * 60)

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip    = start_step if (epoch == start_epoch and start_step > 0) else 0
        loader  = DataLoader(
            train_ds,
            batch_sampler=SkipBatchSampler(train_sampler or indices, args.batch_size, skip),
            num_workers=args.num_workers,
            pin_memory=True,
        )
        dapo_train_epoch(
            epoch, loader, len(loader),
            old_actor_model, ref_model,
            actor_scheduler, reward_model, reward_tokenizer,
            skip, wandb,
        )

    if dist.is_initialized():
        dist.destroy_process_group()