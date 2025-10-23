import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import re
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModel
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def calculate_rewards(prompts, responses, reward_model, reward_tokenizer):
    """整合所有奖励函数计算总奖励"""

    def reasoning_model_reward(rewards):
        # 1. 格式奖励（仅针对训练推理模型时使用）
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"

        matches_pattern = [re.match(pattern, response, re.S) for response in responses]
        matches_pattern2 = [re.match(pattern2, response, re.S) for response in responses]

        format_rewards = []
        for match_pattern, match_pattern2 in zip(matches_pattern, matches_pattern2):
            if match_pattern:
                format_rewards.append(0.5)
            elif match_pattern2:
                format_rewards.append(0.5)
            else:
                format_rewards.append(0.0)
        rewards += torch.tensor(format_rewards, device=args.device)

        # 2. 标记奖励（防止严格奖励稀疏，仅针对训练推理模型时使用）
        def mark_num(text):
            reward = 0
            if text.count("<think>") == 1:
                reward += 0.25
            if text.count("</think>") == 1:
                reward += 0.25
            if text.count("<answer>") == 1:
                reward += 0.25
            if text.count("</answer>") == 1:
                reward += 0.25
            return reward

        mark_rewards = [mark_num(response) for response in responses]
        rewards += torch.tensor(mark_rewards, device=args.device)
        return rewards

    rewards = torch.zeros(len(responses), device=args.device)

    # 格式奖励
    if args.reasoning == 1:
        rewards = reasoning_model_reward(rewards)  # 训练推理模型时使用

    # 使用reward model计算整个response的奖励
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

            # 当args.reasoning=1时，额外计算<answer>内容的奖励
            if args.reasoning == 1:
                answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                if answer_match:
                    answer_content = answer_match.group(1).strip()
                    # 对answer内容单独计算reward
                    tmp_chat = messages + [{"role": "assistant", "content": answer_content}]
                    answer_score = reward_model.get_score(reward_tokenizer, tmp_chat)
                    answer_score = max(min(answer_score, scale), -scale)

                    score = score * 0.4 + answer_score * 0.6
            reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


def ppo_train_epoch(epoch: int, wandb_run, old_actor_model, ref_model, actor_scheduler, critic_scheduler):
    actor_model.train()
    critic_model.train()
    is_master = (not ddp) or dist.get_rank() == 0

    for step, batch in enumerate(train_loader):
        prompts = batch["prompt"]  # list[str], length B
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, 
                       max_length=args.max_seq_len).to(args.device)  # input_ids: [B, P], attention_mask: [B, P]
        prompt_lengths = enc.attention_mask.sum(dim=1)  # [B]

        with torch.no_grad():
            gen_out = actor_model.generate(
                input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                max_new_tokens=args.max_gen_len, do_sample=True, temperature=0.8,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)  # [B, P+R]

        responses_text = [tokenizer.decode(gen_out[i, prompt_lengths[i]:], skip_special_tokens=True) for i in range(len(prompts))]
        rewards = calculate_rewards(prompts, responses_text, reward_model, reward_tokenizer)  # [B]

        full_mask = (gen_out != tokenizer.pad_token_id).long()  # [B, P+R]
        values_seq = critic_model(input_ids=gen_out, attention_mask=full_mask)  # [B, P+R]
        last_indices = full_mask.sum(dim=1) - 1  # [B]
        values = values_seq[torch.arange(values_seq.size(0), device=values_seq.device), last_indices]  # [B]
        advantages = rewards - values.detach()  # [B]

        logits = actor_model(input_ids=gen_out, attention_mask=full_mask).logits  # [B, P+R, V]
        labels = gen_out[:, 1:].clone()  # [B, P+R-1]
        logp_tokens = F.log_softmax(logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [B, P+R-1]
        seq_len = gen_out.size(1) - 1
        resp_mask = torch.arange(seq_len, device=gen_out.device).unsqueeze(0) >= prompt_lengths.unsqueeze(1)
        final_mask = resp_mask & (~labels.eq(tokenizer.pad_token_id))  # [B, P+R-1]
        actor_logp = (logp_tokens * final_mask).sum(dim=1)  # [B]

        with torch.no_grad():
            old_logits = old_actor_model(input_ids=gen_out, attention_mask=full_mask).logits  # [B, P+R, V]
            old_logp_tokens = F.log_softmax(old_logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [B, P+R-1]
            old_logp = (old_logp_tokens * final_mask).sum(dim=1)  # [B]
            
            ref_logits = ref_model(input_ids=gen_out, attention_mask=full_mask).logits  # [B, P+R, V]
            ref_logp_tokens = F.log_softmax(ref_logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [B, P+R-1]
            ref_logp = (ref_logp_tokens * final_mask).sum(dim=1)  # [B]

        kl = (actor_logp - old_logp).mean()  # scalar
        kl_ref = (actor_logp - ref_logp).mean()  # scalar
        ratio = torch.exp(actor_logp - old_logp)  # [B]
        surr1 = ratio * advantages  # [B]
        surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages  # [B]
        policy_loss = -torch.min(surr1, surr2).mean()  # scalar
        value_loss = F.mse_loss(values, rewards)  # scalar
        loss = policy_loss + args.vf_coef * value_loss + args.kl_coef * kl_ref  # scalar
        loss.backward()

        if (step + 1) % args.accumulation_steps == 0:
            clip_grad_norm_(actor_model.parameters(), args.grad_clip)
            clip_grad_norm_(critic_model.parameters(), args.grad_clip)
            actor_optimizer.step()
            critic_optimizer.step()
            actor_scheduler.step()
            critic_scheduler.step()
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()

        if is_master:
            response_ids = gen_out[:, enc.input_ids.shape[1]:]
            is_eos = (response_ids == tokenizer.eos_token_id)
            eos_indices = torch.argmax(is_eos.int(), dim=1)
            has_eos = is_eos.any(dim=1)
            lengths = torch.where(has_eos, eos_indices + 1, torch.tensor(response_ids.shape[1], device=is_eos.device))
            avg_len = lengths.float().mean()

            actor_loss_val = policy_loss.item()
            critic_loss_val = value_loss.item()
            reward_val = rewards.mean().item()
            kl_val = kl.item()
            kl_ref_val = kl_ref.item()
            avg_len_val = avg_len.item()
            actor_lr = actor_optimizer.param_groups[0]['lr']
            critic_lr = critic_optimizer.param_groups[0]['lr']

            if wandb_run is not None:
                wandb_run.log({
                    "actor_loss": actor_loss_val,
                    "critic_loss": critic_loss_val,
                    "reward": reward_val,
                    "kl": kl_val,
                    "kl_ref": kl_ref_val,
                    "avg_response_len": avg_len_val,
                    "actor_lr": actor_lr,
                })

            Logger(f"Epoch: {epoch}, Step: {step + 1}/{len(train_loader)}, "
                   f"Actor Loss: {actor_loss_val:.6f}, Critic Loss: {critic_loss_val:.6f}, "
                   f"Reward: {reward_val:.6f}, KL: {kl_val:.6f}, KL_ref: {kl_ref_val:.6f}, "
                   f"Avg Response Len: {avg_len_val:.2f}, Actor LR: {actor_lr:.2e}, Critic LR: {critic_lr:.2e}")

        if (step + 1) % args.update_old_actor_freq == 0:
            state_dict = actor_model.module.state_dict() if isinstance(actor_model, torch.nn.parallel.DistributedDataParallel) else actor_model.state_dict()
            old_actor_model.load_state_dict({k: v.detach().cpu() for k, v in state_dict.items()})
            old_actor_model.to(args.device)

        if ((step + 1) % args.save_interval == 0 or step == iter_per_epoch - 1) and (not ddp or dist.get_rank() == 0):
            actor_model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/ppo_actor_{lm_config.hidden_size}{moe_path}.pth'

            if isinstance(actor_model, torch.nn.parallel.DistributedDataParallel):
                state_dict = actor_model.module.state_dict()
            else:
                state_dict = actor_model.state_dict()

            state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
            torch.save(state_dict, ckp)
            actor_model.train()


# 自定义的Critic模型，继承自MiniMindLM
class CriticModel(MiniMindForCausalLM):
    def __init__(self, params):
        super().__init__(params)
        # 替换lm_head为输出单一价值的线性层
        self.value_head = nn.Linear(params.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # 使用基础模型获取隐藏状态
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        # self.model 返回的是一个元组，第一个元素是 last_hidden_state
        hidden_states = self.model.norm(outputs[0])
        # 使用value_head获取价值估计
        values = self.value_head(hidden_states).squeeze(-1)
        return values


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('../model/', padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    moe_path = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/{"reason" if args.reasoning == 1 else "full_sft"}_{lm_config.hidden_size}{moe_path}.pth'
    state_dict = torch.load(ckp, map_location=args.device)

    actor_model = MiniMindForCausalLM(lm_config)
    actor_model.load_state_dict(state_dict, strict=False)
    actor_model = actor_model.to(args.device)

    old_actor_model = MiniMindForCausalLM(lm_config)
    old_actor_model.load_state_dict(state_dict, strict=False)
    old_actor_model = old_actor_model.eval().requires_grad_(False).to(args.device)

    ref_model = MiniMindForCausalLM(lm_config)
    ref_model.load_state_dict(state_dict, strict=False)
    ref_model = ref_model.eval().requires_grad_(False).to(args.device)

    critic_model = CriticModel(lm_config)
    critic_model.load_state_dict(state_dict, strict=False)
    critic_model = critic_model.to(args.device)

    reward_name = "../../internlm2-1_8b-reward"
    reward_model = AutoModel.from_pretrained(
        reward_name, device_map="cuda", torch_dtype=torch.float32, trust_remote_code=True
    ).to(args.device).eval().requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_name, trust_remote_code=True)

    Logger(f'Actor模型总参数量：{sum(p.numel() for p in actor_model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    Logger(f'Critic模型总参数量：{sum(p.numel() for p in critic_model.parameters() if p.requires_grad) / 1e6:.3f} 百万')

    return actor_model, old_actor_model, ref_model, critic_model, reward_model, tokenizer, reward_tokenizer


def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE
    dist.init_process_group(backend="nccl")
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=8e-8)
    parser.add_argument("--critic_learning_rate", type=float, default=8e-8)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-PPO")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument('--max_seq_len', default=66, type=int)
    parser.add_argument("--max_gen_len", type=int, default=1536)
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif-mini.jsonl")
    parser.add_argument("--clip_epsilon", type=float, default=0.1)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--kl_coef", type=float, default=0.02, help="KL散度惩罚系数")
    parser.add_argument("--reasoning", type=int, default=1, help='0:普通模型，1:推理模型')
    parser.add_argument("--update_old_actor_freq", type=int, default=4, help="频率：每处理n个batch后更新old_actor_model")
    args = parser.parse_args()

    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               use_moe=args.use_moe)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    ddp = int(os.environ.get("RANK", -1)) != -1
    ddp_local_rank, DEVICE = 0, "cuda:0"
    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        # 同时设置 CUDA 的随机种子
        torch.cuda.manual_seed(base_seed + rank)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import swanlab as wandb

        wandb.init(project=args.wandb_project)
    else:
        wandb = None

    # 初始化所有模型
    actor_model, old_actor_model, ref_model, critic_model, reward_model, tokenizer, reward_tokenizer = init_model(lm_config=lm_config)

    # 准备数据集和数据加载器
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=(args.max_seq_len + args.max_gen_len))
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, pin_memory=True,
                              drop_last=False, shuffle=(train_sampler is None),
                              num_workers=args.num_workers, sampler=train_sampler)

    # 初始化优化器
    actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.AdamW(critic_model.parameters(), lr=args.critic_learning_rate)

    iter_per_epoch = len(train_loader)
    total_optimizer_steps = (iter_per_epoch // args.accumulation_steps) * args.epochs
    actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    critic_scheduler = CosineAnnealingLR(critic_optimizer, T_max=total_optimizer_steps,
                                         eta_min=args.critic_learning_rate / 10)

    # 如果使用分布式训练，包装模型
    if ddp:
        actor_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        critic_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        actor_model = DistributedDataParallel(actor_model, device_ids=[ddp_local_rank])
        critic_model = DistributedDataParallel(critic_model, device_ids=[ddp_local_rank])
        # old_actor_model 不需要DDP包装，因为它只在主进程上用于计算，并且不进行梯度更新
        old_actor_model.to(args.device)

    for epoch in range(args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        ppo_train_epoch(epoch, wandb, old_actor_model, ref_model, actor_scheduler, critic_scheduler)

    if ddp:
        dist.destroy_process_group()
