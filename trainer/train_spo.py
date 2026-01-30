import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import re
import gc
import warnings
import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import SPODataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model

warnings.filterwarnings('ignore')


# --- 1. 自定义优先采样器 (带分布式同步) ---
class WeightedDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, epsilon=1e-5, device='cpu'):
        super().__init__(dataset, num_replicas, rank, shuffle, seed)
        # [修改点]：允许指定 weights 存放的设备，默认 CPU
        self.weights = torch.ones(len(dataset), dtype=torch.float32).to(device) 
        self.epsilon = epsilon
        self.device = device

    def update_weights(self, indices, new_v_estimates):
        """更新权重"""
        # 如果 self.weights 在 GPU，这里就不再产生同步阻塞
        self.weights[indices] = new_v_estimates.to(self.weights.device)

    def sync_weights(self):
        """核心：在 Epoch 结束时同步所有卡的权重，防止采样漂移"""
        if dist.is_initialized():
            dist.all_reduce(self.weights, op=dist.ReduceOp.SUM)
            self.weights /= dist.get_world_size()

    def __iter__(self):
        # 优先级公式：sqrt(v * (1-v)) + epsilon
        priority = torch.sqrt(self.weights * (1 - self.weights) + 1e-8) + self.epsilon
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        # 全局加权采样
        indices = torch.multinomial(priority, self.total_size, replacement=True, generator=g).tolist()
        # 分配到当前进程 (Rank)
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)


# --- 2. SPO 自适应价值追踪器 ---
class AutoAdaptiveValueTracker:
    def __init__(self, rho_mode='kl', rho_const=0.9, D_half=0.06, clip_lower=0.5, clip_upper=0.96):
        self.rho_mode = rho_mode
        self.rho_const = rho_const
        self.D_half = D_half
        self.clip_lower = clip_lower
        self.clip_upper = clip_upper
        N_init = 1.0 / (1.0 - self.clip_lower)
        self.alpha = 0.5 * N_init
        self.beta = 0.5 * N_init
        self.old_mean_logprob = None

    def get_baselines(self, batch_size):
        baseline = self.alpha / (self.alpha + self.beta)
        return torch.full((batch_size,), baseline, dtype=torch.float32)

    def compute_rho(self, cur_mean_logprob):
        if self.rho_mode == 'constant' or self.old_mean_logprob is None:
            return self.rho_const
        kl = abs(self.old_mean_logprob - cur_mean_logprob)
        rho = 2 ** (-kl / self.D_half)
        return max(min(rho, self.clip_upper), self.clip_lower)

    def update(self, rewards, cur_logprobs=None, response_masks=None):
        if cur_logprobs is not None and response_masks is not None:
            mean_logprob = ((cur_logprobs * response_masks).sum() / response_masks.sum()).item()
            rho = self.compute_rho(mean_logprob)
            self.old_mean_logprob = mean_logprob
        else:
            rho = self.rho_const

        scale = 3.0
        normalized_rewards = (rewards + scale) / (2 * scale)
        avg_normalized_reward = normalized_rewards.mean().item()
        self.alpha = rho * self.alpha + avg_normalized_reward
        self.beta = rho * self.beta + (1 - avg_normalized_reward)
        return rho
        

def calculate_rewards(prompts, responses, reward_model, reward_tokenizer):
    """整合所有奖励函数计算总奖励"""
    def reasoning_model_reward(rewards):
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"
        matches_pattern = [re.match(pattern, response, re.S) for response in responses]
        matches_pattern2 = [re.match(pattern2, response, re.S) for response in responses]

        format_rewards = []
        for match_pattern, match_pattern2 in zip(matches_pattern, matches_pattern2):
            if match_pattern or match_pattern2:
                format_rewards.append(0.5)
            else:
                format_rewards.append(0.0)
        rewards += torch.tensor(format_rewards, device=args.device)

        def mark_num(text):
            reward = 0
            if text.count("<think>") == 1: reward += 0.25
            if text.count("</think>") == 1: reward += 0.25
            if text.count("<answer>") == 1: reward += 0.25
            if text.count("</answer>") == 1: reward += 0.25
            return reward

        mark_rewards = [mark_num(response) for response in responses]
        rewards += torch.tensor(mark_rewards, device=args.device)
        return rewards

    rewards = torch.zeros(len(responses), device=args.device)
    if args.reasoning == 1:
        rewards = reasoning_model_reward(rewards)

    with torch.no_grad():
        reward_model_scores = []
        scale = 3.0

        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": content.strip()} for role, content in matches]

            tmp_chat = messages + [{"role": "assistant", "content": response}]
            score = reward_model.get_score(reward_tokenizer, tmp_chat)
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


# --- 4. 核心训练循环 ---
def spo_train_epoch(epoch, loader, iters, model, ref_model, reward_model, reward_tokenizer, 
                    value_tracker, sampler, tokenizer, args, autocast_ctx, wandb=None):
    model.train()
    for step, batch in enumerate(loader, start=1):
        prompts = batch['prompt']
        indices = batch['index']
        
        # 数据预处理
        prompt_inputs = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            padding_side="left", 
            add_special_tokens=False,
            return_token_type_ids=False 
        ).to(args.device)

        if args.max_seq_len:
            prompt_inputs = {k: v[:, -args.max_seq_len:] for k, v in prompt_inputs.items()}

        # 1. 采样生成
        with torch.no_grad(), autocast_ctx:
            model_for_gen = model.module if isinstance(model, DistributedDataParallel) else model
            outputs = model_for_gen.generate(
                **prompt_inputs, max_new_tokens=args.max_gen_len, do_sample=True, temperature=0.8,
                pad_token_id=tokenizer.pad_token_id)    # use_cache = False
            completion_ids = outputs[:, prompt_inputs["input_ids"].size(1):]
            completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
            
        # 2. 计算 Logprobs
        def get_per_token_logps(mdl, input_ids, n_keep):
            logits = mdl(input_ids, logits_to_keep=n_keep + 1).logits[:, :-1, :]
            target_ids = input_ids[:, -n_keep:]
            log_probs = logits.log_softmax(dim=-1)
            return torch.gather(log_probs, 2, target_ids.unsqueeze(2)).squeeze(2)    
        
        with autocast_ctx:
            per_token_logps = get_per_token_logps(model, outputs, completion_ids.size(1))
            with torch.no_grad():
                ref_per_token_logps = get_per_token_logps(ref_model, outputs, completion_ids.size(1))

            # 3. 奖励与优势计算
            rewards = calculate_rewards(prompts, completions, reward_model, reward_tokenizer)
            baselines = value_tracker.get_baselines(len(prompts)).to(args.device)

            advantages = (rewards - baselines).clamp(-5.0, 5.0)

            # 4. 更新采样权重 (EMA)
            norm_rewards = ((rewards.detach() + 3.0) / 6.0).clamp(0, 1)
            current_v_gpu = sampler.weights[indices].to(args.device) # 把 CPU 上的旧权重拉到 GPU
            updated_v_gpu = 0.7 * current_v_gpu + 0.3 * norm_rewards # 全程 GPU 计算
            sampler.update_weights(indices, updated_v_gpu.cpu())    # 计算完结果传回 CPU 存储

            # 5. 计算损失
            is_eos = completion_ids == tokenizer.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=args.device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            completion_mask = (torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)).int()

            kl_div = ref_per_token_logps - per_token_logps
            per_token_kl = torch.exp(kl_div) - kl_div - 1
            
            per_token_loss = -per_token_logps * advantages.unsqueeze(1) + args.beta * per_token_kl
            loss = ((per_token_loss * completion_mask).sum(dim=1) / (completion_mask.sum(dim=1) + 1e-8)).mean() / args.accumulation_steps

        # 6. 反向传播
        loss.backward()
        
        rho = value_tracker.update(rewards, per_token_logps.detach(), completion_mask.float())

        if step % args.accumulation_steps == 0:
            if args.grad_clip > 0: 
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

        # 7. 日志打印
        if step % args.log_interval == 0 or step == iters:
            policy_loss_val = loss.item() * args.accumulation_steps # 恢复显示量级
            avg_reward_val = rewards.mean().item()
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            kl_val = ((per_token_kl * completion_mask).sum() / (completion_mask.sum() + 1e-8)).item()
            avg_baseline_val = baselines.mean().item()
            current_lr = optimizer.param_groups[0]['lr']

            Logger(f'Step: {step}/{iters}, Loss: {policy_loss_val:.4f}, Reward: {avg_reward_val:.4f}, '
                   f'Baseline: {avg_baseline_val:.4f}, KL: {kl_val:.4f}, Rho: {rho:.4f}')

            if wandb and is_main_process():
                wandb.log({
                    "policy_loss": policy_loss_val,
                    "reward": avg_reward_val,
                    "kl": kl_val,
                    "rho": float(rho),
                    "baseline": avg_baseline_val,
                    "advantages_mean": advantages.mean().item(),
                    "learning_rate": current_lr,

                })

        # 8. 模型保存逻辑
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            # ### <--- 修改点 5: 确保 lm_config 在作用域内 (通常从 args 或 model 获取)
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(model.module.config if hasattr(model, 'module') else model.config, 
                          weight=args.save_weight, model=model, optimizer=optimizer, 
                          epoch=epoch, step=step, wandb=wandb, save_dir=args.save_dir, scheduler=scheduler)
            model.train()
            del state_dict

        # 清理内存
        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, advantages, completion_mask, baselines
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind SPO (Self-Play Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='spo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-7, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_seq_len', default=66, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1536, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif-mini.jsonl", help="RLAIF数据路径")
    parser.add_argument("--beta", type=float, default=0.02, help="KL惩罚系数")
    parser.add_argument("--reasoning", type=int, default=1, choices=[0, 1], help='推理模型类型（0=普通模型，1=推理模型）')
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-SPO", help="wandb项目名")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
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
        wandb_run_name = f"MiniMind-SPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume, mode="local")
    
# ========== 5. 初始化模型（Policy, Ref, Reward）和Value Tracker、数据 ==========
    base_weight = "reason" if args.reasoning == 1 else "full_sft"
    # Policy模型
    model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    # Reference模型
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    
    # Reward模型
    reward_model = AutoModel.from_pretrained(
        args.reward_model_path, torch_dtype=torch.float16, trust_remote_code=True
    )
    reward_model = reward_model.to(args.device).eval().requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)
    
    # Value Tracker
    value_tracker = AutoAdaptiveValueTracker(rho_mode='kl', rho_const=0.9, D_half=0.06, clip_lower=0.5, clip_upper=0.96)
    
    # --- 关键改动标注 ---
    train_ds = SPODataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    
    # [标注 1]: 必须显式传入 num_replicas 和 rank，确保分布式采样逻辑正确
    train_sampler = WeightedDistributedSampler(
        train_ds, 
        num_replicas=dist.get_world_size() if dist.is_initialized() else 1, 
        rank=local_rank
    )
    
    # [标注 2]: DataLoader 必须绑定这个 train_sampler
    loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        sampler=train_sampler, 
        num_workers=args.num_workers, 
        pin_memory=True
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # 直接使用 loader 的长度即可，无需额外创建一个 loader_for_count，节省开销
    iters = len(loader)
    
    # 确保 total_optimizer_steps 考虑了梯度累积
    total_optimizer_steps = (iters // args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)    
 
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. DDP包模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
# ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # 必须调用 set_epoch 使得 priority 随 epoch 重新洗牌
        train_sampler.set_epoch(epoch)
        
        if epoch == start_epoch and start_step > 0:
            # 续训逻辑
            batch_sampler = SkipBatchSampler(train_sampler, args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
            
            # [修改点 1]: 补全了大量缺失的参数。原代码漏掉了 model, tokenizer, args, autocast_ctx 等。
            # 注意：这里的 iters 应该是全局步数，修正为 len(loader) + start_step
            spo_train_epoch(
                epoch, loader, len(loader) + start_step + 1, 
                model, ref_model, reward_model, reward_tokenizer, 
                value_tracker, train_sampler, tokenizer,          
                args, autocast_ctx, wandb                         
            )
            train_sampler.sync_weights()
        
        else:
            # 常规训练
            loader = DataLoader(train_ds, batch_size=args.batch_size, pin_memory=True,
                               drop_last=False, num_workers=args.num_workers, sampler=train_sampler)
            
            spo_train_epoch(
                epoch, loader, len(loader), 
                model, ref_model, reward_model, reward_tokenizer, 
                value_tracker, train_sampler, tokenizer,          
                args, autocast_ctx, wandb                         
            )
            # sync_weights 必须在每个 epoch 结束时调用，以同步多卡的采样权重
            train_sampler.sync_weights()