import os
import sys

# 设置包名
__package__ = "trainer"
# 将项目根目录添加到系统路径, 确保能够导入同级目录的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import re
import gc
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
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model

# 忽略警告信息
import warnings
warnings.filterwarnings('ignore')


def calculate_rewards(prompts, responses, reward_model, reward_tokenizer):
    """
    整合所有奖励函数计算总奖励

    根据配置使用 reasoning 模型奖励函数和/或 reward model 评分来计算总奖励
    支持推理模型的格式检查和普通模型的 reward model 评分

    Args:
    - prompts:          输入 prompt 列表, list[str], 长度为 batch_size
    - responses:        模型生成的回复列表, list[str], 长度为 batch_size * num_generations
    - reward_model:     reward model 实例, 用于计算语义相似度分数
    - reward_tokenizer: reward model 的分词器

    Returns:
    - torch.Tensor:     每个生成样本的奖励分数, shape [B*num_gen]
    """
    def reasoning_model_reward(rewards):
        """
        如果使用 reasoning 模型, 应用 reasoning 奖励函数
        
        if args.reasoning == 1:
            rewards = reasoning_model_reward(rewards)
        """
        # 定义推理模型的格式匹配模式: <think> 块后跟 <answer> 块
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        # 备选模式: 允许 <think> 和 <answer> 之间有一个空行
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"
        # 检查每个回复是否匹配格式
        matches_pattern = [re.match(pattern, response, re.S) for response in responses]
        matches_pattern2 = [re.match(pattern2, response, re.S) for response in responses]

        # 计算格式奖励: 格式正确得 0.5 分, 否则 0 分
        format_rewards = []
        for match_pattern, match_pattern2 in zip(matches_pattern, matches_pattern2):
            if match_pattern or match_pattern2:
                format_rewards.append(0.5)
            else:
                format_rewards.append(0.0)
        rewards += torch.tensor(format_rewards, device=args.device)

        def mark_num(text):
            # 统计标签出现次数, 每个正确出现的标签奖励 0.25 分
            reward = 0
            if text.count("<think>") == 1: reward += 0.25
            if text.count("</think>") == 1: reward += 0.25
            if text.count("<answer>") == 1: reward += 0.25
            if text.count("</answer>") == 1: reward += 0.25
            return reward

        # 计算标签数量奖励
        mark_rewards = [mark_num(response) for response in responses]
        rewards += torch.tensor(mark_rewards, device=args.device)
        return rewards

    # 初始化奖励张量为 0
    rewards = torch.zeros(len(responses), device=args.device)
    # 如果使用 reasoning 模型, 应用 reasoning 奖励函数
    if args.reasoning == 1:
        rewards = reasoning_model_reward(rewards)

    # 使用 reward model 计算语义奖励
    with torch.no_grad():
        reward_model_scores = []
        batch_size = len(prompts)
        scale = 3.0

        # 遍历每个 prompt 和每个生成的回复
        for i in range(batch_size):
            for j in range(args.num_generations):
                response_idx = i * args.num_generations + j
                response = responses[response_idx]
                prompt = prompts[i]

                # 解析 prompt 中的对话历史
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]

                # 构建完整的对话上下文并计算 reward model 分数
                tmp_chat = messages + [{"role": "assistant", "content": response}]
                score = reward_model.get_score(reward_tokenizer, tmp_chat)
                # 将分数裁剪到 [-scale, scale] 范围
                score = max(min(score, scale), -scale)

                # 对于 reasoning 模型, 额外计算 answer 部分的分数
                if args.reasoning == 1:
                    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                    if answer_match:
                        answer_content = answer_match.group(1).strip()
                        tmp_chat = messages + [{"role": "assistant", "content": answer_content}]
                        answer_score = reward_model.get_score(reward_tokenizer, tmp_chat)
                        answer_score = max(min(answer_score, scale), -scale)
                        # 综合分数: 40% 完整回复 + 60% answer 部分
                        score = score * 0.4 + answer_score * 0.6

                reward_model_scores.append(score)

        # 将 reward model 分数添加到总奖励
        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


def grpo_train_epoch(epoch, loader, iters, ref_model, reward_model, reward_tokenizer, start_step=0, wandb=None):
    """
    GRPO (Group Relative Policy Optimization) 单 epoch 训练函数

    实现 GRPO 算法的核心训练循环, 包括:
    - 生成多组回复样本
    - 计算 reward 和 advantage
    - 计算 policy loss 和 KL divergence
    - 更新模型参数

    Args:
    - epoch:            当前训练轮次
    - loader:           数据加载器
    - iters:            总迭代次数
    - ref_model:        reference model (冻结参数), 用于计算 KL divergence
    - reward_model:     reward model, 用于计算奖励分数
    - reward_tokenizer: reward model 的分词器
    - start_step:       从第几步开始 (用于断点续训)
    - wandb:            日志记录对象
    """
    for step, batch in enumerate(loader, start=start_step + 1):
        # 获取 prompt 列表并编码
        prompts = batch['prompt']  # list[str], length B
        prompt_inputs = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            return_token_type_ids=False,
            padding_side="left", 
            add_special_tokens=False).to(args.device)  # input_ids: [B, P], attention_mask: [B, P]
        # 截断过长的 prompt
        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

        # 使用当前 model 生成多个回复
        with torch.no_grad():
            # DDP 模型需要使用 .module 访问 generate 方法
            model_for_gen = model.module if isinstance(model, DistributedDataParallel) else model
            outputs = model_for_gen.generate(
                **prompt_inputs, 
                max_new_tokens=args.max_gen_len, 
                do_sample=True, temperature=0.8,
                num_return_sequences=args.num_generations, 
                pad_token_id=tokenizer.pad_token_id)  # [B*num_gen, P+R]

        # 提取生成的回复部分 (去掉 prompt)
        completion_ids = outputs[:, prompt_inputs["input_ids"].size(1):]  # [B*num_gen, R]
        
        def get_per_token_logps(mdl, input_ids, n_keep):
            """
            获取每个 token 的 log probability
            """
            input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
            logits = mdl(input_ids, logits_to_keep=n_keep + 1).logits[:, :-1, :]
            per_token_logps = []
            for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
                ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
                per_token_logps.append(torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1))
            return torch.stack(per_token_logps)

        # 计算 policy model 的 per-token log probabilities
        with autocast_ctx:
            per_token_logps = get_per_token_logps(model, outputs, completion_ids.size(1))  # [B*num_gen, R]
            # 如果使用 MoE, 计算辅助损失
            res = model(outputs) if lm_config.use_moe else None
            aux_loss = res.aux_loss if res is not None else torch.tensor(0.0, device=args.device)
        
        # 计算 reference model 的 per-token log probabilities (无梯度)
        with torch.no_grad():
            ref_per_token_logps = get_per_token_logps(ref_model, outputs, completion_ids.size(1))  # [B*num_gen, R]

        # 解码生成的回复并计算奖励
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        rewards = calculate_rewards(prompts, completions, reward_model, reward_tokenizer).to(args.device)  # [B*num_gen]

        # 按 group 分组计算 advantage (GRPO 核心)
        grouped_rewards = rewards.view(-1, args.num_generations)                        # [B, num_gen]
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)    # [B*num_gen]
        std_r = grouped_rewards.std(dim=1).repeat_interleave(args.num_generations)      # [B*num_gen]
        advantages = torch.clamp((rewards - mean_r) / (std_r + 1e-4), -10, 10)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)       # [B*num_gen]

        # 构建 completion mask (处理 EOS 提前结束的情况)
        is_eos = completion_ids == tokenizer.eos_token_id                               # [B*num_gen, R]
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=args.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        completion_mask = (torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)).int()  # [B*num_gen, R]

        # 计算 KL divergence
        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1                                   # [B*num_gen, R]
        # 计算 GRPO loss: advantage-weighted loss - beta * KL penalty
        per_token_loss = -(torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1) - args.beta * per_token_kl)  # [B*num_gen, R]
        policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        loss = (policy_loss + aux_loss) / args.accumulation_steps                       # scalar
        loss.backward()

        # 梯度累积更新
        if (step + 1) % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # 日志记录
        if step % args.log_interval == 0 or step == iters:
            policy_loss_val = loss.item() * args.accumulation_steps
            current_aux_loss = aux_loss.item()
            avg_reward_val = rewards.mean().item()
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            current_lr = optimizer.param_groups[0]['lr']

            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'Actor Loss: {policy_loss_val:.4f}, Aux Loss: {current_aux_loss:.4f}, Reward: {avg_reward_val:.4f}, '
                   f'Avg Response Len: {avg_len_val:.2f}, Learning Rate: {current_lr:.8f}')

            if wandb and is_main_process():
                wandb.log({
                    "policy_loss": policy_loss_val,
                    "aux_loss": current_aux_loss,
                    "reward": avg_reward_val,
                    "avg_response_len": avg_len_val,
                    "advantages_mean": advantages.mean().item(),
                    "learning_rate": current_lr
                })

        # 模型保存检查点
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # 解包 DDP 或 torch.compile 包装的模型
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            # 保存半精度权重到 CPU
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # 保存完整的检查点 (用于断点续训)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scheduler=scheduler)
            model.train()
            del state_dict

        # 清理中间变量释放内存
        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind GRPO (Group Relative Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='grpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=8e-8, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构 (0=否, 1=是)")
    parser.add_argument('--max_seq_len', default=66, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1536, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif-mini.jsonl", help="RLAIF数据路径")
    parser.add_argument("--num_generations", type=int, default=8, help="每个prompt生成的样本数")
    parser.add_argument("--beta", type=float, default=0.02, help="KL惩罚系数")
    parser.add_argument("--reasoning", type=int, default=1, choices=[0, 1], help='推理模型类型 (0=普通模型, 1=推理模型)')
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训 (0=否, 1=是)")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-GRPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速 (0=否, 1=是)")
    args = parser.parse_args()

    # ========== 1.初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    # 如果启用了分布式训练, 根据 local_rank 设置设备
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 设置随机种子, 分布式训练时根据 rank 偏移确保不同进程使用不同种子
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2.配置目录、模型参数、检查 ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    # 创建模型配置, max_seq_len 包含 prompt 和生成的最大长度
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers,
        max_seq_len=args.max_seq_len + args.max_gen_len, 
        use_moe=bool(args.use_moe)
    )
    # 如果启用断点续训, 尝试加载检查点
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3.设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # https://docs.pytorch.ac.cn/docs/stable/amp
    # 新版废弃: autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=dtype)

    # ========== 4.配置 wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-GRPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5.初始化模型和数据 ==========
    # 根据是否为 reasoning 模型选择基础权重
    base_weight = "reason" if args.reasoning == 1 else "full_sft"
    # Policy 模型 (将被训练的模型)
    model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    # 如果启用 torch.compile 加速
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    # Reference 模型 (冻结参数, 用于计算 KL divergence)
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    # Reward 模型 (用于计算奖励分数, 冻结参数)
    reward_model = AutoModel.from_pretrained(args.reward_model_path, torch_dtype=torch.float16, trust_remote_code=True)
    reward_model = reward_model.to(args.device).eval().requires_grad_(False)
    # Reward 模型分词器
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)
    # 加载数据集和创建优化器
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    # 计算训练迭代次数和学习率调度器
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_optimizer_steps = (iters // args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    
    # ========== 6.从 ckp 恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        # 加载模型权重、优化器状态和学习率调度器状态
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7.DDP 包装模型 ==========
    if dist.is_initialized():
        # 忽略位置编码参数不参与 DDP 同步
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8.开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # 分布式采样器设置 epoch 以确保不同进程数据不同
        train_sampler and train_sampler.set_epoch(epoch)
        # 每轮使用不同的随机种子打乱数据
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # 如果是续训的第一个 epoch, 跳过已训练的 step
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            grpo_train_epoch(epoch, loader, len(loader) + skip, ref_model, reward_model, reward_tokenizer, start_step, wandb)
        else:
            grpo_train_epoch(epoch, loader, len(loader), ref_model, reward_model, reward_tokenizer, 0, wandb)
    
    # ========== 9.清理分布式进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()