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
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model

warnings.filterwarnings('ignore')


def calculate_rewards(prompts, responses, reward_model, reward_tokenizer):
    """整合所有奖励函数计算总奖励"""

    def reasoning_model_reward(rewards):
        # --- 1. 格式奖励：鼓励模型学会“思考”格式 ---
        # 使用正则表达式检查回答是否遵循 <think>...</think><answer>...</answer> 的标准格式。
        # 这种格式化的输出有助于模型学习链式思考（Chain-of-Thought）。
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"

        matches_pattern = [re.match(pattern, response, re.S) for response in responses]
        matches_pattern2 = [re.match(pattern2, response, re.S) for response in responses]

        format_rewards = []
        for match_pattern, match_pattern2 in zip(matches_pattern, matches_pattern2):
            if match_pattern:
                format_rewards.append(0.5) # 格式正确则给予奖励
            elif match_pattern2:
                format_rewards.append(0.5)
            else:
                format_rewards.append(0.0)
        rewards += torch.tensor(format_rewards, device=args.device)

        # --- 2. 标记奖励：提供更密集的引导信号 ---
        # 检查<think>, </think>, <answer>, </answer>四个标签是否都只出现一次。
        # 这是一个“稀疏奖励”的补充，即使格式不完全对，只要包含了正确的标签，也能得到一些分数，
        # 避免了模型在初期因无法生成完整正确格式而得不到任何正反馈的问题。
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

    # 初始化一个长度为 responses 的张量，用于保存奖励
    rewards = torch.zeros(len(responses), device=args.device)

    # --- 3. 如果是推理模式，则添加上述的格式和标记奖励 ---
    if args.reasoning == 1:
        rewards = reasoning_model_reward(rewards)  # 训练推理模型时使用

    # --- 4. 核心奖励：使用外部奖励模型打分 ---
    # with torch.no_grad() 确保这部分计算不产生梯度
    with torch.no_grad():
        reward_model_scores = []
        batch_size = len(prompts)
        scale = 3.0
        # 遍历每一个Prompt
        for i in range(batch_size):
            # 每一个Prompt对应num_generations个生成结果
            for j in range(args.num_generations):
                # 取该个Prompt的j个生成结果
                response_idx = i * args.num_generations + j
                response = responses[response_idx]
                prompt = prompts[i]

                # --- 准备奖励模型的输入 ---
                # 解析prompt，将其转换为奖励模型能理解的对话格式列表。
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]

                # 将当前response作为assistant的最新回复，添加到对话历史中。
                tmp_chat = messages + [{"role": "assistant", "content": response}]
                # 调用奖励模型，获取对整个对话（包括当前response）的评分。
                score = reward_model.get_score(reward_tokenizer, tmp_chat)
                # 将分数裁剪到[-3.0, 3.0]区间，防止极端值影响训练稳定性。
                score = max(min(score, scale), -scale)

                # --- 对推理任务的特殊处理 ---
                # 如果是推理模式...
                if args.reasoning == 1:
                    #...尝试从response中提取<answer>标签内的最终答案。
                    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                    # 如果找到了最终答案...
                    if answer_match:
                        #...提取答案内容。
                        answer_content = answer_match.group(1).strip()
                        #...构造一个新的对话历史，其中assistant的回复仅包含这个最终答案。
                        tmp_chat_answer = messages + [{"role": "assistant", "content": answer_content}]
                        #...单独对这个最终答案进行评分。
                        answer_score = reward_model.get_score(reward_tokenizer, tmp_chat_answer)
                        answer_score = max(min(answer_score, scale), -scale)
                        # 最终分数是“完整回答”和“核心答案”的加权平均，核心答案权重更高(0.6)。
                        score = score * 0.4 + answer_score * 0.6

                # 将计算出的最终分数添加到列表中。
                reward_model_scores.append(score)

    # 4. 将所有分数转换为PyTorch张量，并累加到总奖励上。
    reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
    rewards += reward_model_scores

    # 5. 返回包含了所有维度奖励的最终总奖励张量。
    return rewards


def grpo_train_epoch(epoch, wandb):
    for step, batch in enumerate(train_loader):
        # --- 准备输入 ---
        prompts = batch['prompt']  # list[str], 一维List 长度 =  Batch_size
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
                                  padding_side="left", add_special_tokens=False).to(args.device)  # input_ids: [B, P], attention_mask: [B, P]
        # 如果超出长度则截断
        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]
            
        # --- 步骤1: 生成一组响应 (Rollout) ---
        # 对于每个prompt，让当前策略模型生成 num_generations 个不同的回答。
        # do_sample=True 和 temperature=0.8 保证了生成的多样性。
        with torch.no_grad():
            outputs = (model.module if ddp else model).generate(
                **prompt_inputs, max_new_tokens=args.max_gen_len, do_sample=True, temperature=0.8,
                num_return_sequences=args.num_generations, pad_token_id=tokenizer.pad_token_id)  # [B*num_gen, P+R]

        # 从生成结果中分离出回答部分 (completion) 的 token IDs
        completion_ids = outputs[:, prompt_inputs["input_ids"].size(1):]  # [B*num_gen, R]
        
        # --- 步骤2: 计算对数概率 ---
        # 定义一个辅助函数来计算给定序列中每个token的对数概率
        def get_per_token_logps(mdl, input_ids, n_keep):
            # input_ids [B*num_gen, P+R]
            # 判断是否为推理模式，如果是，则 detaching input_ids，以防止反向传播
            input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
            # 计算模型的 logits，logits 的形状是 [B, P+R, vocab_size]
            logits = mdl(input_ids, logits_to_keep=n_keep + 1).logits[:, :-1, :]

            per_token_logps = []
            # 遍历每一行 logits 和输入的 id，计算每个 token 的 log 概率
            for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]): # [B*num_gen , R]
                # 如果是推理阶段，detach input_ids 防止反向传播 
                ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
                # 使用 torch.gather 从 logits 中选出 ids_row 对应的列，然后对 logits 做 softmax 转换得到 log 概率
                per_token_logps.append(torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1))
            return torch.stack(per_token_logps) # [B*num_gen, R]

        # 计算“策略模型”对于这批生成的responses的log_probs。
        per_token_logps = get_per_token_logps(model, outputs, completion_ids.size(1))  # [B*num_gen, R]
        # 在不计算梯度的模式下，计算“参考模型”的log_probs。
        with torch.no_grad():
            ref_per_token_logps = get_per_token_logps(ref_model, outputs, completion_ids.size(1))  # [B*num_gen, R]

        # --- 步骤3: 计算奖励和优势 (Reward & Advantage) ---
        # 将response的token IDs解码回文本。
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True) # [B*num_gen, R]
        rewards = calculate_rewards(prompts, completions, reward_model, reward_tokenizer).to(args.device)  # [B*num_gen]

        #!!! GRPO算法的核心!!!
        # 将扁平的rewards张量重塑为(batch_size, num_generations)的形状。
        grouped_rewards = rewards.view(-1, args.num_generations)  # [B, num_gen]
        # 沿着num_generations维度计算每组的平均奖励。
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)  # [B*num_gen]
        std_r = grouped_rewards.std(dim=1).repeat_interleave(args.num_generations)  # [B*num_gen]
        # 计算优势值：(当前奖励 - 组内平均奖励) / 组内标准差。
        # 这衡量了每个response比其“兄弟”们好多少（或差多少）。
        advantages = torch.clamp((rewards - mean_r) / (std_r + 1e-4), -10, 10)
        # 对所有优势值进行全局归一化，使训练更加稳定。
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # [B*num_gen]

        # --- 准备阶段 ---
        # completion_ids: 模型生成的回答部分的token ID, 维度
        # tokenizer.eos_token_id: 句子结束符的token ID, 比如 2
        
        # --- 步骤1: 找到每个回答中的结束符(EOS) ---
        # 比较 completion_ids 中的每个元素是否等于结束符ID。
        # is_eos 是一个布尔类型的张量，形状和 completion_ids 一样。
        # 值为 True 的位置表示那里是一个结束符。
        is_eos = completion_ids == tokenizer.eos_token_id  # 维度: [B*num_gen, R]
        
        # --- 步骤2: 定位第一个结束符的位置 ---
        # a. 创建一个初始索引张量 eos_idx，长度为 B*num_gen。
        #    torch.full 用 R (回答的最大长度) 来填充它。
        #    这是一种“悲观”初始化：假设所有回答都没有结束符，那么有效长度就是最大长度R。
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=args.device)
        
        # b. is_eos.any(dim=1) 会检查每一行(每个回答)是否至少包含一个 True (结束符)。
        #    这会返回一个布尔类型的一维张量，标记了哪些回答是“完整”的。
        # c. is_eos.int().argmax(dim=1) 会找到每一行中第一个 1 (即第一个True) 的索引。
        #    这就是我们想要的第一个结束符的位置。
        # d. 最后，用 b 作为索引，将 c 中计算出的真实结束符位置，更新到 a 中对应的位置。
        #    对于那些没有结束符的回答，它们在 eos_idx 中的值依然是初始的 R。
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        
        # --- 步骤3: 根据结束符位置创建掩码 ---
        # a. torch.arange(is_eos.size(1),...): 创建一个从 0 到 R-1 的序列:。
        # b..expand(is_eos.size(0), -1): 将这个序列复制 B*num_gen 次，形成一个矩阵。
        #    每一行都是。
        # c. eos_idx.unsqueeze(1): 将结束符索引张量变形，以便进行广播比较。
        # d. <=: 逐元素比较。对于每一行，只有当列索引小于或等于该行的结束符索引时，结果才为 True。
        # e..int(): 将布尔结果 (True/False) 转换为整数 (1/0)。
        completion_mask = (torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)).int()  # 维度: [B*num_gen, R]

        # --- 步骤1: 计算KL散度惩罚 ---
        # KL散度(Kullback-Leibler divergence)衡量两个概率分布的差异。
        # 在这里，它衡量“策略模型”的输出分布与“参考模型”的输出分布有多么不同。
        # kl_div 是对数概率的直接相减，这是计算KL散度的基础。
        kl_div = ref_per_token_logps - per_token_logps
        # 这行代码是KL散度惩罚项的一种常见且数值稳定的近似计算方法。
        # 它的作用是：当 kl_div 远离0时，per_token_kl 的值会迅速增大，形成一个惩罚。
        per_token_kl = torch.exp(kl_div) - kl_div - 1  # 维度: [B*num_gen, R]
        
        # --- 步骤2: 计算PPO/GRPO核心损失 ---
        # 这是整个算法最核心的公式，它由两部分组成：
        # 1. 策略梯度项: torch.exp(...) * advantages.unsqueeze(1)
        #    - per_token_logps.detach() 创建一个不带梯度的副本，代表“旧策略”的概率。
        #    - exp(新logp - 旧logp) = exp(log(新p/旧p)) = 新p/旧p，这就是“重要性采样”中的概率比(ratio)。
        #    - 用这个概率比乘以“优势”，就是策略梯度的核心思想：
        #      - 如果优势为正（好行为），则鼓励增大这个概率比（即提高新策略的概率）。
        #      - 如果优势为负（坏行为），则鼓励减小这个概率比（即降低新策略的概率）。
        # 2. KL惩罚项: args.beta * per_token_kl
        #    - 用超参数 beta 缩放KL惩罚，加到损失中。这会阻止策略模型为了追求高优势而变得与初始模型差异过大，从而保证了训练的稳定性。
        # 整个表达式前的负号是因为优化器通常是最小化(minimize)损失，而我们的目标是最大化(maximize)这个策略目标。
        per_token_loss = -(torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1) - args.beta * per_token_kl)  # 维度:
        
        # --- 步骤3: 聚合损失并反向传播 ---
        # a. (per_token_loss * completion_mask): 用我们之前计算的掩码，将所有填充位置的损失清零。
        # b..sum(dim=1): 计算每个回答的“有效”总损失。
        # c. / completion_mask.sum(dim=1): 除以每个回答的“有效”长度，得到每个回答的平均损失。
        # d..mean(): 计算整个batch的平均损失，得到一个最终的标量值。
        # e. / args.accumulation_steps: 如果使用梯度累积，将损失进行缩放。
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean() / args.accumulation_steps  # 最终为标量(scalar)
        
        # --- 步骤4: 启动优化 ---
        # 根据计算出的最终损失值，计算所有模型参数的梯度。
        loss.backward()

        if (step + 1) % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % args.log_interval == 0 or step == iters:
            policy_loss_val = loss.item()
            avg_reward_val = rewards.mean().item()
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            current_lr = optimizer.param_groups[0]['lr']

            Logger(f'Epoch: {epoch+1}, Step: {step}/{iters}, '
                   f'Actor Loss: {policy_loss_val:.6f}, Reward: {avg_reward_val:.6f}, '
                   f'Avg Response Len: {avg_len_val:.2f}, LR: {current_lr:.2e}')

            if wandb and is_main_process():
                wandb.log({
                    "policy_loss": policy_loss_val,
                    "reward": avg_reward_val,
                    "avg_response_len": avg_len_val,
                    "advantages_mean": advantages.mean().item(),
                    "learning_rate": current_lr
                })

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
            torch.save({k: v.half() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scheduler=scheduler)
            model.train()

        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind GRPO (Group Relative Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='grpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=8e-8, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_seq_len', default=66, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1536, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif-mini.jsonl", help="RLAIF数据路径")
    parser.add_argument("--num_generations", type=int, default=8, help="每个prompt生成的样本数")
    parser.add_argument("--beta", type=float, default=0.02, help="KL惩罚系数")
    parser.add_argument("--reasoning", type=int, default=1, choices=[0, 1], help='推理模型类型（0=普通模型，1=推理模型）')
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-GRPO", help="wandb项目名")
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
        wandb_run_name = f"MiniMind-GRPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 初始化模型和数据 ==========
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
    # 数据和优化器
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
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
        train_sampler and train_sampler.set_epoch(epoch)
        if epoch == start_epoch and start_step > 0:  # 第一个epoch且存在检查点
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            grpo_train_epoch(epoch, loader, len(loader) + start_step + 1, ref_model, reward_model, reward_tokenizer, start_step, wandb)
        else:  # 默认从头开始
            loader = DataLoader(train_ds, batch_size=args.batch_size, pin_memory=True,
                              drop_last=False, shuffle=(train_sampler is None),
                              num_workers=args.num_workers, sampler=train_sampler)
            grpo_train_epoch(epoch, loader, len(loader), ref_model, reward_model, reward_tokenizer, 0, wandb)
