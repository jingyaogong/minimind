import os
import sys

# 设置包名
__package__ = "trainer"
# 将项目根目录添加到系统路径, 确保能够导入同级目录的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import re
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model

# 忽略警告信息, 避免输出干扰
import warnings
warnings.filterwarnings('ignore')


class CriticModel(MiniMindForCausalLM):
    """
     Critic 模型, 继承自 MiniMindForCausalLM
     用于估计状态价值, 为 PPO 算法提供优势函数计算基础
    """
    def __init__(self, params):
        super().__init__(params)
        # 添加价值头, 将隐藏层输出映射为单一标量价值
        self.value_head = nn.Linear(params.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # 使用基础模型获取隐藏状态输出
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        # 对最后一层隐藏状态进行 LayerNorm 归一化
        hidden_states = self.model.norm(outputs[0])
        # 通过 value_head 计算每个位置的价值估计, 并移除最后一个维度
        values = self.value_head(hidden_states).squeeze(-1)
        return values


def calculate_rewards(prompts, responses, reward_model, reward_tokenizer):
    """
    整合所有奖励函数计算总奖励

    Args:
    - prompts:          输入提示列表, list[str]
    - responses:        模型生成的响应列表, list[str]
    - reward_model:     奖励模型, 用于评估回答质量
    - reward_tokenizer: 奖励模型的分词器

    Returns:
    - torch.Tensor:     每个样本的总奖励值, shape [B]
    """
    def reasoning_model_reward(rewards):
        """
        推理模型的奖励计算

        包含两部分奖励:
        1. 格式奖励: 检查回答是否符合 <think>...</think><answer>...</answer> 格式
        2. 标记奖励: 检查关键标记 (think, answer 标签) 的出现次数

        Args:
        - rewards: 当前奖励张量, shape [B]

        Returns:
        - torch.Tensor: 更新后的奖励张量
        """
        # 定义正则表达式模式, 匹配标准的推理格式
        # 模式1: <think> 和 <answer> 之间只有一个换行
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        # 模式2: <think> 和 <answer> 之间有两个换行
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"

        # 对每个响应进行正则匹配
        matches_pattern = [re.match(pattern, response, re.S) for response in responses]
        matches_pattern2 = [re.match(pattern2, response, re.S) for response in responses]

        # 计算格式奖励: 匹配任一模式得 0.5 分, 否则 0 分
        format_rewards = []
        for match_pattern, match_pattern2 in zip(matches_pattern, matches_pattern2):
            if match_pattern:
                format_rewards.append(0.5)
            elif match_pattern2:
                format_rewards.append(0.5)
            else:
                format_rewards.append(0.0)
        rewards += torch.tensor(format_rewards, device=args.device)

        # 计算标记奖励, 防止严格格式奖励过于稀疏
        def mark_num(text):
            """统计文本中关键标记的出现次数, 每类标记正确出现一次得 0.25 分"""
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

    # 初始化奖励张量为零, shape [B]
    rewards = torch.zeros(len(responses), device=args.device)

    # 如果是推理模型, 添加格式和标记奖励
    if args.reasoning == 1:
        rewards = reasoning_model_reward(rewards)

    # 使用奖励模型计算 response 的整体质量分数
    with torch.no_grad():
        reward_model_scores = []
        for prompt, response in zip(prompts, responses):
            # 解析 prompt 中的对话历史
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": content.strip()} for role, content in matches]

            # 构建完整的对话, 包含当前 response
            tmp_chat = messages + [{"role": "assistant", "content": response}]
            # 调用奖励模型获取评分
            score = reward_model.get_score(reward_tokenizer, tmp_chat)

            # 将分数裁剪到 [-scale, scale] 范围内, 避免极端值
            scale = 3.0
            score = max(min(score, scale), -scale)

            # 如果是推理模型, 额外计算 <answer> 内容部分的奖励
            if args.reasoning == 1:
                answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                if answer_match:
                    # 提取 answer 标签内的内容
                    answer_content = answer_match.group(1).strip()
                    # 对 answer 内容单独计算奖励
                    tmp_chat = messages + [{"role": "assistant", "content": answer_content}]
                    answer_score = reward_model.get_score(reward_tokenizer, tmp_chat)
                    answer_score = max(min(answer_score, scale), -scale)
                    # 综合评分: 整体回答 40% + answer 内容 60%
                    score = score * 0.4 + answer_score * 0.6
            reward_model_scores.append(score)

        # 将奖励模型分数转换为张量并累加到总奖励
        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


def ppo_train_epoch(epoch, loader, iters, old_actor_model, ref_model, actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, start_step=0, wandb=None):
    """
    PPO 单 epoch 训练函数

    Args:
    - epoch:                当前训练轮次
    - loader:               数据加载器
    - iters:                总迭代次数
    - old_actor_model:      旧策略模型, 用于计算重要性采样比率
    - ref_model:            参考模型, 用于计算 KL 散度惩罚
    - actor_scheduler:      Actor 学习率调度器
    - critic_scheduler:     Critic 学习率调度器
    - reward_model:         奖励模型
    - reward_tokenizer:     奖励模型分词器
    - start_step:           起始步数 (用于断点续训)
    - wandb:                日志记录工具

    Note:
    使用全局变量: actor_model, critic_model, tokenizer, lm_config, args, actor_optimizer, critic_optimizer, autocast_ctx
    """
    # 设置模型为训练模式
    actor_model.train()
    critic_model.train()

    for step, batch in enumerate(loader, start=start_step + 1):
        # 获取批次中的 prompt 列表
        prompts = batch["prompt"]  # list[str], length B
        # 对 prompt 进行编码, 左填充以支持生成
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                       max_length=args.max_seq_len, padding_side="left").to(args.device)  # input_ids: [B, P], attention_mask: [B, P]
        prompt_length = enc.input_ids.shape[1]

        # 使用当前策略生成响应 (不计算梯度)
        with torch.no_grad():
            # DDP 包装模型需要通过 .module 访问 generate 方法
            model_for_gen = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            gen_out = model_for_gen.generate(
                input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                max_new_tokens=args.max_gen_len, do_sample=True, temperature=0.8,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)  # [B, P+R]

        # 解码生成的响应文本 (去除 prompt 部分)
        responses_text = [tokenizer.decode(gen_out[i, prompt_length:], skip_special_tokens=True) for i in range(len(prompts))]
        # 计算每个样本的奖励值
        rewards = calculate_rewards(prompts, responses_text, reward_model, reward_tokenizer)  # [B]

        # 创建完整序列的注意力掩码 (非填充位置为 1)
        full_mask = (gen_out != tokenizer.pad_token_id).long()  # [B, P+R]
        # 使用 Critic 模型评估生成序列的价值
        values_seq = critic_model(input_ids=gen_out, attention_mask=full_mask)  # [B, P+R]
        # 找到每个序列最后一个非填充位置的索引
        last_indices = (full_mask * torch.arange(full_mask.size(1), device=gen_out.device)).argmax(dim=1)
        # 提取每个序列最终位置的价值估计
        values = values_seq[torch.arange(values_seq.size(0), device=values_seq.device), last_indices]  # [B]
        # 计算优势函数: 奖励 - 价值 (使用 detach 阻止梯度流向 Critic)
        advantages = rewards - values.detach()  # [B]

        # 使用混合精度计算 Actor 模型输出
        with autocast_ctx:
            res = actor_model(input_ids=gen_out, attention_mask=full_mask)
            logits = res.logits  # [B, P+R, V]
            # 如果是 MoE 模型, 获取辅助损失; 否则为 0
            aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)

        # 准备标签 (向右移动一位, 用于计算每个位置的 log prob)
        labels = gen_out[:, 1:].clone()  # [B, P+R-1]
        # 计算每个 token 的对数概率
        logp_tokens = F.log_softmax(logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [B, P+R-1]
        seq_len = gen_out.size(1) - 1
        # 创建响应部分的掩码 (只计算生成部分, 不计算 prompt)
        resp_mask = torch.arange(seq_len, device=gen_out.device).unsqueeze(0) >= prompt_length - 1
        # 最终掩码: 响应部分且非填充位置
        final_mask = resp_mask & (~labels.eq(tokenizer.pad_token_id))  # [B, P+R-1]
        # 计算 Actor 策略下整个响应的总对数概率
        actor_logp = (logp_tokens * final_mask).sum(dim=1)  # [B]

        # 使用旧策略和参考模型计算对数概率 (不计算梯度)
        with torch.no_grad():
            # 旧策略模型的输出
            old_logits = old_actor_model(input_ids=gen_out, attention_mask=full_mask).logits  # [B, P+R, V]
            old_logp_tokens = F.log_softmax(old_logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [B, P+R-1]
            old_logp = (old_logp_tokens * final_mask).sum(dim=1)  # [B]

            # 参考模型的输出 (用于 KL 散度计算)
            ref_logits = ref_model(input_ids=gen_out, attention_mask=full_mask).logits  # [B, P+R, V]
            ref_logp_tokens = F.log_softmax(ref_logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [B, P+R-1]
            ref_logp = (ref_logp_tokens * final_mask).sum(dim=1)  # [B]

        # 计算与旧策略的 KL 散度
        kl = (actor_logp - old_logp).mean()  # scalar
        # 计算与参考模型的 KL 散度 (作为惩罚项)
        kl_ref = (actor_logp - ref_logp).mean()  # scalar
        # 计算重要性采样比率
        ratio = torch.exp(actor_logp - old_logp)  # [B]
        # PPO 裁剪目标的第一项
        surr1 = ratio * advantages  # [B]
        # PPO 裁剪目标的第二项 (裁剪比率防止过大更新)
        surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages  # [B]
        # 策略损失: 取裁剪后的最小值, 取负号后求平均 (因为我们要最大化目标)
        policy_loss = -torch.min(surr1, surr2).mean()  # scalar
        # 价值损失: Critic 预测值与奖励的均方误差
        value_loss = F.mse_loss(values, rewards)  # scalar
        # 总损失: 策略损失 + 价值系数 * 价值损失 + KL 系数 * KL 惩罚 + 辅助损失
        loss = (policy_loss + args.vf_coef * value_loss + args.kl_coef * kl_ref + aux_loss) / args.accumulation_steps  # scalar
        # 反向传播计算梯度
        loss.backward()

        # 梯度累积: 达到指定步数后执行参数更新
        if (step + 1) % args.accumulation_steps == 0:
            # 对 Actor 和 Critic 的梯度进行裁剪, 防止梯度爆炸
            clip_grad_norm_(actor_model.parameters(), args.grad_clip)
            clip_grad_norm_(critic_model.parameters(), args.grad_clip)
            # 执行优化器参数更新
            actor_optimizer.step()
            critic_optimizer.step()
            # 更新学习率
            actor_scheduler.step()
            critic_scheduler.step()
            # 清空梯度
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()

        # 只在主进程打印日志和记录指标
        if is_main_process():
            # 提取响应部分的 token IDs
            response_ids = gen_out[:, enc.input_ids.shape[1]:]
            # 检测每个响应是否包含 EOS 标记
            is_eos = (response_ids == tokenizer.eos_token_id)
            # 找到第一个 EOS 的位置
            eos_indices = torch.argmax(is_eos.int(), dim=1)
            # 检查每个响应是否实际包含 EOS
            has_eos = is_eos.any(dim=1)
            # 计算实际响应长度 (如果有 EOS 则为 EOS 位置+1, 否则为完整长度)
            lengths = torch.where(has_eos, eos_indices + 1, torch.tensor(response_ids.shape[1], device=is_eos.device))
            # 计算平均响应长度
            avg_len = lengths.float().mean()

            # 提取各项指标数值
            actor_loss_val = policy_loss.item()
            critic_loss_val = value_loss.item()
            current_aux_loss = aux_loss.item()
            reward_val = rewards.mean().item()
            kl_val = kl.item()
            kl_ref_val = kl_ref.item()
            avg_len_val = avg_len.item()
            actor_lr = actor_optimizer.param_groups[0]['lr']
            critic_lr = critic_optimizer.param_groups[0]['lr']

            # 记录到 wandb
            if wandb is not None:
                wandb.log({
                    "actor_loss": actor_loss_val,
                    "critic_loss": critic_loss_val,
                    "aux_loss": current_aux_loss,
                    "reward": reward_val,
                    "kl": kl_val,
                    "kl_ref": kl_ref_val,
                    "avg_response_len": avg_len_val,
                    "actor_lr": actor_lr,
                })

            # 打印训练日志
            Logger(f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                   f"Actor Loss: {actor_loss_val:.4f}, Critic Loss: {critic_loss_val:.4f}, Aux Loss: {current_aux_loss:.4f}, "
                   f"Reward: {reward_val:.4f}, KL: {kl_val:.4f}, KL_ref: {kl_ref_val:.4f}, "
                   f"Avg Response Len: {avg_len_val:.2f}, Actor LR: {actor_lr:.8f}, Critic LR: {critic_lr:.8f}")

        # 按指定频率更新旧策略模型
        if (step + 1) % args.update_old_actor_freq == 0:
            # 解包 DDP 模型获取原始模型
            raw_actor = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            raw_actor = getattr(raw_actor, '_orig_mod', raw_actor)
            # 获取当前策略状态字典
            state_dict = raw_actor.state_dict()
            # 将参数 detached 后加载到旧策略模型
            old_actor_model.load_state_dict({k: v.detach().cpu() for k, v in state_dict.items()})
            old_actor_model.to(args.device)

        # 按指定间隔保存检查点 (只在主进程执行)
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            actor_model.eval()
            # 根据是否使用 MoE 构建文件名后缀
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # 解包 DDP 模型获取原始 Actor 模型
            raw_actor = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            raw_actor = getattr(raw_actor, '_orig_mod', raw_actor)
            # 获取状态字典并转换为半精度后保存
            actor_state = raw_actor.state_dict()
            torch.save({k: v.half().cpu() for k, v in actor_state.items()}, ckp)

            # 使用 lm_checkpoint 保存完整训练状态 (包括 Critic)
            lm_checkpoint(lm_config, weight=args.save_weight, model=actor_model, optimizer=actor_optimizer,
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints',
                         scheduler=actor_scheduler, critic_model=critic_model,
                         critic_optimizer=critic_optimizer, critic_scheduler=critic_scheduler)
            actor_model.train()
            del actor_state

        # 显式删除中间变量, 释放显存
        del enc, gen_out, responses_text, rewards, full_mask, values_seq, values, advantages
        del logits, labels, logp_tokens, final_mask, actor_logp, old_logits, old_logp, ref_logits, ref_logp
        del kl, kl_ref, ratio, surr1, surr2, policy_loss, value_loss, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind PPO (Proximal Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='ppo_actor', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=8e-8, help="Actor学习率")
    parser.add_argument("--critic_learning_rate", type=float, default=8e-8, help="Critic学习率")
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
    parser.add_argument("--clip_epsilon", type=float, default=0.1, help="PPO裁剪参数")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function系数")
    parser.add_argument("--kl_coef", type=float, default=0.02, help="KL散度惩罚系数")
    parser.add_argument("--reasoning", type=int, default=1, choices=[0, 1], help='推理模型类型 (0=普通模型, 1=推理模型)')
    parser.add_argument("--update_old_actor_freq", type=int, default=4, help="更新old_actor_model的频率")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训 (0=否, 1=是)")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-PPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速 (0=否, 1=是)")
    args = parser.parse_args()

    # ========== 1.初始化环境和随机种子 ==========
    # 初始化分布式训练环境, 获取本地 rank
    local_rank = init_distributed_mode()
    # 如果处于分布式训练环境, 根据 local_rank 设置设备
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 设置随机种子, 分布式环境下根据 rank 添加偏移以确保不同进程种子不同
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2.配置目录、模型参数、检查 ckp ==========
    # 创建模型保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    # 创建模型配置对象
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # 如果启用断点续训, 尝试加载检查点数据
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None

    # ========== 3.设置混合精度 ==========
    # 判断设备类型
    device_type = "cuda" if "cuda" in args.device else "cpu"
    # 设置数据类型
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # https://docs.pytorch.ac.cn/docs/stable/amp
    # 新版废弃: autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=dtype)

    # ========== 4. 配置 wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        # 使用 swanlab 作为 wandb 的替代
        import swanlab as wandb
        # 尝试从检查点恢复 wandb ID
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        # 构建运行名称
        wandb_run_name = f"MiniMind-PPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        # 初始化 wandb
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 初始化模型和数据 ==========
    # 根据是否训练推理模型选择基础权重名称
    base_weight = "reason" if args.reasoning == 1 else "full_sft"

    # Actor 模型 (策略网络, 需要训练)
    actor_model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    if args.use_compile == 1:
        # 使用 torch.compile 加速模型 (需要 PyTorch 2.0+)
        actor_model = torch.compile(actor_model)
        Logger('torch.compile enabled')

    # Old Actor 模型 (旧策略, 用于 PPO 的重要性采样)
    old_actor_model, _ = init_model(lm_config, base_weight, device=args.device)
    # 设置为评估模式, 不计算梯度
    old_actor_model = old_actor_model.eval().requires_grad_(False)

    # Reference 模型 (参考模型, 用于 KL 散度惩罚)
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)

    # Critic 模型 (价值网络, 估计状态价值)
    moe_suffix = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/{base_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    # 加载基础模型权重初始化 Critic
    state_dict = torch.load(ckp, map_location=args.device)
    critic_model = CriticModel(lm_config)
    critic_model.load_state_dict(state_dict, strict=False)
    critic_model = critic_model.to(args.device)

    # Reward 模型 (奖励模型, 提供奖励信号)
    reward_model = AutoModel.from_pretrained(args.reward_model_path, torch_dtype=torch.float16, trust_remote_code=True)
    reward_model = reward_model.to(args.device).eval().requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)

    # 数据集和优化器配置
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=(args.max_seq_len + args.max_gen_len))
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.AdamW(critic_model.parameters(), lr=args.critic_learning_rate)
    # 计算训练迭代次数
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    # 计算总优化步数 (考虑梯度累积)
    total_optimizer_steps = (iters // args.accumulation_steps) * args.epochs
    # 学习率调度器 (余弦退火)
    actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    critic_scheduler = CosineAnnealingLR(critic_optimizer, T_max=total_optimizer_steps, eta_min=args.critic_learning_rate / 10)

    # ==========6. 从ckp 恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        # 恢复模型权重
        actor_model.load_state_dict(ckp_data['model'])
        critic_model.load_state_dict(ckp_data['critic_model'])
        # 恢复优化器状态
        actor_optimizer.load_state_dict(ckp_data['optimizer'])
        critic_optimizer.load_state_dict(ckp_data['critic_optimizer'])
        # 恢复学习率调度器状态
        actor_scheduler.load_state_dict(ckp_data['scheduler'])
        critic_scheduler.load_state_dict(ckp_data['critic_scheduler'])
        # 恢复训练进度
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7.DDP 包装模型 ==========
    if dist.is_initialized():
        # 设置 DDP 忽略的参数 (位置编码相关, 不需要梯度同步)
        actor_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        critic_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        # 包装为 DDP 模型
        actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank])
        critic_model = DistributedDataParallel(critic_model, device_ids=[local_rank])
        # 确保旧策略模型在正确设备上
        old_actor_model.to(args.device)

    # ========== 8.开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # 设置分布式采样器的 epoch (确保每个 epoch 数据打乱方式不同)
        train_sampler and train_sampler.set_epoch(epoch)
        # 每个 epoch 使用不同的随机种子
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # 如果是续训的第一个 epoch, 需要跳过已经训练过的 step
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # 创建支持跳过指定批次数量的采样器
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        # 创建数据加载器
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step, 从step {start_step + 1}开始')
            ppo_train_epoch(
                epoch, 
                loader, 
                len(loader) + skip, 
                old_actor_model, 
                ref_model,
                actor_scheduler, 
                critic_scheduler, 
                reward_model, 
                reward_tokenizer, 
                start_step, 
                wandb)
        else:
            ppo_train_epoch(
                epoch, 
                loader, 
                len(loader), 
                old_actor_model, 
                ref_model,
                actor_scheduler, 
                critic_scheduler, 
                reward_model, 
                reward_tokenizer, 
                0, 
                wandb)

    # ========== 9.清理分布式进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()