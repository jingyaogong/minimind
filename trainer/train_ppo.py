import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import math
import re
import warnings
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
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_minimind_mla import MiniMindMLAConfig, MiniMindMLAForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model, LMForRewardModel, get_model_suffix, build_lm_config, resolve_attention_type, log_training_setup
from trainer.rollout_engine import create_rollout_engine

warnings.filterwarnings('ignore')


def rep_penalty(text, n=3, cap=0.5):
    #处理重复回答，是因为PPO/RL训练里模型很容易钻奖励函数空子，比如reward有长度奖励，奖励模型打分，格式奖励等

    #reward hacking 表示模型没有真正学好任务，而是学会了钻奖励函数的空子
    #解决reward hacking方法：规则函数做约束，分项监控reward来源，加kl约束防止偏离参考模型，降低ppo更新强度，还要监控行为指标

    #参数n代表，检查n-gram，也就是连续n个token的重复，cap表示最多惩罚的分数
    #计算重复惩罚，回答的越重复，reward扣除的越多，用于惩罚模型回答里重复出现的n-gram
    #先把文本转为小写， \w+表示匹配连续的单词字符 [^\w\s]表示既不是单词，又不是空白字符的东西，一般是标点符号
    #中间的 ｜ 表示或者，所以整体意思是，匹配一个单词或者匹配一个标点
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


def _make_critic_model(base_class):
    class CriticModel(base_class):
        def __init__(self, params):
            super().__init__(params)
            # 替换lm_head为输出单一价值的线性层
            self.value_head = nn.Linear(params.hidden_size, 1)

        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            # 使用基础模型获取隐藏状态
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            hidden_states = self.model.norm(outputs[0])
            # 使用value_head获取价值估计
            values = self.value_head(hidden_states).squeeze(-1)
            return values
    return CriticModel


def calculate_rewards(prompts, responses, reward_model):
    rewards = torch.zeros(len(responses), device=args.device)

    with torch.no_grad():
        reward_model_scores = []
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            """
            下面这两行是在从prompt字符串中解析出来对话消息，比如把
            <|im_start|>system
            你是一个助手<|im_end|>
            <|im_start|>user
            你好<|im_end|>
            <|im_start|>assistant
            你好，有什么可以帮你？<|im_end|>
            解析为
            [
                ("system", "你是一个助手"),
                ("user", "你好"),
                ("assistant", "你好，有什么可以帮你？"),
            ]

            """
            #前面的r表示raw string ，｜在正则里是特殊字符，或的意思，所以要写成\|
            #(system|user|assistant) 这一段匹配角色名，括号()表示捕获组，匹配三者之一
            #\s+表示匹配一个或多个空白字符，(.*?)表示非贪婪匹配内容直到下一个<|im_end|>，也是一个捕获组
            #.表示匹配任意字符，*表示匹配0或多次，？表示变成非贪婪模式，直到遇到下一个<|im_end|>为止
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            #默认情况下，正则里的.不能匹配换行符，加上re.DOTALL参数后，.就能匹配换行符了，这样就能正确解析多行的消息内容
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": content.strip()} for role, content in matches]
            answer = response
            #如果回答的长度在20到800之间则奖励分数，限制模型回答长度
            rewards[i] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
            if '</think>' in response:
                #这里的1表示最多切割一次，根据</think>切成两部分
                thinking_content, answer_content = response.split('</think>', 1)
                #限制思考内容的长度
                rewards[i] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                #限制模型有且只有一个</think>
                rewards[i] += 0.25 if response.count('</think>') == 1 else -0.25
                answer = answer_content.strip()
            rewards[i] -= rep_penalty(answer)

            score = reward_model.get_score(messages, answer)
            reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


def ppo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, actor_scheduler, critic_scheduler, reward_model, start_step=0, wandb=None, use_sglang=False):
    actor_model.train()
    critic_model.train()
    grad_accum_step = 0

    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompt"]  # list[str], length B
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_len,
                        padding_side="left").to(args.device)  # input_ids: [B, P], attention_mask: [B, P]

        rollout_result = rollout_engine.rollout(
            prompt_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            num_generations=1,
            max_new_tokens=args.max_gen_len,
            temperature=0.8,
        )
        gen_out = rollout_result.output_ids
        completion_ids = rollout_result.completion_ids
        prompt_lens = rollout_result.prompt_lens.to(args.device)
        responses_text = rollout_result.completions
        old_resp_logp = rollout_result.per_token_logps.to(args.device)
        #responses_text [B]  为每条回答计算一个reward
        rewards = calculate_rewards(prompts, responses_text, reward_model)  # [B]

        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            #debug模式就是程序运行时额外打开的一种调试开关
            for i in range(len(prompts)):
                Logger(f"[DEBUG] step={step}, sample[{i}]")
                Logger('-'*100)
                Logger(f"{'=' * 30} [DEBUG] sample[{i}] CONTEXT_BEGIN {'=' * 30}")
                Logger(prompts[i])
                Logger(f"{'=' * 31} [DEBUG] sample[{i}] CONTEXT_END {'=' * 31}")
                Logger(f"[DEBUG] prompt_len={prompt_lens[i].item()}, response_len={len(responses_text[i])}")
                Logger(f"{'=' * 28} [DEBUG] sample[{i}] RESPONSE_BEGIN {'=' * 28}")
                Logger(responses_text[i])
                Logger(f"{'=' * 29} [DEBUG] sample[{i}] RESPONSE_END {'=' * 29}")
                Logger(f"[DEBUG] reward={rewards[i].item():.4f}")
                Logger('='*100)

        full_mask = (gen_out != tokenizer.pad_token_id).long()  # [B, P+R]
        labels = gen_out[:, 1:].clone()  # [B, P+R-1]
        B = len(prompts)
        #[B*num_gen,R]
        resp_labels = completion_ids
        #[1,R]
        resp_idx = torch.arange(resp_labels.size(1), device=gen_out.device).unsqueeze(0)
        #[B*num_gen,R]，第0个位置的logits预测第1个token，所以要-1
        logp_pos = prompt_lens.unsqueeze(1) - 1 + resp_idx
        #[B*num_gen,R]
        resp_pad_mask = rollout_result.completion_mask.to(args.device).bool()
        #[B*num_gen]，找真实eos_token,是真实生成的那种
        resp_lengths = resp_pad_mask.sum(dim=1); valid_resp = resp_lengths > 0; eos_mask = resp_labels.eq(tokenizer.eos_token_id) & resp_pad_mask
        #判断每个batch是否有eos，找到eos的位置
        has_eos = eos_mask.any(dim=1); eos_pos = torch.argmax(eos_mask.int(), dim=1)
        resp_lengths = torch.where(has_eos, eos_pos + 1, resp_lengths).long().clamp(min=1)#clamp把张量裁剪到至少为1
        resp_policy_mask = ((resp_idx < resp_lengths.unsqueeze(1)) & resp_pad_mask).float()
        resp_value_mask = resp_policy_mask.clone()

        with torch.no_grad():  # Rollout阶段只需推理获取old_logp和old_values，切断梯度省显存
            critic_for_rollout = critic_model.module if isinstance(critic_model, DistributedDataParallel) else critic_model
            values_seq = critic_for_rollout(input_ids=gen_out, attention_mask=full_mask)
            #[B，R]
            old_resp_values = values_seq.gather(1, logp_pos) * resp_value_mask
            #返回输出的logp
            ref_resp_logp = F.log_softmax(ref_model(input_ids=gen_out, attention_mask=full_mask).logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)
            token_rewards = torch.zeros_like(old_resp_logp)
            last_idx = resp_lengths - 1  # [B]
            token_rewards[torch.arange(B, device=args.device)[valid_resp], last_idx[valid_resp]] += rewards[valid_resp]  # 末尾加外部奖励

            #old_resp_values [R]
            gen_len = old_resp_values.size(1); lastgaelam = torch.zeros(B, device=args.device); advs_rev = []
            for t in reversed(range(gen_len)):
                #GAE过程，GAE其实就是两个公式，别害怕
                nv = old_resp_values[:, t + 1] if t < gen_len - 1 else 0.0
                delta = token_rewards[:, t] + args.gamma * nv - old_resp_values[:, t]
                lastgaelam = delta + args.gamma * args.lam * lastgaelam
                advs_rev.append(lastgaelam)
            #stack是把多个tensor堆叠为一个新tensor
            #advs_rev[::-1] 是把列表反过来 start:end:step
            advantages = torch.stack(advs_rev[::-1], dim=1)  # [B, R]
            returns = advantages + old_resp_values  # [B, R]

            #对adv做归一化
            adv_mean = (advantages * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
            adv_var = ((advantages - adv_mean) ** 2 * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
            #rsqrt等价于1/sqrt
            advantages = (advantages - adv_mean) * torch.rsqrt(adv_var + 1e-8) * resp_policy_mask

        mb_size = max(1, min(args.mini_batch_size, B))
        #ppo 早停标志
        stop_ppo = False
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        kl_sum = 0.0
        kl_ref_sum = 0.0
        clipfrac_sum = 0.0
        aux_loss_sum = 0.0
        log_count = 0
        actor_unwrapped = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
        critic_unwrapped = critic_model.module if isinstance(critic_model, DistributedDataParallel) else critic_model
        for ppo_epoch in range(args.ppo_update_iters):
            if stop_ppo:
                break
            #生成当前batch样本编号的随机排列。后面会用它取数据
            #这里重新算的意义是前面算的old_resp_values是参考值
            b_inds = torch.randperm(B, device=args.device)
            for i in range(0, B, mb_size):
                inds = b_inds[i:i + mb_size]
                
                mb_values_seq = critic_unwrapped(input_ids=gen_out[inds], attention_mask=full_mask[inds])
                mb_resp_values = mb_values_seq.gather(1, logp_pos[inds])

                with autocast_ctx:
                    res = actor_unwrapped(input_ids=gen_out[inds], attention_mask=full_mask[inds])
                    aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)

                mb_resp_logp = F.log_softmax(res.logits[:, :-1], dim=-1).gather(2, labels[inds].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos[inds])

                #policy ratio：π_new / π_old
                log_ratio = mb_resp_logp - old_resp_logp[inds]
                #算出每个位置上平均的kl散度，resp_policy_mask是为了只计算有效位置的kl散度，最后除以有效位置的数量得到平均kl散度
                approx_kl = (0.5 * (log_ratio ** 2) * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
                
                # 同步各卡的 approx_kl，防止某卡 break 而其它卡继续导致 DDP 死锁
                #把approx_kl从计算图中切出来，不参与反向传播
                approx_kl_val = approx_kl.detach().clone()
                if dist.is_initialized():
                    #在多卡之间做平均，然后同步到每张卡上
                    dist.all_reduce(approx_kl_val, op=dist.ReduceOp.AVG)
                    
                if approx_kl_val > args.early_stop_kl:
                    stop_ppo = True
                
                ratio = torch.exp(log_ratio)
                #计算有多少比例的token触发了PPO clip范围
                clipfrac = ((((ratio - 1.0).abs() > args.clip_epsilon).float() * resp_policy_mask[inds]).sum()
                            / resp_policy_mask[inds].sum().clamp(min=1))
                #这里用的是k3 KL estimator，k3_t=r-log(r)-1 其中r为policy ratio
                kl_ref_penalty = ((torch.exp(ref_resp_logp[inds] - mb_resp_logp) - (ref_resp_logp[inds] - mb_resp_logp) - 1.0)
                                  * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
                #计算ppo loss，kl_coef是kl惩罚项的系数，当actor偏离ref model时 要罚多重
                policy_loss = ((torch.max(-advantages[inds] * ratio,
                                          -advantages[inds] * torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon))
                               * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
                               + args.kl_coef * kl_ref_penalty)
                #value loss 是critic的损失，用来训练critic模型，让他学会预测，当前生成到某个token时，后面大概能得到多少reward
                #计算两版本的loss，一个是有裁剪，当critic预测的token价值超过[old value - cliprange, old value + cliprange]范围时，就把预测值裁剪到这个范围内，另一个是直接用critic预测的值，两者取最大，保证更新的稳定性
                #不让v_new相比v_old跳的太大，取max就是在两者选择更猛的惩罚，防止critic更新太猛
                value_loss = 0.5 * (torch.max((mb_resp_values - returns[inds]) ** 2,
                                              (torch.clamp(mb_resp_values, old_resp_values[inds] - args.cliprange_value,
                                                           old_resp_values[inds] + args.cliprange_value) - returns[inds]) ** 2)
                                    * resp_value_mask[inds]).sum() / resp_value_mask[inds].sum().clamp(min=1)

                kl = approx_kl_val
                kl_ref = kl_ref_penalty.detach()

                # 早停时必须保证 forward-backward 闭环，故只截断 loss 不中断 DDP 通信
                if stop_ppo:
                    loss = (policy_loss + args.vf_coef * value_loss + aux_loss) * 0.0
                else:
                    loss = (policy_loss + args.vf_coef * value_loss + aux_loss) / args.accumulation_steps
                
                loss.backward()

                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                kl_sum += kl.item()
                kl_ref_sum += kl_ref.item()
                clipfrac_sum += clipfrac.item()
                aux_loss_sum += aux_loss.item()
                log_count += 1

                grad_accum_step += 1

                if grad_accum_step % args.accumulation_steps == 0:
                    clip_grad_norm_(actor_model.parameters(), args.grad_clip)
                    clip_grad_norm_(critic_model.parameters(), args.grad_clip)
                    #优化器更新参数
                    actor_optimizer.step()
                    critic_optimizer.step()
                    #优化器的学习率调度参数
                    actor_scheduler.step()
                    critic_scheduler.step()
                    actor_optimizer.zero_grad()
                    critic_optimizer.zero_grad()

        if grad_accum_step % args.accumulation_steps != 0:
            clip_grad_norm_(actor_model.parameters(), args.grad_clip)
            clip_grad_norm_(critic_model.parameters(), args.grad_clip)
            actor_optimizer.step()
            critic_optimizer.step()
            actor_scheduler.step()
            critic_scheduler.step()
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
        #定期把训练后的actor权重同步给rollout引擎
        #如果当前到了save_interval间隔或者当前step是本轮的最后一步
        if step % args.save_interval == 0 or step == iters: rollout_engine.update_policy(actor_model)

        if is_main_process():
            critic_loss_val = value_loss_sum / max(log_count, 1)
            reward_val = rewards.mean().item()
            approx_kl_val = kl_sum / max(log_count, 1)
            kl_ref_val = kl_ref_sum / max(log_count, 1)
            clipfrac_val = clipfrac_sum / max(log_count, 1)
            avg_len_val = resp_lengths.float().mean().item()
            actor_lr, critic_lr = actor_optimizer.param_groups[0]['lr'], critic_optimizer.param_groups[0]['lr']

            if wandb is not None:
                wandb.log({
                    "reward": reward_val,
                    "kl_ref": kl_ref_val,
                    "approx_kl": approx_kl_val,
                    "clipfrac": clipfrac_val,
                    "critic_loss": critic_loss_val,
                    "avg_response_len": avg_len_val,
                    "actor_lr": actor_lr,
                    "critic_lr": critic_lr,
                })

            Logger(f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                   f"Reward: {reward_val:.4f}, KL_ref: {kl_ref_val:.4f}, Approx KL: {approx_kl_val:.4f}, "
                   f"ClipFrac: {clipfrac_val:.4f}, Critic Loss: {critic_loss_val:.4f}, "
                   f"Avg Response Len: {avg_len_val:.2f}, Actor LR: {actor_lr:.8f}, Critic LR: {critic_lr:.8f}")

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            actor_model.eval()
            model_suffix = get_model_suffix(lm_config)
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{model_suffix}.pth'
            raw_actor = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            raw_actor = getattr(raw_actor, '_orig_mod', raw_actor)
            actor_state = raw_actor.state_dict()
            torch.save({k: v.half().cpu() for k, v in actor_state.items()}, ckp)
            
            # 使用 lm_checkpoint 保存完整状态（包括 critic）
            lm_checkpoint(lm_config, weight=args.save_weight, model=actor_model, optimizer=actor_optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints',
                         scheduler=actor_scheduler, critic_model=critic_model, 
                         critic_optimizer=critic_optimizer, critic_scheduler=critic_scheduler)
            actor_model.train()
            del actor_state

        del enc, gen_out, completion_ids, responses_text, rewards, full_mask, values_seq, advantages
        del labels, resp_labels, resp_idx, resp_pad_mask, valid_resp, eos_mask, has_eos, eos_pos, resp_lengths, resp_policy_mask, resp_value_mask, old_resp_logp, ref_resp_logp
        del kl, kl_ref, policy_loss, value_loss, loss, token_rewards, returns, old_resp_values, prompt_lens, logp_pos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind PPO (Proximal Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='ppo_actor', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    #这里的batchsize是一次rollout多少条prompt，batch_size影响的是采样质量和统计稳定性，比如reward均值，advantage，kl估计，critic学的的valude，actor更新的方向都会受到影响
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="Actor学习率")
    parser.add_argument("--critic_learning_rate", type=float, default=5e-7, help="Critic学习率")
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
    parser.add_argument('--attention_type', default='gqa', choices=['gqa', 'mha', 'mqa', 'mla'], help="注意力架构")
    parser.add_argument('--use_mla', default=0, type=int, choices=[0, 1], help="兼容旧参数：是否使用MLA注意力架构（0=否，1=是）")
    parser.add_argument('--kv_lora_rank', default=128, type=int, help="MLA的KV压缩秩（仅use_mla=1时生效）")
    parser.add_argument('--max_seq_len', default=768, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1024, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif.jsonl", help="RLAIF数据路径")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO裁剪参数")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function系数")
    parser.add_argument("--kl_coef", type=float, default=0.02, help="KL散度惩罚系数")
    parser.add_argument("--gamma", type=float, default=1.0, help="GAE折扣因子")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda参数")
    parser.add_argument("--cliprange_value", type=float, default=0.2, help="Value function裁剪范围")
    parser.add_argument("--ppo_update_iters", type=int, default=2, help="同一批rollout重复更新次数")
    parser.add_argument("--early_stop_kl", type=float, default=0.25, help="PPO early stop 的 KL 阈值")
    #mini_batch_size是每次ppo更新时，计算loss的mini-batch大小，通常小于batch_size，mini_batch_size解决的是更新显存问题
    parser.add_argument("--mini_batch_size", type=int, default=2, help="PPO每次更新的minibatch大小")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-PPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--debug_mode", action="store_true", help="是否打印训练调试采样")
    parser.add_argument("--debug_interval", type=int, default=20, help="debug模式下每隔多少step打印一次采样")
    parser.add_argument("--thinking_ratio", type=float, default=0.9, help="按概率开启thinking（0.0~1.0）")
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"], help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_ppo", help="SGLang共享存储路径")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = build_lm_config(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        attention_type=resolve_attention_type(args),
        kv_lora_rank=args.kv_lora_rank
    )
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
        wandb_run_name = f"MiniMind-PPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 初始化模型和数据 ==========
    base_weight = args.from_weight
    # Actor模型
    actor_model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    model_suffix = get_model_suffix(lm_config)
    ckp = f'{args.save_dir}/{base_weight}_{lm_config.hidden_size}{model_suffix}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    CriticModel = _make_critic_model(MiniMindMLAForCausalLM if isinstance(lm_config, MiniMindMLAConfig) else MiniMindForCausalLM)
    critic_model = CriticModel(lm_config)
    critic_model.load_state_dict(state_dict, strict=False)
    critic_model = critic_model.to(args.device)
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
    # Rollout引擎
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=actor_model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=(args.max_seq_len + args.max_gen_len), thinking_ratio=args.thinking_ratio)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.AdamW(critic_model.parameters(), lr=args.critic_learning_rate)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    #在计算一个batch能分出来几个minibatchsize，math.ceil向上取整
    mb_factor = max(1, math.ceil(args.batch_size / args.mini_batch_size))
    #优化器更新的次数
    total_optimizer_steps = math.ceil(iters * args.epochs * args.ppo_update_iters * mb_factor / args.accumulation_steps)
    #让learning_rate按余弦曲线下降到最小值，T_max是最大更新步数（学习率从初始值降到最小值需要多少个step），eta_min是最小学习率，让学习率按照余弦曲线下降到eta_min
    actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    critic_scheduler = CosineAnnealingLR(critic_optimizer, T_max=total_optimizer_steps, eta_min=args.critic_learning_rate / 10)
    log_training_setup(args, lm_config, stage="ppo", dataset_len=len(train_ds), iters=iters,
                       tokens_per_sample=args.max_seq_len + args.max_gen_len,
                       extra={
                           "rollout_engine": args.rollout_engine,
                           "ppo_update_iters": args.ppo_update_iters,
                           "mini_batch_size": args.mini_batch_size,
                           "kl_coef": args.kl_coef,
                           "total_optimizer_steps": total_optimizer_steps,
                       })

    start_epoch, start_step = 0, 0
    if ckp_data:
        actor_model.load_state_dict(ckp_data['model'])
        critic_model.load_state_dict(ckp_data['critic_model'])
        actor_optimizer.load_state_dict(ckp_data['optimizer'])
        critic_optimizer.load_state_dict(ckp_data['critic_optimizer'])
        actor_scheduler.load_state_dict(ckp_data['scheduler'])
        critic_scheduler.load_state_dict(ckp_data['critic_scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        #会把actor_model包一层编译后的模型，让forward更快
        #torch.compile()包装后的模型，原始模型通常都在_orig_mod里
        actor_model = torch.compile(actor_model)
        Logger('torch.compile enabled')
        #把当前的actor同步给rollout engine
        rollout_engine.update_policy(actor_model)
    if dist.is_initialized():
        #DDP包装
        actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank])
        critic_model = DistributedDataParallel(critic_model, device_ids=[local_rank])
    #将DDP包装后的模型同步给rollout engine
    rollout_engine.update_policy(actor_model)
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            ppo_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, actor_scheduler, critic_scheduler, reward_model, start_step, wandb, use_sglang = (args.rollout_engine == "sglang"))
        else:
            ppo_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, actor_scheduler, critic_scheduler, reward_model, 0, wandb, use_sglang = (args.rollout_engine == "sglang"))
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()
