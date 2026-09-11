# 第 28 课：总复盘与小项目

这一课不引入新算法。它把前 27 课收束成一个可检查、可复述、可展示的小项目：你要能从 MiniMind 源码出发，解释一条 LLM 链路如何从数据走到模型、训练、后训练、Agent，再走到推理接口。

本节的重点不是“把所有命令都跑一遍”，而是建立一套最终验收标准：哪些源码必须能读懂，哪些张量必须能说清楚，哪些教学版实现必须能和原项目对齐，哪些内容可以写进 portfolio。

## 目录

- [0. 本节主线](#l28-mainline)
- [1. 本节要懂的 6 个复盘原则](#l28-principles)
- [2. 完整原理：一条 LLM 链路如何闭环](#l28-complete-principle)
- [3. 源码总地图](#l28-source-map)
- [4. MiniMind 源码证据复盘](#l28-source-evidence)
- [5. 最终小项目](#l28-capstone)
- [6. 验收实验](#l28-experiments)
- [7. Portfolio 记录](#l28-portfolio)
- [8. 本节检查](#l28-check)
- [9. 课程结束后的下一步](#l28-next)

<a id="l28-mainline"></a>
## 0. 本节主线

MiniMind 这门课最终要你能独立讲清楚：

```text
jsonl 样本
-> tokenizer / chat template
-> input_ids / labels / masks
-> MiniMindForCausalLM
-> logits / next-token loss
-> optimizer / lr / amp / checkpoint
-> pretrain / SFT / LoRA / distillation
-> DPO / PPO / GRPO / CISPO
-> Agent rollout / tool observation / trajectory reward
-> transformers model directory
-> CLI / OpenAI API / WebUI
```

一句话：

```text
学完 MiniMind，不是记住每个脚本参数，而是能解释每个阶段把什么数据形态变成了什么数据形态，以及这个转换在哪段源码里发生。
```

最终小项目的产物是：

```text
1. 一张主链路图；
2. 一组 tiny 实验输出；
3. 一份教学版实现与原项目差异说明；
4. 一份可放进 portfolio 的项目总结。
```

<a id="l28-principles"></a>
## 1. 本节要懂的 6 个复盘原则

| 原则 | 要理解什么 | 源码位置 |
|---|---|---|
| 数据不是文本，数据会变成训练张量 | pretrain/SFT/DPO/RL 的样本格式不同，最终都要变成 token、label、mask 或 rollout 输入 | `dataset/lm_dataset.py:37-252` |
| CausalLM 的核心是不变的 next-token prediction | 不管 pretrain、SFT、DPO、RL，底层都离不开 logits 对下一个 token 的 logprob | `model/model_minimind.py:234-255`, `trainer/rollout_engine.py:24-36` |
| 模型结构是一组可追踪的 shape 变换 | Embedding、RoPE、Attention、FFN/MoE、LM head 都能用 shape 串起来 | `model/model_minimind.py:10-255` |
| 训练脚本主要组织状态变化 | optimizer、lr、amp、grad accumulation、checkpoint 是训练工程状态，不是模型结构 | `trainer/train_pretrain.py:24-80`, `trainer/trainer_utils.py:63-139` |
| 后训练改变的是优化目标，不是 CausalLM 本质 | DPO、PPO、GRPO/CISPO 都围绕 logprob、reward、reference、mask 重组 loss | `trainer/train_dpo.py:34-49`, `trainer/train_ppo.py:79-299`, `trainer/train_grpo.py:71-196` |
| 部署侧复用训练时的协议 | chat template、special tokens、tool_call、thinking、decode 都必须和训练协议一致 | `eval_llm.py:71-88`, `scripts/serve_openai_api.py:83-102`, `scripts/web_demo.py:350-413` |

学完本节，你应该能用自己的话解释这些变量：

```text
input_ids:
  模型真正看到的 token id 序列。

labels:
  causal LM loss 的目标 token；不训练的位置用 -100。

attention_mask:
  哪些 token 是有效上下文，哪些是 padding。

loss_mask / completion_mask / response_mask:
  哪些目标 token 参与某个训练目标。

logits:
  每个位置对词表中下一个 token 的未归一化分数。

per_token_logps:
  从 logits 里 gather 出来的目标 token logprob。

old_logps / ref_logps / policy_logps:
  rollout policy、reference model、current policy 对同一批 token 的 logprob。

reward / advantage:
  外部评价信号，以及它进入 policy loss 前的权重形态。
```

<a id="l28-complete-principle"></a>
## 2. 完整原理：一条 LLM 链路如何闭环

### 2.1 数据层：所有训练目标先从样本格式开始分化

MiniMind 不是只有一种数据集：

```text
Pretrain:
  {"text": "..."}

SFT:
  {"conversations": [{"role": "...", "content": "..."}]}

DPO:
  {"chosen": [...], "rejected": [...]}

RLAIF / Agent:
  prompt 或 messages，用于在线 rollout。
```

分化发生在数据集类里。不同数据集返回不同字段：

```text
PretrainDataset:
  input_ids, labels

SFTDataset:
  input_ids, labels，其中只训练 assistant 区域

DPODataset:
  chosen/rejected 的 x/y/mask

RLAIFDataset:
  prompt，用于后续生成 response

AgentRLDataset:
  messages/tools/gt，用于多轮工具 rollout
```

所以第一条能力标准是：

```text
看到任意训练脚本，先问它的 DataLoader 每个 batch 到底是什么结构。
```

### 2.2 模型层：底层永远是 decoder-only CausalLM

MiniMind 的模型主线是：

```text
input_ids
-> embedding
-> N 个 decoder block
   -> RMSNorm
   -> Attention + RoPE + causal mask / KV cache
   -> residual
   -> RMSNorm
   -> FFN 或 MoE
   -> residual
-> final RMSNorm
-> lm_head
-> logits
```

只要你能解释：

```text
[B, T] input_ids
-> [B, T, H] hidden_states
-> [B, T, vocab] logits
```

后面的 pretrain、SFT、DPO、RL 才有共同地基。

### 2.3 训练层：loss 都在回答“哪些 token 应该提高概率”

Pretrain 和 SFT 的差异不是模型不同，而是 loss 区域不同：

```text
Pretrain:
  训练整段普通文本的下一个 token。

SFT:
  仍然是 next-token prediction，但只训练 assistant 回复区域。
```

DPO、PPO、GRPO 也没有脱离 logprob：

```text
DPO:
  比较 chosen 和 rejected 的 sequence logprob 差异。

PPO:
  用 reward / value / advantage 更新 response token 的 policy logprob。

GRPO:
  用同 prompt 多条 response 的 group reward 标准化得到 advantage。

CISPO:
  使用 CISPO 分支调整 ratio / policy logprob 的使用方式。
```

所以第三条能力标准是：

```text
任何训练目标，都要能说清楚：
哪些 token 参与 loss？
logprob 来自哪里？
mask 怎么对齐？
reference / old policy 是否参与？
```

### 2.4 Agent 层：工具执行是环境反馈，不是模型直接学标签

Agentic RL 不是 SFT labels 训练。它的核心是环境循环：

```text
messages + tools
-> policy 生成 <tool_call>
-> Python 执行工具
-> tool observation 拼回 messages
-> policy 继续生成
-> 整条 trajectory 得 reward
-> GRPO/CISPO 更新 policy
```

这里最关键的边界是：

```text
policy token:
  模型生成，参与 policy loss。

tool observation token:
  环境产生，进入上下文，但不参与 policy loss。
```

这就是第 26 课一直强调 `response_mask`、`response_old_logps`、`completion_mask` 对齐的原因。

### 2.5 部署层：把同一套文本协议接到不同入口

部署侧没有重新定义模型能力。它只是把同一套训练协议接到不同外壳：

```text
CLI:
  终端输入输出。

OpenAI API:
  HTTP request / response，支持 stream、reasoning_content、tool_calls。

WebUI:
  浏览器交互，维护 session history，显示 thinking 和 tool result。

Transformers / vLLM / Ollama:
  依赖 config、tokenizer、chat_template、weights 的模型目录。
```

所以最后一条能力标准是：

```text
部署不是魔法。部署只是：
load model/tokenizer
-> render prompt
-> generate
-> decode
-> package response
```

<a id="l28-source-map"></a>
## 3. 源码总地图

按学习目的看源码，而不是按目录顺序看：

| 目的 | 先读 | 再读 | 最后读 |
|---|---|---|---|
| 看懂 tokenizer / template | `model/tokenizer_config.json`, `minimind-3/chat_template.jinja` | `dataset/lm_dataset.py` | `eval_llm.py` |
| 看懂模型结构 | `MiniMindConfig` | `MiniMindModel`, `MiniMindBlock`, `Attention` | `MiniMindForCausalLM.forward`, `generate` |
| 看懂 pretrain/SFT | `PretrainDataset`, `SFTDataset` | `train_pretrain.py`, `train_full_sft.py` | `trainer_utils.py` |
| 看懂 LoRA / distillation | `model_lora.py` | `train_lora.py` | `train_distillation.py` |
| 看懂 DPO | `DPODataset` | `train_dpo.py:dpo_loss` | `course/impl/train_dpo_impl.py` |
| 看懂 PPO/GRPO | `rollout_engine.py` | `train_ppo.py`, `train_grpo.py` | `course/impl/core/ppo.py`, `course/impl/core/grpo.py` |
| 看懂 Agent | `AgentRLDataset` | `train_agent.py:rollout_single` | `train_agent.py:rl_train_epoch` |
| 看懂部署 | `convert_model.py` | `serve_openai_api.py` | `web_demo.py`, `chat_api.py` |

<a id="l28-source-evidence"></a>
## 4. MiniMind 源码证据复盘

### 4.1 数据集决定 batch 的语义

File: `dataset/lm_dataset.py:37-252`

Read this to understand: 每个训练阶段从 DataLoader 里拿到的 batch 字段不同，后面的训练脚本必须围绕这些字段组织。

Code/config/template excerpt:

```python
class PretrainDataset(Dataset):
    ...
    def __getitem__(self, index):
        ...
        return input_ids, labels

class SFTDataset(Dataset):
    ...
    def __getitem__(self, index):
        ...
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

class DPODataset(Dataset):
    ...
    def __getitem__(self, index):
        ...
        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

class AgentRLDataset(Dataset):
    ...
    def __getitem__(self, index):
        ...
        return {'messages': messages, 'tools': tools, 'gt': sample['gt']}
```

This code shows:

- Pretrain/SFT 直接返回 supervised training 张量。
- DPO 返回 chosen/rejected 两条路径。
- Agent RL 不返回 labels，而是返回在线 rollout 需要的环境输入。

### 4.2 CausalLM forward 是所有训练目标的共同地基

File: `model/model_minimind.py:234-255`

Read this to understand: MiniMind 的语言模型头如何把 hidden states 变成 logits，并在有 labels 时计算 next-token loss。

Code/config/template excerpt:

```python
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    ...
    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
```

This code shows:

- `lm_head` 把 `[B, T, H]` 变成 `[B, T, vocab]`。
- loss 使用 `logits[..., :-1, :]` 预测 `labels[..., 1:]`。
- `ignore_index=-100` 是 SFT mask 生效的位置。

### 4.3 训练脚本组织 optimizer、lr、checkpoint

File: `trainer/train_pretrain.py:24-80`

Read this to understand: 一个基础训练 epoch 如何把 batch、forward、loss、backward、clip、step 串起来。

Code/config/template excerpt:

```python
def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    for step, (X, Y) in enumerate(loader, start=start_step + 1):
        X = X.to(args.device)
        Y = Y.to(args.device)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        with autocast_ctx:
            res = model(X, labels=Y)
            loss = res.loss
            loss = loss / args.accumulation_steps
        loss.backward()
        if step % args.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
```

This code shows:

- batch 先搬到 device。
- lr 每 step 按 schedule 更新。
- loss 会除以 gradient accumulation steps。
- 反向、裁剪、优化器 step 是训练工程的核心状态变化。

File: `trainer/trainer_utils.py:63-139`

Read this to understand: checkpoint 不只是保存模型参数，还保存 optimizer、epoch、step、world size 等恢复训练需要的状态。

Code/config/template excerpt:

```python
resume_data = {
    'model': state_dict,
    'optimizer': optimizer.state_dict(),
    'epoch': epoch,
    'step': step,
    'world_size': dist.get_world_size() if dist.is_initialized() else 1,
    'wandb_id': wandb_id
}
...
if os.path.exists(resume_path):
    ckp_data = torch.load(resume_path, map_location='cpu')
    saved_ws = ckp_data.get('world_size', 1)
    current_ws = dist.get_world_size() if dist.is_initialized() else 1
    if saved_ws != current_ws:
        ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
    return ckp_data
```

This code shows:

- 可恢复训练必须保存 optimizer 和 step，不只是模型权重。
- world size 变化时会调整 step。
- 这是训练工程状态，不是模型结构。

### 4.4 DPO 把偏好学习写成 logprob 差异

File: `trainer/train_dpo.py:34-49`

Read this to understand: DPO 的 loss 如何比较 chosen/rejected 在 policy 相对 reference 上的提升。

Code/config/template excerpt:

```python
def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    seq_lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
    ref_log_probs = (ref_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    policy_log_probs = (policy_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    chosen_ref, rejected_ref = ref_log_probs[::2], ref_log_probs[1::2]
    chosen_policy, rejected_policy = policy_log_probs[::2], policy_log_probs[1::2]
    pi_logratios = chosen_policy - rejected_policy
    ref_logratios = chosen_ref - rejected_ref
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()
```

This code shows:

- DPO 先用 mask 聚合 response token 的 logprob。
- chosen/rejected 必须成对排列。
- policy 比 reference 更偏向 chosen 时，loss 会变小。

### 4.5 PPO/GRPO 都围绕 old/current/ref logprob

File: `trainer/rollout_engine.py:24-36`

Read this to understand: token logprob 的核心操作是 `log_softmax + gather`，后面 PPO/GRPO/Agent 都复用这个思想。

Code/config/template excerpt:

```python
def compute_per_token_logps(model, input_ids, n_keep, attention_mask=None):
    logits = unwrapped(input_ids, attention_mask=attention_mask, logits_to_keep=n_keep + 1).logits[:, :-1, :]
    per_token_logps = []
    for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
        per_token_logps.append(
            torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1)
        )
    return torch.stack(per_token_logps)
```

This code shows:

- `logits[:, :-1, :]` 对齐 next-token targets。
- `input_ids[:, -n_keep:]` 是要取 logprob 的 response/completion token。
- 返回 shape 是 `[B, n_keep]`。

File: `trainer/train_grpo.py:121-143`

Read this to understand: GRPO/CISPO 如何把 group reward、ratio、reference KL 合成 policy loss。

Code/config/template excerpt:

```python
grouped_rewards = rewards.view(-1, args.num_generations)
mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
advantages = (rewards - mean_r) / (std_r + 1e-4)

kl_div = ref_per_token_logps - per_token_logps
per_token_kl = torch.exp(kl_div) - kl_div - 1
ratio = torch.exp(per_token_logps - old_per_token_logps)
...
per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)
policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)).mean()
```

This code shows:

- group reward 标准化得到 sequence-level advantage。
- `ratio` 比较 current policy 和 rollout old policy。
- `per_token_kl` 比较 current policy 和 frozen reference。
- `completion_mask` 决定哪些 token 参与 loss。

### 4.6 Agent rollout 把环境 observation 接回上下文

File: `trainer/train_agent.py:98-157`

Read this to understand: Agentic RL 的关键不只是 loss，而是多轮环境循环如何构造 trajectory。

Code/config/template excerpt:

```python
for turn in range(max_turns):
    context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools, open_thinking=open_thinking)
    inputs = tokenizer(context, return_tensors="pt", add_special_tokens=False).to(device)
    rollout_result = rollout_engine.rollout(...)
    new_ids = rollout_result.completion_ids[0].tolist()
    new_logps = rollout_result.per_token_logps[0].tolist()
    response_ids.extend(new_ids)
    response_mask.extend([1] * len(new_ids))
    response_old_logps.extend(new_logps)
    calls = parse_tool_calls(new_text)
    if not calls:
        break
    messages.append({"role": "assistant", "content": new_text})
    ...
    messages.append({"role": "tool", "content": result_str})
    observe_context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=not unfinished, tools=tools, open_thinking=open_thinking)
    obs_delta = observe_ids[current_len:]
    response_ids.extend(obs_delta)
    response_mask.extend([0] * len(obs_delta))
    response_old_logps.extend([0.0] * len(obs_delta))
```

This code shows:

- policy 生成 token 的 mask 是 1。
- tool observation delta 会进入 `response_ids`，因为它是后续上下文。
- observation token 的 mask 是 0，old logprob 是占位 0.0。
- Agent 的难点在 trajectory 构造，而不只是 GRPO/CISPO loss。

### 4.7 部署复用同一套 template 和 generate

File: `eval_llm.py:71-88`

Read this to understand: 最小部署闭环就是 messages、template、tokenizer、generate、decode。

Code/config/template excerpt:

```python
conversation.append({"role": "user", "content": prompt})
inputs = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, open_thinking=bool(args.open_thinking))
inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)
generated_ids = model.generate(
    inputs=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=args.max_new_tokens,
    do_sample=True,
    streamer=streamer,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
```

This code shows:

- 部署侧仍然依赖 chat template。
- 模型生成的是完整 `[prompt + response]` token 序列。
- decode 时只取 prompt 之后的新 token。

<a id="l28-capstone"></a>
## 5. 最终小项目

小项目名称：

```text
MiniMind 源码驱动 LLM 链路复盘
```

目标不是训练出强模型，而是产出一份能证明你真的看懂链路的报告。

### 5.1 必交产物

产物 A：主链路图

```text
dataset jsonl
-> tokenizer/chat template
-> input_ids/labels/masks
-> MiniMindForCausalLM
-> logits/loss/logprob
-> optimizer/checkpoint
-> pretrain/SFT/LoRA/DPO/PPO/GRPO/Agent
-> eval/API/WebUI
```

每个箭头旁边写一个源码位置。例如：

```text
SFT labels: dataset/lm_dataset.py:SFTDataset.generate_labels
next-token loss: model/model_minimind.py:MiniMindForCausalLM.forward
GRPO ratio: trainer/train_grpo.py:grpo_train_epoch
```

产物 B：实验记录

至少跑 6 个实验，每个实验记录：

```text
命令：
关键输出：
这个实验验证了哪条源码原则：
我曾经容易错在哪里：
```

产物 C：教学版实现说明

对 `course/impl/` 写一份差异说明：

```text
我们手写实现了什么：
对齐 MiniMind 哪段源码：
省略了什么工程细节：
为什么可以省略：
如果要真实训练，必须回到原项目哪些能力：
```

产物 D：portfolio 总结

写进 `course/portfolio/`：

```text
progress.md:
  各阶段是否跑通。

experiments.md:
  关键实验命令和输出。

implementation.md:
  教学版实现和原项目差异。

resume_bullets.md:
  一段准确不过度包装的项目描述。
```

### 5.2 推荐报告结构

```markdown
# MiniMind 源码驱动 LLM 链路复盘报告

## 1. 项目目标
用 MiniMind 源码解释并验证一个小型 LLM 从数据到部署的完整链路。

## 2. 主调用链
画出 jsonl -> tokenizer -> model -> loss -> optimizer -> post-training -> deploy。

## 3. 关键源码证据
列 8-12 个源码点，每个点说明它证明了什么。

## 4. 实验记录
记录 tiny 实验命令、shape、loss、mask、logprob、reward 等输出。

## 5. 教学版实现
说明 course/impl 实现了哪些核心模块，以及和原项目差异。

## 6. 仍未覆盖的工程能力
列出分布式训练、真实大数据训练、SGLang/vLLM 服务化、线上监控等。

## 7. 下一步改进
选择 1-2 个真正要继续做的方向。
```

### 5.3 不合格的小项目长什么样

```text
只贴 README 命令；
只说“实现了 Transformer”但没有源码位置；
只展示生成效果，不解释 input_ids / labels / masks；
只讲 PPO/GRPO 概念，不说明 old/current/ref logprob；
把 WebUI 当成模型能力本身；
没有任何 tiny 实验输出。
```

合格标准是：

```text
每个结论都能回到一个源码位置；
每个实验都能验证一个变量流；
每个实现差异都能说明为什么省略。
```

<a id="l28-experiments"></a>
## 6. 验收实验

### 6.1 环境与资源检查

命令：

```bash
cd /home/sun/minimind
python course/labs/check_project_readiness.py
```

记录：

```text
torch =
transformers =
cuda =
pretrain_t2t_mini =
sft_t2t_mini =
minimind_3_safetensors =
```

这个实验验证：当前工作区能跑哪些课程实验，哪些真实训练资源仍然缺失。

### 6.2 tokenizer / SFT labels / loss shift

命令：

```bash
cd /home/sun/minimind
python course/labs/inspect_tokenizer.py
python course/labs/inspect_sft_dataset.py
python course/labs/trace_loss_shift.py
```

记录：

```text
special token ids =
chat template 片段 =
assistant label 区域 =
logits shift =
ignore_index=-100 的位置 =
```

这个实验验证：文本如何变成 token，哪些 token 参与 SFT loss，以及 causal LM 为什么右移。

### 6.3 模型结构与 generation

命令：

```bash
cd /home/sun/minimind
python course/labs/inspect_model_config_params.py
python course/labs/trace_block_shapes.py
python course/labs/trace_attention_shapes.py
python course/labs/trace_rope_kv_cache.py
```

记录：

```text
hidden_size =
num_layers =
q/k/v shape =
causal mask shape =
kv cache length growth =
```

这个实验验证：模型结构不是黑盒，所有核心模块都能用 shape 追踪。

### 6.4 pretrain / SFT / train loop

命令：

```bash
cd /home/sun/minimind
python course/labs/compare_pretrain_sft.py
python course/labs/trace_pretrain_step.py
```

记录：

```text
pretrain labels =
sft labels =
loss before step =
grad norm =
参数是否更新 =
```

这个实验验证：训练脚本不是只调用 `model.train()`，而是明确组织 batch、loss、backward、optimizer step。

### 6.5 DPO / PPO / GRPO / Agent

命令：

```bash
cd /home/sun/minimind
python course/labs/trace_dpo_dataset_reference.py
python course/labs/trace_dpo_loss.py
python course/labs/trace_ppo_batch_flow.py
python course/labs/trace_grpo_cispo_loss.py
python course/labs/trace_reward_logprob_kl.py
python course/labs/trace_agent_tool_reward.py
```

记录：

```text
chosen/rejected mask =
dpo logratio =
old/current/ref logprob shape =
ratio =
reference KL penalty =
reward =
advantages =
tool reward case 对比 =
```

这个实验验证：后训练算法的共同底层是 logprob、mask、reward、reference。

### 6.6 教学版实现测试

命令：

```bash
cd /home/sun/minimind
pytest course/impl/tests -q
```

记录：

```text
通过的测试数 =
失败的测试名 =
失败原因 =
对应源码原则 =
```

这个实验验证：手写实现不是孤立练习，而是要通过对齐测试。

如果当前环境没有 `pytest`，先只记录：

```text
pytest missing
```

不要把缺少测试工具误判为模型实现错误。

### 6.7 本地 transformers 推理

命令：

```bash
cd /home/sun/minimind
python course/labs/run_minimind3_once.py --device cpu --max_new_tokens 32 --prompt "MiniMind 是什么？"
python course/labs/trace_deploy_surfaces.py
```

记录：

```text
input_ids.shape =
generated_ids.shape =
new_tokens =
contains_tools_tag =
has_open_think_prompt =
tool_calls =
```

这个实验验证：训练协议最终能被 transformers 模型目录和部署接口复用。

<a id="l28-portfolio"></a>
## 7. Portfolio 记录

### 7.1 `course/portfolio/progress.md`

建议更新成这种状态：

```markdown
| 阶段 | 状态 | 产物 | 备注 |
|---|---|---|---|
| 模型结构 | 已完成 | `core/model_parts.py`, `core/causal_lm.py`, `core/generation.py` | 已通过 shape / loss / cache 相关实验 |
| Pretrain | 已完成 | `core/train_loop.py`, `train_pretrain_impl.py` | 已跑通 tiny pretrain step |
| SFT | 已完成 | `core/datasets.py`, `train_sft_impl.py` | 已验证 assistant labels mask |
| LoRA | 已完成 | `core/lora.py`, `train_lora_impl.py` | 已验证只更新 LoRA 参数 |
| DPO | 已完成 | `train_dpo_impl.py` | 已验证 chosen/rejected logratio |
| PPO/GRPO | 已完成 | `core/ppo.py`, `core/grpo.py`, `train_grpo_impl.py` | 已验证 reward/logprob/KL/advantage |
| Agent/Deploy | 已观察验证 | labs + 原项目脚本 | 复用原项目工程外围 |
```

只有实际跑通过的阶段才写“已完成”。没跑过就写“已实现，待验收”。

### 7.2 `course/portfolio/experiments.md`

每条实验记录建议这样写：

```markdown
日期：YYYY-MM-DD
阶段：DPO
命令：python course/labs/trace_dpo_loss.py
关键输出：
- chosen_logp =
- rejected_logp =
- dpo_loss =
结论：
这个实验验证 DPO loss 比较的是 policy 相对 reference 的 chosen/rejected 偏好差。
问题：
一开始容易把 token 平均 logprob 和 sequence sum logprob 混在一起。
```

### 7.3 `course/portfolio/implementation.md`

建议每个模块用同一模板：

```markdown
## 模块：GRPO/CISPO

我们实现了什么：
- group reward 标准化
- ratio = exp(policy_logps - old_logps)
- reference KL penalty
- GRPO/CISPO loss 分支

对齐的原源码：
- `trainer/train_grpo.py:121-143`
- `course/impl/core/grpo.py`

省略了什么：
- 真实 reward model
- rollout server
- DDP
- wandb/swanlab

为什么可以省略：
- 教学版目标是验证 loss 公式和 mask 规约，不是复刻完整训练系统。

什么时候必须回到原源码方案：
- 需要真实在线生成、多卡训练、长时间 checkpoint、真实数据集。
```

### 7.4 `course/portfolio/resume_bullets.md`

候选表达：

```text
基于 MiniMind 源码完成小型 LLM 全链路学习与教学版复现，覆盖 tokenizer/chat template、Decoder-only CausalLM、RoPE/KV cache、Pretrain/SFT、LoRA、DPO、PPO/GRPO/CISPO、Agent tool-use rollout 与 OpenAI-compatible 推理接口。手写实现核心模型部件、训练循环、SFT mask、LoRA、DPO/GRPO/PPO loss，并通过 tiny 实验验证 input_ids/labels/mask/logprob/reward/KL 等关键变量流。
```

不要写：

```text
训练了一个媲美大模型的 LLM。
```

除非你真的完成了大规模训练和评测。

<a id="l28-check"></a>
## 8. 本节检查

1. 给你一个 `trainer/train_*.py`，你第一步应该找哪个 dataset？为什么？
2. 为什么 SFT、DPO、PPO、GRPO 都离不开 next-token logprob？
3. `labels=-100`、`loss_mask`、`completion_mask`、`response_mask` 分别解决什么问题？
4. 为什么 Agent tool observation 要进入上下文，但不能像 policy token 一样参与 policy loss？
5. 教学版 `course/impl/` 省略了哪些真实工程能力？这些省略什么时候会变成问题？
6. transformers 模型目录为什么必须包含 tokenizer / chat template，而不只是权重文件？

<a id="l28-next"></a>
## 9. 课程结束后的下一步

课程结束后，建议只选一个方向继续，不要同时开太多分支。

方向 A：把教学版实现跑得更完整

```text
目标：
  pytest course/impl/tests -q 全部通过；
  tiny pretrain -> tiny SFT -> tiny DPO/GRPO 能顺序跑通。

重点：
  保持小而可解释，不追求模型效果。
```

方向 B：做一次真实数据小训练

```text
目标：
  下载 pretrain_t2t_mini / sft_t2t_mini；
  用小 batch 跑完整 pretrain/SFT；
  记录 loss 曲线、生成样例和资源消耗。

重点：
  先验证训练系统稳定，不要急着调效果。
```

方向 C：深入一个后训练算法

```text
目标：
  只选 DPO、PPO、GRPO 或 Agent RL 之一；
  写出公式、源码、实验、失败案例和改进方向。

重点：
  把 logprob/mask/reward/reference 对齐讲清楚。
```

方向 D：做部署小闭环

```text
目标：
  用本地 minimind-3 启动 OpenAI-compatible API；
  写一个客户端请求；
  验证 thinking/tool_call 字段。

重点：
  分清模型生成协议和 API 包装协议。
```

无论选哪条，继续遵守这条规则：

```text
先说明变量流，再解释源码，再跑最小实验，最后才谈效果。
```
