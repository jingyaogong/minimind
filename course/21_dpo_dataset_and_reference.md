# 第 21 课：DPO 数据和 reference model

这一课只解决一个问题：MiniMind 的 DPO 训练如何把 `chosen/rejected` 偏好样本变成模型可计算的 logprob，并为什么需要一个冻结的 reference model。

## 目录

- [0. 本节主线](#l21-mainline)
- [1. 本节要懂的 6 个原理](#l21-principles)
- [2. DPO 数据和 reference model 完整原理](#l21-complete-principle)
- [3. 源码阅读顺序图](#l21-reading-order)
- [4. MiniMind 源码走读](#l21-source-walkthrough)
- [5. 本节必须会写 / 暂时不要求](#l21-must-write)
- [6. 实验验证](#l21-experiment)
- [7. 阶段组装](#l21-stage-assembly)
- [8. 本节检查](#l21-check)
- [9. 下一课](#l21-next)

<a id="l21-mainline"></a>
## 0. 本节主线

DPO 数据和 reference model 的流程是：

```text
dpo.jsonl 里每条样本有 chosen / rejected
-> chosen 和 rejected 都是完整对话
-> DPODataset 分别渲染 chat template
-> 分别 tokenize + padding
-> 分别切成 x=input_ids[:-1], y=input_ids[1:]
-> 分别生成 assistant 区域 mask
-> train_epoch 把 chosen 和 rejected 拼到同一个 batch
-> policy model 计算 token logprob
-> reference model 计算 token logprob，但不更新
-> 第 22 课再用这些 logprob 计算 DPO loss
```

这节课先不推完整 DPO loss。你只需要抓住四件事：

```text
1. DPO 数据是一对回复：chosen 更好，rejected 更差。
2. policy model 是正在训练的模型。
3. reference model 是冻结的基准模型。
4. 两个模型都要对 chosen/rejected 算 logprob，后面才能比较偏好。
```

一句话：

```text
DPO 不是让模型直接模仿 chosen，而是让 policy 相对 reference 更偏向 chosen、远离 rejected。
```

<a id="l21-principles"></a>
## 1. 本节要懂的 6 个原理

| 原理 | 要理解什么 | 源码位置 |
|---|---|---|
| DPO 数据是偏好对 | `chosen` 是更好回复，`rejected` 是较差回复 | `README.md:461-478` |
| DPODataset 分别处理两条对话 | chosen/rejected 各自渲染 chat template 并 tokenize | `dataset/lm_dataset.py:135-153` |
| DPO 样本已经提前 shift | dataset 返回 `x=input_ids[:-1]`、`y=input_ids[1:]`、`mask=loss_mask[1:]` | `dataset/lm_dataset.py:155-174` |
| mask 只保留 assistant 区域 | DPO 只比较回复 token，不训练 user/system prompt | `dataset/lm_dataset.py:176-192` |
| policy/ref 从同一权重开始 | policy 被训练，ref 冻结作为基准 | `trainer/train_dpo.py:182-194` |
| chosen/rejected 会拼到同一个 batch | 脚本把 chosen 放前半，rejected 放后半 | `trainer/train_dpo.py:57-83` |

学完本节，你应该能说明：

```text
DPODataset 返回的六个 tensor 各是什么；
为什么 DPO 里 x/y 不再需要模型 forward 内部 shift；
reference model 为什么和 policy 初始权重相同但不更新；
为什么 train_epoch 要把 chosen/rejected 在 batch 维拼接。
```

<a id="l21-complete-principle"></a>
## 2. DPO 数据和 reference model 完整原理

### 2.1 chosen/rejected 是一对完整答案

SFT 数据通常是一条对话：

```json
{"conversations": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}
```

DPO 数据是一对对话：

```json
{
  "chosen": [
    {"role": "user", "content": "Q"},
    {"role": "assistant", "content": "good answer"}
  ],
  "rejected": [
    {"role": "user", "content": "Q"},
    {"role": "assistant", "content": "bad answer"}
  ]
}
```

它表达的是：

```text
同一个或相近 prompt 下，chosen 比 rejected 更符合偏好。
```

DPO 不需要 reward model 在线打分。偏好信息已经写在静态数据里：

```text
chosen > rejected
```

### 2.2 DPO 仍然要把文本变成 token 序列

chosen 和 rejected 都是 messages 列表，不能直接喂给模型。

它们仍然要经过：

```text
messages
-> apply_chat_template
-> prompt string
-> tokenizer
-> input_ids
```

和 SFT 一样，prompt 里会包含 user、assistant、特殊分隔符等结构。不同的是，DPO 一条样本会得到两条 token 序列：

```text
chosen_input_ids
rejected_input_ids
```

### 2.3 DPO dataset 直接返回 x/y/mask

第 10 课讲过，causal LM 训练要做 shift：

```text
logits[..., :-1, :] 对齐 labels[..., 1:]
```

DPO 的 `DPODataset` 不返回完整 `input_ids` 和 `labels`，而是提前切好：

```text
x = input_ids[:-1]
y = input_ids[1:]
mask = assistant_loss_mask[1:]
```

所以 DPO 训练脚本里：

```python
logits_to_log_probs(logits, y)
```

不再额外 shift。因为：

```text
model(x) 的 logits 位置 i
正好预测 y 位置 i
```

这是本节最容易错的地方。

### 2.4 mask 决定比较哪些 token

DPO 不应该比较 user/system prompt token 的概率，因为这些 token 是条件，不是要优化的回复。

所以 `DPODataset` 生成：

```text
mask_chosen
mask_rejected
```

mask 为 1 的位置是 assistant 回复区域；mask 为 0 的位置是 prompt、padding 或其它不参与偏好比较的位置。

后面算 sequence logprob 时会做：

```text
token_logprob * mask
```

再沿 seq 维求和。

### 2.5 policy model 和 reference model 的区别

DPO 训练里有两套同结构模型：

```text
policy model: 正在训练的模型
reference model: 冻结的基准模型
```

它们通常从同一个 SFT 权重开始：

```text
from_weight = full_sft
```

训练开始时：

```text
policy 和 reference 的行为一样
```

训练过程中：

```text
policy 更新；
reference 不更新。
```

reference 的作用是给 policy 一个“不要偏离基座太多”的基准。DPO loss 会比较：

```text
policy 对 chosen/rejected 的偏好变化
reference 对 chosen/rejected 的原始偏好
```

第 22 课会完整推这个公式。

### 2.6 为什么 chosen/rejected 要拼在同一个 batch

MiniMind 的 `train_dpo.py` 把 chosen 和 rejected 在 batch 维拼起来：

```text
x = concat([x_chosen, x_rejected], dim=0)
y = concat([y_chosen, y_rejected], dim=0)
mask = concat([mask_chosen, mask_rejected], dim=0)
```

这样 policy 和 ref 都只需要各 forward 一次：

```text
前半 batch: chosen
后半 batch: rejected
```

后面的 DPO loss 再按前后半切开。

这个做法简洁，但有一个隐含约定：

```text
chosen 和 rejected 必须按相同顺序成对排列。
```

如果 batch 里顺序乱了，chosen/rejected 配对就错了。

<a id="l21-reading-order"></a>
## 3. 源码阅读顺序图

建议按这个顺序读：

```text
1. README.md:461-478
   看 dpo.jsonl 的数据格式。

2. dataset/lm_dataset.py:122-174
   看 DPODataset 如何把 chosen/rejected 转成 x/y/mask。

3. dataset/lm_dataset.py:176-192
   看 assistant 区域 mask 如何生成。

4. trainer/train_dpo.py:131-155
   看 DPO 脚本默认参数。

5. trainer/train_dpo.py:182-194
   看 policy/ref 如何初始化和冻结。

6. trainer/train_dpo.py:57-83
   看 chosen/rejected 如何拼 batch，policy/ref 如何算 logprob。

7. trainer/train_dpo.py:34-50
   只预览 DPO loss 的输入，不深入公式。
```

和前几课的关系：

```text
第 5 课：assistant mask
第 10 课：logits 和 labels shift
第 16 课：SFTDataset 如何训练 assistant
第 20 课：teacher/reference 都是冻结模型，但用途不同
第 21 课：DPO 偏好对和 reference model
第 22 课：DPO loss 公式和手算
```

<a id="l21-source-walkthrough"></a>
## 4. MiniMind 源码走读

### 4.1 DPO 数据格式是 chosen/rejected

#### 源码证据 A：README 中的数据格式

文件：`README.md:461-478`

看它是为了理解：DPO 数据不是普通 SFT conversation，而是一对偏好答案。

代码摘录：

```json
{
  "chosen": [
    {"content": "Q", "role": "user"},
    {"content": "good answer", "role": "assistant"}
  ],
  "rejected": [
    {"content": "Q", "role": "user"},
    {"content": "bad answer", "role": "assistant"}
  ]
}
```

这段配置说明：

- `chosen` 和 `rejected` 都是 messages 列表。
- 两边通常共享同一个用户问题。
- `chosen` 是更符合偏好的回复。
- `rejected` 是相对较差的回复。

理解到这一步就够：

```text
DPO 的监督信号不是一个正确答案，而是“chosen 比 rejected 好”。
```

### 4.2 DPODataset 分别渲染 chosen/rejected

#### 源码证据 A：读取样本并渲染 prompt

文件：`dataset/lm_dataset.py:135-147`

看它是为了理解：chosen/rejected 两边都要走 chat template。

代码摘录：

```python
sample = self.samples[index]
chosen = sample['chosen']
rejected = sample['rejected']
chosen_prompt = self.tokenizer.apply_chat_template(
    chosen, tokenize=False, add_generation_prompt=False
)
chosen_prompt = post_processing_chat(chosen_prompt)

rejected_prompt = self.tokenizer.apply_chat_template(
    rejected, tokenize=False, add_generation_prompt=False
)
rejected_prompt = post_processing_chat(rejected_prompt)
```

这段代码说明：

- `chosen` 和 `rejected` 分开处理。
- 二者都要转成完整 prompt 字符串。
- `add_generation_prompt=False`，因为答案已经在数据里。
- `post_processing_chat` 会按规则处理空 thinking 标签。

#### 源码证据 B：tokenize 和 padding

文件：`dataset/lm_dataset.py:148-153`

看它是为了理解：两边序列会被整理到固定长度。

代码摘录：

```python
chosen_encoding = self.tokenizer(
    chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
)
rejected_encoding = self.tokenizer(
    rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
)
```

这段代码说明：

- chosen/rejected 都会截断到 `max_length`。
- 不足部分会 padding。
- 后续 mask 会避免 padding 参与偏好比较。

### 4.3 DPODataset 提前完成 x/y shift

#### 源码证据：返回 x/y/mask

文件：`dataset/lm_dataset.py:155-174`

看它是为了理解：DPO batch 里的 `x_chosen` 和 `y_chosen` 为什么长度是 `max_length - 1`。

代码摘录：

```python
chosen_input_ids = chosen_encoding['input_ids']
chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

rejected_input_ids = rejected_encoding['input_ids']
rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)
x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)
```

这段代码说明：

- `x_*` 是模型输入。
- `y_*` 是每个位置要预测的下一个 token。
- `mask_*` 和 `y_*` 对齐，所以也取 `[1:]`。
- 这里已经完成了第 10 课讲过的 shift。

关键判断：

```text
DPO 的 logits_to_log_probs 不再做 shift；
因为 dataset 已经把 x/y 对齐好了。
```

### 4.4 generate_loss_mask 只标出 assistant 回复

#### 源码证据：assistant mask

文件：`dataset/lm_dataset.py:176-192`

看它是为了理解：DPO 为什么只比较回答区域的 logprob。

代码摘录：

```python
loss_mask = [0] * len(input_ids)
...
if input_ids[i:i + len(self.bos_id)] == self.bos_id:
    start = i + len(self.bos_id)
    ...
    for j in range(start, min(end + len(self.eos_id), self.max_length)):
        loss_mask[j] = 1
...
return loss_mask
```

这段代码说明：

- 初始所有位置都是 0。
- 找到 assistant 起始标记后，把 assistant 内容区设为 1。
- 到 eos 为止，包括 eos 区域。
- user/system/padding 仍然是 0。

和 SFT 的共同点：

```text
都只训练 assistant 回复区域。
```

和 SFT 的不同点：

```text
SFT 用 label=-100 忽略非 assistant 区域；
DPO 用 mask=0 忽略非 assistant 区域。
```

### 4.5 DPO 脚本同时构造 policy 和 reference

#### 源码证据 A：脚本默认参数

文件：`trainer/train_dpo.py:131-155`

看它是为了理解：DPO 从哪个权重开始，默认用什么数据和 beta。

代码摘录：

```python
parser.add_argument('--save_weight', default='dpo', type=str)
parser.add_argument("--learning_rate", type=float, default=4e-8)
parser.add_argument("--data_path", type=str, default="../dataset/dpo.jsonl")
parser.add_argument('--from_weight', default='full_sft', type=str)
parser.add_argument('--beta', default=0.15, type=float)
```

这段代码说明：

- DPO 默认保存为 `dpo_{hidden_size}.pth`。
- 默认从 `full_sft` 开始。
- 默认数据是 `dpo.jsonl`。
- DPO 学习率很小，源码注释也提示避免遗忘。
- `beta` 是第 22 课要讲的 DPO 强度参数。

#### 源码证据 B：policy/ref 初始化

文件：`trainer/train_dpo.py:182-194`

看它是为了理解：policy model 和 reference model 的关系。

代码摘录：

```python
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
Logger(f'策略模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
ref_model.eval()
ref_model.requires_grad_(False)
Logger(f'参考模型总参数量：{sum(p.numel() for p in ref_model.parameters()) / 1e6:.3f} M')

train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
```

这段代码说明：

- `model` 是 policy，要被 optimizer 更新。
- `ref_model` 从同一个 `from_weight` 加载，但被冻结。
- ref_model 用 eval 模式。
- optimizer 只接收 policy 参数。
- DPO 使用 `DPODataset`，不是 `SFTDataset`。

和第 20 课 distillation 的相似点：

```text
都有一个冻结模型参与 forward。
```

关键区别：

```text
Distillation 的 teacher 给 soft target；
DPO 的 reference 给偏离基座的参照项。
```

### 4.6 train_epoch 拼接 chosen/rejected

#### 源码证据 A：batch 取出六个 tensor

文件：`trainer/train_dpo.py:57-67`

看它是为了理解：DPODataset 返回值如何进入训练循环。

代码摘录：

```python
x_chosen = batch['x_chosen'].to(args.device)
x_rejected = batch['x_rejected'].to(args.device)
y_chosen = batch['y_chosen'].to(args.device)
y_rejected = batch['y_rejected'].to(args.device)
mask_chosen = batch['mask_chosen'].to(args.device)
mask_rejected = batch['mask_rejected'].to(args.device)
x = torch.cat([x_chosen, x_rejected], dim=0)
y = torch.cat([y_chosen, y_rejected], dim=0)
mask = torch.cat([mask_chosen, mask_rejected], dim=0)
```

这段代码说明：

- 一个 batch 里有 chosen 和 rejected 两套序列。
- 拼接后 batch 维变成原来的 2 倍。
- 前半部分是 chosen。
- 后半部分是 rejected。
- `y` 和 `mask` 与 `x` 同样拼接，顺序必须一致。

#### 源码证据 B：ref/policy 都计算 logprob

文件：`trainer/train_dpo.py:73-83`

看它是为了理解：reference model 虽然冻结，但仍然参与 forward。

代码摘录：

```python
with torch.no_grad():
    ref_outputs = ref_model(x)
    ref_logits = ref_outputs.logits
ref_log_probs = logits_to_log_probs(ref_logits, y)

outputs = model(x)
logits = outputs.logits
policy_log_probs = logits_to_log_probs(logits, y)

dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
```

这段代码说明：

- ref forward 用 `torch.no_grad()`。
- policy forward 正常构建反向图。
- 两边都把 logits 转成 token logprob。
- DPO loss 接收 ref/policy logprob 和 mask。

第 21 课只需要知道：

```text
logprob 是 DPO loss 的输入。
```

第 22 课再展开：

```text
chosen/rejected logprob 如何变成 DPO loss。
```

### 4.7 logits_to_log_probs 从 logits 中取目标 token 概率

#### 源码证据：gather 目标 token logprob

文件：`trainer/train_dpo.py:25-31`

看它是为了理解：模型输出 `[batch, seq, vocab]` 后，如何得到每个目标 token 的 logprob。

代码摘录：

```python
def logits_to_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=2)
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token
```

这段代码说明：

- `log_softmax` 把 vocab 维 logits 变成 log probability。
- `labels.unsqueeze(2)` 把 `[batch, seq]` 变成 `[batch, seq, 1]`。
- `gather` 取出每个位置目标 token 的 logprob。
- 返回 shape 是 `[batch, seq]`。

注意：

```text
这里没有 ignore_index=-100。
```

原因是 DPO dataset 的 `y` 是真实 token id，包括 padding token id；不参与训练的位置通过 `mask=0` 处理，而不是通过 `-100` 处理。

<a id="l21-must-write"></a>
## 5. 本节必须会写 / 暂时不要求

必须会写：

```text
1. 画出 DPO 数据格式：
   chosen:   [user, assistant_good]
   rejected: [user, assistant_bad]

2. 说明 DPODataset 返回的六个 tensor：
   x_chosen, y_chosen, mask_chosen
   x_rejected, y_rejected, mask_rejected

3. 说明 x/y/mask 的 shift 关系：
   x = input_ids[:-1]
   y = input_ids[1:]
   mask = loss_mask[1:]

4. 说明 policy/ref 的关系：
   同一 from_weight 初始化；
   policy 训练；
   ref 冻结。

5. 说明 logits_to_log_probs：
   logits -> log_softmax -> gather labels -> token logprob。
```

暂时不要求：

```text
1. DPO loss 公式推导。
2. beta 对训练强度的影响。
3. chosen/rejected logratio 的手算。
4. reference output 预缓存。
5. PPO 到 DPO 的理论推导。
6. 多轮偏好数据质量评估。
```

<a id="l21-experiment"></a>
## 6. 实验验证

本节新增两个文件：

```text
course/labs/tiny_dpo.jsonl
course/labs/trace_dpo_dataset_reference.py
```

这个实验验证：

```text
1. tiny DPO 数据能被 DPODataset 读取。
2. chosen/rejected 都会变成 x/y/mask。
3. x/y/mask 的长度是 max_length - 1。
4. chosen/rejected 拼接后 batch 维翻倍。
5. policy/ref 对同一批 token 计算 logprob。
6. ref_model 被冻结。
```

运行命令：

```bash
cd /home/sun/minimind
python course/labs/trace_dpo_dataset_reference.py \
  --data_path course/labs/tiny_dpo.jsonl \
  --max_length 96
```

重点记录：

```text
x_chosen.shape =
y_chosen.shape =
mask_chosen_active =
x_cat.shape =
policy_log_probs.shape =
ref_log_probs.shape =
initial_policy_ref_logprob_max_abs_diff =
ref_requires_grad =
```

你应该看到：

```text
x/y/mask 的 seq 长度是 max_length - 1；
拼接后的 batch 维是 2；
policy/ref 初始权重相同时 logprob diff 接近 0；
ref_requires_grad=False。
```

这个实验不验证 DPO loss 是否正确。第 22 课会在这些 logprob 基础上手算 DPO loss。

### 6.1 真实训练命令

如果本地已经有：

```text
out/full_sft_768.pth
dataset/dpo.jsonl
```

可以跑原项目 DPO：

```bash
cd /home/sun/minimind/trainer
python train_dpo.py \
  --from_weight full_sft \
  --data_path ../dataset/dpo.jsonl \
  --epochs 1
```

当前课程环境如果缺真实权重和数据，就不要把这个作为本节必跑验收。本节重点是：

```text
DPODataset 的输出结构；
reference model 的冻结；
policy/ref logprob 的输入形状。
```

<a id="l21-stage-assembly"></a>
## 7. 阶段组装

第 21 课是 DPO 阶段的第一半：

```text
数据和模型准备：
  tiny_dpo.jsonl
  DPODataset
  policy model
  reference model
  token logprob
```

第 22 课会补第二半：

```text
sequence logprob
chosen/rejected 分组
policy logratio
reference logratio
DPO loss
beta
```

### 7.1 教学版 DPO 后续实现边界

后续如果进入 `course/impl/`，DPO 阶段可以拆成：

```text
course/impl/core/datasets.py::CourseDPODataset
course/impl/core/losses.py::sequence_log_probs
course/impl/core/losses.py::dpo_loss
course/impl/train_dpo_impl.py
```

第 21 课只要求你能把数据准备对。

第 22 课再要求你手写：

```text
sequence_log_probs
dpo_loss
```

### 7.2 和原项目的差异

| 内容 | 原项目 | 教学版建议 |
|---|---|---|
| 数据 | `dataset/dpo.jsonl` | `course/labs/tiny_dpo.jsonl` |
| 模型 | 真实 full_sft 权重 | tiny 随机模型或已完成的 course_sft |
| 训练 | 完整 DPO 训练 | 先 trace 数据和 logprob |
| loss | `train_dpo.py::dpo_loss` | 第 22 课手写和对齐 |
| reference | 每步 forward | 可以先每步 forward，不做缓存 |

### 7.3 Portfolio 记录

完成本节后，可以在 `course/portfolio/experiments.md` 记录：

```text
DPO data/reference trace:
- 构造 tiny chosen/rejected 偏好样本。
- 用 DPODataset 生成 x/y/mask。
- 验证 x/y/mask 已经 shift 对齐。
- 拼接 chosen/rejected batch。
- 初始化 policy/ref，冻结 ref。
- 对同一批序列计算 policy/ref token logprob。
```

在 `course/notes/mistakes.md` 记录容易错的点：

```text
1. DPO dataset 已经 shift，训练脚本里不要再 shift 一次。
2. chosen/rejected 拼接后顺序不能乱。
3. mask 要和 y 对齐，所以取 loss_mask[1:]。
4. DPO 的 y 不能含 -100；忽略区域靠 mask。
5. reference model 需要冻结，但仍然要 forward。
```

<a id="l21-check"></a>
## 8. 本节检查

1. DPO 的 `chosen` 和 `rejected` 分别表示什么？
2. `DPODataset` 为什么返回 `x=input_ids[:-1]` 和 `y=input_ids[1:]`？
3. `mask_chosen` 为什么取 `chosen_loss_mask[1:]`？
4. DPO 里为什么不用 `label=-100`，而是用 mask？
5. policy model 和 reference model 为什么从同一个 `from_weight` 初始化？
6. reference model 为什么要冻结，但仍然要 forward？
7. chosen/rejected 拼接到同一个 batch 后，后续如何知道哪一半是哪一类？

<a id="l21-next"></a>
## 9. 下一课

第 22 课进入 DPO loss 源码：

- `22_dpo_loss_source.md`

下一课要解决：

```text
logits_to_log_probs 如何得到 token logprob；
mask 后的 sequence logprob 如何求和；
chosen/rejected 如何拆成两组；
policy logratio 和 reference logratio 分别是什么；
DPO loss = -logsigmoid(beta * logits) 如何手算。
```
