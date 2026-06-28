# 第 20 课：Distillation 训练链路

这一课只解决一个问题：MiniMind 的 `train_distillation.py` 如何让 student 同时学习真实 assistant label 和 teacher 的 soft logits。

## 目录

- [0. 本节主线](#l20-mainline)
- [1. 本节要懂的 6 个原理](#l20-principles)
- [2. 蒸馏训练完整原理](#l20-complete-principle)
- [3. 源码阅读顺序图](#l20-reading-order)
- [4. MiniMind 源码走读](#l20-source-walkthrough)
- [5. 本节必须会写 / 暂时不要求](#l20-must-write)
- [6. 实验验证](#l20-experiment)
- [7. 阶段组装](#l20-stage-assembly)
- [8. 本节检查](#l20-check)
- [9. 下一课](#l20-next)

<a id="l20-mainline"></a>
## 0. 本节主线

蒸馏训练流程是：

```text
加载 student 模型
-> 加载 teacher 模型
-> 冻结 teacher
-> 用 SFTDataset 得到 input_ids / labels
-> student 前向得到 student_logits
-> teacher no_grad 前向得到 teacher_logits
-> labels 生成 CE loss
-> teacher soft distribution 生成 KL distillation loss
-> 总 loss = alpha * CE + (1 - alpha) * distill
-> 只反传和更新 student
-> 保存 student 权重
```

这节课的核心不是“teacher 直接给答案”，而是：

```text
teacher 给每个位置、每个 token 的概率分布；
student 学真实 label，也学 teacher 对其它 token 的相对偏好。
```

一句话：

```text
SFT 只告诉 student 正确 token 是什么，蒸馏还告诉 student teacher 认为哪些 token 也接近合理。
```

<a id="l20-principles"></a>
## 1. 本节要懂的 6 个原理

| 原理 | 要理解什么 | 源码位置 |
|---|---|---|
| student/teacher 是两套模型 | student 被训练，teacher 只提供 soft target | `trainer/train_distillation.py:204-214` |
| teacher 必须冻结 | teacher 用 eval/no_grad/requires_grad_(False) 避免更新和节省显存 | `trainer/train_distillation.py:43-45`, `trainer/train_distillation.py:207-209` |
| 蒸馏仍用 SFTDataset | 数据、chat template、assistant-only labels 仍沿用 SFT | `trainer/train_distillation.py:211`, `dataset/lm_dataset.py:88-119` |
| CE loss 学真实 label | student logits 与 labels shift 后计算 token-level CE | `trainer/train_distillation.py:68-80` |
| KL loss 学 teacher 分布 | teacher softmax 后作为 target，student log_softmax 后做 KL | `trainer/train_distillation.py:25-36`, `trainer/train_distillation.py:82-90` |
| alpha/temperature 控制训练信号 | alpha 混合 CE/KL，temperature 平滑 teacher 分布 | `trainer/train_distillation.py:92-93`, `trainer/train_distillation.py:171-173` |

学完本节，你应该能说明：

```text
distillation_loss 的输入 shape 是什么；
teacher 为什么不 backward；
loss_mask 为什么来自 labels[..., 1:]；
alpha 和 temperature 分别控制什么；
最终保存的是 student，不是 teacher。
```

<a id="l20-complete-principle"></a>
## 2. 蒸馏训练完整原理

### 2.1 SFT 的监督信号太硬

普通 SFT 对每个位置只有一个目标 token。

假设当前位置的正确 token 是：

```text
北京
```

CE loss 看的是：

```text
模型给“北京”的概率够不够高
```

它不会告诉 student：

```text
“上海”是不是比“苹果”更接近合理；
“中国”是不是也有一定语义相关性；
标点、停用词、同义表达之间有什么相对偏好。
```

真实 label 是 hard target：

```text
正确 token = 1
其它 token = 0
```

蒸馏引入 teacher 的 soft target：

```text
teacher 对整个 vocab 的概率分布
```

这让 student 不只学习“唯一答案”，还学习 teacher 的分布形状。

### 2.2 teacher logits 是什么

teacher 和 student 都是 causal LM。

对同一个 `input_ids`，它们都会输出：

```text
student_logits: [batch, seq, vocab]
teacher_logits: [batch, seq, vocab]
```

第 10 课讲过 next-token shift，所以训练时真正使用：

```text
student_logits[..., :-1, :]
teacher_logits[..., :-1, :]
labels[..., 1:]
```

含义是：

```text
位置 i 的 logits 用来预测位置 i+1 的 label。
```

### 2.3 CE loss 和 distill loss 分工

MiniMind 的蒸馏总损失是：

```text
loss = alpha * CE + (1 - alpha) * Distill
```

CE loss 来自真实数据：

```text
student_logits -> labels
```

Distill loss 来自 teacher：

```text
student_logits -> teacher_logits
```

两者都只在 assistant label 区域上计算。这个区域由：

```python
labels[..., 1:] != -100
```

决定。

为什么不在 user/system 区域算 teacher KL？

因为 SFTDataset 把 user/system/prompt 区域设为 `-100`，表示这些 token 只是条件，不是训练目标。蒸馏训练沿用这个边界，避免 student 被训练去复述 prompt。

### 2.4 temperature 的作用

蒸馏时常把 logits 除以 temperature：

```text
softmax(logits / T)
```

当 `T > 1` 时，概率分布会更平滑：

```text
最大 token 的概率会降低；
次优 token 的概率会抬高；
student 更容易看到 teacher 的相对偏好。
```

MiniMind 默认：

```text
temperature = 1.5
```

源码最后乘回：

```text
temperature ** 2
```

这是常见蒸馏写法，用来补偿 temperature 对梯度尺度的影响。第 20 课只需要理解它保持 loss 尺度更稳定，不必推完整梯度公式。

### 2.5 teacher 为什么不更新

teacher 是训练目标的来源，不是被训练对象。

所以 teacher 必须：

```text
eval()
requires_grad_(False)
with torch.no_grad()
```

这三件事分别解决：

```text
eval(): 关闭 dropout 等训练态行为
requires_grad_(False): teacher 参数不需要梯度
no_grad(): teacher forward 不构建反向图
```

如果 teacher 也更新，目标本身会变化，student 学的分布不稳定；同时显存和计算都会浪费。

<a id="l20-reading-order"></a>
## 3. 源码阅读顺序图

建议按这个顺序读：

```text
1. train_distillation.py:146-176
   看脚本参数：student/teacher 配置、alpha、temperature。

2. train_distillation.py:184-214
   看 student 和 teacher 如何初始化、冻结、建 optimizer。

3. train_distillation.py:39-67
   看一个 batch 里 student/teacher logits 如何取得。

4. train_distillation.py:68-93
   看 CE loss、distill loss、总 loss 如何组合。

5. train_distillation.py:25-36
   单独看 KL 蒸馏损失函数。

6. train_distillation.py:124-143
   看保存 student 和尾批 gradient accumulation。
```

和前几课的关系：

```text
第 5 课：SFTDataset labels mask
第 10 课：logits/labels shift
第 14 课：lr、grad accumulation、AMP
第 16 课：Full SFT 训练入口
第 20 课：在 SFT 训练上额外加 teacher KL
```

<a id="l20-source-walkthrough"></a>
## 4. MiniMind 源码走读

### 4.1 脚本参数决定 student、teacher、alpha 和 temperature

#### 源码证据 A：student/teacher 参数

文件：`trainer/train_distillation.py:146-176`

看它是为了理解：蒸馏脚本和普通 SFT 脚本相比多了哪些训练对象和超参数。

代码摘录：

```python
parser.add_argument('--student_hidden_size', default=768, type=int)
parser.add_argument('--student_num_layers', default=8, type=int)
parser.add_argument('--teacher_hidden_size', default=768, type=int)
parser.add_argument('--teacher_num_layers', default=8, type=int)
parser.add_argument('--student_use_moe', default=0, type=int, choices=[0, 1])
parser.add_argument('--teacher_use_moe', default=1, type=int, choices=[0, 1])
parser.add_argument('--from_student_weight', default='full_sft', type=str)
parser.add_argument('--from_teacher_weight', default='full_sft', type=str)
parser.add_argument('--alpha', default=0.5, type=float)
parser.add_argument('--temperature', default=1.5, type=float)
```

这段代码说明：

- student 和 teacher 可以有不同结构。
- 默认场景是 teacher 用 MoE，student 用 dense。
- 两者默认都从 `full_sft` 权重开始。
- `alpha` 控制 CE 和 KL 的权重。
- `temperature` 控制 teacher soft target 的平滑程度。

理解到这一步就够：

```text
蒸馏训练至少要同时关心两套模型配置，而不是只关心一个 lm_config。
```

暂时不要看：

```text
teacher 比 student 大多少效果最好；
alpha/temperature 搜参策略。
```

### 4.2 student 被训练，teacher 被冻结

#### 源码证据 A：构造两套 config

文件：`trainer/train_distillation.py:184-188`

看它是为了理解：student 和 teacher 在代码里是两套独立配置。

代码摘录：

```python
lm_config_student = MiniMindConfig(hidden_size=args.student_hidden_size, num_hidden_layers=args.student_num_layers, use_moe=bool(args.student_use_moe))
lm_config_teacher = MiniMindConfig(hidden_size=args.teacher_hidden_size, num_hidden_layers=args.teacher_num_layers, use_moe=bool(args.teacher_use_moe))
ckp_data = lm_checkpoint(lm_config_student, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
```

这段代码说明：

- student 和 teacher 都用 `MiniMindConfig` 构建。
- resume checkpoint 按 student 配置保存。
- 最终被训练和恢复的是 student。

#### 源码证据 B：加载并冻结 teacher

文件：`trainer/train_distillation.py:204-214`

看它是为了理解：optimizer 为什么只更新 student。

代码摘录：

```python
model, tokenizer = init_model(lm_config_student, args.from_student_weight, device=args.device)
teacher_model, _ = init_model(lm_config_teacher, args.from_teacher_weight, device=args.device)
teacher_model.eval()
teacher_model.requires_grad_(False)
train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
```

这段代码说明：

- `model` 是 student。
- `teacher_model` 单独加载。
- teacher 被设置为 eval 并冻结参数。
- optimizer 只接收 `model.parameters()`，也就是 student 参数。
- 数据仍然用 student tokenizer 构造。

### 4.3 一个 batch 同时跑 student 和 teacher

#### 源码证据 A：student 前向

文件：`trainer/train_distillation.py:47-60`

看它是为了理解：student logits 如何对齐 next-token 目标。

代码摘录：

```python
input_ids = input_ids.to(args.device)
labels = labels.to(args.device)
loss_mask = (labels[..., 1:] != -100).float()

with autocast_ctx:
    res = model(input_ids)
    student_logits = res.logits[..., :-1, :].contiguous()
```

这段代码说明：

- `loss_mask` 来自 shifted labels。
- student forward 没传 `labels`，因为脚本要手写 CE 和 KL。
- `student_logits[..., :-1, :]` 去掉最后一个位置。
- 这和第 10 课的 next-token shift 一致。

#### 源码证据 B：teacher no_grad 前向

文件：`trainer/train_distillation.py:61-67`

看它是为了理解：teacher logits 如何参与 loss 但不参与反向传播。

代码摘录：

```python
if teacher_model is not None:
    with torch.no_grad():
        teacher_logits = teacher_model(input_ids).logits[..., :-1, :].contiguous()
        vocab_size_student = student_logits.size(-1)
        teacher_logits = teacher_logits[..., :vocab_size_student]
```

这段代码说明：

- teacher forward 包在 `torch.no_grad()` 里。
- teacher logits 和 student logits 一样做 `[..., :-1, :]`。
- teacher vocab 会被截断到 student vocab 大小。
- KL loss 需要 student 和 teacher 最后一维一致。

注意这个截断只解决 vocab 维度对齐：

```text
teacher_logits[..., :vocab_size_student]
```

它不解决 tokenizer 不一致问题。真正训练时 student 和 teacher 应该使用兼容 tokenizer，否则同一个 token id 的含义可能不同。

### 4.4 CE loss 只在 assistant 区域上算

#### 源码证据：手写 CE loss

文件：`trainer/train_distillation.py:68-80`

看它是为了理解：为什么 distillation 脚本没有直接用 `model(input_ids, labels=labels)` 的 `res.loss`。

代码摘录：

```python
shift_labels = labels[..., 1:].contiguous()
loss_mask_flat = loss_mask.view(-1)
ce_loss = F.cross_entropy(
    student_logits.view(-1, student_logits.size(-1)),
    shift_labels.view(-1),
    ignore_index=-100,
    reduction='none'
)
ce_loss_raw = torch.sum(ce_loss * loss_mask_flat) / (loss_mask_flat.sum() + 1e-8)
if lm_config_student.use_moe: ce_loss = ce_loss_raw + res.aux_loss
else: ce_loss = ce_loss_raw
```

这段代码说明：

- CE loss 的 logits 和 labels 仍然 shift 对齐。
- `reduction='none'` 保留每个 token 的 loss。
- `loss_mask_flat` 只保留 assistant 训练区域。
- `ce_loss_raw` 是 active token 上的平均 CE。
- 如果 student 是 MoE，还要加 `res.aux_loss`。

为什么不用 `res.loss`？

因为蒸馏脚本后面还要对同一批 active token 计算 KL。手写 CE 能让 CE 和 KL 使用同一个 `loss_mask_flat`，日志里也能分别打印 `ce_loss` 和 `distill_loss`。

### 4.5 KL distillation loss 学 teacher 分布

#### 源码证据 A：distillation_loss 函数

文件：`trainer/train_distillation.py:25-36`

看它是为了理解：teacher logits 如何变成 student 的 soft target。

代码摘录：

```python
def distillation_loss(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction=reduction
    )
    return (temperature ** 2) * kl
```

这段代码说明：

- teacher logits 先除以 temperature，再 softmax 成概率。
- teacher 概率 detach，不参与梯度。
- student logits 变成 log probability。
- `F.kl_div` 的输入是 student log probs，target 是 teacher probs。
- 最后乘 `temperature ** 2`。

#### 源码证据 B：只在 active token 上算 KL

文件：`trainer/train_distillation.py:82-90`

看它是为了理解：teacher KL 和 SFT labels mask 如何结合。

代码摘录：

```python
distill_loss = distillation_loss(
    student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],
    teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask_flat == 1],
    temperature=temperature
)
```

这段代码说明：

- logits 被展平成 `[batch * (seq - 1), vocab]`。
- 只取 `loss_mask_flat == 1` 的 token。
- user/system/padding 区域不会进入 KL。
- CE 和 KL 使用同一批 active token。

### 4.6 总 loss 混合 CE 和 distill

#### 源码证据 A：loss 组合

文件：`trainer/train_distillation.py:92-102`

看它是为了理解：alpha 对最终训练目标的影响。

代码摘录：

```python
loss = (alpha * ce_loss + (1 - alpha) * distill_loss) / args.accumulation_steps

scaler.scale(loss).backward()

if step % args.accumulation_steps == 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

这段代码说明：

- `alpha=1.0` 时接近普通 SFT。
- `alpha=0.0` 时只学 teacher 分布。
- 默认 `alpha=0.5`，CE 和 KL 各占一半。
- 梯度累积、AMP、clip、optimizer step 沿用普通训练脚本。
- optimizer 只管理 student，所以只更新 student。

#### 源码证据 B：日志同时打印 CE 和 distill

文件：`trainer/train_distillation.py:104-122`

看它是为了理解：训练时应该观察哪些指标。

代码摘录：

```python
Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, ce: {current_ce_loss:.4f}, aux_loss: {current_aux_loss:.4f}, distill: {distill_loss.item():.4f}, learning_rate: {current_lr:.8f}, epoch_time: {eta_min:.3f}min')
```

这段代码说明：

- `loss` 是混合后的总损失。
- `ce` 是真实 label 的监督损失。
- `distill` 是 teacher 分布的 KL 损失。
- MoE student 时还要观察 `aux_loss`。

### 4.7 保存的是 student 权重

#### 源码证据：保存 checkpoint

文件：`trainer/train_distillation.py:124-143`

看它是为了理解：蒸馏训练产物是什么。

代码摘录：

```python
ckp = f'{args.save_dir}/{args.save_weight}_{lm_config_student.hidden_size}{moe_suffix}.pth'
raw_model = model.module if isinstance(model, DistributedDataParallel) else model
raw_model = getattr(raw_model, '_orig_mod', raw_model)
state_dict = raw_model.state_dict()
torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
lm_checkpoint(lm_config_student, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
```

这段代码说明：

- 保存路径使用 student config。
- 保存的是当前训练中的 `model`，也就是 student。
- teacher 不会被保存成训练产物。
- resume checkpoint 也按 student 权重名保存。

<a id="l20-must-write"></a>
## 5. 本节必须会写 / 暂时不要求

必须会写：

```text
1. 说明 distillation_loss 的输入输出：
   student_logits: [active_tokens, vocab]
   teacher_logits: [active_tokens, vocab]
   output: scalar KL loss

2. 手动写出 teacher soft target：
   teacher_probs = softmax(teacher_logits / temperature)

3. 手动写出 student log probability：
   student_log_probs = log_softmax(student_logits / temperature)

4. 说明 loss_mask 的来源：
   loss_mask = labels[..., 1:] != -100

5. 说明总 loss：
   loss = alpha * ce_loss + (1 - alpha) * distill_loss

6. 说明 teacher 为什么 eval/no_grad/frozen。
```

暂时不要求：

```text
1. 蒸馏效果评测。
2. teacher/student 架构搜索。
3. 多 teacher ensemble。
4. hidden states / attention map 蒸馏。
5. tokenizer 不一致时的 vocab 映射。
6. temperature 梯度尺度公式推导。
```

<a id="l20-experiment"></a>
## 6. 实验验证

本节新增实验：

```text
course/labs/trace_distillation_loss.py
```

这个实验不需要真实 `full_sft` 权重。它用随机初始化的 tiny student 和 tiny teacher 跑一次前向，只验证：

```text
input_ids / labels shape
student_logits / teacher_logits shape
loss_mask active token 数量
CE loss
distill loss
alpha 混合后的 total loss
```

运行命令：

```bash
cd /home/sun/minimind
python course/labs/trace_distillation_loss.py \
  --data_path course/labs/tiny_sft.jsonl \
  --max_length 96 \
  --alpha 0.5 \
  --temperature 1.5
```

重点记录：

```text
student_logits.shape =
teacher_logits.shape =
active_tokens =
ce_loss_raw =
distill_loss =
total_loss =
```

你应该看到：

```text
student_logits.shape 和 teacher_logits.shape 的最后一维都是 vocab_size；
active_tokens 大于 0；
ce_loss_raw、distill_loss、total_loss 都是标量；
total_loss 约等于 alpha * ce_loss_raw + (1-alpha) * distill_loss。
```

这个实验验证的是计算路径，不验证模型效果。随机 teacher 不会教出好 student，但足够让你看懂源码里的张量流。

### 6.1 真实训练命令

如果本地已经有 student 和 teacher 权重，例如：

```text
out/full_sft_768.pth
out/full_sft_768_moe.pth
dataset/sft_t2t_mini.jsonl
```

可以跑原项目蒸馏训练：

```bash
cd /home/sun/minimind/trainer
python train_distillation.py \
  --from_student_weight full_sft \
  --from_teacher_weight full_sft \
  --student_use_moe 0 \
  --teacher_use_moe 1 \
  --alpha 0.5 \
  --temperature 1.5 \
  --epochs 1
```

当前课程环境如果缺默认权重和数据，就不要把真实训练作为本节必跑验收。先跑 tiny trace 实验，确认 CE/KL 的张量和数值关系。

<a id="l20-stage-assembly"></a>
## 7. 阶段组装

第 20 课不新增 `course/impl/` 的手写训练阶段。原因是课程主线里的手写阶段已经覆盖：

```text
Pretrain
SFT
LoRA
DPO
PPO/GRPO
```

蒸馏在这里作为源码训练链路学习：

```text
理解 train_distillation.py
-> 用 lab 复现 CE + KL 的核心计算
-> 知道真实训练需要两套权重
```

如果后续要扩展教学版蒸馏，可以新增：

```text
course/impl/core/losses.py::distillation_loss
course/impl/train_distillation_impl.py
course/impl/tests/test_distillation_loss.py
```

但第 20 课先不强制做这件事，避免在 DPO 之前引入太多额外阶段。

### 7.1 和前面课程的连接

| 已学内容 | 在蒸馏里的作用 |
|---|---|
| SFTDataset | 提供 assistant-only labels 和 loss mask |
| loss shift | 让 logits[..., :-1, :] 对齐 labels[..., 1:] |
| MoE aux loss | student 是 MoE 时加到 CE loss |
| AMP / grad accumulation | 蒸馏训练循环继续复用 |
| checkpoint | 保存和恢复 student 训练状态 |

### 7.2 Portfolio 记录

完成本节后，可以在 `course/portfolio/experiments.md` 记录：

```text
Distillation trace:
- 构造 tiny student/teacher。
- 对同一批 SFT input_ids 计算 student_logits 和 teacher_logits。
- 用 labels[..., 1:] != -100 只保留 assistant active tokens。
- 计算 CE loss、KL distillation loss 和 alpha 混合总 loss。
```

在 `course/notes/mistakes.md` 记录容易错的点：

```text
1. 忘记 teacher no_grad，导致 teacher 也进入反向图。
2. loss_mask 用 labels 而不是 shifted labels。
3. CE 和 KL 使用的 token 区域不一致。
4. student/teacher vocab 维度不一致时直接做 KL。
5. 把 alpha 理解反了：源码里 alpha 是 CE 权重。
```

<a id="l20-check"></a>
## 8. 本节检查

1. 蒸馏训练为什么要同时有 CE loss 和 distill loss？
2. `loss_mask = labels[..., 1:] != -100` 为什么要用 shifted labels？
3. teacher 为什么要 `eval()`、`requires_grad_(False)` 和 `torch.no_grad()`？
4. `temperature > 1` 对 teacher soft target 有什么影响？
5. 源码里的 `alpha` 是 CE 权重还是 distill 权重？
6. 为什么 KL 只在 assistant active token 上计算？
7. 蒸馏训练最终保存的是 student 还是 teacher？

<a id="l20-next"></a>
## 9. 下一课

第 21 课进入 DPO 数据和 reference model：

- `21_dpo_dataset_and_reference.md`

下一课要解决：

```text
DPO 数据为什么需要 chosen/rejected；
policy model 和 reference model 分别是什么；
DPODataset 如何构造 chosen/rejected 的 input、label 和 mask；
reference model 为什么冻结；
DPO 和 SFT/Distillation 的训练信号有什么不同。
```
