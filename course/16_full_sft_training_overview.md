# 第 16 课：Full SFT 训练脚本总览

这一课只解决一个问题：MiniMind 如何从预训练权重出发，用对话数据做 Full SFT。

## 目录

- [0. 本节主线](#l16-mainline)
- [1. 原理讲解](#l16-principle)
- [2. 源码阅读顺序图](#l16-reading-order)
- [3. MiniMind 源码走读](#l16-source-walkthrough)
- [4. 本节必须会写 / 暂时不要求](#l16-must-write)
- [5. 手写模块](#l16-handwrite)
- [6. 对齐测试](#l16-alignment-test)
- [7. 阶段组装](#l16-stage-assembly)
- [8. 本节检查](#l16-check)
- [9. 下一课](#l16-next)

<a id="l16-mainline"></a>
## 0. 本节主线

Full SFT 的训练链路是：

```text
读取 sft_t2t_mini.jsonl
-> SFTDataset 把 conversations 渲染成 chat prompt
-> tokenizer 编码成 input_ids
-> generate_labels 只保留 assistant 区域，其他位置设为 -100
-> 从 pretrain 权重初始化模型
-> 复用 pretrain 的 train_epoch 训练循环
-> 保存 full_sft 权重和 resume checkpoint
```

一句话：

```text
Full SFT 和 Pretrain 的训练循环几乎一样，真正不同的是数据目标：SFT 只让 assistant 回复区域参与 loss。
```

本节要把两条线连起来：

```text
第 5 课：SFT 数据如何构造 input_ids / labels
第 14-15 课：训练循环和 checkpoint 如何工作
```

<a id="l16-principle"></a>
## 1. 原理讲解

### 1.1 Full SFT 是什么

Full SFT 指的是“全参数监督微调”。

它和 LoRA 不同：

```text
Full SFT:
更新模型全部可训练参数。

LoRA:
通常冻结原模型，只训练额外插入的小矩阵。
```

它和 Pretrain 也不同：

```text
Pretrain:
用普通文本学习 next-token prediction。

SFT:
用对话数据学习 assistant 应该怎么回答。
```

训练目标仍然是 causal LM next-token loss，但 labels 的有效区域不同。

### 1.2 SFT 为什么仍然是 causal LM loss

SFT 并没有把模型改成分类器。模型输入仍然是：

```text
input_ids: [B, S]
```

模型输出仍然是：

```text
logits: [B, S, V]
```

loss 仍然是 shifted next-token cross entropy：

$$
L_{\text{lm}}
= -\frac{1}{|\mathcal{A}|}\sum_{t \in \mathcal{A}}
\log p_{\theta}(x_t \mid x_{<t})
$$

其中：

| 符号 | 形状/类型 | 含义 |
|---|---|---|
| $B$ | 标量 | batch size |
| $S$ | 标量 | sequence length |
| $V$ | 标量 | vocab size |
| $x_t$ | 标量 token id | 第 `t` 个目标 token |
| $\mathcal{A}$ | token 位置集合 | assistant 回复区域 |
| $p_{\theta}$ | 概率分布 | 模型对下一个 token 的预测 |

关键是 $\mathcal{A}$。Pretrain 里大部分非 padding token 都参与 loss；SFT 里通常只有 assistant 回复 token 参与 loss。

### 1.3 `labels=-100` 决定哪些 token 不训练

PyTorch 的 `cross_entropy` 支持 `ignore_index=-100`。

所以 SFTDataset 会构造：

```text
input_ids: 完整 prompt，包括 system/user/assistant/padding
labels:    只有 assistant 区域保留 token id，其它位置都是 -100
```

形状是：

| 变量 | 形状 | 含义 |
|---|---|---|
| `input_ids` | `[S]` | 一条样本的完整 token 序列 |
| `labels` | `[S]` | 同长度训练目标 |
| `input_ids batch` | `[B, S]` | dataloader 组成 batch 后 |
| `labels batch` | `[B, S]` | dataloader 组成 batch 后 |

可以把一条样本理解成：

```text
system/user token -> labels = -100
assistant token   -> labels = token_id
padding token     -> labels = -100
```

这样模型能看到完整上下文，但只因为 assistant 回复区域产生梯度。

### 1.4 SFTDataset 如何找到 assistant 区域

MiniMind 的 tokenizer chat template 会把 assistant 回复包在特殊标记附近。

SFTDataset 初始化时构造两个模式：

```text
bos_id = tokenizer(f"{bos_token}assistant\n", add_special_tokens=False).input_ids
eos_id = tokenizer(f"{eos_token}\n", add_special_tokens=False).input_ids
```

然后扫描 `input_ids`：

```text
找到 bos_id
-> bos_id 后面是 assistant 内容开始
-> 找到 eos_id
-> assistant 内容 + eos_id 这段 labels 保留 token id
-> 其它位置保持 -100
```

用公式表示 label mask：

$$
y_t =
\begin{cases}
x_t, & t \in \mathcal{A} \\
-100, & t \notin \mathcal{A}
\end{cases}
$$

其中 $\mathcal{A}$ 是 assistant 回复和结束标记所在位置。

### 1.5 Full SFT 从哪个权重开始

MiniMind 的 Full SFT 默认：

```text
from_weight = pretrain
save_weight = full_sft
```

意思是：

```text
读取 out/pretrain_768.pth
-> 训练 SFT
-> 保存 out/full_sft_768.pth
```

这体现了后训练的基本顺序：

```text
pretrain 权重
-> SFT 权重
-> DPO/RL 等偏好或强化学习阶段
```

如果 `from_weight='none'`，那就是从随机初始化直接 SFT。能跑，但通常不是正常路线。

### 1.6 Full SFT 训练循环和 Pretrain 有什么不同

训练循环几乎一样：

```text
get_lr
autocast forward
loss = res.loss + res.aux_loss
loss / accumulation_steps
backward
clip
optimizer.step
checkpoint
resume
```

不同主要在配置和数据：

| 项目 | Pretrain | Full SFT |
|---|---|---|
| dataset | `PretrainDataset` | `SFTDataset` |
| 默认数据 | `pretrain_t2t_mini.jsonl` | `sft_t2t_mini.jsonl` |
| 默认权重来源 | `none` | `pretrain` |
| 默认保存名 | `pretrain` | `full_sft` |
| 默认学习率 | `5e-4` | `1e-5` |
| 默认 max_seq_len | `340` | `768` |
| 默认 accumulation | `8` | `1` |

SFT 学习率更小，是因为它是在已有预训练能力上微调，不希望剧烈破坏已有语言建模能力。

<a id="l16-reading-order"></a>
## 2. 源码阅读顺序图

这节源码按这个顺序读：

```text
train_full_sft.py 参数默认值
-> init_model(..., from_weight='pretrain')
-> SFTDataset 初始化
-> SFTDataset.create_chat_prompt
-> SFTDataset.generate_labels
-> train_full_sft.py::train_epoch
-> checkpoint / resume 流程
```

对应文件：

```text
trainer/train_full_sft.py
dataset/lm_dataset.py
trainer/trainer_utils.py
model/model_minimind.py
```

先看训练脚本，是为了知道 SFT 阶段怎么启动。  
再看 `SFTDataset`，是为了知道 SFT 和 Pretrain 的真正差异。  
最后回到 `train_epoch`，确认它复用的是同一套 causal LM 训练机制。

<a id="l16-source-walkthrough"></a>
## 3. MiniMind 源码走读

### 第 1 步：Full SFT 的默认训练参数

File: `trainer/train_full_sft.py:84-107`

Read this to understand: SFT 阶段默认用什么数据、从什么权重开始、保存成什么名字。

Code/config/template excerpt:

```python
parser.add_argument('--save_weight', default='full_sft', type=str, help="保存权重的前缀名")
parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument("--learning_rate", type=float, default=1e-5, help="初始学习率")
parser.add_argument('--max_seq_len', default=768, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
parser.add_argument("--data_path", type=str, default="../dataset/sft_t2t_mini.jsonl", help="训练数据路径")
parser.add_argument('--from_weight', default='pretrain', type=str, help="基于哪个权重训练，为none则不基于任何权重训练")
```

This code shows:

- SFT 默认保存为 `full_sft`。
- SFT 默认从 `pretrain` 权重开始。
- SFT 默认数据是 `sft_t2t_mini.jsonl`。
- SFT 学习率比 Pretrain 小。

### 第 2 步：SFT 初始化模型和数据

File: `trainer/train_full_sft.py:115-139`

Read this to understand: Full SFT 的模型来自 `from_weight`，数据来自 `SFTDataset`。

Code/config/template excerpt:

```python
lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
...
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
```

This code shows:

- `init_model(..., args.from_weight)` 会加载 `pretrain` 权重。
- `SFTDataset` 替代了 `PretrainDataset`。
- optimizer、scaler、sampler 的结构和 pretrain 一样。
- 如果 `from_resume=1`，会优先查 `full_sft` 的 resume checkpoint。

### 第 3 步：`init_model` 如何加载 pretrain 权重

File: `trainer/trainer_utils.py:119-131`

Read this to understand: `from_weight='pretrain'` 到底加载哪个文件。

Code/config/template excerpt:

```python
if from_weight!= 'none':
    moe_suffix = '_moe' if lm_config.use_moe else ''
    weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    weights = torch.load(weight_path, map_location=device)
    model.load_state_dict(weights, strict=False)
```

This code shows:

- `from_weight='pretrain'` 会加载 `../out/pretrain_768.pth`。
- MoE 模型会加载带 `_moe` 后缀的权重。
- `strict=False` 允许有些 key 不完全匹配。
- 这里加载的是普通权重文件，不是 resume checkpoint。

### 第 4 步：SFTDataset 声明样本格式

File: `dataset/lm_dataset.py:58-66`

Read this to understand: SFT 数据以 `conversations` 为核心字段。

Code/config/template excerpt:

```python
features = Features({'conversations': [{'role': Value('string'), 'content': Value('string'), 'reasoning_content': Value('string'), 'tools': Value('string'), 'tool_calls': Value('string')}]})
self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)
self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
```

This code shows:

- 每条样本包含一个 `conversations` 列表。
- `bos_id` 用来识别 assistant 回复开始。
- `eos_id` 用来识别 assistant 回复结束。
- `generate_labels` 后面会用这两个 token 序列扫描 assistant 区域。

### 第 5 步：conversations 渲染成 prompt

File: `dataset/lm_dataset.py:71-86`

Read this to understand: SFT 训练时不是直接拼字符串，而是使用 tokenizer 的 chat template。

Code/config/template excerpt:

```python
def create_chat_prompt(self, conversations):
    messages = []
    tools = None
    for message in conversations:
        message = dict(message)
        if message.get("role") == "system" and message.get("tools"):
            tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
        if message.get("tool_calls") and isinstance(message["tool_calls"], str):
            message["tool_calls"] = json.loads(message["tool_calls"])
        messages.append(message)
    return self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=tools
    )
```

This code shows:

- `add_generation_prompt=False`，因为 SFT 样本已经包含 assistant 回复。
- 普通对话只需要 `role/content`。
- 工具调用字段会在 template 前转成 JSON 对象。
- 输出是 prompt 字符串，还不是 token id。

### 第 6 步：generate_labels 只保留 assistant 区域

File: `dataset/lm_dataset.py:88-104`

Read this to understand: SFT labels mask 的核心实现。

Code/config/template excerpt:

```python
labels = [-100] * len(input_ids)
i = 0
while i < len(input_ids):
    if input_ids[i:i + len(self.bos_id)] == self.bos_id:
        start = i + len(self.bos_id)
        end = start
        while end < len(input_ids):
            if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                break
            end += 1
        for j in range(start, min(end + len(self.eos_id), self.max_length)):
            labels[j] = input_ids[j]
        i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
    else:
        i += 1
```

This code shows:

- labels 初始全是 `-100`。
- 只有匹配到 assistant 开始标记后，才开始保留 label。
- 保留区域包含 assistant 内容和 eos 标记。
- user/system/padding 默认都不参与 loss。

### 第 7 步：`__getitem__` 产出训练 batch 的单样本

File: `dataset/lm_dataset.py:106-119`

Read this to understand: DataLoader 最终拿到的是 `input_ids` 和 `labels` 两个 tensor。

Code/config/template excerpt:

```python
sample = self.samples[index]
conversations = pre_processing_chat(sample['conversations'])
prompt = self.create_chat_prompt(conversations)
prompt = post_processing_chat(prompt)
input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
labels = self.generate_labels(input_ids)
return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
```

This code shows:

- prompt 会被截断到 `max_length`。
- 不足 `max_length` 的部分用 pad token 补齐。
- labels 和 input_ids 等长。
- 返回值和 PretrainDataset 一样都是 `(input_ids, labels)`，所以训练循环可以复用。

### 第 8 步：Full SFT 复用 causal LM 训练循环

File: `trainer/train_full_sft.py:24-81`

Read this to understand: SFT 的训练循环和 Pretrain 几乎同构。

Code/config/template excerpt:

```python
for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
    input_ids = input_ids.to(args.device)
    labels = labels.to(args.device)
    lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
    ...
    with autocast_ctx:
        res = model(input_ids, labels=labels)
        loss = res.loss + res.aux_loss
        loss = loss / args.accumulation_steps
    scaler.scale(loss).backward()
```

This code shows:

- SFT 仍然调用 `model(input_ids, labels=labels)`。
- `labels=-100` 决定哪些 token 不算 loss。
- optimizer、lr、AMP、checkpoint 逻辑和 Pretrain 一样。
- 因为 dataset 返回格式一致，训练循环不用关心这是 pretrain 还是 SFT。

<a id="l16-must-write"></a>
## 4. 本节必须会写 / 暂时不要求

必须会写：

1. SFT prompt 渲染：

```text
conversations
-> tokenizer.apply_chat_template(..., tokenize=False, add_generation_prompt=False)
-> prompt
```

2. assistant 区域 labels mask：

$$
y_t =
\begin{cases}
x_t, & t \in \mathcal{A} \\
-100, & t \notin \mathcal{A}
\end{cases}
$$

3. padding/truncation：

```text
input_ids = tokenizer(prompt).input_ids[:max_length]
input_ids += [pad_token_id] * (max_length - len(input_ids))
labels = generate_labels(input_ids)
```

4. Full SFT 初始化：

```text
from_weight = pretrain
save_weight = full_sft
dataset = SFTDataset
train_loop = 和 pretrain 同构
```

暂时不要求：

```text
1. tools / tool_calls 的完整训练细节
2. thinking 数据比例控制
3. 多轮复杂截断策略优化
4. SFT 训练质量评估
5. LoRA SFT
```

<a id="l16-handwrite"></a>
## 5. 手写模块

本节你要补的是：

```text
course/impl/core/datasets.py::CourseSFTDataset
```

### 5.1 补 `__init__`

对齐源码：

```text
dataset/lm_dataset.py:58-66
```

你要实现的行为：

- 读取 jsonl 数据。
- 保存 `tokenizer` 和 `max_length`。
- 构造 `bos_id`：

```python
tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
```

- 构造 `eos_id`：

```python
tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids
```

第一版可以直接用 `datasets.load_dataset`，和原源码对齐。

### 5.2 补 `create_chat_prompt`

接口：

```python
def create_chat_prompt(self, conversations):
    ...
```

对齐源码：

```text
dataset/lm_dataset.py:71-86
```

你要实现的行为：

- 遍历 `conversations`。
- 普通 message 用 `dict(message)` 复制一份。
- 如果有 `tools/tool_calls` 字符串，转成 JSON。
- 调用 `tokenizer.apply_chat_template`。
- `tokenize=False`。
- `add_generation_prompt=False`。

### 5.3 补 `generate_labels`

接口：

```python
def generate_labels(self, input_ids: list[int]) -> list[int]:
    ...
```

对齐源码：

```text
dataset/lm_dataset.py:88-104
```

你要实现的行为：

- 创建 `labels = [-100] * len(input_ids)`。
- 扫描 `input_ids`。
- 找到 `bos_id` 后，从 assistant 内容开始。
- 找到 `eos_id` 后停止。
- 把 assistant 内容和 eos 区域的 labels 设为对应 `input_ids[j]`。
- 返回 labels。

### 5.4 补 `__getitem__`

对齐源码：

```text
dataset/lm_dataset.py:106-119
```

你要实现的行为：

```text
sample
-> pre_processing_chat
-> create_chat_prompt
-> post_processing_chat
-> tokenizer
-> truncate
-> pad
-> generate_labels
-> return LongTensor(input_ids), LongTensor(labels)
```

注意：

- 要从原项目导入并复用 `pre_processing_chat` 和 `post_processing_chat`，不要重写这两个工程细节。
- 返回 tensor dtype 必须是 `torch.long`。
- `input_ids` 和 `labels` 长度都必须等于 `max_length`。

<a id="l16-alignment-test"></a>
## 6. 对齐测试

本节新增对齐测试：

```text
course/impl/tests/test_sft_dataset_alignment.py
```

运行命令：

```bash
cd /home/sun/minimind
python course/impl/tests/test_sft_dataset_alignment.py
```

现在还没有实现时，这个测试会因为 `NotImplementedError` 失败。等你补完 `CourseSFTDataset` 后，它应该打印类似：

```text
sft_sample_0_input_diff=0
sft_sample_0_label_diff=0
sft_sample_0_non_ignored_labels=...
...
sft_dataset_alignment=passed
```

这个测试做了什么：

```text
1. 用同一个 tokenizer 读取 tiny_sft.jsonl。
2. 实例化原版 SFTDataset。
3. 实例化 CourseSFTDataset。
4. 对每条样本比较 input_ids。
5. 对每条样本比较 labels。
6. 确认至少有 assistant label 没被 mask。
```

如果 input diff 是 0，但 label diff 不是 0，通常说明 `generate_labels` 的 assistant 区域边界写错了。

<a id="l16-stage-assembly"></a>
## 7. 阶段组装

本节完成后，SFT 阶段会多出数据处理能力：

```text
course/impl/core/datasets.py::CourseSFTDataset
```

后续 `course/impl/train_sft_impl.py` 会这样组装：

```text
load tokenizer
-> load CourseMiniMindForCausalLM
-> load course_pretrain checkpoint 或 MiniMind pretrain 权重
-> CourseSFTDataset
-> DataLoader
-> train_one_epoch
-> save_course_checkpoint(save_weight="course_sft")
```

当前 SFT 阶段还缺：

```text
1. 教学版模型 forward/loss 完整组装
2. train_sft_impl.py 脚本入口
3. 从 course_pretrain 到 course_sft 的权重加载策略
4. SFT 后推理验证
```

第 16 课先把 SFT 数据目标和训练脚本结构对齐。

<a id="l16-check"></a>
## 8. 本节检查

1. Full SFT 和 Pretrain 的训练循环为什么可以几乎一样？
2. Full SFT 真正不同的地方在哪里？
3. 为什么 SFT 仍然使用 causal LM next-token loss？
4. `labels=-100` 在 SFT 里屏蔽了哪些 token？
5. `add_generation_prompt=False` 的原因是什么？
6. `from_weight='pretrain'` 会加载哪个文件？
7. `save_weight='full_sft'` 会保存成什么文件？
8. 如果 `input_ids.shape = [B, S]`，写出 `labels`、`logits`、`loss` 的形状。

<a id="l16-next"></a>
## 9. 下一课

第 17 课讲从 pretrain 权重到 full_sft 权重。

下一课要解决：

```text
如何真正跑一条 tiny pretrain -> tiny SFT 链路；
from_weight 和 from_resume 在阶段切换时有什么区别；
训练前后同一个 prompt 的输出如何变化；
教学版 train_sft_impl.py 如何组装并保存 course_sft 权重。
```
