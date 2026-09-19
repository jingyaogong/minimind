# 第 6 课：Pretrain 与 SFT 的数据目标对比

这一课只解决一个问题：Pretrain 和 SFT 都用 causal LM next-token loss，为什么训练目标却不一样。

## 目录

- [0. 本节主线](#l06-mainline)
- [1. 本节要懂的 5 个原理](#l06-principles)
- [2. 变量流转](#l06-flow)
- [3. 原理一：同一个训练接口](#l06-same-interface)
- [4. 原理二：Pretrain 的训练目标](#l06-pretrain-target)
- [5. 原理三：SFT 的训练目标](#l06-sft-target)
- [6. 原理四：同一个 causal LM loss](#l06-causal-lm-loss)
- [7. 原理五：真正差异在哪里](#l06-real-difference)
- [8. 实验验证](#l06-experiment)
- [9. 本节检查](#l06-check)
- [10. 下一课](#l06-next)

<a id="l06-mainline"></a>
## 0. 本节主线

Pretrain 和 SFT 的共同点是：

```text
都把文本变成 input_ids
-> 都构造 labels
-> 都调用 model(input_ids, labels=labels)
-> 都用 causal LM shift 后的 cross entropy
```

它们的关键差异是：

```text
Pretrain: 普通文本几乎全段参与 next-token loss
SFT: 对话 prompt 里只让 assistant 区域参与 loss
```

所以本节的核心不是训练循环，而是：

```text
同一个 loss 函数
不同的数据格式
不同的 labels mask
不同的学习目标
```

<a id="l06-principles"></a>
## 1. 本节要懂的 5 个原理

| 原理 | 要理解什么 | 源码证据 |
|---|---|---|
| 同一个训练接口 | Pretrain 和 SFT 最后都返回 `(input_ids, labels)` | `trainer/train_pretrain.py:27-37`, `trainer/train_full_sft.py:27-37` |
| Pretrain 目标 | 普通文本加 BOS/EOS，非 padding token 基本都参与 loss | `dataset/lm_dataset.py:37-55` |
| SFT 目标 | 对话 prompt 中只保留 assistant 区域作为 labels | `dataset/lm_dataset.py:58-119` |
| 同一个 causal LM loss | 模型 forward 不知道数据来自 Pretrain 还是 SFT，只看 labels | `model/model_minimind.py:245-253` |
| 真正差异 | 两阶段的训练差异主要由 Dataset 构造的 labels 决定 | `trainer/train_pretrain.py:134-136`, `trainer/train_full_sft.py:134-137` |

学完本节，你应该能解释：为什么换一个 Dataset，就能把同一个模型 loss 变成“预训练目标”或“SFT 目标”。

<a id="l06-flow"></a>
## 2. 变量流转

Pretrain：

```text
{"text": "..."}
-> tokenizer(text)
-> [bos] + text_tokens + [eos] + padding
-> labels = input_ids.clone()
-> padding labels = -100
```

SFT：

```text
{"conversations": [...]}
-> chat template prompt
-> tokenizer(prompt)
-> padding
-> labels 默认全 -100
-> assistant 区域 labels = 对应 input_ids
```

共同进入训练循环后：

```text
model(input_ids, labels=labels)
-> logits[..., :-1, :]
-> labels[..., 1:]
-> cross_entropy(ignore_index=-100)
```

<a id="l06-same-interface"></a>
## 3. 原理一：同一个训练接口

### 原理讲解

Pretrain 和 SFT 看起来是不同训练阶段，但进入训练循环时，接口是一样的：

```text
DataLoader 产出 input_ids, labels
模型接收 input_ids, labels
模型返回 loss
优化器反向传播
```

也就是说，训练脚本不需要知道“这条样本是普通文本还是对话”。差异已经提前被 Dataset 编码进了 `labels`。

### 源码证据 A：Pretrain 训练循环接收 input_ids, labels

文件：`trainer/train_pretrain.py:27-37`

看它是为了理解：预训练循环最终也是把 labels 传给模型。

```python
for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
    input_ids = input_ids.to(args.device)
    labels = labels.to(args.device)
    ...
    res = model(input_ids, labels=labels)
    loss = res.loss + res.aux_loss
```

这段代码说明：

- 预训练 DataLoader 输出 `(input_ids, labels)`。
- 模型 forward 接收 `labels` 并计算 loss。
- 训练循环本身没有特殊的“预训练 loss”逻辑。

### 源码证据 B：SFT 训练循环接口相同

文件：`trainer/train_full_sft.py:27-37`

看它是为了理解：SFT 训练循环和预训练循环在核心调用上相同。

```python
for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
    input_ids = input_ids.to(args.device)
    labels = labels.to(args.device)
    ...
    res = model(input_ids, labels=labels)
    loss = res.loss + res.aux_loss
```

这段代码说明：

- SFT 也是 `model(input_ids, labels=labels)`。
- 训练循环不负责区分 user/assistant。
- 区分逻辑已经在 `SFTDataset` 的 labels 里完成。

### 理解到这一步就够

你应该能说清楚：

```text
Pretrain 和 SFT 的训练循环核心接口一样；
差异主要在 Dataset 如何构造 input_ids 和 labels。
```

暂时不用看：

- 学习率、混合精度、梯度累积。
- checkpoint 保存逻辑。

<a id="l06-pretrain-target"></a>
## 4. 原理二：Pretrain 的训练目标

### 原理讲解

Pretrain 的数据是一段普通文本。

目标很直接：

```text
给模型一段文本，让它学会预测下一个 token。
```

所以 labels 基本就是 input_ids 的复制。只有 padding 位置不该训练，因此 padding label 会设成 `-100`。

注意：虽然 `labels = input_ids.clone()`，模型 forward 里会做 shift：

```text
logits[..., :-1, :] 对齐 labels[..., 1:]
```

所以真正监督的是“前一个位置预测下一个 token”。

### 源码证据 A：PretrainDataset 读取 text 字段

文件：`dataset/lm_dataset.py:47-50`

看它是为了理解：预训练样本从普通 `text` 字段开始。

```python
sample = self.samples[index]
tokens = self.tokenizer(
    str(sample['text']),
    add_special_tokens=False,
    max_length=self.max_length - 2,
    truncation=True
).input_ids
tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
```

这段代码说明：

- 预训练样本只需要 `text` 字段。
- 文本先 tokenized。
- 代码手动加 BOS 和 EOS。
- `max_length - 2` 是为了给 BOS/EOS 留位置。

### 源码证据 B：Pretrain labels 基本复制 input_ids

文件：`dataset/lm_dataset.py:51-55`

看它是为了理解：预训练目标几乎覆盖整段文本。

```python
input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
input_ids = torch.tensor(input_ids, dtype=torch.long)
labels = input_ids.clone()
labels[input_ids == self.tokenizer.pad_token_id] = -100
return input_ids, labels
```

这段代码说明：

- `input_ids` 会被 padding 到固定长度。
- `labels = input_ids.clone()`，非 padding 位置都保留 token id。
- padding 位置改成 `-100`，不参与 loss。

### 理解到这一步就够

你应该能说清楚：

```text
Pretrain 学的是普通文本的 next-token prediction；
labels 基本复制 input_ids；
只有 padding 位置被 -100 忽略。
```

暂时不用看：

- 数据集来源和清洗流程。
- 超长文本如何分块采样。

<a id="l06-sft-target"></a>
## 5. 原理三：SFT 的训练目标

### 原理讲解

SFT 的数据是对话 prompt，不是普通文本。

一条 SFT prompt 里既有 user 输入，也有 assistant 回复：

```text
<|im_start|>user
问题<|im_end|>
<|im_start|>assistant
回答<|im_end|>
```

训练时，模型应该看到完整 prompt 作为上下文，但只学习 assistant 回复区域。

所以 SFT 的 labels 是：

```text
user/system 区域 -> -100
assistant 区域 -> token id
padding 区域 -> -100
```

这和 Pretrain 最大的差异就是 labels mask。

### 源码证据 A：SFT prompt 先由 chat template 生成

文件：`dataset/lm_dataset.py:71-86`

看它是为了理解：SFT 先把对话变成完整 prompt。

```python
return self.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
    tools=tools
)
```

这段代码说明：

- SFT 使用 chat template。
- `add_generation_prompt=False`，因为训练样本里已经有 assistant 回复。
- prompt 里同时包含 user 和 assistant。

### 源码证据 B：SFT labels 默认全部忽略

文件：`dataset/lm_dataset.py:88-90`

看它是为了理解：SFT 默认不训练任何位置。

```python
def generate_labels(self, input_ids):
    labels = [-100] * len(input_ids)
    i = 0
```

这段代码说明：

- SFT labels 初始全是 `-100`。
- 只有后面识别出的 assistant 区域会被恢复成 token id。

### 源码证据 C：只训练 assistant span

文件：`dataset/lm_dataset.py:91-104`

看它是为了理解：assistant 区域如何被选出来。

```python
if input_ids[i:i + len(self.bos_id)] == self.bos_id:
    start = i + len(self.bos_id)
    ...
    for j in range(start, min(end + len(self.eos_id), self.max_length)):
        labels[j] = input_ids[j]
```

这段代码说明：

- 找到 assistant 消息开头后，才开始恢复 labels。
- assistant 回复和结尾 `<|im_end|>` 会参与训练。
- user/system 区域仍然是 `-100`。

### 理解到这一步就够

你应该能说清楚：

```text
SFT 让模型看到完整对话上下文；
但 loss 只训练 assistant 回复区域；
这靠 labels 中的 -100 mask 实现。
```

暂时不用看：

- Tool use SFT 样本。
- 多轮 assistant 区域的复杂情况。

<a id="l06-causal-lm-loss"></a>
## 6. 原理四：同一个 causal LM loss

### 原理讲解

模型 forward 并不知道数据来自 Pretrain 还是 SFT。

它只做一件事：

```text
如果传了 labels，就计算 next-token cross entropy。
```

Pretrain 和 SFT 的差异不是 loss 函数不同，而是 labels 不同。

```text
Pretrain labels：大部分位置是 token id
SFT labels：只有 assistant 位置是 token id
```

`ignore_index=-100` 让同一个 loss 可以适配两种训练目标。

### 源码证据 A：模型 forward 做统一的 shift loss

文件：`model/model_minimind.py:245-253`

看它是为了理解：模型只根据 labels 计算统一的 causal LM loss。

```python
if labels is not None:
    x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
    loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
```

这段代码说明：

- `logits[..., :-1, :]` 用前面位置的输出。
- `labels[..., 1:]` 是下一个 token 的目标。
- `ignore_index=-100` 让 masked 位置不参与 loss。
- 这段代码没有区分 Pretrain/SFT。

### 理解到这一步就够

你应该能说清楚：

```text
同一个 causal LM loss 可以训练 Pretrain 和 SFT；
因为 Dataset 已经通过 labels 决定哪些位置参与 loss。
```

暂时不用看：

- `aux_loss`。
- MoE router loss。
- label smoothing 等其它 loss 变体。

<a id="l06-real-difference"></a>
## 7. 原理五：真正差异在哪里

### 原理讲解

现在可以把两阶段差异归纳成一句话：

```text
Pretrain 和 SFT 的模型 forward 与训练循环几乎一样；
真正差异在 Dataset 构造 input_ids/labels 的方式。
```

这也是很多 LLM 训练代码的常见结构：

```text
模型保持 causal LM
loss 保持 next-token prediction
通过数据格式和 mask 改变训练目标
```

### 源码证据 A：预训练脚本使用 PretrainDataset

文件：`trainer/train_pretrain.py:134-136`

看它是为了理解：预训练阶段选择的是普通文本数据集。

```python
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
```

这段代码说明：

- 预训练脚本把数据交给 `PretrainDataset`。
- 也就是由 `PretrainDataset` 决定 labels 基本复制 input_ids。

### 源码证据 B：SFT 脚本使用 SFTDataset

文件：`trainer/train_full_sft.py:134-137`

看它是为了理解：SFT 阶段选择的是对话数据集。

```python
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
```

这段代码说明：

- SFT 脚本把数据交给 `SFTDataset`。
- 也就是由 `SFTDataset` 决定只训练 assistant 区域。

### 理解到这一步就够

你应该能说清楚：

```text
训练阶段名称不同，不代表模型 forward 一定不同；
Pretrain 与 SFT 的主要区别在 Dataset 和 labels mask。
```

暂时不用看：

- 从 pretrain 权重加载到 SFT 的完整流程。
- 训练超参数如何选择。

<a id="l06-experiment"></a>
## 8. 实验验证

### 实验 A：对比 tiny pretrain 和 tiny SFT

这个实验验证：

```text
同样输出 input_ids / labels；
Pretrain 和 SFT 的 labels mask 完全不同。
```

运行：

```bash
cd /home/sun/minimind
PYTHONDONTWRITEBYTECODE=1 python course/labs/compare_pretrain_sft.py --show_rows 65
```

记录：

```text
Pretrain non_ignored_labels =
SFT non_ignored_labels =
Pretrain 第一个 padding label =
SFT 第一个 assistant label != -100 的位置 =
SFT user token 的 label =
```

你应该看到：

```text
Pretrain: labels 大部分复制 input_ids，padding 是 -100。
SFT: user/system/padding 是 -100，assistant 回复区才是 token id。
```

### 实验 B：换一条 SFT 样本

运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python course/labs/compare_pretrain_sft.py --sft_index 2 --show_rows 80
```

记录：

```text
system token 是否参与 SFT loss？
assistant 区域从哪个 token 后开始？
padding 是否参与 loss？
```

<a id="l06-check"></a>
## 9. 本节检查

如果你真懂了本节，应该能不看答案说清楚：

1. PretrainDataset 的 labels 为什么基本等于 input_ids。
2. SFTDataset 为什么不能让 user token 参与 loss。
3. Pretrain 和 SFT 为什么可以共用同一个 `model.forward` loss。
4. `ignore_index=-100` 在两个数据集里分别忽略了什么。
5. 为什么说两阶段差异主要在 Dataset，而不是训练循环。

<a id="l06-next"></a>
## 10. 下一课

第 7 课进入 `train_pretrain.py`：我们会从训练入口开始，看模型、tokenizer、dataset、optimizer、loss、保存权重是如何串起来的。
