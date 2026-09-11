# 第 10 课：Causal LM forward 与 loss shift

这一课只解决一个问题：`MiniMindForCausalLM.forward` 如何把 `hidden_states` 变成 `logits`，再把 `logits` 和 `labels` shift 后算出 next-token loss。

## 目录

- [0. 本节主线](#l10-mainline)
- [1. 本节要懂的 5 个原理](#l10-principles)
- [2. 变量流转](#l10-flow)
- [3. 原理一：`lm_head` 把 hidden 映射成 logits](#l10-lm-head-logits)
- [4. 原理二：next-token loss 必须 shift](#l10-loss-shift)
- [5. 原理三：`-100` 决定哪些位置不参与 loss](#l10-ignore-index)
- [6. 原理四：Pretrain 和 SFT 共享同一个 forward loss](#l10-shared-loss)
- [7. 原理五：训练目标差异来自 labels](#l10-labels-target)
- [8. 实验验证](#l10-experiment)
- [9. 本节检查](#l10-check)
- [10. 下一课](#l10-next)

<a id="l10-mainline"></a>
## 0. 本节主线

Causal LM loss 的本质是：

```text
input_ids
-> MiniMindModel
-> hidden_states [batch, seq, hidden_size]
-> lm_head
-> logits [batch, seq, vocab_size]
-> logits[..., :-1, :]
-> labels[..., 1:]
-> cross_entropy(ignore_index=-100)
-> loss
```

最关键的一句话：

```text
logits 在位置 i 的输出，用来预测 labels 在位置 i+1 的 token。
```

所以模型不是预测“当前位置 token”，而是用当前位置之前的信息预测“下一个 token”。

<a id="l10-principles"></a>
## 1. 本节要懂的 5 个原理

| 原理 | 要理解什么 | 源码证据 |
|---|---|---|
| `lm_head` 把 hidden 映射成 logits | 主干输出 hidden_states，外层 CausalLM 头输出词表分数 | `model/model_minimind.py:245-249` |
| next-token loss 必须 shift | `logits[..., :-1, :]` 对齐 `labels[..., 1:]` | `model/model_minimind.py:249-252` |
| `-100` 决定哪些位置不参与 loss | cross entropy 用 `ignore_index=-100` 忽略被 mask 的 label | `model/model_minimind.py:250-252` |
| Pretrain 和 SFT 共享同一个 forward loss | 训练脚本都只是传入 `(input_ids, labels)` | `trainer/train_pretrain.py:35-40`, `trainer/train_full_sft.py:35-40` |
| 训练目标差异来自 labels | Pretrain 基本全段监督，SFT 只监督 assistant 区域 | `dataset/lm_dataset.py:47-55`, `dataset/lm_dataset.py:88-119` |

学完本节，你应该能解释：为什么 SFT 仍然是 causal LM next-token loss，以及为什么 user token 的 label 要设成 `-100`。

<a id="l10-flow"></a>
## 2. 变量流转

先看 shape：

```text
input_ids:
  [batch, seq]

hidden_states:
  [batch, seq, hidden_size]

logits:
  [batch, seq, vocab_size]

labels:
  [batch, seq]
```

再看 shift：

```text
logits[..., :-1, :]
  [batch, seq - 1, vocab_size]
  位置 0 到 seq-2 的预测分数

labels[..., 1:]
  [batch, seq - 1]
  位置 1 到 seq-1 的目标 token
```

对齐关系是：

```text
logits[:, 0, :] -> labels[:, 1]
logits[:, 1, :] -> labels[:, 2]
logits[:, 2, :] -> labels[:, 3]
...
```

所以：

```text
labels[:, 0] 不参与 loss
logits[:, -1, :] 不参与 loss
```

<a id="l10-lm-head-logits"></a>
## 3. 原理一：`lm_head` 把 hidden 映射成 logits

### 原理讲解

第 9 课讲到，`MiniMindModel` 的输出还是 hidden states：

```text
[batch, seq, hidden_size]
```

但训练语言模型时，我们需要在每个位置预测下一个 token。预测 token 就要输出词表大小的分数：

```text
[batch, seq, vocab_size]
```

这一步由 `lm_head` 完成：

```text
hidden_size -> vocab_size
```

每个位置都会得到一组长度为 `vocab_size` 的 logits。logits 还不是概率，它只是未归一化分数；cross entropy 内部会处理 softmax。

### 源码证据 A：forward 先得到 hidden，再调用 lm_head

文件：`model/model_minimind.py:245-249`

看它是为了理解：logits 是从哪里来的。

源码：

```python
def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
    hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :])
    loss = None
```

这段代码说明：

- `self.model(...)` 返回的是 `hidden_states`。
- `self.lm_head(...)` 把 hidden 映射成词表 logits。
- logits 的最后一维是 `vocab_size`。
- `loss` 初始为 `None`，只有传入 labels 才计算训练损失。

### 理解到这一步就够

你应该能说清楚：

```text
MiniMindModel 负责产生 hidden 表示；
MiniMindForCausalLM 负责把 hidden 变成 logits，并在有 labels 时计算 loss。
```

暂时不用看：

- `logits_to_keep` 的推理节省计算用途。
- softmax 的数值稳定实现。

<a id="l10-loss-shift"></a>
## 4. 原理二：next-token loss 必须 shift

### 原理讲解

Causal LM 的训练目标是：

```text
看到前面的 token，预测下一个 token。
```

如果输入是：

```text
input_ids = [A, B, C, D]
```

那么监督关系应该是：

```text
看到 A      -> 预测 B
看到 A B    -> 预测 C
看到 A B C  -> 预测 D
```

这就是为什么要 shift：

```text
logits positions: 0, 1, 2
labels positions: 1, 2, 3
```

模型输出的 `logits[:, i, :]` 不是用来预测 `input_ids[:, i]`，而是用来预测下一位 `labels[:, i+1]`。

### 源码证据 A：shift logits 和 labels

文件：`model/model_minimind.py:249-252`

看它是为了理解：next-token 对齐关系在源码里是哪一行实现的。

源码：

```python
loss = None
if labels is not None:
    x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
    loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
```

这段代码说明：

- `logits[..., :-1, :]` 去掉最后一个位置的预测。
- `labels[..., 1:]` 去掉第一个位置的目标。
- `x` 和 `y` 的 seq 维都变成 `seq-1`。
- 这样 `x[:, i, :]` 就对齐 `y[:, i]`，也就是原始的 `labels[:, i+1]`。

### 理解到这一步就够

你应该能回答：

```text
为什么 labels[0] 不参与 loss？
因为没有前一个位置的 logits 去预测它。

为什么 logits 最后一个位置不参与 loss？
因为序列里没有下一个 label 给它预测。
```

暂时不用看：

- teacher forcing 的训练范式。
- 推理时如何一步步 autoregressive generate。

<a id="l10-ignore-index"></a>
## 5. 原理三：`-100` 决定哪些位置不参与 loss

### 原理讲解

`labels` 不只是目标 token id，也是一张 loss mask。

规则是：

```text
label 是正常 token id -> 这个位置参与 loss
label 是 -100       -> 这个位置被忽略
```

PyTorch 的 cross entropy 支持 `ignore_index`。MiniMind 设置：

```text
ignore_index=-100
```

所以只要 Dataset 把某些位置的 label 写成 `-100`，这些位置就不会对 loss 和梯度产生贡献。

注意 shift 后才判断是否忽略：

```text
真正参与 loss 的是 labels[..., 1:] 里的非 -100 位置。
```

### 源码证据 A：cross entropy 忽略 `-100`

文件：`model/model_minimind.py:250-252`

看它是为了理解：`-100` 为什么不是普通 token id，而是 loss mask。

源码：

```python
if labels is not None:
    x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
    loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
```

这段代码说明：

- `y.view(-1)` 里有正常 token id，也可能有 `-100`。
- `ignore_index=-100` 让 `-100` 位置不参与 loss。
- Dataset 决定哪些位置是 `-100`，模型 forward 只负责执行这个规则。

### 理解到这一步就够

你应该能说清楚：

```text
-100 不是 tokenizer 词表里的特殊 token；
它是 PyTorch loss 的忽略标记。
```

暂时不用看：

- reduction='mean' 时忽略位置如何影响平均值。
- label smoothing。

<a id="l10-shared-loss"></a>
## 6. 原理四：Pretrain 和 SFT 共享同一个 forward loss

### 原理讲解

从训练循环看，Pretrain 和 SFT 都是同一种调用：

```python
model(input_ids, labels=labels)
```

训练脚本没有写两套 loss。模型 forward 也没有判断“现在是 pretrain 还是 SFT”。

这说明一个关键事实：

```text
阶段差异不在 loss 函数；
阶段差异在 labels 是怎么构造的。
```

Pretrain 传进来的 labels 和 SFT 传进来的 labels 不一样，所以同一个 causal LM loss 会产生不同训练目标。

### 源码证据 A：Pretrain 训练循环

文件：`trainer/train_pretrain.py:35-40`

看它是为了理解：pretrain 训练时如何调用模型 loss。

源码：

```python
with autocast_ctx:
    res = model(input_ids, labels=labels)
    loss = res.loss + res.aux_loss
    loss = loss / args.accumulation_steps

scaler.scale(loss).backward()
```

这段代码说明：

- pretrain 阶段直接把 `input_ids` 和 `labels` 交给模型。
- `res.loss` 来自 `MiniMindForCausalLM.forward`。
- 训练循环本身没有手写 next-token loss。

### 源码证据 B：SFT 训练循环

文件：`trainer/train_full_sft.py:35-40`

看它是为了理解：SFT 阶段是否用了另一套 loss。

源码：

```python
with autocast_ctx:
    res = model(input_ids, labels=labels)
    loss = res.loss + res.aux_loss
    loss = loss / args.accumulation_steps

scaler.scale(loss).backward()
```

这段代码说明：

- SFT 调用方式和 pretrain 一样。
- 模型 forward 不知道当前阶段名称。
- 训练目标差异要去 Dataset 的 labels 里找。

### 理解到这一步就够

你应该能回答：

```text
为什么 SFT 仍然可以使用 causal LM next-token loss？
因为 SFT 只是把对话 prompt 的 assistant 区域设成监督目标，其余位置设成 -100。
```

暂时不用看：

- SFT 超参数选择。
- SFT 是否会注入新知识。

<a id="l10-labels-target"></a>
## 7. 原理五：训练目标差异来自 labels

### 原理讲解

Pretrain 和 SFT 的 `input_ids` 都是 token 序列，但 labels 的含义不同。

Pretrain：

```text
labels = input_ids.clone()
padding labels = -100
```

这表示普通文本几乎每个 token 都参与 next-token prediction。

SFT：

```text
labels 默认全是 -100
只把 assistant 回答区改成 token id
```

这表示模型只在 assistant 回答区域被监督。user/system/tool prompt 只是条件上下文，不作为要模仿的答案。

关键点：

```text
SFT 不是让模型学习“复读 user 问题”；
SFT 是让模型在 user/system 上下文之后，学习 assistant 应该怎么回答。
```

### 源码证据 A：Pretrain labels

文件：`dataset/lm_dataset.py:47-55`

看它是为了理解：pretrain 为什么几乎全段参与 loss。

源码：

```python
def __getitem__(self, index):
    sample = self.samples[index]
    tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
    tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
    input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    labels = input_ids.clone()
    labels[input_ids == self.tokenizer.pad_token_id] = -100
    return input_ids, labels
```

这段代码说明：

- `labels = input_ids.clone()` 是 pretrain 的核心。
- padding 位置改成 `-100`，不参与 loss。
- 非 padding 文本 token 基本都作为 next-token 监督目标。

### 源码证据 B：SFT labels

文件：`dataset/lm_dataset.py:88-119`

看它是为了理解：SFT 如何只监督 assistant 区域。

源码：

```python
def generate_labels(self, input_ids):
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
    return labels
...
labels = self.generate_labels(input_ids)
return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
```

这段代码说明：

- SFT labels 初始全部是 `-100`。
- 代码找到 assistant 回答区域后，才把对应 label 改成 token id。
- user/system/padding 默认保持 `-100`，所以不参与 loss。

### 理解到这一步就够

你应该能说清楚：

```text
同一个 forward loss；
不同 Dataset 构造出来的 labels；
最终得到不同训练目标。
```

暂时不用看：

- tool call 样本的完整标签细节。
- 多轮对话中多个 assistant span 的边界情况。

<a id="l10-experiment"></a>
## 8. 实验验证

### 实验 A：手动复现 SFT 的 shifted loss

这个实验验证：

```text
model.forward 的 loss
等于
logits[..., :-1, :] 与 labels[..., 1:] 手动 cross_entropy 的 loss
```

运行：

```bash
cd /home/sun/minimind
PYTHONDONTWRITEBYTECODE=1 python course/labs/trace_loss_shift.py --mode sft --show_rows 70
```

记录：

```text
logits.shape =
shift_logits.shape =
shift_labels.shape =
model_forward_loss =
manual_shift_loss =
abs_diff =
active_loss_positions =
ignored_loss_positions =
```

你应该看到：

```text
abs_diff 接近 0
shift_logits.shape = [1, seq-1, vocab_size]
shift_labels.shape = [1, seq-1]
SFT 里大量位置 participates=False
```

这说明模型源码里的 loss 就是 shifted next-token cross entropy。

### 实验 B：对比 Pretrain 的参与位置

运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python course/labs/trace_loss_shift.py --mode pretrain --show_rows 50
```

记录：

```text
active_loss_positions =
ignored_loss_positions =
哪些位置是 padding 导致的 ignored
```

你应该看到：

- pretrain 的非 padding 位置基本都参与 loss。
- SFT 的 user/system/prompt 大量不参与 loss。
- 两者 `manual_shift_loss` 都和 `model_forward_loss` 一致。

<a id="l10-check"></a>
## 9. 本节检查

如果你真懂了本节，应该能不看答案说清楚：

1. 为什么 `logits[..., :-1, :]` 要对齐 `labels[..., 1:]`。
2. `logits[:, i, :]` 预测的是哪个位置的 token。
3. 为什么 `labels[:, 0]` 不参与 loss。
4. `ignore_index=-100` 在 MiniMind 里起什么作用。
5. 为什么 SFT 不需要单独写一个 SFT loss。
6. Pretrain 和 SFT 的训练目标差异到底来自哪里。

<a id="l10-next"></a>
## 10. 下一课

第 11 课进入 `Attention 原理与源码`：我们会看 `q_proj / k_proj / v_proj` 如何把 `[batch, seq, hidden_size]` 拆成多头 Q/K/V，并开始解释 causal mask 与注意力权重。
