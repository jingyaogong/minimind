# 第 5 课：SFT 数据、Prompt 与 Labels

这一课只解决一个问题：SFT 数据里的 `conversations` 如何变成模型训练用的 `input_ids` 和 `labels`。

## 目录

- [0. 本节主线](#l05-mainline)
- [1. 本节要懂的 5 个原理](#l05-principles)
- [2. 变量流转](#l05-flow)
- [3. 原理一：SFT 样本格式](#l05-sft-format)
- [4. 原理二：conversations 如何变成 prompt](#l05-conversations-prompt)
- [5. 原理三：labels 为什么要 mask](#l05-label-mask)
- [6. 原理四：padding 和 truncation](#l05-padding-truncation)
- [7. 原理五：loss 如何使用 labels](#l05-loss-labels)
- [8. 实验验证](#l05-experiment)
- [9. 本节检查](#l05-check)
- [10. 下一课](#l05-next)

<a id="l05-mainline"></a>
## 0. 本节主线

SFT 数据处理的本质是：

```text
读取 conversations
-> 用 chat template 渲染成 prompt
-> tokenizer 编码成 input_ids
-> 构造 labels
-> user/system/padding 位置设为 -100
-> assistant 回复位置保留 token id
-> 交给 causal LM loss 训练 assistant 回复
```

这条链对应本节的 5 个原理：

```text
SFT 样本格式 -> prompt 构造 -> labels mask -> padding/truncation -> loss 如何使用 labels
```

本节不是讲“怎么训练出好模型”，而是讲清楚 SFT 数据在进入模型前到底变成了什么。

<a id="l05-principles"></a>
## 1. 本节要懂的 5 个原理

| 原理 | 要理解什么 | 源码证据 |
|---|---|---|
| SFT 样本格式 | SFT 数据以多轮 `conversations` 表达 user/assistant 消息 | `course/labs/tiny_sft.jsonl:1-3`, `dataset/lm_dataset.py:59-66` |
| prompt 构造 | conversations 先经过 chat template 变成纯文本 prompt | `dataset/lm_dataset.py:71-86` |
| labels mask | SFT 不是所有 token 都算 loss，只训练 assistant 区域 | `dataset/lm_dataset.py:65-66`, `dataset/lm_dataset.py:88-104` |
| padding/truncation | prompt 被截断/补齐到固定长度，padding 不参与 loss | `dataset/lm_dataset.py:106-119` |
| loss 使用 labels | 模型 forward 会 shift logits/labels，并忽略 `-100` | `model/model_minimind.py:245-253`, `trainer/train_full_sft.py:27-37` |

学完本节，你应该能看懂一条 SFT 样本中哪些 token 参与训练，哪些 token 被忽略。

<a id="l05-flow"></a>
## 2. 变量流转

把本节主线对应到变量：

```text
sample: jsonl 里的一条样本
conversations: [{"role": "...", "content": "..."}]
prompt: chat template 渲染后的字符串
input_ids: prompt 编码后的 token id，并 pad 到 max_length
labels: 和 input_ids 等长；assistant 区域是 token id，其它位置是 -100
loss: 只在 labels != -100 的位置计算
```

你要盯住一个核心差异：

```text
input_ids 是模型看到的完整上下文；
labels 是哪些位置要让模型学习。
```

<a id="l05-sft-format"></a>
## 3. 原理一：SFT 样本格式

### 原理讲解

预训练样本通常是一段普通文本；SFT 样本是对话。

SFT 的目标不是让模型背整段 prompt，而是让模型在看到 user/system 上下文后，学会生成 assistant 回复。

所以 SFT 数据要保留角色信息：

```text
user 问了什么
assistant 应该怎么答
system 是否给了行为约束
```

这些信息在 jsonl 里用 `conversations` 表示。每条 message 至少有：

```text
role: user / assistant / system
content: 消息正文
```

### 源码证据 A：tiny SFT 样本长什么样

文件：`course/labs/tiny_sft.jsonl:1-3`

看它是为了理解：SFT 数据的最小结构。

```json
{"conversations":[
  {"role":"user","content":"MiniMind 是什么？"},
  {"role":"assistant","content":"MiniMind 是一个小型语言模型教学项目，用来学习从预训练到后训练的完整流程。"}
]}
```

这段数据说明：

- 一条样本的核心字段是 `conversations`。
- `user` 是输入上下文的一部分。
- `assistant` 是我们希望模型学会生成的目标回复。

### 源码证据 B：SFTDataset 声明 conversations schema

文件：`dataset/lm_dataset.py:59-66`

看它是为了理解：代码期望 SFT 样本有哪些字段。

```python
features = Features({'conversations': [{
    'role': Value('string'),
    'content': Value('string'),
    'reasoning_content': Value('string'),
    'tools': Value('string'),
    'tool_calls': Value('string')
}]})
self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)
```

这段代码说明：

- `SFTDataset` 会从 jsonl 读取 `conversations`。
- 除了 `role/content`，它还预留了 reasoning、tools、tool_calls 字段。
- 本节只关心普通 user/assistant 对话，工具调用后面再讲。

### 理解到这一步就够

你应该能说清楚：

```text
SFT 数据不是普通文本，而是带 role 的对话列表；
user/system 提供上下文，assistant 提供训练目标。
```

暂时不用看：

- `tools` 和 `tool_calls` 的完整格式。
- DPO/RL 数据格式。

<a id="l05-conversations-prompt"></a>
## 4. 原理二：conversations 如何变成 prompt

### 原理讲解

模型不能直接吃 Python 列表：

```python
[{"role": "user", "content": "..."}]
```

它只能吃 token id，而 token id 来自字符串。所以 SFTDataset 必须先把 `conversations` 渲染成 prompt 字符串。

这一步和第 4 课的 chat template 是同一件事，只是训练时有一个关键差异：

```text
推理时：add_generation_prompt=True，因为还没有 assistant 回复，要提示模型开始回答。
SFT 时：add_generation_prompt=False，因为样本里已经有完整 assistant 回复。
```

### 源码证据 A：SFTDataset 调用 chat template

文件：`dataset/lm_dataset.py:71-86`

看它是为了理解：SFT prompt 是怎么从 conversations 生成的。

```python
def create_chat_prompt(self, conversations):
    messages = []
    tools = None
    for message in conversations:
        message = dict(message)
        ...
        messages.append(message)
    return self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=tools
    )
```

这段代码说明：

- `messages` 是从 `conversations` 整理出来的。
- `apply_chat_template(..., tokenize=False)` 返回 prompt 字符串。
- `add_generation_prompt=False`，因为训练样本已经包含 assistant 回复。
- 如果有 tools，会传给模板；普通 SFT 样本可以先忽略。

### 源码证据 B：训练前可能做轻量 prompt 后处理

文件：`dataset/lm_dataset.py:31-35`

看它是为了理解：prompt 生成后可能会随机移除空 thinking 标签。

```python
def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content
```

这段代码说明：

- SFT prompt 中可能有空 `<think></think>` 块。
- 作者会按概率移除空 thinking 标签。
- 这是轻量数据增强，不影响本节核心：最终训练的是 prompt token 序列。

### 理解到这一步就够

你应该能说清楚：

```text
SFTDataset 先把 conversations 变成完整 prompt；
SFT 用 add_generation_prompt=False，因为 assistant 回复已经在 prompt 里。
```

暂时不用看：

- tool 模板分支。
- random system prompt 的训练策略细节。

<a id="l05-label-mask"></a>
## 5. 原理三：labels 为什么要 mask

### 原理讲解

SFT 的关键不是“把整段 prompt 都学一遍”。

一条 SFT prompt 里通常包含：

```text
system 指令
user 问题
assistant 回复
padding
```

模型训练时应该主要学习 assistant 怎么回答，而不是学习 user 问题本身。

所以 labels 会做 mask：

```text
user/system/padding 位置 -> -100
assistant 回复位置 -> 对应 token id
```

在 PyTorch 的 cross entropy 里，`ignore_index=-100` 表示这个位置不参与 loss。

### 源码证据 A：SFTDataset 先找 assistant 起止标记

文件：`dataset/lm_dataset.py:65-66`

看它是为了理解：代码如何识别 assistant 回复区域。

```python
self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
```

这段代码说明：

- `bos_id` 不是普通 BOS，而是 assistant 消息开头：`<|im_start|>assistant\n`。
- `eos_id` 是消息结束：`<|im_end|>\n`。
- 后面 `generate_labels` 会靠这两个 token 序列定位 assistant 区域。

### 源码证据 B：labels 默认全是 -100

文件：`dataset/lm_dataset.py:88-90`

看它是为了理解：默认所有位置都不参与 loss。

```python
def generate_labels(self, input_ids):
    labels = [-100] * len(input_ids)
    i = 0
```

这段代码说明：

- labels 和 input_ids 等长。
- 初始状态下每个位置都是 `-100`。
- 后面只有 assistant 区域会被改成真实 token id。

### 源码证据 C：只把 assistant 区域写回 labels

文件：`dataset/lm_dataset.py:91-104`

看它是为了理解：哪些 token 参与 SFT loss。

```python
if input_ids[i:i + len(self.bos_id)] == self.bos_id:
    start = i + len(self.bos_id)
    end = start
    while end < len(input_ids):
        if input_ids[end:end + len(self.eos_id)] == self.eos_id:
            break
        end += 1
    for j in range(start, min(end + len(self.eos_id), self.max_length)):
        labels[j] = input_ids[j]
```

这段代码说明：

- 找到 `<|im_start|>assistant\n` 后，assistant 正文从 `start` 开始。
- 一直找到 `<|im_end|>\n` 为止。
- 只有这个区间的 labels 会从 `-100` 改为真实 token id。
- user/system 仍然是 `-100`，不会贡献 loss。

### 理解到这一步就够

你应该能说清楚：

```text
input_ids 是完整 prompt；
labels 是训练目标；
SFT 只让 assistant 回复区域参与 loss，其它位置用 -100 忽略。
```

暂时不用看：

- DPO 的 chosen/rejected mask。
- PPO/GRPO 的 rollout mask。

<a id="l05-padding-truncation"></a>
## 6. 原理四：padding 和 truncation

### 原理讲解

训练通常按 batch 处理样本。一个 batch 里的 tensor 需要统一长度。

所以每条 prompt 会被处理成固定长度：

```text
太长 -> 截断到 max_length
太短 -> 用 pad_token_id 补齐
```

padding 只是为了形状对齐，不应该参与训练。因此 padding 对应的 labels 也应该保持 `-100`。

### 源码证据 A：SFTDataset 截断并补齐 input_ids

文件：`dataset/lm_dataset.py:106-113`

看它是为了理解：prompt 如何变成固定长度的 input_ids。

```python
prompt = self.create_chat_prompt(conversations)
prompt = post_processing_chat(prompt)
input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
labels = self.generate_labels(input_ids)
```

这段代码说明：

- `self.tokenizer(prompt).input_ids` 把 prompt 编码成 token id。
- `[:self.max_length]` 截断过长样本。
- 不足 `max_length` 的部分用 `pad_token_id` 补齐。
- `generate_labels(input_ids)` 在补齐后生成 labels，padding 位置会保持 `-100`。

### 源码证据 B：DataLoader 取到的是 input_ids 和 labels

文件：`dataset/lm_dataset.py:119`

看它是为了理解：SFTDataset 最终向训练脚本返回什么。

```python
return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
```

这段代码说明：

- 每条样本最终返回两个 tensor。
- 两者长度相同。
- 训练脚本不再关心原始文本，只看 `input_ids` 和 `labels`。

### 理解到这一步就够

你应该能说清楚：

```text
SFTDataset 把不同长度的 prompt 变成固定长度 input_ids；
padding 是形状占位，不参与 loss；
labels 和 input_ids 等长。
```

暂时不用看：

- DDP sampler。
- DataLoader 的性能参数。

<a id="l05-loss-labels"></a>
## 7. 原理五：loss 如何使用 labels

### 原理讲解

Causal LM 的训练目标是预测下一个 token。

如果输入是：

```text
[t0, t1, t2, t3]
```

模型在位置 0 的 logits 用来预测 `t1`，位置 1 的 logits 用来预测 `t2`，以此类推。

所以计算 loss 时会做 shift：

```text
logits[..., :-1, :] 对齐 labels[..., 1:]
```

这意味着：`labels[j]` 监督的是前一个位置对 token `j` 的预测。

当 `labels[j] == -100` 时，这个目标会被忽略。SFT 正是通过这个机制只训练 assistant 区域。

### 源码证据 A：训练脚本把 labels 传给模型

文件：`trainer/train_full_sft.py:27-37`

看它是为了理解：SFT 训练真正使用的是 `model(input_ids, labels=labels)`。

```python
for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
    input_ids = input_ids.to(args.device)
    labels = labels.to(args.device)
    ...
    res = model(input_ids, labels=labels)
    loss = res.loss + res.aux_loss
```

这段代码说明：

- DataLoader 从 `SFTDataset` 得到 `(input_ids, labels)`。
- labels 会被传进模型 forward。
- 模型内部计算 `res.loss`。

### 源码证据 B：模型 forward 做 shift 并忽略 -100

文件：`model/model_minimind.py:245-253`

看它是为了理解：labels 如何真正进入 cross entropy。

```python
if labels is not None:
    x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
    loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
```

这段代码说明：

- `logits[..., :-1, :]` 去掉最后一个位置，因为最后一个位置没有下一个 token 可预测。
- `labels[..., 1:]` 去掉第一个 label，让每个位置预测下一个 token。
- `ignore_index=-100` 表示 label 为 `-100` 的位置不参与 loss。

### 理解到这一步就够

你应该能说清楚：

```text
SFTDataset 决定哪些 token 是训练目标；
model.forward 通过 shift 做 next-token loss；
-100 是“这个位置不要算 loss”的标记。
```

暂时不用看：

- `aux_loss` 的 MoE 细节。
- 混合精度、梯度累积、优化器。

<a id="l05-experiment"></a>
## 8. 实验验证

### 实验 A：打印一条 tiny SFT 样本

这个实验验证：

```text
conversations -> prompt -> input_ids -> labels
```

运行：

```bash
cd /home/sun/minimind
PYTHONDONTWRITEBYTECODE=1 python course/labs/inspect_sft_dataset.py --show_rows 70
```

记录：

```text
input_ids.shape =
labels.shape =
non_ignored_labels =
assistant span 的起始 token =
第一段 label != -100 的位置 =
padding 位置的 label =
```

你应该看到：

```text
user token 的 label 是 -100；
assistant 回复 token 的 label 是对应 token id；
padding token 的 label 是 -100。
```

### 实验 B：换一条样本

运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python course/labs/inspect_sft_dataset.py --index 2 --show_rows 90
```

记录：

```text
system 消息是否出现在 prompt 里？
system/user token 是否参与 loss？
assistant token 从哪里开始参与 loss？
```

<a id="l05-check"></a>
## 9. 本节检查

如果你真懂了本节，应该能不看答案说清楚：

1. SFT 的 `conversations` 和普通文本预训练样本有什么不同。
2. 为什么 SFT 用 `add_generation_prompt=False`。
3. `input_ids` 和 `labels` 分别表示什么。
4. 为什么 user/system token 的 label 要设为 `-100`。
5. `labels[..., 1:]` 和 `logits[..., :-1, :]` 为什么要 shift。
6. padding token 为什么不应该参与 loss。

<a id="l05-next"></a>
## 10. 下一课

第 6 课会正式对比 PretrainDataset 和 SFTDataset：同样是 causal LM loss，为什么数据格式、labels 构造和训练目标不同。
