# 第 4 课：Tokenizer 与特殊标记

这一课只解决一个问题：MiniMind 如何把文本、角色和结构标记变成模型能处理的 token id。

## 目录

- [0. 本节主线](#l04-mainline)
- [1. 本节要懂的 5 个原理](#l04-principles)
- [2. 变量流转](#l04-flow)
- [3. 原理一：加载 tokenizer](#l04-load-tokenizer)
- [4. 原理二：文本怎么被切成 token](#l04-token-splitting)
- [5. 原理三：每个 token 怎么对应 id](#l04-token-id)
- [6. 原理四：哪些 token 有特殊含义](#l04-special-tokens)
- [7. 原理五：chat template 如何把 messages 变成 prompt](#l04-chat-template)
- [8. 实验验证](#l04-experiment)
- [9. 本节检查](#l04-check)
- [10. 下一课](#l04-next)

<a id="l04-mainline"></a>
## 0. 本节主线

Tokenizer 的本质是：

```text
加载 tokenizer 配置
-> 按 ByteLevel + BPE 规则切文本
-> 用 vocab 把 token 映射成 id
-> 用特殊 token 表达消息边界、思考、工具调用等结构
-> 用 chat template 把 messages 变成 prompt
-> encode 成 input_ids，必要时 decode 回文本
```

这条链对应本节的 5 个原理：

```text
加载 tokenizer -> 切文本 -> token/id 映射 -> 特殊标记 -> chat template
```

本节不训练 tokenizer。项目作者也明确不建议重复训练 tokenizer，因为 tokenizer 一旦变化，数据、权重、推理格式都会不兼容。

<a id="l04-principles"></a>
## 1. 本节要懂的 5 个原理

| 原理 | 要理解什么 | 源码证据 |
|---|---|---|
| 加载 tokenizer | tokenizer 是文本和 token id 的转换器，不是模型本身 | `eval_llm.py:13`, `model/tokenizer_config.json:334` |
| 切文本 | MiniMind 用 ByteLevel + BPE 切文本，不是按字或空格简单切 | `model/tokenizer.json:332-346`, `trainer/train_tokenizer.py:24-27` |
| token/id 映射 | tokenizer 词表规定每个 token 的 id，模型输出维度必须匹配词表 | `model/tokenizer.json:354-370`, `trainer/train_tokenizer.py:43-48`, `model/model_minimind.py:20` |
| 特殊标记 | 消息边界、padding、thinking、tool use 都是 tokenizer 能识别的 token | `model/tokenizer_config.json:5-29`, `model/tokenizer_config.json:317-325`, `trainer/train_tokenizer.py:28-48` |
| chat template | 对话列表必须先渲染成带角色标记的 prompt，推理时还要加 assistant 起始提示 | `minimind-3/chat_template.jinja:25-45`, `minimind-3/chat_template.jinja:78-84`, `eval_llm.py:76` |

学完本节，你不需要手写 BPE 算法，也不需要完整读懂 `tokenizer.json` 的所有字段。

<a id="l04-flow"></a>
## 2. 变量流转

把本节主线对应到变量：

```text
text: 普通字符串，例如 "MiniMind 是什么？"
messages: [{"role": "user", "content": text}]
prompt: chat template 渲染后的字符串
raw_tokens: token id 对应的 token 字符串
input_ids: tokenizer 编码后的 token id
decoded_text: token id 解码回来的文本
```

第 3 课里这行代码：

```python
inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)
```

背后依赖的就是本节内容：tokenizer 负责把 prompt 字符串变成 `input_ids` 和 `attention_mask`。

<a id="l04-load-tokenizer"></a>
## 3. 原理一：加载 tokenizer

### 原理讲解

推理时，模型不能直接处理字符串。字符串必须先交给 tokenizer，转成一串整数 id。

所以 tokenizer 的职责是双向转换：

```text
文本 -> token id
token id -> 文本
```

它不是模型权重，也不负责“理解问题”。它更像模型使用的字典和切分规则。模型训练时用哪套 tokenizer，推理时也必须用同一套 tokenizer。否则同一个 id 可能代表不同 token，模型输出就会乱。

MiniMind 里 tokenizer 的加载发生在推理入口：

```text
eval_llm.py:13
```

### 源码证据 A：推理代码加载 tokenizer

文件：`eval_llm.py:13`

看它是为了理解：推理入口从哪个目录加载 tokenizer。

```python
tokenizer = AutoTokenizer.from_pretrained(args.load_from)
```

这段代码说明：

- `args.load_from` 可以是 `./model`，也可以是 `./minimind-3`。
- `AutoTokenizer.from_pretrained(...)` 会读取目录里的 tokenizer 文件。
- 加载出来的 `tokenizer` 后面会负责 `apply_chat_template`、encode、decode。

### 源码证据 B：配置声明 tokenizer 类型

文件：`model/tokenizer_config.json:334`

看它是为了理解：transformers 应该按哪类 tokenizer 加载。

```json
{
  "tokenizer_class": "PreTrainedTokenizerFast"
}
```

这段配置说明：

- MiniMind 使用 transformers 的 fast tokenizer 接口。
- 底层切分和映射由 `tokenizers` 库执行。
- 你不需要手动实例化 tokenizer 类，`AutoTokenizer` 会根据配置处理。

### 理解到这一步就够

你应该能说清楚：

```text
tokenizer 是文本和 token id 的转换协议；
模型权重必须和 tokenizer 的词表、特殊标记保持一致；
MiniMind 通过 AutoTokenizer.from_pretrained(...) 加载 tokenizer。
```

暂时不用看：

- `AutoTokenizer` 内部如何选择具体类。
- `tokenizers` Rust 底层实现。

<a id="l04-token-splitting"></a>
## 4. 原理二：文本怎么被切成 token

### 原理讲解

tokenizer 不是简单按“字”切，也不是简单按“空格”切。

MiniMind 的 tokenizer 大致分两步：

```text
第一步：ByteLevel 预切分
第二步：BPE 根据 merge 规则合并常见片段
```

ByteLevel 的好处是覆盖面广。中文、英文、标点、空格、少见字符都能落到字节级表示，不容易出现完全无法处理的字符。

BPE 的作用是压缩常见片段。常见的字符组合可以合并成一个 token，这样同样一句话可以用更少 token 表示。

所以当你看到：

```text
MiniMind 是什么？
```

它不会简单变成 12 个字符，也不会只按空格切。它会经过 ByteLevel + BPE 变成一串 token。

### 源码证据 A：tokenizer.json 记录切分组件

文件：`model/tokenizer.json:332-346`

看它是为了理解：已经训练好的 tokenizer 使用什么预切分器和模型类型。

```json
"pre_tokenizer": {
  "type": "ByteLevel",
  "add_prefix_space": false
},
"model": {
  "type": "BPE"
}
```

这段配置说明：

- `pre_tokenizer.type = ByteLevel`：先按字节级规则处理文本。
- `model.type = BPE`：再使用 BPE 模型进行 token 合并。
- 这两个字段回答了“文本大致怎么被切”的问题。

### 源码证据 B：训练脚本里也是这样创建 tokenizer

文件：`trainer/train_tokenizer.py:24-27`

看它是为了理解：如果从零训练 tokenizer，作者也是显式选择 BPE + ByteLevel。

```python
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
```

这段代码说明：

- `Tokenizer(models.BPE())`：创建 BPE tokenizer。
- `pre_tokenizers.ByteLevel(...)`：设置 ByteLevel 预切分。
- 训练脚本和最终 `tokenizer.json` 是一致的。

### 理解到这一步就够

你应该能说清楚：

```text
MiniMind 的 tokenizer 不是按字或按空格切；
它先做 ByteLevel 处理，再用 BPE merge 把常见片段合并；
具体规则保存在 tokenizer.json 里。
```

暂时不用看：

- 完整 `merges` 表。
- `trim_offsets`、`use_regex` 的细节。
- raw token 里 `Ġ` 或乱码样式片段的完整含义，先通过实验观察即可。

<a id="l04-token-id"></a>
## 5. 原理三：每个 token 怎么对应 id

### 原理讲解

模型不会直接预测文字。模型预测的是 token id。

tokenizer 维护一张词表：

```text
token 字符串 -> 整数 id
```

例如 MiniMind 里：

```text
<|endoftext|> -> 0
<|im_start|>  -> 1
<|im_end|>    -> 2
```

模型最后一层输出 logits，最后一维长度就是 `vocab_size`。如果 `vocab_size=6400`，模型每个位置会输出 6400 个分数：

```text
logits[..., 0] 对应 token id 0
logits[..., 1] 对应 token id 1
...
logits[..., 6399] 对应 token id 6399
```

因此 tokenizer 的词表大小必须和模型的 `vocab_size` 一致。否则模型预测出来的 id，tokenizer 无法按同一套含义 decode；训练好的权重也会和输入输出层不匹配。

### 源码证据 A：token 到 id 的映射表

文件：`model/tokenizer.json:354-370`

看它是为了理解：token 字符串如何对应具体 id。

```json
"vocab": {
  "<|endoftext|>": 0,
  "<|im_start|>": 1,
  "<|im_end|>": 2
}
```

这段配置说明：

- `<|endoftext|>` 的 id 是 0。
- `<|im_start|>` 的 id 是 1。
- `<|im_end|>` 的 id 是 2。
- 这些 id 不是随便来的，而是 tokenizer 词表定义的。

### 源码证据 B：训练 tokenizer 时指定词表大小

文件：`trainer/train_tokenizer.py:43-48`

看它是为了理解：词表大小是在 tokenizer 训练阶段确定的。

```python
VOCAB_SIZE = 6400

trainer = trainers.BpeTrainer(
    vocab_size=vocab_size,
    special_tokens=all_special_tokens
)
```

这段代码说明：

- `VOCAB_SIZE = 6400` 是 MiniMind tokenizer 的目标词表大小。
- `BpeTrainer(... vocab_size=...)` 会在训练 tokenizer 时控制词表规模。
- `special_tokens=all_special_tokens` 保证特殊标记被放进词表。

### 源码证据 C：模型配置也使用同样的 vocab_size

文件：`model/model_minimind.py:20`

看它是为了理解：模型输出维度必须和 tokenizer 词表大小一致。

```python
self.vocab_size = kwargs.get("vocab_size", 6400)
```

这段代码说明：

- MiniMind 模型默认 `vocab_size` 也是 6400。
- 后面的 embedding 和 lm head 都依赖这个大小。
- 模型输出的 token 分数和 tokenizer 的 token id 是一一对应关系。

### 理解到这一步就够

你应该能说清楚：

```text
tokenizer.json 里保存 token -> id 的词表；
模型输出 logits 的最后一维对应这些 id；
tokenizer vocab_size 和模型 vocab_size 必须一致。
```

暂时不用看：

- 全部 6400 个 token。
- 每一条 merge 规则如何形成最终 vocab。

<a id="l04-special-tokens"></a>
## 6. 原理四：哪些 token 有特殊含义

### 原理讲解

普通 token 表示文本片段；特殊 token 表示结构。

在聊天模型里，模型不仅要知道“文字内容”，还要知道：

```text
这是谁说的？
这条消息从哪里开始？
这条消息在哪里结束？
是否有 thinking 区间？
是否有 tool call 或 tool response？
```

这些结构都要通过 token 表达。比如：

```text
<|im_start|>user
MiniMind 是什么？<|im_end|>
```

这里 `<|im_start|>` 和 `<|im_end|>` 不是给人看的装饰，而是模型训练时反复见过的结构 token。

还有一个容易混淆的点：

```text
结构标记 != 一定 special=true
```

`<think>` 是结构标记，但在当前配置里 `special=false`。这意味着它会作为一个完整 token 存在，但 `decode(..., skip_special_tokens=True)` 不会自动把它删掉。

### 源码证据 A：消息边界 token 是 special=true

文件：`model/tokenizer_config.json:5-29`

看它是为了理解：哪些 token 会被 tokenizer 当作 special token。

```json
"1": {
  "content": "<|im_start|>",
  "special": true
},
"2": {
  "content": "<|im_end|>",
  "special": true
}
```

这段配置说明：

- `<|im_start|>` 和 `<|im_end|>` 都是 tokenizer 级别的 special token。
- 当 `decode(..., skip_special_tokens=True)` 时，这类 token 会被跳过。

### 源码证据 B：bos/eos/pad/unk 的指定

文件：`model/tokenizer_config.json:317-325`

看它是为了理解：常用特殊角色分别绑定到哪个 token。

```json
"bos_token": "<|im_start|>",
"eos_token": "<|im_end|>",
"pad_token": "<|endoftext|>",
"unk_token": "<|endoftext|>"
```

这段配置说明：

- BOS 使用 `<|im_start|>`。
- EOS 使用 `<|im_end|>`。
- padding 和未知 token 都复用 `<|endoftext|>`。
- 这会影响 padding、结束判断、decode 等行为。

### 源码证据 C：训练 tokenizer 时预留结构标记

文件：`trainer/train_tokenizer.py:28-48`

看它是为了理解：这些结构标记在训练 tokenizer 时就被固定进词表。

```python
special_tokens_list = [
    "<|endoftext|>", "<|im_start|>", "<|im_end|>",
    ...
]

additional_tokens_list = [
    "<tool_call>", "</tool_call>",
    "<tool_response>", "</tool_response>",
    "<think>", "</think>"
]
```

这段代码说明：

- `<|im_start|>`、`<|im_end|>` 等核心标记不会被拆开。
- `<think>`、`<tool_call>` 等结构标记也作为完整 token 进入词表。
- 后续模型才能稳定学习这些结构。

### 理解到这一步就够

你应该能说清楚：

```text
特殊标记用于表达对话结构；
special=true 会影响 skip_special_tokens；
<think> 这类结构标记虽然是完整 token，但当前不是 special=true。
```

暂时不用看：

- 多模态 token：`vision_*`、`audio_*`。
- buffer token 的扩展用途。
- Tool call 的完整格式，后面 Tool Use 再讲。

<a id="l04-chat-template"></a>
## 7. 原理五：chat template 如何把 messages 变成 prompt

### 原理讲解

模型不能直接吃 Python 里的 messages 列表。

比如：

```python
[{"role": "user", "content": "MiniMind 是什么？"}]
```

这只是程序里的结构化数据。模型真正能处理的是 token id，而 token id 来自字符串。因此在 tokenize 之前，要先把 messages 渲染成一个字符串 prompt。

chat template 就是这个渲染规则。

它把 user 消息变成：

```text
<|im_start|>user
MiniMind 是什么？<|im_end|>
```

推理时还要在最后补上 assistant 开头：

```text
<|im_start|>assistant
<think>

</think>
```

这样模型才知道：接下来应该从 assistant 的位置继续生成。

### 源码证据 A：user / assistant 消息如何被渲染

文件：`minimind-3/chat_template.jinja:25-45`

看它是为了理解：messages 里的 role 如何变成 prompt 中的角色标记。

```jinja
{%- for message in messages %}
    {%- if (message.role == "user") %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
        {{- '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
```

这段模板说明：

- 遍历每条 message。
- user 消息被包装成 `<|im_start|>user ... <|im_end|>`。
- assistant 消息被包装成 `<|im_start|>assistant ... <|im_end|>`，并带 `<think>` 区域。

### 源码证据 B：推理时添加 assistant 起始提示

文件：`minimind-3/chat_template.jinja:78-84`

看它是为了理解：`add_generation_prompt=True` 到底加了什么。

```jinja
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if open_thinking is defined and open_thinking is true %}
        {{- '<think>\n' }}
    {%- else %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}
```

这段模板说明：

- `add_generation_prompt=True` 会在末尾加 `<|im_start|>assistant\n`。
- `open_thinking=False` 时插入空 thinking 块。
- `open_thinking=True` 时只打开 `<think>\n`，让模型自己生成 thinking 内容。

### 源码证据 C：推理代码启用 generation prompt

文件：`eval_llm.py:76`

看它是为了理解：CLI 推理确实打开了这个模板行为。

```python
inputs = tokenizer.apply_chat_template(
    conversation,
    tokenize=False,
    add_generation_prompt=True,
    open_thinking=bool(args.open_thinking)
)
```

这段代码说明：

- `conversation` 会先被渲染成 prompt 字符串。
- `tokenize=False` 表示这一步还不转成 token id。
- `add_generation_prompt=True` 让模板加 assistant 起始提示。

### 理解到这一步就够

你应该能说清楚：

```text
chat template 是 messages 和 prompt 之间的桥；
推理时必须告诉模型“接下来由 assistant 续写”；
这就是 add_generation_prompt=True 的作用。
```

暂时不用看：

- tools 分支。
- tool response 的多轮定位逻辑。
- reasoning_content 的所有兼容处理。

<a id="l04-experiment"></a>
## 8. 实验验证

### 实验 A：观察 tokenizer 基本信息

这个实验验证：

```text
token -> id
id -> token
special token 是否会被 skip
普通文本如何 encode/decode
```

运行：

```bash
cd /home/sun/minimind
PYTHONDONTWRITEBYTECODE=1 python course/labs/inspect_tokenizer.py
```

记录：

```text
vocab_size =
bos/eos/pad/unk token 和 id =
"MiniMind 是什么？" 的 input_ids =
"MiniMind 是什么？" 的 raw_tokens =
decode(skip_special_tokens=False) =
decode(skip_special_tokens=True) =
<think> 在 skip_special_tokens=True 时是否被移除 =
```

你应该理解到：

```text
同一段文本有 input_ids，也有 raw_tokens；
decode 是把 id 转回字符串；
special=true 的 token 会受 skip_special_tokens 影响。
```

### 实验 B：观察 chat template

这个实验验证：

```text
messages -> prompt
add_generation_prompt -> assistant 起始格式
open_thinking -> prompt 末尾变化
```

运行默认版本：

```bash
cd /home/sun/minimind
PYTHONDONTWRITEBYTECODE=1 python course/labs/inspect_tokenizer.py
```

再运行 open thinking 版本：

```bash
PYTHONDONTWRITEBYTECODE=1 python course/labs/inspect_tokenizer.py --open_thinking
```

记录：

```text
open_thinking=False 时 prompt 末尾是什么？
open_thinking=True 时 prompt 末尾是什么？
两者的 prompt_token_count 是否变化？
```

<a id="l04-check"></a>
## 9. 本节检查

如果你真懂了本节，应该能不看答案说清楚：

1. `tokenizer.json` 和 `tokenizer_config.json` 分别主要解决什么问题。
2. ByteLevel + BPE 大致是什么意思。
3. token 到 id 的映射在哪里，为什么它必须和模型 `vocab_size` 匹配。
4. `<|im_start|>`、`<|im_end|>`、`<think>` 分别在对话格式中起什么作用。
5. 为什么 `apply_chat_template(..., tokenize=False)` 返回的是 prompt 字符串，而不是 token id。
6. `add_generation_prompt=True` 为什么对推理重要。

<a id="l04-next"></a>
## 10. 下一课

第 5 课继续数据格式和 chat template：我们会看 SFT 数据中的 `conversations` 如何被 `SFTDataset` 转成 prompt 和 labels。
