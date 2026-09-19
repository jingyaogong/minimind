# 第4节：Tokenizer 与 Special Tokens

> 我的问题笔记 | 聚焦真正不懂的地方

---

## 📚 本节目录

1. [Q1: truncation=True 是什么意思？](#q1-truncationtrue-是什么意思)
2. [Q2: tokenizer 怎么把文字转成 token ids？](#q2-tokenizer-怎么把文字转成-token-ids)
3. [Q3: 词表里 token id 是怎么分配的？](#q3-词表里-token-id-是怎么分配的)
4. [Q4: special tokens 是什么？](#q4-special-tokens-是什么)
5. [Q5: `<|im_start|>` 这些标记怎么工作的？](#q5-im_start-这些标记怎么工作的)
6. [Q6: chat_template 是什么？](#q6-chat_template-是什么)
7. [Q7: `tokenizer.apply_chat_template` 做了什么？](#q7-tokenizerapply_chat_template-做了什么)
8. [Q8: 什么是预训练 tokenizer？](#q8-什么是预训练-tokenizer)
9. [Q9: `from pretrained()` 做了什么？](#q9-from-pretrained-做了什么)
10. [Q10: tokenizer 的训练语料从哪来？](#q10-tokenizer-的训练语料从哪来)
11. [训练流程](#训练流程)
12. [Encode / Decode](#encode--decode)
13. [一句话总结](#一句话总结)

---

## Q1: truncation=True 是什么意思？

**代码** `eval_llm.py:78`
```python
inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)
```

如果输入文本太长（超过 max_position_embeddings），**截断**多余部分。

**举例**：
```python
text = "很长很长的文本..." * 10000  # 假设有5万个token

# 报错：超出模型上限
inputs = tokenizer(text)

# truncation=True：自动截断到模型能处理的长度
inputs = tokenizer(text, truncation=True)
```

---

## Q2: attention_mask 里的"有效输入"是什么？

**代码** `eval_llm.py:83`
```python
attention_mask=inputs["attention_mask"]
```

GPU batch 处理要求固定 shape，会 padding。attention_mask 用 **1=有效，0=padding** 标记哪些是真 token。

**举例**：
```python
# batch里2个句子，长度不同
sentence_a = [1, 2, 3]        # 长度3
sentence_b = [4, 5, 6, 7, 8]  # 长度5

# padding后
token_ids = [[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]]

# attention_mask 标记
attention_mask = [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
#             ← 前3有效，后2是padding →  ← 全部有效
```

Attention 计算时，0的位置会被**忽略**。

---

## Q3: generate 为什么叫"自回归"？

**代码** `eval_llm.py:82-87`
```python
generated_ids = model.generate(inputs=inputs["input_ids"], ...)
```

**自回归** = 每一步的输出会成为下一步的输入。自己生成的东西又喂回去。

**生成过程**：
```
用户: "你好"

Step1 → 输入"你好"     → 输出 "，"
Step2 → 输入"你好，"   → 输出 "我"
Step3 → 输入"你好，我" → 输出 "是"
Step4 → 输入"你好，我是"→ 输出 "AI"
...一直重复直到 EOS 或 max_tokens
```

**对比**：非自回归是一次性输出所有 token，像"看一眼全说完"。

---

## Q4: repeat 是在干嘛？

**代码** `model/model_minimind.py`
```python
input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
attention_mask = attention_mask.repeat(num_return_sequences, 1) if ...
```

复制 batch 维度，**一次生成多个不同回复**。

**举例**：
```python
input_ids = [[1, 2, 3]]  # 1个样本

input_ids.repeat(3, 1)
# → [[1, 2, 3],   # 样本1
#    [1, 2, 3],   # 样本2
#    [1, 2, 3]]   # 样本3

# 一次forward同时生成3个答案，比调3次generate省计算
```

---

## Q5: past_key_values 是什么？

**代码** `model/model_minimind.py`
```python
past_key_values = kwargs.pop("past_key_values", None)
```

这就是 **KV Cache**。把之前算好的 K/V 存起来，下一步**直接复用**，不用重新算所有历史的 attention。

**为什么需要**：省计算。Attention O(n²) 成本高，缓存历史 K/V 可以避免重复计算。

**举例**：
```
生成第1个token后，past_key_values = [(k0,v0), (k1,v1), (k2,v2)]  # 每层

生成第2个token时：
  - 只算新token的K/V
  - 历史K/V从缓存取，不用重算
```

---

## Q6: ByteLevel 预切分是什么？

UTF-8 里任何字符都转成字节。**不存在"这个字无法处理"的情况**。

| 字符 | UTF-8 字节数 |
|------|-------------|
| `h` | 1 byte |
| `你` | 3 bytes (`\xe4\xbd\xa0`) |
| `🎉` | 4 bytes |

**举例**：
```
"hello你"
→ [h, e, l, l, o, Ġ, 你]  # 实际底层是字节
```

混合语言不怕，最终都是字节。

---

## Q7: BPE 需要什么？不然切什么合什么？

**ByteLevel** 先把语料切成字节，然后 **BPE 统计语料中哪些字节对最常出现，频繁出现的就合并成新 token**。

**训练过程举例**：
```
语料中大量 "banana"

Step1: ByteLevel → [b, a, n, a, n, a]

Step2: 统计频率
  "an" 出现4次 ← 最常见，先合
  "na" 出现2次
  "ba" 出现1次

Step3: 迭代合并
  第1次: [b, a, n, a, n, a] → [b, an, an]
  第2次: [b, an, an] → [ban, an]

最终 "banana" = 2个token [ban, an]，而不是6个字节
```

**好处**：常见词合并成1个token（序列短），生僻字拆成字节（不会OOV）。

---

## Q8: from_pretrained 和 tokenizer_config.json 什么关系？

**调用** `eval_llm.py:13`
```python
tokenizer = AutoTokenizer.from_pretrained(args.load_from)
```

**配置** `minimind-3/tokenizer_config.json:330`
```json
{
  "tokenizer_class": "PreTrainedTokenizerFast"
}
```

**关系**：
```
AutoTokenizer.from_pretrained("./minimind-3")
    ↓ 读取 tokenizer_config.json
    ↓ 发现 "tokenizer_class": "PreTrainedTokenizerFast"
    ↓ 自动用这个类加载
```

`tokenizer_config.json` = 配方卡，`from_pretrained` = 自动点餐。

---

## Q9: PreTrainedTokenizerFast 是什么？minimind 实现了吗？

`PreTrainedTokenizerFast` 是 **HuggingFace transformers 库提供的类**（Rust 编写，速度快）。

minimind **没有自己实现**，只是定义了词表和配置，兼容这个接口。

| 类比 | 说明 |
|------|------|
| 公共汽车 | `PreTrainedTokenizerFast`（HuggingFace 提供） |
| 乘客名单 | `tokenizer.json`（minimind 定义） |
| 车辆配置 | `tokenizer_config.json`（minimind 定义） |

---

## Q10: minimind 的 tokenizer 训练代码逻辑是什么？

**文件** `trainer/train_tokenizer.py`

minimind 用 **tokenizers 库 + BPE 算法**训练 tokenizer，训练数据是 `dataset/sft_t2t_mini.jsonl`（取前1万行）。

### 第1步：准备语料

```python
def get_texts(data_path):
    for i, line in enumerate(f):
        if i >= 10000: break  # 取1万行
        data = json.loads(line)
        contents = [item.get('content') for item in ...]
        yield "\n".join(contents)
```

从 jsonl 里提取 conversation 的 content，yield 出来供 BPE 训练用。

### 第2步：创建 tokenizer 对象

```python
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
```

- 用 **BPE 模型**
- 用 **ByteLevel** 做预切分（UTF-8 字节级）

### 第3步：定义 special tokens

```python
special_tokens_list = [
    "<|endoftext|>", "<|im_start|>", "<|im_end|>",
    "<|vision_start|>", "<|vision_end|>", "<|image_pad|>",
    "<tool_call>", "<tool_response>", "<think>", "</think>",
    ...
]

additional_tokens_list = ["<tts_pad>", "<tts_text_bos>", ...]

buffer_tokens = [f"<|buffer{i}|>" for i in range(1, num_buffer+1)]  # 预留扩展位

all_special_tokens = special_tokens_list + additional_tokens_list + buffer_tokens
```

**为什么要有 buffer_tokens**？预留空位，方便以后扩展新的 special token。

### 第4步：训练 BPE

```python
trainer = trainers.BpeTrainer(
    vocab_size=6400,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    special_tokens=all_special_tokens
)

texts = get_texts(data_path)
tokenizer.train_from_iterator(texts, trainer=trainer)
```

- **vocab_size=6400**：最终词表大小
- **ByteLevel.alphabet()**：用字节级字母表
- **special_tokens=all_special_tokens**：固定这些 tokens，不让 BPE 拆分

**special_tokens 的处理方式**：

| 处理 | 说明 |
|------|------|
| 优先加入 vocab | 训练前就占好位置 |
| 不参与统计 | BPE 统计字节对频率时跳过它们 |
| 不被拆分 | 永远保持独立 token，不会被切成碎片 |

**举例**：
```
输入: "hello<|im_start|>你好"
ByteLevel: [h, e, l, l, o, Ġ, <|im_start|>, 你, 好]

BPE 训练后:
- "h", "e", "l", "o", "你", "好" 可能被合并
- "<|im_start|>" 永远不会被合并或拆分
- 它就是一个完整的 token，id 固定
```

这样设计是为了保证 `<|im_start|>`、`<think>` 这些**结构标记**在编码解码时保持完整。

### 训练过程

```
语料: "banana" 大量出现

ByteLevel 切: [b, a, n, a, n, a]

统计频率:
  "an" 出现4次 ← 最常见，先合并
  "na" 出现2次
  "ba" 出现1次

迭代合并:
  第1次: [b, an, an]
  第2次: [ban, an]

最终: "banana" = 2个token [ban, an]
```

### 第5步：保存

```python
tokenizer.decoder = decoders.ByteLevel()
tokenizer.add_special_tokens(special_tokens_list)

tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
tokenizer.model.save(tokenizer_dir)  # 保存 vocab.json 和 merges.txt
```

**`tokenizer.decoder = decoders.ByteLevel()`**：设置解码器。encode 用 ByteLevel 切成字节，decode 也要用 ByteLevel 还原回字节再转字符串。

**`tokenizer.add_special_tokens(special_tokens_list)`**：训练后把 special tokens 强制加入词表。

| 步骤 | 代码 | 作用 |
|------|------|------|
| 训练时 | `BpeTrainer(special_tokens=all_special_tokens)` | 训练过程中**不拆分**这些 tokens |
| 训练后 | `tokenizer.add_special_tokens(special_tokens_list)` | 把这些 tokens **加入词表**，确保能用 `tokenizer.encode("<\|im_start\|>")` 返回对应 id |

**举例**：
```python
# 训练完成后
tokenizer.encode("<|im_start|>")
# 可能报错：这个 token 不在词表里！

# 执行 add_special_tokens 后
tokenizer.add_special_tokens(special_tokens_list)
tokenizer.encode("<|im_start|>")
# 正常返回: [5]（假设 id=5）
```

**`tokenizer.save(...)`**：保存 tokenizer.json（HuggingFace 兼容格式，可直接用 `AutoTokenizer.from_pretrained()` 加载）。

**`tokenizer.model.save(...)`**：保存 BPE 核心文件。

| 文件 | 内容 |
|------|------|
| `vocab.json` | token → id 映射 |
| `merges.txt` | BPE 合并规则（按优先级排列） |

**举例**：
```
原始: [b, a, n, a, n, a]
按 merges.txt 合并:
1. 找 "an" → 合并 → [b, an, an]
2. 找 "ban" → 合并 → [ban, an]
最终: [ban, an]
```

### 第6步：修正 special 标记

```python
# 遍历 tokenizer.json 里的 added_tokens
# special_tokens_list 里的 → special=True
# 其他（buffer 等）→ special=False
# 特别注意: <think> 是 special=False，所以 skip_special_tokens=True 时会保留
```

**`special` vs `skip_special_tokens` 的区别**：

| 概念 | 说明 |
|------|------|
| `special` | token 的**固有属性**（是或不是特殊标记），写在 tokenizer.json 里 |
| `skip_special_tokens` | `decode()` 的**开关参数**，控制是否跳过 `special=True` 的 token |

**skip=True 时会发生什么**：
- 解码时，遇到 `special=True` 的 token → **不显示**，直接忽略
- 遇到 `special=False` 的 token → **正常显示**

**举例**：
```python
# 假设:
# <|im_start|> special=True
# <think> special=False
# <buffer1> special=False

input_ids = [<|im_start|>, 你, 好, ，, <think>, 我, 在, 想, <buffer1>]

# skip=True（跳过 special=True 的标记）
decode(input_ids, skip_special_tokens=True)
# 输出: "你好，我在想<buffer1>"
#           ↑ <|im_start|> 被跳过了，<think> 还在

# skip=False（保留所有）
decode(input_ids, skip_special_tokens=False)
# 输出: "<|im_start|>你好，我在想<buffer1>"
```

### 第7步：生成 tokenizer_config.json

```python
config = {
    "added_tokens_decoder": {...},
    "bos_token": "<|im_start|>",
    "eos_token": "<|im_end|>",
    "pad_token": "<|endoftext|>",
    "chat_template": "{%- ... %}",  # Jinja2 模板
    "tokenizer_class": "PreTrainedTokenizerFast"
}
```

`tokenizer_config.json` 是 tokenizer 的配置文件，告诉 HuggingFace 怎么用这个 tokenizer。

| 配置 | 作用 |
|------|------|
| `tokenizer_class` | 用 `PreTrainedTokenizerFast` 类加载 |
| `bos_token` / `eos_token` | 句子开头/结尾标记 |
| `pad_token` | 批量处理时的填充标记 |
| `added_tokens_decoder` | 每个 token id 对应的详细信息 |
| `chat_template` | 怎么把对话格式化成模型能理解的 prompt |

---

## 训练完怎么用（Encode / Decode）

### Encode（文字 → token ids）

```
输入: "hello你"

Step 1: ByteLevel 预切分
[h, e, l, l, o, Ġ, 你]  ← 字节序列（英文1字节，中文3字节）

Step 2: BPE merge 规则合并
[he, ll, o, Ġ, 你]       ← 查 merges.txt，常见组合合并

Step 3: 查 vocab.json（token → id）
[32, 15, 15, 8, 200]     ← 最终 token ids
```

### Decode（token ids → 文字）

```
输入: [32, 15, 15, 8, 200]

Step 1: 查 vocab.json（id → token）
[he, ll, o, Ġ, 你]

Step 2: BPE 拆分（按 merges.txt 逆序）
[he, ll, o, Ġ, 你]
    ↓ "he" → 查 merges.txt → [h, e]
    ↓ "ll" → 查 merges.txt → [l, l]
    ↓ 最终: [h, e, l, l, o, Ġ, 你]

Step 3: ByteLevel decoder 还原
[h, e, l, l, o, \xe4, \xbd, \xa0]
    ↓ 按 UTF-8 规则组合
"hello你"
```

### ByteLevel 怎么还原成文字？

按 UTF-8 编码规则把字节组合成字符：

| 字符类型 | UTF-8 字节数 | 字节特征 |
|----------|-------------|---------|
| ASCII | 1 字节 | 0xxxxxxx（英文、数字） |
| 中文 | 3 字节 | 1110xxxx 10xxxxxx 10xxxxxx |
| emoji | 4 字节 | 1110xxxx 10xxxxxx ... |

```python
字节序列: [h, e, l, l, o, \xe4, \xbd, \xa0]
          ← ASCII → ←── 中文3字节 ──→

还原: "hello你"
```

---

## 一句话总结

**用 tokenizers 库 + BPE 算法，以 SFT 数据（1万行）为语料，训练出 6400 个 token 的词表**。

流程：ByteLevel 预切分 → BPE 统计字节对频率迭代合并 → 保存 tokenizer.json + tokenizer_config.json。
