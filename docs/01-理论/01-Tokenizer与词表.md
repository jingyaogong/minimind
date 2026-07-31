# 01-理论 / 01 - Tokenizer 与词表

> **TL;DR**：Tokenizer 是 LLM 的"字典"，把文字切成 token id。minimind 用 6400 词表的 BPE+ByteLevel，
> 还预定义了 ``、`<\|im_end\|>`、``、`<\|endoftext\|>` 等对话/思考标记。

## ✅ 你将能

- 解释 BPE + ByteLevel 是什么、为什么用
- 知道 minimind 词表为什么是 6400
- 读懂 `model/tokenizer.json` 与 `tokenizer_config.json` 的 chat_template
- 理解 `` / `` / `` 等特殊标记的作用

---

## 一、Tokenizer 是什么

LLM 不直接处理文字，它处理**整数 ID**。Tokenizer 就是文字 ↔ ID 的翻译器：

```
"你好 minimind"  →  [3245, 1098, 42]  →  模型
   ↑                                          ↓
   └────────  [3245, 1098, 42, 7]  ←  decode ──┘
                                          (7=eos)
```

Tokenizer 由两部分组成：
1. **词表（vocab）**：每个 token 对应一个 ID
2. **切分规则**：怎么把一个字符串切成多个 token

## 二、BPE + ByteLevel：minimind 的选择

### BPE（Byte-Pair Encoding）

从字符开始，反复合并出现频率最高的相邻对，直到达到词表大小。

举例（极简）：
```
初始:  [l, o, w, e, r]
合并 l+o: [lo, w, e, r]
合并 lo+w: [low, e, r]
...
```

最终词表里既有单字符，也有"常见片段"，**高频词切得少、低频词切得多**。

### ByteLevel

不在 Unicode 字符上做 BPE，而是**先转成字节（256 种）再做 BPE**。

好处：
- **零未登录词**：任何字符串都能切（哪怕 emoji、生僻字）
- 跨语言通用：中英日韩同一套算法

坏处：
- 中文压缩比不如 qwen2（约 1.5-1.7 字符/token，纯英文 4-5 字符/token）

### minimind 的取舍

| Tokenizer | 词表大小 | 来源 |
|---|---|---|
| Llama 3 | 128,000 | Meta |
| Qwen2 | 151,643 | 阿里 |
| ChatGLM | 151,329 | 智谱 |
| **minimind** | **6,400** | 自定义 |

为什么 6400？**词表大小直接影响 embedding 层和输出层的参数占比**。

对 minimind 这种 64M 小模型：
- 词表 6400 × dim 768 ≈ 4.9M 参数（embedding）
- 词表 150k × dim 768 ≈ 115M 参数（embedding 就爆了）

> 见 `model/model_minimind.py:242`：`if self.config.tie_word_embeddings: self.model.embed_tokens.weight = self.lm_head.weight`
> —— **tie weights** 让输入和输出共享，再省一半。这是小模型必用的技巧。

## 三、minimind 的特殊标记

打开 `model/tokenizer_config.json`，里面定义了关键 token：

| Token | 作用 | 何时使用 |
|---|---|---|
| `<\|im_start\|>` | 消息开始 | 每条对话消息开头 |
| `<\|im_end\|>` | 消息结束 | 每条对话消息结尾（也是默认 eos） |
| `<\|endoftext\|>` | 文档/批次结束 | 预训练数据分隔 |
| `<\|pad\|>` | padding 填充 | batch 内不同长度对齐 |
| `` | 思考开始 | `open_thinking=1` 时模板注入 |
| `</think>` | 思考结束 | 同上 |
| `` | 工具调用块开始 | Tool Call 训练时 |
| `` | 工具返回块结束 | Tool Call 训练时 |

> 实际名字请打开 `model/tokenizer_config.json` 对照，每个版本的标记名可能微调。

## 四、chat_template：把对话渲染成模型输入

`tokenizer_config.json` 里有一个 `chat_template` 字段（Jinja2 模板），它把：

```python
[
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！"},
]
```

渲染成模型实际看到的字符串：

```
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
你好！<|im_end|>
```

调用方式（在 `dataset/lm_dataset.py:81`）：

```python
self.tokenizer.apply_chat_template(
    messages,
    tokenize=False,            # 只渲染字符串，不切 token
    add_generation_prompt=False,
    tools=tools                # Tool Call 时传入工具列表
)
```

`apply_chat_template` 还接受 `open_thinking=True/False` 来动态插入/不插入 `` 标签 —— 这就是 minimind-3 的"自适应思考"实现。

## 五、动手验证（5 分钟）

```bash
cd /path/to/minimind
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('./model')
print('词表大小:', tok.vocab_size)
print('bos:', tok.bos_token, tok.bos_token_id)
print('eos:', tok.eos_token, tok.eos_token_id)
print('pad:', tok.pad_token, tok.pad_token_id)

text = '你好 minimind，今天天气怎么样'
ids = tok(text).input_ids
print('原文:', text)
print('切分:', tok.convert_ids_to_tokens(ids))
print('IDs:', ids)
print('解码回:', tok.decode(ids))
print('压缩比:', len(text), '字符 →', len(ids), 'token')
"
```

✅ **验证**：
- 中文约 1.5 字符/token
- 英文约 4-5 字符/token
- decode 后和原文一致

## 六、为什么不要重新训练 tokenizer

主 README 里作者反复强调："不建议重新训练 tokenizer"。原因：

1. **权重对齐**：换词表 = embedding 层维度变化 = 整个模型权重作废
2. **数据格式**：所有数据集的 token id 都要重切
3. **生态兼容**：与 transformers / vllm / ollama / llama.cpp 的兼容性下降
4. **指标失真**：PPL（perplexity）按 token 统计，词表不同不可比 → 跨 tokenizer 比 BPB（Bits Per Byte）更公平

想看怎么训 tokenizer 的话 → `trainer/train_tokenizer.py`，但**读懂就好，别真去改它**。

## 七、关键代码对照

| 概念 | 代码位置 |
|---|---|
| 词表加载 | `eval_llm.py:13` `AutoTokenizer.from_pretrained(args.load_from)` |
| 词表大小配置 | `model/model_minimind.py:18` `vocab_size=6400` |
| bos/eos 配置 | `model/model_minimind.py:19-20` |
| tie weights | `model/model_minimind.py:30, 242` |
| chat_template 调用 | `dataset/lm_dataset.py:81-86` |
| Pretrain 数据切分 | `dataset/lm_dataset.py:47-55` |
| 自适应思考模板 | `dataset/lm_dataset.py:31-35` `post_processing_chat` |

## ✅ 本篇完成自检

<details>
<summary>点开自检（先想 30 秒）</summary>

1. 为什么 minimind 词表只有 6400？
   - 答：小模型要把参数预算留给 attention/FFN，词表太大会让 embedding 层占比过高。
2. ByteLevel 解决了什么问题？
   - 答：零未登录词，任何字符（emoji、生僻字）都能切，因为是先转字节再做 BPE。
3. `apply_chat_template` 在做什么？
   - 答：把 `[{role, content}, ...]` 的结构化对话渲染成模型实际看到的带特殊标记的字符串。
4. `tie_word_embeddings=True` 节省了多少参数？
   - 答：embedding 和 lm_head 共享一份权重，省了一份 ≈ vocab_size × hidden_size。
5. `open_thinking` 是模型本身的能力，还是模板层的能力？
   - 答：模板层。训练时混入空 think 和显式 think 数据，推理时由模板注入 `` 决定输出。

</details>

下一篇：[02-Transformer核心架构](./02-Transformer核心架构.md) —— 逐行读懂 288 行模型代码。
