# 第5节：Features 与数据集加载

> 我的问题笔记 | 聚焦真正不懂的地方

---

## 📚 本节目录

1. [Q1: Features 是什么？](#q1-features-是什么)
2. [Q2: logits 是什么？](#q2-logits-是什么)
3. [Q3: logits 和 labels 的 shift 对齐](#q3-logits-和-labels-的-shift-对齐)
4. [Q4: loss 计算代码解释](#q4-loss-计算代码解释)
5. [Q5: create_chat_prompt 做了什么？](#q5-create_chat_prompt-做了什么)
6. [Q6: apply_chat_template 做了什么？](#q6-apply_chat_template-做了什么)

---

## Q1: Features 是什么？

**代码** `dataset/lm_dataset.py:63`
```python
from datasets import load_dataset, Features, Sequence, Value

features = Features({
    'conversations': [
        {'role': Value('string'), 'content': Value('string'), ...}
    ]
})
```

`Features` 是 **HuggingFace datasets 库**的一个类，用于**定义数据集的表结构**（schema）。

---

## Features 的作用

| 作用 | 说明 |
|------|------|
| 类型检查 | 加载数据时检查是否符合声明的类型 |
| 自动转换 | 比如字符串 "123" 能自动转成 int |
| IDE 提示 | 方便看有哪些字段 |

就像数据库的 **CREATE TABLE** 语句，声明"这个数据集有哪些列、每列是什么类型"。

---

## 举例

```python
from datasets import Features, Value, Sequence

# 定义数据结构
features = Features({
    'name': Value('string'),      # 字符串字段
    'age': Value('int32'),        # 整数字段
    'tags': Sequence(Value('string')),  # 字符串数组
})
```

---

## Features 和 self.samples 的关系

| 概念 | 作用 |
|------|------|
| `features` | **声明**数据长什么样（类型检查） |
| `self.samples` | **实际加载**的数据 |

就像： `features` = 填表时的**表头**，`self.samples` = 实际填的**数据**。

---

## load_dataset 加载的数据集长什么样？

```python
self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)
```

把 jsonl 文件加载成 datasets 库的数据集对象。

**举例**：
```python
# jsonl 文件内容
{"conversations": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好，有什么帮助？"}]}
{"conversations": [{"role": "user", "content": "你叫什么？"}, {"role": "assistant", "content": "我叫 MiniMind"}]}

# load_dataset 后
self.samples = [
    {"conversations": [{"role": "user", ...}, {"role": "assistant", ...}]},  # 第1条
    {"conversations": [{"role": "user", ...}, {"role": "assistant", ...}]},  # 第2条
]
```

---

## Q2: logits 是什么？

**代码** `model/model_minimind.py`
```python
logits = model(input_ids)  # shape: [batch, seq_len, vocab_size]
```

`logits` 是模型 **softmax 之前的原始预测分数**。

| 阶段 | 名称 | 说明 |
|------|------|------|
| softmax 前 | **logits** | 原始分数，可正可负 |
| softmax 后 | **probs** | 0~1 的概率 |

**每个位置的 logits 是什么意思？**

```
输入: [你, 好]
         ↓
       模型
         ↓
位置 0 的 logits → 根据 "你" 预测下一个词（从 vocab 里选）
位置 1 的 logits → 根据 "你、好" 预测下一个词（从 vocab 里选）
```

每个位置的 logits **不是从输入序列中选词，是从整个 vocab 词表中选下一个词**。

**举例**：
```
vocab = ["你", "好", "世", "界", "，"]  # 词表

输入: [你]
         ↓
位置 0 的 logits = [1.5, 3.2, -0.3, -0.1, 0.8]
                  ↑你   ↑好   ↑世   ↑界   ↑，
                  "好" 的分数最高 → 模型预测下一个词是 "好"
```

**logits 可以看成前一部分句子对下一个词的预测分数，分数越高越可能被选中。**

这里 logits[0]=1.5 对应 "你"，logits[1]=3.2 对应 "好"，因为 "好" 分数最高，所以模型预测下一个词是 "好"（而不是 "你"）。

---

## Q3: logits 和 labels 的 shift 对齐

**代码**
```python
logits[..., :-1, :]  # 去掉最后一个位置
labels[..., 1:]       # 去掉第一个位置
```

**为什么需要 shift？**

```
输入: [t0, t1, t2, t3, t4]  # 5个 token

logits: [L0, L1, L2, L3, L4]  # 5个位置的预测

L0 预测 t1
L1 预测 t2
L2 预测 t3
L3 预测 t4
L4 预测 t5（没有 t5！）← 无效，要去掉
```

**位置 N 的 logits 预测的是位置 N+1 的 token**，所以 loss 计算时 logits 和 labels 要错开一位对齐。

**对齐后**：
| logits 位置 | 预测目标 |
|-------------|---------|
| logits[0] | labels[1] |
| logits[1] | labels[2] |
| logits[2] | labels[3] |
| logits[3] | labels[4] |

**Python 切片验证**：
```python
logits = [L0, L1, L2, L3, L4]  # 5个元素
logits[:-1]  # [L0, L1, L2, L3]  ← 去掉了 L4（无效预测）
labels[1:]   # [t1, t2, t3, t4]  ← 去掉了 t0（不需要预测）
```

---

## Q4: loss 计算代码解释

**代码**
```python
if labels is not None:
    x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
    loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
```

### 第1步：shift 对齐

```python
x = logits[..., :-1, :]    # [batch, seq_len-1, vocab_size]  去掉最后一个位置
y = labels[..., 1:]        # [batch, seq_len-1]              去掉第一个位置
```

位置 N 的 logits 预测位置 N+1 的 label。

### 第2步：contiguous()

```python
x = x.contiguous()  # 确保内存连续，view 操作需要
y = y.contiguous()
```

`view()` 需要张量在内存中是连续存储的。

### 第3步：展平

```python
x.view(-1, x.size(-1))  # [batch, seq_len-1, vocab_size] → [N, vocab_size]
y.view(-1)               # [batch, seq_len-1]              → [N]
```

- `x.view(-1, x.size(-1))`：展成 `[N, vocab_size]`，N = batch * (seq_len-1)
- `y.view(-1)`：展成一维 `[N]`

### 第4步：交叉熵 loss

**交叉熵公式**：

$$H(p, q) = - \sum_{i} p_i \cdot \log(q_i)$$

| 符号 | 含义 |
|------|------|
| $p_i$ | 真实分布（one-hot，目标 token 的位置为 1，其他为 0） |
| $q_i$ | 模型预测分布（softmax 后的概率） |
| $H(p, q)$ | 交叉熵，越小越好 |

**代码实现**：
```python
F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
```

| 参数 | 说明 |
|------|------|
| `x` | 预测 logits，shape `[N, vocab_size]` |
| `y` | 目标 token ids，shape `[N]` |
| `ignore_index=-100` | label=-100 的位置不计算 loss |

**计算过程**：
1. 对 logits 做 softmax，得到每个词的概率 $q_i$
2. 取目标 token 对应的 $\log(q_i)$
3. 求和/平均得到 loss

### ignore_index=-100 的作用

SFT 训练时，user 的 label 设为 -100，表示"这部分不计算 loss，只训练 assistant"。

```python
# labels 示例
labels = [-100, -100, -100, 1024, 2048, 2068, -100, ...]
#            ↑ user部分    ↑ assistant部分      ↑ user部分
#          不算loss           算loss            不算loss
```

---

## Q5: create_chat_prompt 做了什么？

**代码** `dataset/lm_dataset.py:71-86`
```python
def create_chat_prompt(self, conversations):
    messages = []
    tools = None
    for message in conversations:
        message = dict(message)
        # 如果是 system 消息且有 tools 字段，解析工具定义
        if message.get("role") == "system" and message.get("tools"):
            tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
        # 如果有 tool_calls 字段且是字符串，转成对象
        if message.get("tool_calls") and isinstance(message.get("tool_calls"), str):
            message["tool_calls"] = json.loads(message["tool_calls"])
        messages.append(message)
    return self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=tools
    )
```

`create_chat_prompt` 把原始对话数据转成 **prompt 字符串**。

**输入输出**：
```python
输入: [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好，有什么帮助？"}]
输出: "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n你好，有什么帮助？<|im_end|>\n"
```

**每一步在干嘛**：
| 步骤 | 作用 |
|------|------|
| `message = dict(message)` | 确保是字典格式 |
| `tools = json.loads(...)` | 解析 system 消息里的工具定义 |
| `tool_calls = json.loads(...)` | 解析工具调用（如果有） |
| `apply_chat_template(...)` | 用模板转成 prompt 字符串 |

---

## Q6: apply_chat_template 做了什么？

**代码** `minimind-3/chat_template.jinja`

`apply_chat_template` 内部使用 Jinja2 模板引擎，把 messages 列表转换成特定格式的字符串。

### 模板核心逻辑（简化版）

```jinja
{%- for message in messages %}
    {%- if message.role == "user" %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {{- '\n<tool_response>\n' + content + '\n</tool_response>' }}
    {%- endif %}
{%- endfor %}
```

### 输入输出示例

**输入（messages 列表）**：
```python
[
    {"role": "system", "content": "你是一个有帮助的助手。"},
    {"role": "user", "content": "MiniMind 是什么？"},
    {"role": "assistant", "content": "MiniMind 是一个小型语言模型..."}
]
```

**输出（prompt 字符串）**：
```
<|im_start|>system
你是一个有帮助的助手。<|im_end|>
<|im_start|>user
MiniMind 是什么？<|im_end|>
<|im_start|>assistant
<think>

</think>

MiniMind 是一个小型语言模型...<|im_end|>
```

### 各 role 的格式化规则

| role | 格式 |
|------|------|
| `system` | `<\|im_start\|>system\n{content}<\|im_end\|>\n` |
| `user` | `<\|im_start\|>user\n{content}<\|im_end\|>\n` |
| `assistant` | `<\|im_start\|>assistant\n[<think>\n{reasoning}\n</think>\n\n]{content}<\|im_end\|>\n` |
| `tool` | `<\|im_start\|>user\n<tool_response>\n{content}\n</tool_response><\|im_end\|>\n` |

### add_generation_prompt 参数

`add_generation_prompt=False` 时，输出到 assistant 回答结束就停。

`add_generation_prompt=True` 时，还会在最后加上 `<|im_start|>assistant\n<think>\n`，表示开始生成的位置。

```python
# add_generation_prompt=True 时的输出末尾
<|im_start|>assistant
<think>

</think>


```

### tokenize=False 的作用

| 参数值 | 效果 |
|--------|------|
| `tokenize=False` | 返回**字符串**，用于查看 prompt 格式 |
| `tokenize=True`（默认） | 返回 **token ids 列表**，直接用于训练 |

### 关键 special tokens

| 完整字符串 | 含义 | 组成 |
|-----------|------|------|
| `<\|im_start\|>system\n` | system 角色开头 | `<\|im_start\|>` (id=1) + "system" + 换行 |
| `<\|im_start\|>user\n` | user 角色开头 | `<\|im_start\|>` (id=1) + "user" + 换行 |
| `<\|im_start\|>assistant\n` | assistant 角色开头 | `<\|im_start\|>` (id=1) + "assistant" + 换行 |
| `<\|im_end\|>` | 消息结束 | id=2 |

**注意**：`<|im_start|>` 只是分隔符（id=1），后面跟的 role 名称（system/user/assistant）是普通文本，由 tokenizer 继续编码。
