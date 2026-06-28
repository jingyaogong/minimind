# 第 3 课：CLI 推理流程

这一课只解决一个问题：用户输入一句话后，MiniMind 如何把它变成模型输入，再生成回复。

## 目录

- [0. 本节主线](#l03-mainline)
- [1. 本节要懂的 5 个原理](#l03-principles)
- [2. 变量流转](#l03-flow)
- [3. 原理一：模型加载](#l03-model-loading)
- [4. 原理二：对话模板](#l03-chat-template)
- [5. 原理三：Tokenization](#l03-tokenization)
- [6. 原理四：自回归生成](#l03-generation)
- [7. 原理五：Decode](#l03-decode)
- [8. 实验验证](#l03-experiment)
- [9. 本节检查](#l03-check)
- [10. 下一课](#l03-next)

<a id="l03-mainline"></a>
## 0. 本节主线

CLI 推理的本质是：

```text
加载 tokenizer 和模型
-> 把用户消息用 chat template 包装成 prompt
-> 把 prompt tokenizer 成 input_ids
-> 让模型自回归生成新 token
-> 只 decode 新生成 token 作为回复
```

这条链对应本节的 5 个原理：

```text
模型加载 -> 对话模板 -> Tokenization -> 自回归生成 -> Decode
```

本节不是命令手册。你要学的是这 5 个推理原理，以及源码里哪里实现了它们。

<a id="l03-principles"></a>
## 1. 本节要懂的 5 个原理

| 原理 | 你要懂什么 | 源码实现 |
|---|---|---|
| 模型加载 | 推理必须同时有 tokenizer、模型结构、模型权重 | `eval_llm.py:12-30` |
| 对话模板 | 聊天模型不是直接吃用户原文，而是吃 chat template 包装后的 prompt | `eval_llm.py:71-78` |
| Tokenization | 模型不处理字符串，只处理 token id 和 attention mask | `eval_llm.py:78` |
| 自回归生成 | 模型一次预测一个 next token，循环追加到已有 token 后面 | `eval_llm.py:82-87`, `model/model_minimind.py:257-287` |
| Decode | 生成结果仍是 token id，需要切掉 prompt 部分再 decode 成文本 | `eval_llm.py:88` |

学完本节，你不需要懂 attention、loss、SFT labels、LoRA、RL。那些是后面的课。

<a id="l03-flow"></a>
## 2. 变量流转

把主线对应到代码里的变量名：

```text
prompt: 用户输入的字符串
conversation: [{"role": "user", "content": prompt}]
inputs: tokenizer 输出的字典，包含 input_ids / attention_mask
generated_ids: prompt token + 新生成 token
response: 新生成 token decode 后的文本
```

<a id="l03-model-loading"></a>
## 3. 原理一：模型加载

推理要准备三件东西：

```text
tokenizer：负责字符串和 token id 互转
model config / architecture：负责创建模型结构
model weights：训练好的参数
```

看哪段源码：

```text
eval_llm.py:12-30
```

先看这几行：

```text
AutoTokenizer.from_pretrained(...)
MiniMindForCausalLM(MiniMindConfig(...))
model.load_state_dict(...)
AutoModelForCausalLM.from_pretrained(...)
return model.half().eval().to(...)
```

源码：

```python
def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        model = MiniMindForCausalLM(MiniMindConfig(...))
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    return model.half().eval().to(args.device), tokenizer
```

关键解释：

- `AutoTokenizer.from_pretrained(args.load_from)`：从指定目录加载 tokenizer。这个目录可以是 `./model`，也可以是 `./minimind-3`。
- `if 'model' in args.load_from`：作者用字符串判断加载方式。路径里包含 `model` 时，走本仓库原生 PyTorch 权重加载。
- `MiniMindForCausalLM(MiniMindConfig(...))`：先创建一个空模型结构，此时还没有训练好的能力。
- `model.load_state_dict(...)`：把 `.pth` 权重填进模型结构里。没有这一步，模型只是随机初始化。
- `AutoModelForCausalLM.from_pretrained(...)`：transformers 格式的一步式加载，会根据目录里的 `config.json` 和 `model.safetensors` 加载模型。
- `model.half().eval().to(args.device)`：切到半精度、推理模式，并移动到 CPU 或 GPU。

读这段源码时只抓一个点：**推理不是只有模型文件，还必须同时匹配 tokenizer、模型结构和权重。**

暂时不用看：

- `apply_lora` / `load_lora`：LoRA 后面单独讲。
- `get_model_params`：这里只是打印参数量。

这里有两条分支：

```text
--load_from ./model
-> 用本仓库的 MiniMindForCausalLM 创建结构
-> 从 ./out/full_sft_768.pth 加载权重

--load_from ./minimind-3
-> 用 transformers 目录里的 config/tokenizer/model.safetensors 加载
```

我们现在已经下载了 `./minimind-3`，所以课程优先使用 transformers 格式模型。

<a id="l03-chat-template"></a>
## 4. 原理二：对话模板

聊天模型通常不是直接吃：

```text
MiniMind 是什么？
```

而是吃带角色标记的 prompt，例如：

```text
<|im_start|>user
MiniMind 是什么？<|im_end|>
<|im_start|>assistant
<think>

</think>
```

这就是 chat template 的作用：把结构化对话消息转成模型训练时见过的文本格式。

看哪段源码：

```text
eval_llm.py:71-76
```

先看这几行：

```text
conversation.append(...)
tokenizer.apply_chat_template(...)
tokenize=False
add_generation_prompt=True
```

源码：

```python
conversation = conversation[-args.historys:] if args.historys else []
conversation.append({"role": "user", "content": prompt})

inputs = tokenizer.apply_chat_template(
    conversation,
    tokenize=False,
    add_generation_prompt=True,
    open_thinking=bool(args.open_thinking)
)
```

关键解释：

- `conversation = conversation[-args.historys:] if args.historys else []`：控制是否携带历史对话。默认 `historys=0`，所以每轮都清空历史。
- `conversation.append(...)`：把当前用户问题加入消息列表。此时它还是结构化 Python 对象，不是字符串。
- `role: "user"`：告诉模板这句话来自用户。聊天模型依赖角色标记区分 user / assistant / system。
- `apply_chat_template(...)`：把消息列表渲染成一段 prompt 字符串。
- `tokenize=False`：只返回字符串，不直接返回 token id。
- `add_generation_prompt=True`：在 prompt 末尾加上 assistant 开始标记，提示模型“接下来该助手回答了”。
- `open_thinking=...`：控制 prompt 里 `<think>` 部分的格式，后面讲 tokenizer/template 时再细看。

读这段源码时只抓一个点：**聊天推理的输入不是裸问题，而是带角色格式的 prompt。**

暂时不用看：

- `open_thinking` 的模板细节：下一课 tokenizer/chat template 会展开。
- 多轮历史裁剪策略：现在只知道 `historys=0` 时不带历史即可。

这里要分清：

```text
conversation 是 Python 消息列表
apply_chat_template 的输出是 prompt 字符串
```

`tokenize=False` 的意思是：这一步只生成字符串，还不转 token id。

<a id="l03-tokenization"></a>
## 5. 原理三：Tokenization

模型不能直接处理字符串。字符串必须经过 tokenizer 变成数字。

看哪段源码：

```text
eval_llm.py:78
```

先看这几个参数：

```text
return_tensors="pt"
truncation=True
.to(args.device)
```

源码：

```python
inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)
```

关键解释：

- 第一个 `inputs`：这里变量名有点绕。上一行的 `inputs` 其实还是 prompt 字符串。
- `tokenizer(inputs, ...)`：把 prompt 字符串编码成 token id。
- `return_tensors="pt"`：返回 PyTorch tensor，而不是普通 Python list。
- `truncation=True`：如果 prompt 太长，会按 tokenizer/model 的规则截断。
- `.to(args.device)`：把 `input_ids` 和 `attention_mask` 移到 CPU 或 GPU。

这行执行前后，`inputs` 的含义变了：

```text
执行前：inputs 是 prompt 字符串
执行后：inputs 是 tokenizer 输出的 BatchEncoding 字典
```

输出里的两个核心字段：

```text
input_ids: token id 序列，形状通常是 [1, seq_len]
attention_mask: 哪些位置是有效 token，形状通常也是 [1, seq_len]
```

你在本节只需要理解 shape：

```text
input_ids.shape = [batch_size, prompt_token_count]
```

CLI 单条推理时，`batch_size` 通常是 1。

<a id="l03-generation"></a>
## 6. 原理四：自回归生成

LLM 不是一次性写完整段答案。它每次只预测下一个 token。

逻辑是：

```text
已有 token
-> 模型 forward 得到最后位置的 logits
-> 从 logits 里选出 next_token
-> 把 next_token 拼到 input_ids 后面
-> 重复
```

看哪段源码：

```text
eval_llm.py:82-87
model/model_minimind.py:257-287
```

第一遍先看 `eval_llm.py` 怎么调用 `generate`：

```python
generated_ids = model.generate(
    inputs=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=args.max_new_tokens,
    do_sample=True,
    streamer=streamer,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    top_p=args.top_p,
    temperature=args.temperature,
    repetition_penalty=1
)
```

关键解释：

- `inputs=inputs["input_ids"]`：把 prompt token 交给模型。
- `attention_mask=inputs["attention_mask"]`：告诉模型哪些 token 是有效输入。
- `max_new_tokens`：最多新生成多少个 token，不包括 prompt 本身。
- `do_sample=True`：使用采样生成，而不是总选概率最高的 token。
- `streamer=streamer`：边生成边打印，CLI 里能看到字逐步出来。
- `pad_token_id` / `eos_token_id`：告诉生成过程 padding token 和结束 token 是什么。
- `top_p` / `temperature`：控制采样分布。temperature 越高越随机，top_p 控制候选 token 的概率质量范围。

读这段源码时只抓一个点：**`generate` 输入的是 prompt token，输出的是 prompt token 加新 token。**

第二遍再看本仓库原生 PyTorch 模型的 `generate` 实现：

```python
for _ in range(max_new_tokens):
    outputs = self.forward(...)
    logits = outputs.logits[:, -1, :] / temperature
    next_token = ...
    input_ids = torch.cat([input_ids, next_token], dim=-1)
```

这段循环的关键含义：

- `for _ in range(max_new_tokens)`：最多循环生成这么多次。
- `self.forward(...)`：用当前已有 token 计算 logits。
- `outputs.logits[:, -1, :]`：只取最后一个位置的 logits，因为它对应“下一个 token”的预测。
- `next_token = ...`：从 logits 中选择或采样一个 token。
- `torch.cat([input_ids, next_token], dim=-1)`：把新 token 接到序列末尾，下一轮继续用。

现在下载的 `./minimind-3` 是 transformers 格式，实际会走 transformers 模型的生成机制。但原理一致：循环预测 next token。

<a id="l03-decode"></a>
## 7. 原理五：Decode

`generated_ids` 包含两部分：

```text
原始 prompt token + 新生成 token
```

所以不能直接把整个 `generated_ids` decode 给用户，否则会把用户问题也输出一遍。

看哪段源码：

```text
eval_llm.py:88
```

先看这几个点：

```text
generated_ids[0]
len(inputs["input_ids"][0])
[prompt_len:]
skip_special_tokens=True
```

源码：

```python
response = tokenizer.decode(
    generated_ids[0][len(inputs["input_ids"][0]):],
    skip_special_tokens=True
)
```

关键解释：

- `generated_ids[0]`：取 batch 中第 1 条生成结果。
- `len(inputs["input_ids"][0])`：prompt 原本的 token 数量，也就是 prompt 长度。
- `[prompt_len:]`：跳过 prompt，只保留新生成部分。
- `skip_special_tokens=True`：decode 时去掉 `<|im_start|>`、`<|im_end|>` 等特殊标记。

关键是这段切片：

```python
generated_ids[0][len(inputs["input_ids"][0]):]
```

它的含义是：

```text
从 prompt 结束之后开始取，只取新生成 token
```

<a id="l03-experiment"></a>
## 8. 实验验证

### 实验 A：真实模型推理

这个实验验证：下载好的 `./minimind-3` 可以真实生成回复。

```bash
cd /home/sun/minimind
PYTHONDONTWRITEBYTECODE=1 python course/labs/run_minimind3_once.py --device cpu --max_new_tokens 64
```

记录：

```text
input_ids.shape =
generated_ids.shape =
new_tokens =
response =
```

注意：MiniMind-3 是小模型，回答可能有事实错误。本实验只验证推理链路。

### 实验 B：形状追踪

这个实验验证：prompt、token、logits、generated_ids 的形状关系。

```bash
cd /home/sun/minimind
PYTHONDONTWRITEBYTECODE=1 python course/labs/trace_cli_inference.py
```

记录：

```text
prompt 中出现的特殊标记 =
input_ids.shape =
attention_mask.shape =
logits.shape =
generated_ids.shape =
```

这个脚本用的是随机 tiny 模型，所以 decode 出来的文字没有意义。它的价值是让你看清 shape。

<a id="l03-check"></a>
## 9. 本节检查

如果你真懂了本节，应该能不看答案说清楚：

1. `conversation`、`prompt`、`input_ids` 分别是什么类型。
2. 为什么聊天模型要使用 chat template。
3. `tokenizer(...)` 输出里的 `input_ids` 和 `attention_mask` 分别做什么。
4. `generate` 为什么叫自回归生成。
5. 为什么 decode 前要切掉 prompt token。

<a id="l03-next"></a>
## 10. 下一课

第 4 课进入 tokenizer。我们会专门拆开看 special tokens、编码/解码、chat template 里的 `<|im_start|>`、`<|im_end|>`、`<think>` 等标记。
