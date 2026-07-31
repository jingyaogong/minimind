# 02-实战 / 02 - 指令微调 SFT

> **TL;DR**：基于 `pretrain_768.pth`，用对话数据让模型学会"回答+停止"。跑通 `trainer/train_full_sft.py`，
> 单卡 4090 约 1.1 小时，产出 `out/full_sft_768.pth`，能进行多轮对话。
>
> **心理学技巧**：**首因效应** —— 当模型第一次"答上"你的问题时，会形成强烈印象：
> "我真的训出了一个 LLM"。这个瞬间会驱动你继续走完后续 RL / 部署。

## ✅ 你将能

- 说出 SFT 与 Pretrain 在数据/loss/参数上的全部区别
- 跑通 `train_full_sft.py`，产出能对话的 `full_sft_768.pth`
- 解释 chat_template 套用流程和 labels mask 机制
- 用 `eval_llm.py --weight full_sft` 验证多轮对话
- 知道 minimind-3 的 SFT 数据已混入 Tool Call 主线

---

## 一、SFT 在做什么

预训练模型只会"接着写"，**SFT（Supervised Fine-Tuning，监督微调）让模型学会"对话"**：

```
输入: <|im_start|>user\n天空为什么是蓝色的？<|im_end|>\n<|im_start|>assistant\n
期望输出: 因为大气层对阳光的短波部分（蓝紫）散射更强...<|im_end|>
```

SFT 教模型：
1. 看到 `<|im_start|>user` 就知道是"用户提问"
2. 看到 `<|im_start|>assistant` 后要"回答"而不是"续写"
3. 答完打 `<|im_end|>` 表示结束（推理时碰到就停）

### 1.1 SFT 与 Pretrain 的核心差异

| 维度 | Pretrain | SFT |
|---|---|---|
| 数据格式 | 纯文本（`{"text": ...}`） | 对话（`{"conversations": [...]}`） |
| 调用模板 | 直接 tokenize | `apply_chat_template` 渲染 |
| 算 loss 的位置 | 整段（除 pad） | **只在 assistant 回复部分** |
| 起始权重 | 随机 | **基于 `pretrain_768.pth`** |
| 学习率 | 5e-4（大） | 1e-5（小，避免破坏预训练知识） |
| batch_size | 32 | 16（序列更长） |
| max_seq_len | 340 | 768（对话更长） |
| accumulation_steps | 8 | 1（不累积） |

> 学习率小 50 倍是关键 —— SFT 是"精修"，5e-4 会把预训练学到的东西冲掉。

### 1.2 labels 的 -100 mask 机制（核心）

代码：`dataset/lm_dataset.py:88-104` `SFTDataset.generate_labels`

```python
def generate_labels(self, input_ids):
    labels = [-100] * len(input_ids)   # 默认全部不算 loss
    i = 0
    while i < len(input_ids):
        if input_ids[i:i + len(self.bos_id)] == self.bos_id:
            # 找到 <|im_start|>assistant\n 标记
            start = i + len(self.bos_id)
            end = start
            while end < len(input_ids):
                if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                    break
                end += 1
            # 只把 assistant 内容段设成有 label
            for j in range(start, min(end + len(self.eos_id), self.max_length)):
                labels[j] = input_ids[j]
            i = end + len(self.eos_id)
        else:
            i += 1
    return labels
```

可视化（`-` 表示 -100，`A` 表示有 label 的 assistant 部分）：

```
input_ids: [<im_start|>user\n天空蓝<im_end|><im_start|>assistant\n因为散射<im_end|>pad pad pad]
labels:    [------------------------------------------------------A A A A A A A A A---]
```

**为什么这么做？**
- 不在 user/system 部分算 loss：那些是输入，模型不该学着"生成"用户的提问
- 只在 assistant 部分算 loss：模型只需要学"怎么回答"
- `-100` 是 PyTorch `F.cross_entropy(ignore_index=-100)` 的默认忽略值（见 `model_minimind.py:252`）

> Pretrain 没有这个 mask，整段都算 loss —— 因为整段都是"知识"。

### 1.3 chat_template 套用流程

代码：`dataset/lm_dataset.py:71-86` `create_chat_prompt`

```python
def create_chat_prompt(self, conversations):
    messages = []
    tools = None
    for message in conversations:
        # 解析 system 里的 tools 字段
        if message.get("role") == "system" and message.get("tools"):
            tools = json.loads(message["tools"])
        # 解析 assistant 的 tool_calls
        if message.get("tool_calls") and isinstance(message["tool_calls"], str):
            message["tool_calls"] = json.loads(message["tool_calls"])
        messages.append(message)
    return self.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, tools=tools
    )
```

`apply_chat_template` 把结构化对话渲染成带特殊标记的字符串，详见 [01-理论/01-Tokenizer与词表](../01-理论/01-Tokenizer与词表.md) 第四节。

### 1.4 数据增强：自适应思考 + 随机 system

代码：`dataset/lm_dataset.py:9-35`

- `pre_processing_chat`（`lm_dataset.py:9`）：20% 概率随机加 system prompt，增强鲁棒性
- `post_processing_chat`（`lm_dataset.py:31`）：80% 概率移除空 `<think></think>` 标签

> 这是 minimind-3 "自适应思考"训练技巧：让模型见过"开思考"和"不开思考"两种样本，推理时通过模板切换。详见 [03-进阶/03-工具调用与思考](../03-进阶/03-工具调用与思考.md)。

### 1.5 minimind-3 已经把 Tool Call 混入 sft_t2t 主线

`dataset/sft_t2t_mini.jsonl` 已经包含 Tool Call 样本（带 `tools` / `tool_calls` 字段）。
所以 `full_sft_768.pth` 训完后**就有基础的 Tool Call 能力**，不用再单独训练。

> 想深入 Tool Call 机制和评测，见 [03-进阶/03-工具调用与思考](../03-进阶/03-工具调用与思考.md)。

---

## 步骤 1：确认前置条件

```bash
# 1. 预训练权重已存在
ls -lh /path/to/minimind/out/pretrain_768.pth

# 2. SFT 数据已下载
ls -lh /path/to/minimind/dataset/sft_t2t_mini.jsonl
```

✅ **验证**：两个文件都在。没有 pretrain 权重回 [01-预训练Pretrain](./01-预训练Pretrain.md) 训练；没数据回 [00-起步/03-跑通现成模型](../00-起步/03-跑通现成模型.md) 步骤 7 下载。

## 步骤 2：单卡 SFT（最简命令）

```bash
cd /path/to/minimind/trainer
python train_full_sft.py
```

默认参数（见 `trainer/train_full_sft.py:84-108`）：

| 参数 | 默认 | 与 pretrain 区别 |
|---|---|---|
| `--epochs` | 2 | 相同 |
| `--batch_size` | 16 | pretrain 是 32（SFT 序列更长） |
| `--accumulation_steps` | 1 | pretrain 是 8（不累积） |
| `--learning_rate` | 1e-5 | pretrain 是 5e-4（小 50 倍） |
| `--max_seq_len` | 768 | pretrain 是 340（对话更长） |
| `--data_path` | ../dataset/sft_t2t_mini.jsonl | SFT 数据 |
| `--from_weight` | pretrain | **基于 pretrain 权重** |
| `--save_weight` | full_sft | 输出名 |
| `--use_moe` | 0 | MoE 切换 |

> `from_weight='pretrain'` 在 `train_full_sft.py:103`，`init_model` 会自动拼接 `out/pretrain_768.pth`（`trainer_utils.py:119-127`）。

启动后看到：
```
Model Params: 64.31M
Trainable Params: 64.309M
Epoch:[1/2](100/7634), loss: 1.2345, logits_loss: 1.2345, aux_loss: 0.0000, lr: 0.0000093, epoch_time: 28.3min
```

✅ **验证**：loss 从约 1-2 起步（不是 10，因为基于 pretrain 了），降到约 0.5-1。

## 步骤 3：开 swanlab 看曲线对比

```bash
cd /path/to/minimind/trainer
python train_full_sft.py --use_wandb
```

对比 pretrain 曲线的关键差异：
- SFT **起始 loss 远低于 pretrain**（继承知识，不再学 token 分布）
- SFT **学习率小 50 倍**，曲线更平
- SFT 有效 batch = 16（pretrain 是 256），step 数更多

## 步骤 4：训练完成后查看产物

```bash
ls -lh /path/to/minimind/out/full_sft_768.pth
ls -lh /path/to/minimind/checkpoints/full_sft_768_resume.pth
```

✅ **验证**：两个文件都在。`full_sft_768.pth` 是最终产物，能直接推理对话。

## 步骤 5：验证对话能力

```bash
cd /path/to/minimind
python eval_llm.py --load_from ./model --weight full_sft
```

启动后选 `[1] 手动输入`，试几个问题：

```
💬: 你好
🧠: 你好！我是 minimind，一个由学习者训练的小型语言模型，很高兴认识你。

💬: 为什么天空是蓝色的
🧠: 天空呈蓝色是因为大气层对阳光的瑞利散射...

💬: 请用Python写一个计算斐波那契数列的函数
🧠: def fib(n):
    ...
```

✅ **验证**：
- 模型**会回答+停止**（不再像 pretrain 那样续写到 max_new_tokens）
- 答完自动打 `<|im_end|>` 让 generate 停下
- 中文流畅、有逻辑、能写代码

## 步骤 6：多轮对话验证

```bash
cd /path/to/minimind
python eval_llm.py --load_from ./model --weight full_sft --historys 4
```

`--historys 4` 携带最近 2 轮（4 条消息）历史（`eval_llm.py:71`）。

```
💬: 我叫 TJK
🧠: 你好 TJK！很高兴认识你...

💬: 我叫什么名字？    ← 测试模型是否记得上文
🧠: 你叫 TJK。
```

✅ **验证**：模型能引用前文对话内容，说明 history 机制工作正常。

---

## 二、关键机制补充

### 2.1 SFT 训练循环与 Pretrain 几乎一致

对比 `train_pretrain.py` 和 `train_full_sft.py`：
- 训练循环（`train_epoch` 函数）**完全一样**，包括 AMP / 梯度累积 / DDP / ckp
- 唯一区别在数据集类（`SFTDataset` vs `PretrainDataset`）和默认参数

这就是"训练框架统一"的好处 —— 看懂一个就都懂了。

### 2.2 SFT 的断点续训

```bash
cd /path/to/minimind/trainer
python train_full_sft.py --from_resume 1
```

机制和 pretrain 完全一致（`train_full_sft.py:69-70` 调用 `lm_checkpoint`）：
- 检查点存 `checkpoints/full_sft_768_resume.pth`
- 自动恢复 model/optimizer/scaler/epoch/step

### 2.3 SFT 的多卡训练

```bash
cd /path/to/minimind/trainer
torchrun --nproc_per_node 2 train_full_sft.py
```

### 2.4 SFT 的 MoE 版本

```bash
cd /path/to/minimind/trainer
# 前提：pretrain 也得是 MoE，即先跑 train_pretrain.py --use_moe 1
python train_full_sft.py --use_moe 1 --from_weight pretrain
```

输出 `out/full_sft_768_moe.pth`。

---

## 三、训练曲线对比

| 阶段 | dense 模型 loss 范围 | 说明 |
|---|---|---|
| step 0 | ~1.5-2 | 起步低（基于 pretrain） |
| step 1000 | ~1.0-1.2 | 学对话格式 |
| step 5000 | ~0.7-0.9 | 学回答质量 |
| 2 epochs 后 | ~0.5-0.7 | 收敛 |

> loss 数值**不能跨数据集比较**（token 分布不同）。比 pretrain 的 ~2-3 低是因为 mask 掉了一半序列不算 loss，并不是模型变好了。

---

## 🧯 踩坑提示

### Q1：`RuntimeError: size mismatch for lm_head.weight`
学习率太大把权重搞乱了。SFT lr 必须 ≤ 1e-5，默认就是这个，不要手动调到 1e-4 以上。

### Q2：训练完模型回答里全是 `<|im_start|>` 之类的特殊标记
数据生成有 bug。检查 `dataset/sft_t2t_mini.jsonl` 是否包含完整 assistant 回复（不能为空）。

### Q3：SFT 后模型反而**不如 pretrain 接龙好**
正常 —— SFT 是把模型从"通用接龙器"调成"对话助手"，接龙能力会下降一些。如果想保持通用能力，需要更大数据 + 更小 lr + 更少 epoch。

### Q4：模型对话里出现重复（如"我我是是是minimind"）
- 调高 `--max_seq_len`（768 → 1024），数据可能被截断了
- 推理时调 `--temperature 0.5` 或加 `repetition_penalty`（见下篇）

### Q5：显存不够（OOM）
SFT 比 pretrain 更吃显存（max_seq_len 翻倍）。降 `--batch_size 16 → 4`，加 `--accumulation_steps 4`。

### Q6：4090 时间预算
- dense（64M）：约 1.1h / 2 epochs
- MoE（198M-A64M）：约 1.4h
- 2 卡 DDP：约 0.65h
- 完整数据集 `sft_t2t.jsonl`（14GB）：约 10x 时间

### Q7：模型能做 Tool Call 吗
能。minimind-3 把 tool call 数据混入了 `sft_t2t_mini.jsonl` 主线（见 `dataset/lm_dataset.py:11` 注释"tool use 数据完整保留"）。`full_sft_768.pth` 已具备基础 Tool Call 能力。深度测试见 [03-进阶/03-工具调用与思考](../03-进阶/03-工具调用与思考.md)。

### Q8：SFT 学完后还有"思考链"吗
mini 数据训出的 Zero 模型思考能力很弱，需要完整 SFT + RLAIF 才有明显思考链。模板里 `<think></think>` 标签已就绪，推理时 `--open_thinking 1` 开启。

---

## ✅ 本篇完成自检

<details>
<summary>点开自检（先想 30 秒）</summary>

1. SFT 的 labels 为什么 user/system 部分要设 -100？
   - 那些是输入，模型不该学"生成用户的提问"。只在 assistant 部分算 loss 让模型聚焦学"如何回答"。
2. SFT 学习率为什么比 pretrain 小 50 倍？
   - SFT 是基于 pretrain 的"精修"，大 lr 会破坏预训练学到的知识（灾难性遗忘）。
3. `SFTDataset` 的 `bos_id` 和 `eos_id` 是什么？
   - `bos_id` = `tokenizer("<|im_start|>assistant\n")`（`lm_dataset.py:65`），用于定位"assistant 回复开始"。
   - `eos_id` = `tokenizer("<|im_end|>\n")`（`lm_dataset.py:66`），用于定位"回复结束"。
4. SFT 后模型为什么能"答完就停下"？
   - 数据里每个 assistant 回复都以 `<|im_end|>` 结尾。模型学会打这个标记后，`generate` 检测到 eos_token_id 就停止（`model_minimind.py:283-285`）。
5. 为什么 SFT 的起始 loss 远低于 pretrain？
   - 因为基于 pretrain 权重续训，模型已经掌握语言知识，不需要从零学 token 分布。
6. minimind-3 的 SFT 数据为什么已经能做 Tool Call？
   - `sft_t2t_mini.jsonl` 主线已混入 tool use 样本（带 `tools` / `tool_calls` 字段），`SFTDataset.create_chat_prompt` 会自动 `apply_chat_template(..., tools=tools)` 渲染。
7. `pre_processing_chat` 20% 加 system prompt 的作用？
   - 数据增强，让模型见过带 system / 不带 system 两种情况，提高鲁棒性（`lm_dataset.py:9-29`）。

</details>

下一篇：[03-推理与采样](./03-推理与采样.md) —— 把刚训好的模型用起来。
