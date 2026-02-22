# MiniMind 新手学习路线（含基础知识速查）

这份指南面向 **LLM 零基础/初学者**，目标是用最短路径理解并跑通 MiniMind 的训练闭环，然后逐步深入源码细节。

**你将得到什么**
- 一条清晰的学习路线（从“能跑通”到“看懂源码”）
- 训练前需要的最少基础知识
- 常见报错与排错方向
- 必要的概念速查（形状、loss、混合精度、DDP 等）

---

## 1. 最短可行路线（先跑通再深入）

**目标**：先完整跑通一次预训练，再回头读源码。

1. 确保能运行 `trainer/train_pretrain.py`（最小数据集也行）
2. 跑一次训练并看到 loss 输出
3. 用 `eval_llm.py` 做一次推理
4. 再开始读源码（按推荐顺序）

> 经验：先跑通再读源码，比一开始就硬啃模型代码更高效。

---

## 2. 源码阅读顺序（强烈推荐）

1. `trainer/train_pretrain.py`  
2. `dataset/lm_dataset.py`  
3. `model/model_minimind.py`  
4. `trainer/trainer_utils.py`  
5. `eval_llm.py`  

**原因**：这条顺序刚好对应“数据 → 模型 → 训练 → 保存/恢复 → 推理”。

---

## 3. 必备基础知识（最少集合）

**Python 基础**
- 列表重复：`[0] * N`
- 切片：`a[:-1]` / `a[1:]`
- `...`（Ellipsis）用于多维切片
- 类与 `self` 的含义

**PyTorch 最小闭环**
- forward → loss → backward → step
- `Dataset / DataLoader`
- `Tensor` 的 shape / dtype / device
- `torch.nn.functional.cross_entropy`

**LLM 训练最小概念**
- “预测下一个 token”的 shift 逻辑  
- `labels` 中 `-100` 表示忽略 loss  
- `batch_size / seq_len / vocab_size` 的形状关系

---

## 4. 形状速查（理解训练最关键）

**常见张量形状**
- `input_ids`: `(batch_size, seq_len)`  
- `logits`: `(batch_size, seq_len, vocab_size)`  
- `shift_logits`: `(batch_size, seq_len-1, vocab_size)`  
- `shift_labels`: `(batch_size, seq_len-1)`  

**loss 逻辑**
- 模型学的是“预测下一个 token”
- 所以要让 `logits[:, :-1]` 对齐 `labels[:, 1:]`

---

## 5. 训练参数怎么理解（新手版）

- `batch_size`: 越大越吃显存  
- `max_seq_len`: 越大显存增长很快  
- `hidden_size / num_hidden_layers`: 决定模型规模  
- `learning_rate`: 太大容易发散  
- `accumulation_steps`: 让“显存小”也能模拟大 batch  
- `dtype`: `bfloat16` 稳，`float16` 快但更易不稳  

---

## 6. 常见问题与排错顺序

**1) 找不到数据文件**
- 优先检查 `--data_path` 是否存在  

**2) 报错提示 `../model` 或 `../dataset`**
- 说明运行路径不对  
- 建议在 `trainer/` 目录下运行脚本  

**3) OOM（显存不够）**
- 优先降低 `batch_size`  
- 再降低 `max_seq_len`  
- 再减小 `hidden_size/num_hidden_layers`  

**4) loss 变 NaN**
- 先把 `learning_rate` 降 10 倍  
- 检查数据是否异常  

---

## 7. 推荐练习（快速提升理解）

1. 用 5 行数据过拟合（loss 应快速下降）  
2. 把 `max_seq_len` 改大，观察显存变化  
3. 把 `learning_rate` 调大，观察 loss 发散  
4. 改 `hidden_size`，理解模型规模与速度关系  

---

## 8. 下一步建议

当你能解释下面三件事，就说明已经入门：
- 文本是如何变成 `input_ids/labels` 的  
- loss 在哪里算、为什么要 shift  
- 训练循环里参数是如何更新的  

如果还不确定，可以回到：
- `trainer/train_pretrain.py`  
- `dataset/lm_dataset.py`  
- `model/model_minimind.py`  

