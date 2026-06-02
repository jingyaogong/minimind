# Day 1 Prereading: 环境、项目地图、Decoder-only 基础

## 今日目标

你今天要建立项目全局地图，知道 MiniMind 的模型结构、训练脚本、数据集、评测入口分别在哪里。不要急着长时间训练，先确认环境、GPU、依赖和最小 forward 能跑通。

## 必须掌握的基础概念

- Decoder-only Transformer：输入 token 序列，预测下一个 token。
- Causal LM loss：第 `t` 个位置的 logits 预测第 `t+1` 个 token。
  > **笔记：** Causal = 只能看前面的 token（上三角 mask）。Loss = `CrossEntropy(logits[:-1], labels[1:])`，即位置 t 预测 t+1。与 BERT (Masked LM, 双向) 相对。
- Tokenizer：把文本变成 token id，模型只看 token id。

  > **笔记：** Tokenizer 由两个文件构成：`tokenizer.json`（词表+切词规则）和 `tokenizer_config.json`（特殊 token 定义 + chat_template 对话格式模板）。关键特殊 token：id0=PAD, id1=BOS(`<|im_start|>`), id2=EOS(`<|im_end|>`), id21-22=tool_call, id25-26=think。
  > 切词方式：BPE + ByteLevel。先把文本转 UTF-8 bytes（任何语言都不会 UNK），再按 6108 条训练好的合并规则把高频 byte 组合合并成 token。词表 6400，中文约 1.5-1.7 字符/token。
- Embedding / LM Head：输入词表映射到 hidden，输出 hidden 映射回 vocab logits。
  > **笔记：** 接力关系：Tokenizer (文本→id, 模型外部) → Embedding (id→向量, 入口) → LM Head (向量→logits, 出口)。MiniMind 中 Embedding 和 LM Head 共享权重 (`tie_word_embeddings=True`)。
  > Hidden = Embedding 只是起点（查表），经过 8 层 Transformer 融合上下文后的向量才是真正的 hidden (768维)。Vocab logits = hidden @ LM_Head矩阵(768×6400) 得到的得分向量 (6400维)，softmax 后变概率。
  > **Weight Tying 原理：** Embedding `[6400,768]` 和 LM Head `[768,6400]` 互为逆操作（id→向量 vs 向量→得分），共享同一份权重（LM Head = Embedding 转置）。省 490 万参数（占 MiniMind 约 20%），且让编码/解码共用语义空间，效果更好。GPT-2/LLaMA/Qwen 均采用。
- Pre-Norm：attention/MLP 前先做 RMSNorm，训练更稳定。
- Flash Attention：优化注意力计算的算法，通过分块（tiling）在 GPU SRAM 内完成计算，避免将 O(n²) 注意力矩阵反复搬运到慢速显存（HBM），数学等价但速度快 2-4x、显存降至 O(n)。PyTorch 2.0+ 通过 `F.scaled_dot_product_attention` 内置支持。
- RoPE：旋转位置编码，把位置信息注入 Q/K。
  > **笔记：** RoPE 不是专为 Causal LM 发明，而是解决通用问题：Transformer 本身无位置感知。演进：Learned Absolute (固定长度) → Sinusoidal → RoPE (旋转 Q/K，天然编码相对距离，可外推)。Causal LM 对长序列需求大，所以 RoPE 在此场景尤其流行。
  > **YaRN 外推：** `max_position_embeddings`(32K) > 训练长度 `max_seq_len`(8K)，靠 YaRN 实现。原理：把 RoPE 频率分三区——低频（长距离）压缩适配、高频（短距离）保持不变、中频渐进过渡。避免简单等比缩放丢失近距离精度。对应参数：`yarn_scale`、`yarn_beta_fast/slow`。
- GQA：query heads 多于 key/value heads，降低 KV cache 和计算成本。
  > **笔记：** MHA 8Q:8KV → GQA 8Q:4KV → MQA 8Q:1KV。MiniMind 用 GQA (每2个Q头共享1对KV)，KV cache 省50%。
- SwiGLU：现代 LLM 常用 FFN 结构。
  > **笔记：** `Down(SiLU(Gate(x)) * Up(x))`。比标准 FFN 多一个 Gate 矩阵 (参数+50%)，Gate 起可学习门控作用，选择性保留信息。LLaMA/Qwen/DeepSeek 均采用。

## 面试考点

- 为什么 causal LM 训练时 logits 和 labels 要 shift？
  > **答：** 自回归目标是"用位置 t 预测 t+1"。logits 的最后一个位置没有下一个 token 可预测，labels 的第一个 token 没有上文来预测它，所以 `logits[:-1]` 对齐 `labels[1:]`。不 shift 的话预测目标就错位了。
- Decoder-only 和 Encoder-Decoder 在训练目标上有什么区别？
  > **答：** Decoder-only（GPT）：所有 token 统一做从左到右的 next-token prediction，输入和输出共享同一个序列。Encoder-Decoder（T5）：Encoder 双向编码输入（无因果 mask），Decoder 自回归生成输出，两者是分开的序列。Decoder-only 更简单，规模扩展更直接，是当前主流。
- RMSNorm 和 LayerNorm 的区别是什么？
  > **答：** LayerNorm: `(x - mean) / std * γ + β`，需要均值和方差，有偏置 β。RMSNorm: `x / RMS(x) * γ`，只用均方根，无偏置。RMSNorm 省掉 mean 计算和 β 参数，速度快 ~10-15%，效果基本无损，所以现代 LLM 都用 RMSNorm。
- RoPE 相比 learned absolute position embedding 的优势是什么？
  > **答：** ① 天然编码相对距离（Q·K 只依赖 |i-j|，而非绝对位置）；② 可外推到训练时没见过的长度（配合 YaRN 更好）；③ 不增加可学习参数；④ Learned absolute 长度固定，超出即崩，且只编码绝对位置，对相对关系的建模隐式且弱。

- GQA/MQA/MHA 的区别和工程取舍是什么？
  > **答：** MHA: Q=K=V 头数相同，表达力最强但 KV cache 最大。MQA: 所有 Q 头共享 1 个 KV 头，KV cache 最小但质量可能下降。GQA: 折中，多个 Q 头分组共享 KV 头（MiniMind: 8Q:4KV）。工程取舍核心是推理时 KV cache 显存 vs 模型质量——大规模部署时 KV cache 是瓶颈，GQA 是当前最佳平衡点。

- 为什么小模型里 vocab size 会显著影响参数量？
  > **答：** Embedding 层参数 = `vocab_size × hidden_size`。MiniMind: 6400×768 = 490 万。如果用 Qwen2 的 151K 词表: 151643×768 = 1.16 亿，**仅 Embedding 就超过整个模型 (64M)**。小模型总参数少，Embedding 占比极高，所以词表必须精简。MiniMind 用 6400 词表正是这个原因。

## 工业要求
- 训练前先确定可复现环境：Python、CUDA、PyTorch、GPU 显存。
- 所有实验记录命令、git 状态、数据版本、模型参数、随机种子。
- 第一天不追求效果，只追求可运行、可定位问题。
- 任何训练前先做小 batch smoke test，避免长时间跑错配置。

## 项目中要看的文件
- `README.md`：快速开始、数据、训练说明。
- `model/model_minimind.py`：模型结构。
- `trainer/train_pretrain.py`：预训练入口。
- `trainer/train_full_sft.py`：SFT 入口。
- `dataset/lm_dataset.py`：数据到 tensor 的转换。
- `eval_llm.py`：训练后推理评测。

## 今日动手任务

1. 确认环境：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

2. 阅读 `model/model_minimind.py` 中这些类：

- `MiniMindConfig`
- `Attention`
- `FeedForward`
- `MOEFeedForward`
- `MiniMindForCausalLM`

3. 写一段自己的结构笔记：输入 `[B, T]` 如何变成 logits `[B, T, vocab]`。

## 训练注意事项

- 今天不要直接跑完整训练。
- 如果没有 CUDA，后续训练计划必须缩小模型或只做代码理解。
- 如果用 Mac/MPS，混合精度、性能和部分算子兼容性要单独确认。
