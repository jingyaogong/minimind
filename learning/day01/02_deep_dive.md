# Day 1 Deep Dive: MiniMind 模型结构拆解

## 深挖问题 1：MiniMind 的最小 forward 路径是什么？

打开 `model/model_minimind.py`：

1. `MiniMindForCausalLM.forward`
2. `MiniMindModel.forward`
3. `MiniMindBlock.forward`
4. `Attention.forward`
5. `FeedForward.forward`

你要能追踪：

```text
input_ids
-> embed_tokens
-> dropout
-> N x MiniMindBlock
-> RMSNorm
-> lm_head
-> logits
-> cross_entropy loss
```

重点观察：

- `input_ids.shape == [batch, seq_len]`
- `hidden_states.shape == [batch, seq_len, hidden_size]`
- `logits.shape == [batch, seq_len, vocab_size]`

## 深挖问题 2：为什么 forward 里是这样算 loss？

代码位置：`MiniMindForCausalLM.forward`

核心逻辑：

```python
x = logits[..., :-1, :]
y = labels[..., 1:]
loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
```

理解方式：

- 位置 0 的 logits 预测 token 1。
- 位置 1 的 logits 预测 token 2。
- 最后一个位置没有下一个 token，所以丢掉。
- label 的第一个 token 没有被任何上文预测，所以丢掉。

`ignore_index=-100` 是训练中最重要的 mask 约定之一。pretrain 用它忽略 padding；SFT 用它忽略 user/system/prompt 部分。

## 深挖问题 3：Attention 里 GQA 是怎么实现的？

看 `Attention.__init__`：

- `num_attention_heads`
- `num_key_value_heads`
- `n_rep = q_heads // kv_heads`

看 `repeat_kv`：

- K/V heads 先少算。
- attention 前把 K/V 复制到和 Q heads 对齐。

工业理解：

- 训练时省一部分参数和计算。
- 推理时 KV cache 更小。
- 对小模型通常是合理取舍。

## 深挖问题 4：MoE 为什么有 aux loss？

看 `MOEFeedForward.forward`：

- gate 给每个 token 分配 expert。
- `topk_idx` 决定 token 去哪个 expert。
- `aux_loss` 鼓励 expert 负载更均衡。

工业风险：

- 没有负载均衡，部分 expert 可能被过度使用。
- 原生 PyTorch MoE 训练不一定快，因为 expert 分派有额外调度开销。

## 今日小实验

写一个临时 Python 片段，不修改项目文件：

```bash
python - <<'PY'
import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

config = MiniMindConfig(hidden_size=128, num_hidden_layers=2)
model = MiniMindForCausalLM(config)
input_ids = torch.randint(0, config.vocab_size, (2, 16))
labels = input_ids.clone()
out = model(input_ids, labels=labels)
print(out.logits.shape)
print(out.loss.item())
PY
```

你要解释：

- 为什么随机模型也有 loss？
- loss 大概应接近 `log(vocab_size)`，为什么？
