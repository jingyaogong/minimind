# Day 1 Exam

## A. 代码阅读题

1. `MiniMindForCausalLM.forward` 中 `logits[..., :-1, :]` 的含义是什么？
2. `labels[..., 1:]` 为什么要右移？
3. `ignore_index=-100` 在 pretrain 和 SFT 中分别解决什么问题？
4. `Attention.forward` 中 `past_key_value` 的作用是什么？
5. `repeat_kv` 为什么只重复 K/V，不重复 Q？

## B. 手写题

给定：

```text
input_ids = [BOS, 我, 爱, AI, EOS, PAD, PAD]
```

请写出 causal LM 中每个位置预测的目标 token。

再写出 padding 后的 labels，其中 PAD 对应 `-100`。

## C. 面试题

1. RoPE 为什么能支持比训练长度更长的外推？
2. RMSNorm 为什么比 LayerNorm 更轻？
3. 小模型中为什么 vocab size 不能无限大？
4. GQA 相比 MHA 的主要收益是什么？

## D. 实验题

运行 Day 1 deep dive 的随机模型 forward 实验，记录：

- logits shape
- loss 数值
- `log(6400)` 的值

回答：随机初始化模型的 loss 为什么和 `log(vocab_size)` 接近？

## E. 通过标准

你能不看代码讲出：

```text
input_ids -> embeddings -> blocks -> lm_head -> logits -> shifted CE loss
```

并能解释 `-100`、shift、RoPE、GQA。
