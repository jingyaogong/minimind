# Day 2 Exam

## A. 代码阅读题

1. `PretrainDataset.__getitem__` 为什么要手动加 BOS/EOS？
2. `SFTDataset.create_chat_prompt` 中 `tools` 是从哪里来的？
3. `SFTDataset.generate_labels` 如何定位 assistant 内容？
4. `post_processing_chat` 为什么会概率性移除空 `<think>`？
5. `RLAIFDataset.__getitem__` 为什么只返回 prompt，不返回标准 answer？

## B. 手写题

给定 SFT 对话：

```json
[
  {"role": "user", "content": "2+2等于几？"},
  {"role": "assistant", "content": "等于4。"}
]
```

请说明：

- 哪些部分是条件上下文？
- 哪些部分参与 loss？
- 如果 assistant 回答被截断到只剩 `"等于"`，训练信号有什么问题？

## C. 工业题

1. 如何发现 SFT mask 写错了？
2. 如果训练 loss 很低但模型推理不会聊天，可能是什么原因？
3. 如果 SFT 数据中 tool call JSON 经常不合法，会导致什么问题？
4. 为什么数据抽样检查比直接看 loss 更重要？

## D. 实验题

抽 3 条真实 SFT 样本，打印：

- chat template 后的文本
- token 数量
- loss token 数量
- loss token 占比

写出你的观察：样本长度分布和 loss token 占比是否合理？

## E. 通过标准

你能解释：

```text
同样是 cross entropy，pretrain 和 SFT 的差异主要来自 labels。
```
