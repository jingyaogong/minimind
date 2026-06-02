# Day 4 Exam

## A. 代码阅读题

1. `train_full_sft.py` 和 `train_pretrain.py` 的训练循环有哪些完全相同的地方？
2. SFT 的 `from_weight` 默认值是什么？为什么？
3. `max_seq_len=768` 相比 pretrain 默认 `340` 的意义是什么？
4. `SFTDataset.generate_labels` 是否会训练多轮对话里的多个 assistant？
5. `eval_llm.py` 中 pretrain 权重和 SFT 权重的 prompt 构造有什么区别？

## B. 对比题

用同一个 prompt 比较 `pretrain` 和 `full_sft`：

```text
解释什么是机器学习。
```

回答：

- 哪个更像聊天助手？
- 哪个更可能只是续写文本？
- 这种差异来自模型结构还是训练数据？

## C. 工业题

1. SFT 数据中如果有大量低质量长回答，会对模型造成什么影响？
2. 为什么 SFT 前后必须固定一组评测 prompt？
3. 如何判断模型是在学能力，还是只是在学模板？
4. 领域 SFT 为什么要混合通用数据？

## D. 实验题

完成 SFT 后，运行：

```bash
python eval_llm.py --weight full_sft
cd scripts && python eval_toolcall.py --weight full_sft
```

记录：

- 普通对话 8 个 prompt 的表现。
- tool call 自动测试中成功和失败样例。
- 失败原因是格式、工具选择、参数、还是最终回答。

## E. 通过标准

你能解释：

```text
SFT 训练循环不是魔法，真正关键是 chat template 和 assistant-only labels。
```
