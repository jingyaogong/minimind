# Day 5 Exam

## A. 代码阅读题

1. `eval_llm.py` 中 `open_thinking` 如何传给 chat template？
2. `temperature` 在 `model.generate` 中如何影响 logits？
3. `top_p` 和 `top_k` 可以同时使用吗？有什么效果？
4. `eval_toolcall.py` 的多轮工具调用循环何时停止？
5. tool call 失败时，应该先看模型输出还是工具执行？

## B. 评测设计题

为 MiniMind 设计一个 20 题小评测集，覆盖：

- 身份/自我认知
- 基础事实
- 中文表达
- 代码
- 总结
- 简单数学
- tool call

每题给出评分标准：0/1/2 分。

## C. 失败分析题

对于以下现象，写出至少 3 个可能原因：

1. 模型回答流畅但事实错误。
2. 模型重复同一句话。
3. 模型不按 JSON 输出 tool call。
4. 模型回答特别短。
5. 模型输出大量 `<think>` 但内容无意义。

## D. 实验题

用同一组 prompt，分别测试：

```bash
python eval_llm.py --weight full_sft --temperature 0.3 --top_p 0.9
python eval_llm.py --weight full_sft --temperature 0.9 --top_p 0.95
```

记录不同采样参数下：

- 稳定性
- 创造性
- 幻觉
- 重复

## E. 通过标准

你能给一个模型输出做结构化诊断，而不是只说“好”或“不好”。
