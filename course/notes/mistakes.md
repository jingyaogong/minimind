# Mistakes

这里记录写代码时真正犯过的错误。

格式：

```text
日期：
课程：
错误：
为什么错：
正确源码：
以后怎么判断：
```

重点记录：

- shape 维度混淆
- attention mask / causal mask 错误
- labels 和 `-100` 错误
- logits/labels shift 错误
- KV cache 的 `start_pos` 或拼接维度错误
- logprob gather 维度错误
- chosen/rejected 对齐错误
