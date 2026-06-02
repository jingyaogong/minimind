# Day 5 Deep Dive: 生成质量为什么不等于训练 loss

## 深挖问题 1：eval_llm 如何构造输入？

打开 `eval_llm.py`。

关键差异：

```python
if 'pretrain' in args.weight:
    inputs = tokenizer.bos_token + prompt
else:
    inputs = tokenizer.apply_chat_template(...)
```

说明：

- pretrain 模型只适合文本续写。
- SFT 模型适合 chat template。
- 如果推理模板和训练模板不一致，效果会明显变差。

## 深挖问题 2：采样参数如何影响输出？

`generate` 中关键参数：

- `temperature`：logits 除以温度，越高越随机。
- `top_p`：只在累计概率前 p 的 token 中采样。
- `top_k`：只在概率最高 k 个 token 中采样。
- `repetition_penalty`：惩罚已出现 token。

实验建议：

```bash
python eval_llm.py --weight full_sft --temperature 0.3 --top_p 0.9
python eval_llm.py --weight full_sft --temperature 0.9 --top_p 0.95
```

观察：

- 低温更稳定，但可能死板。
- 高温更多样，但更容易胡编。

## 深挖问题 3：为什么 loss 低但生成差？

常见原因：

- 数据分布简单，loss 易降但泛化差。
- teacher-forcing 训练，推理时错误会累积。
- 评测 prompt 超出训练分布。
- 模板不一致。
- 采样参数过激。
- 小模型容量不足。

## 深挖问题 4：Tool call 如何评测？

打开 `scripts/eval_toolcall.py`。

关注：

- 工具 schema 如何给模型。
- 模型如何输出 `<tool_call>`。
- `parse_tool_calls` 如何解析。
- `execute_tool` 如何执行。
- 工具结果如何追加回 messages。

失败类型分类：

```text
格式失败：JSON 不合法或标签不闭合
选择失败：调用了错误工具
参数失败：工具对但参数错
利用失败：工具结果对，但最终回答错
```

## 今日产出

写一份 `eval_report.md`，至少包含：

- 评测命令
- prompt 列表
- 每个 prompt 的回答
- 评分表
- 失败样例
- 下一步改进建议
