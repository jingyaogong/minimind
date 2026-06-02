# Day 4 Deep Dive: SFT 为什么能让模型会聊天

## 深挖问题 1：SFT 训练循环和 pretrain 有什么不同？

打开：

- `trainer/train_pretrain.py`
- `trainer/train_full_sft.py`

你会发现训练循环几乎一样。核心差异：

```text
PretrainDataset -> 所有正文 token 学 next token
SFTDataset      -> 只有 assistant span 学 next token
```

这说明：训练算法变化不大，监督信号变了。

## 深挖问题 2：`from_weight=pretrain` 为什么重要？

SFT 默认：

```python
parser.add_argument('--from_weight', default='pretrain')
```

如果从 `none` 开始 SFT，模型只看到问答格式，缺少基础语言建模能力，效果会差很多。

工程理解：

- pretrain 建立语言和世界知识底座。
- SFT 对齐交互格式和回答风格。
- 小数据 SFT 不能替代 pretrain。

## 深挖问题 3：为什么 SFT 学习率低？

默认：

- pretrain：`5e-4`
- SFT：`1e-5`

原因：

- SFT 是在已有模型上微调。
- 学习率太大会破坏 pretrain 已学到的分布。
- SFT 数据更窄，过大学习率容易过拟合风格或格式。

## 深挖问题 4：Tool Call 能力来自哪里？

README 说明当前 `sft_t2t_mini` 已混入 tool call 样本。

数据格式大致是：

```json
{"role": "system", "tools": "..."}
{"role": "user", "content": "..."}
{"role": "assistant", "tool_calls": "..."}
{"role": "tool", "content": "..."}
{"role": "assistant", "content": "..."}
```

模板会展开为：

```text
<tool_call>...</tool_call>
<tool_response>...</tool_response>
```

因此 full_sft 权重已经具备基础 tool call 格式能力。

## 今日实验

对比：

```bash
python eval_llm.py --weight pretrain
python eval_llm.py --weight full_sft
```

固定 prompt：

```text
你是谁？
解释什么是机器学习。
请用 Python 写一个斐波那契函数。
推荐一些中国美食。
```

记录：

- 回答是否像 assistant。
- 是否能按指令组织结构。
- 是否重复或胡编。
- 是否出现模板标签。

## 复盘问题

- 为什么 SFT loss 下降不代表模型事实性更强？
- 如果 SFT 后模型回答变短，可能是什么数据分布导致？
- 如果模型总输出 `<think>` 标签，可能和哪些训练样本有关？
