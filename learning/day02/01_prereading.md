# Day 2 Prereading: Tokenizer、数据格式、Label Mask

## 今日目标

你要理解数据怎样进入训练。MiniMind 里 pretrain 和 SFT 的训练循环很像，真正关键差异在 dataset 和 label mask。

## 必须掌握的基础概念

- JSONL 数据：一行一个样本。
- Pretrain 样本：`{"text": "..."}`
- SFT 样本：`{"conversations": [{"role": "...", "content": "..."}]}`
- Chat template：把多轮消息序列变成模型输入文本。
- Label mask：哪些 token 参与 loss，哪些 token 不参与。
- `-100`：PyTorch cross entropy 中忽略 label 的约定。

## 面试考点

- SFT 为什么通常只对 assistant answer 计算 loss？
- 如果 user prompt 也计算 loss，会有什么副作用？
- `max_seq_len` 截断会造成什么训练问题？
- 为什么 tool call 要通过 chat template 统一成特殊标签？
- padding side 在训练和生成中有什么影响？

## 工业要求

- 数据质量比模型小改动更重要。
- 训练前必须抽样查看 tokenized 后的样本。
- SFT 必须检查 assistant span mask 是否正确。
- 长样本截断不能只看字符数，要看 token 数。
- Tool call / reasoning / normal chat 混合数据要保持模板一致。

## 项目中要看的文件

- `dataset/lm_dataset.py`
- `model/tokenizer.json`
- `model/tokenizer_config.json`
- `README.md` 的数据章节

重点类：

- `PretrainDataset`
- `SFTDataset`
- `RLAIFDataset`
- `AgentRLDataset`

## 今日动手任务

1. 下载或确认数据：

```text
dataset/pretrain_t2t_mini.jsonl
dataset/sft_t2t_mini.jsonl
```

2. 打印一条 pretrain 样本和一条 SFT 样本。

3. 用 tokenizer 对一条 SFT 样本做 `apply_chat_template`，观察模板输出。

4. 手动判断哪些 token 应该参与 loss。

## 训练注意事项

- 今天仍然不需要完整训练。
- 如果数据文件不存在，先只用 README 样例做理解。
- 任何 SFT 训练异常，优先怀疑 label mask 和 chat template。
