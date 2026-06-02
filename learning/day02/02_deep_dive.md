# Day 2 Deep Dive: Dataset 到 Loss 的路径

## 深挖问题 1：PretrainDataset 做了什么？

代码位置：`dataset/lm_dataset.py` 的 `PretrainDataset`

流程：

```text
sample["text"]
-> tokenizer(..., add_special_tokens=False)
-> [BOS] + tokens + [EOS]
-> pad 到 max_length
-> labels = input_ids.clone()
-> labels[PAD] = -100
```

关键点：

- pretrain 让模型学习所有非 PAD token 的 next-token prediction。
- BOS/EOS 也参与训练，帮助模型学习开始和结束。

## 深挖问题 2：SFTDataset 为什么更复杂？

代码位置：`SFTDataset.create_chat_prompt` 和 `SFTDataset.generate_labels`

SFT 输入包含 system/user/assistant/tool 等角色。训练目标不是复述完整对话，而是让模型在看到上下文后生成 assistant 输出。

因此：

- user/system/tool context 作为条件。
- assistant 内容作为监督目标。
- 非 assistant token label 设为 `-100`。

## 深挖问题 3：`generate_labels` 如何找到 assistant span？

代码里构造：

```python
self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
```

然后扫描 input_ids：

```text
找到 assistant 起点
-> 从 assistant 内容开始
-> 直到 eos
-> 这些 token 的 label = input_ids[j]
-> 其他位置 label = -100
```

你要特别注意：模板变化会直接影响这段扫描逻辑。如果 tokenizer template 改了，但 `bos_id/eos_id` 匹配没改，SFT loss mask 就可能错。

## 深挖问题 4：Tool Call 样本为什么要完整保留？

`pre_processing_chat` 里：

```python
if any(conv.get('tools') for conv in conversations): return conversations
```

原因：

- tool use 样本依赖 system 中的 tools schema。
- 随机加 system prompt 可能破坏工具调用上下文。
- tool call 训练要求模板结构严格。

## 今日小实验

不修改文件，运行：

```bash
python - <<'PY'
from transformers import AutoTokenizer
from dataset.lm_dataset import SFTDataset

tokenizer = AutoTokenizer.from_pretrained("model")
sample = {
    "conversations": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好，我是 MiniMind。"}
    ]
}

ds = SFTDataset("dataset/sft_t2t_mini.jsonl", tokenizer, max_length=128)
prompt = ds.create_chat_prompt(sample["conversations"])
ids = tokenizer(prompt).input_ids[:128]
labels = ds.generate_labels(ids)
print(prompt)
for i, (tid, lab) in enumerate(zip(ids, labels)):
    tok = tokenizer.decode([tid])
    flag = "LOSS" if lab != -100 else "MASK"
    print(i, repr(tok), flag)
PY
```

如果数据文件不存在，可以先等下载后再做。

## 复盘问题

- assistant 的 eos 是否参与 loss？为什么？
- 如果 `max_length` 在 assistant 回答中间截断，会发生什么？
- 多轮对话中每个 assistant span 是否都参与 loss？
