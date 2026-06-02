# Day 4 Prereading: SFT 实战

## 今日目标

今天从 `pretrain` 权重继续训练 `full_sft`，得到能按对话模板回答的模型。

## 必须掌握的基础概念

- SFT：Supervised Fine-Tuning，用高质量指令/对话数据训练助手行为。
- Instruction following：理解用户意图并按格式回答。
- Chat template：训练和推理必须一致。
- Assistant-only loss：只让模型学习 assistant 输出，不学习复述 user。
- Catastrophic forgetting：微调过窄数据可能损伤原有能力。

## 面试考点

- Pretrain 和 SFT 的本质区别是什么？
- SFT 能不能注入新知识？
- 为什么 SFT 学到的不只是格式，也可能改变知识分布？
- 为什么学习率比 pretrain 小很多？
- 为什么 SFT 后模型更会聊天，但不一定更真实？

## 工业要求

- SFT 数据必须检查格式一致性。
- 训练和推理的 chat template 必须一致。
- SFT 评测不能只看 loss，要看真实生成。
- 小模型 SFT 容易学表面风格，但事实性仍弱。
- 如果领域微调，要混合通用数据避免能力坍缩。

## 项目中要看的文件

- `trainer/train_full_sft.py`
- `dataset/lm_dataset.py::SFTDataset`
- `eval_llm.py`
- `scripts/eval_toolcall.py`

## 今日推荐命令

```bash
cd trainer
python train_full_sft.py \
  --epochs 2 \
  --from_weight pretrain \
  --batch_size 16 \
  --accumulation_steps 1 \
  --max_seq_len 768 \
  --learning_rate 1e-5 \
  --log_interval 100 \
  --save_interval 1000
```

训练后：

```bash
cd ..
python eval_llm.py --weight full_sft
```

## 今日记录模板

```text
SFT 命令：
from_weight：
数据：
max_seq_len：
初始 loss：
最终 loss：
pretrain 回答样例：
full_sft 回答样例：
明显提升：
明显问题：
```
