# Day 3 Prereading: Pretrain 实战

## 今日目标

今天正式完成预训练。你要理解训练循环、学习率、梯度累积、混合精度、checkpoint，并产出 `out/pretrain_768.pth`。

## 必须掌握的基础概念

- Epoch：完整遍历一次数据。
- Step：一个 batch 的训练迭代。
- Gradient accumulation：多个 micro-batch 累积后再 optimizer step。
- Effective batch size：`batch_size * accumulation_steps * world_size`。
- AdamW：主流 LLM 优化器。
- Grad clip：限制梯度范数，降低训练爆炸风险。
- Mixed precision：用 bf16/fp16 降低显存和提升吞吐。
- Cosine LR schedule：学习率从高到低平滑衰减。

## 面试考点

- 为什么 LLM pretrain 通常需要较大的学习率？
- 梯度累积和直接大 batch 是否完全等价？
- bf16 相比 fp16 的优势是什么？
- 为什么要做 gradient clipping？
- checkpoint 应该保存哪些状态，为什么只保存模型不够？

## 工业要求

- 必须记录完整训练命令。
- 必须记录 loss、lr、吞吐或耗时。
- 必须测试断点恢复。
- 训练中如果 loss NaN，先检查学习率、精度、数据异常、grad clip。
- 每次保存权重时确认输出文件名，避免覆盖误解。

## 项目中要看的文件

- `trainer/train_pretrain.py`
- `trainer/trainer_utils.py`
- `dataset/lm_dataset.py::PretrainDataset`

## 今日推荐命令

```bash
cd trainer
python train_pretrain.py \
  --epochs 2 \
  --batch_size 32 \
  --accumulation_steps 8 \
  --max_seq_len 340 \
  --learning_rate 5e-4 \
  --log_interval 100 \
  --save_interval 1000
```

如果显存不足：

```bash
python train_pretrain.py --batch_size 8 --accumulation_steps 32
```

## 今日记录模板

```text
命令：
GPU：
数据：
hidden_size / layers：
batch_size：
accumulation_steps：
max_seq_len：
初始 loss：
中段 loss：
最终 loss：
是否发生 OOM：
checkpoint 是否成功：
```
