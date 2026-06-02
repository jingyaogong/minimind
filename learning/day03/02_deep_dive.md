# Day 3 Deep Dive: Pretrain 训练循环

## 深挖问题 1：训练循环每一步发生了什么？

代码位置：`trainer/train_pretrain.py::train_epoch`

流程：

```text
for input_ids, labels in loader:
    move to device
    update lr
    autocast forward
    loss = res.loss + res.aux_loss
    loss /= accumulation_steps
    backward
    if accumulation boundary:
        unscale
        grad clip
        optimizer step
        scaler update
        zero grad
    log
    save
```

你要能解释每一行为什么存在。

## 深挖问题 2：学习率怎么变？

代码位置：`trainer/trainer_utils.py::get_lr`

```python
return lr * (0.1 + 0.45 * (1 + cos(pi * current_step / total_steps)))
```

含义：

- 初始接近 `lr`。
- 最终接近 `0.1 * lr`。
- 中间 cosine 平滑下降。

工业注意：

- 这个实现没有 warmup。
- 小模型和小数据通常能跑，但大规模训练一般会加 warmup。
- 如果 early loss 抖动严重，可以考虑降低 lr 或加 warmup。

## 深挖问题 3：checkpoint 保存了什么？

代码位置：`trainer/trainer_utils.py::lm_checkpoint`

保存两类文件：

- `out/pretrain_768.pth`：主要用于推理/下游训练。
- `checkpoints/pretrain_768_resume.pth`：用于断点续训。

resume checkpoint 包含：

- model
- optimizer
- epoch
- step
- world_size
- wandb id
- scaler 等额外状态

为什么 optimizer 也要保存？

因为 AdamW 有动量状态。如果只恢复模型权重，优化器状态丢失，训练动态会突然改变。

## 深挖问题 4：MoE aux_loss 为什么加在 pretrain loss 上？

如果 `use_moe=1`，模型输出 `aux_loss`。训练总 loss：

```python
loss = res.loss + res.aux_loss
```

这表示主目标仍是 next-token prediction，但额外鼓励 expert 负载均衡。

## 今日实验

1. 先跑 200 step smoke test：

```bash
cd trainer
python train_pretrain.py --epochs 1 --save_interval 200 --log_interval 20
```

2. 中断训练。

3. 恢复：

```bash
python train_pretrain.py --epochs 2 --from_resume 1
```

4. 检查日志是否显示跳过 step。

## 复盘问题

- 为什么 `loss / accumulation_steps` 后再 backward？
- 为什么 `scaler.unscale_` 后再 clip grad？
- 为什么保存权重前切 `model.eval()`，保存后再 `model.train()`？
