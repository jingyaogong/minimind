# Day 3 Exam

## A. 代码阅读题

1. `train_pretrain.py` 中 `autocast_ctx` 是如何根据 device 决定的？
2. `scaler` 什么时候启用？bf16 下是否启用？
3. `optimizer.zero_grad(set_to_none=True)` 的好处是什么？
4. `SkipBatchSampler` 在 resume 时解决什么问题？
5. `torch.save({k: v.half().cpu() ...})` 为什么要转 half 和 cpu？

## B. 计算题

假设：

```text
batch_size = 32
accumulation_steps = 8
world_size = 1
max_seq_len = 340
```

回答：

- effective batch size 是多少条样本？
- 每次 optimizer step 理论上看过多少 token？
- 如果 world_size 变成 4，effective batch size 如何变化？

## C. 故障排查题

1. 训练第一个 step 就 OOM，你优先改哪些参数？
2. loss 变成 NaN，可能原因有哪些？
3. resume 后 loss 突然跳变，可能是什么状态没恢复？
4. 训练很慢，如何判断瓶颈在数据加载还是 GPU 计算？

## D. 实验题

完成一次 pretrain，并记录：

- loss 起点
- loss 终点
- 每 100 step 的 lr
- 每个 epoch 耗时
- 最终权重路径

然后运行：

```bash
python eval_llm.py --weight pretrain
```

评价 pretrain 模型是否已经具备聊天能力。为什么？

## E. 通过标准

你能解释完整 pretrain loop，并成功得到 `out/pretrain_768.pth`。
