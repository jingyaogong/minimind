# 第15节：数据加载器构建（DDP + 断点续训 + batch 化）

> 核心目标：在 DDP 或单卡环境下，构造一个"可 shuffle + 可恢复 + 可 batch 化"的数据加载器

---

## 📚 本节目录

1. [DDP sampler + epoch shuffle 控制](#1-ddp-sampler--epoch-shuffle-控制)
2. [单卡随机索引 fallback](#2-单卡随机索引-fallback)
3. [断点续训 step 计算](#3-断点续训-step-计算)
4. [SkipBatchSampler = 数据源 + batch切分 + resume控制](#4-skipbatchsampler--数据源--batch切分--resume控制)
5. [总体流程图](#5-总体流程图)
6. [一句话终极总结](#6-一句话终极总结)

---

## 1. DDP sampler + epoch shuffle 控制

```python
train_sampler and train_sampler.set_epoch(epoch)
```

**含义**：如果是 DDP（`train_sampler` 不为空），告诉 sampler 当前是第几个 epoch

**作用**：让每个 epoch 的 shuffle 在所有 GPU 上"同步变化"

**原理**：DistributedSampler 在 shuffle 时用随机种子 `seed = base_seed + epoch`

---

## 2. 单卡随机索引 fallback

```python
setup_seed(42 + epoch)
indices = torch.randperm(len(train_ds)).tolist()
```

**含义**：
- 设置随机种子（保证可复现）
- 生成 shuffle 后的 index 序列

**作用**：单卡情况下模拟"shuffle dataset"

---

## 3. 断点续训 step 计算

```python
skip = start_step if (epoch == start_epoch and start_step > 0) else 0
```

**含义**：
- 只有在恢复训练的起始 epoch 才跳过 batch
- 如果是恢复 epoch → skip 前面 step
- 否则 → 不跳

**作用**：实现"从某个 global step 继续训练"

---

## 4. SkipBatchSampler = 数据源 + batch切分 + resume控制

### 构造方式

```python
batch_sampler = SkipBatchSampler(
    train_sampler or indices,  # 数据源：DDP切分 or 单卡shuffle列表
    args.batch_size,           # batch大小
    skip                       # 跳过前N个batch（resume用）
)
```

### 三合一机制

| 组件 | 作用 |
|------|------|
| `train_sampler or indices` | 数据从哪来：DDP分布式切分 / 单卡shuffle |
| `batch_size` | 把单个index聚成batch |
| `skip` | 跳过前N个batch（断点续训） |

### 源码解析

**跳过的是 batch，不是单条样本**

```python
class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler      # 数据源（DDP切分 or shuffle列表）
        self.batch_size = batch_size # 每个batch多少条
        self.skip_batches = skip_batches  # 跳几个batch

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:   # 遍历数据源
            batch.append(idx)
            if len(batch) == self.batch_size:  # 凑满一个batch
                if skipped < self.skip_batches:  # 还在跳过区间
                    skipped += 1
                    batch = []
                    continue
                yield batch         # 正常输出batch
                batch = []
        # 处理尾巴（不足batch_size的剩余部分）
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch
```

### 执行流程举例

```
数据: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
batch_size = 3
skip_batches = 2

凑batch过程:
[0], [1], [2] → 凑满 → 第1个batch → 跳过
[3], [4], [5] → 凑满 → 第2个batch → 跳过
[6], [7], [8] → 凑满 → 第3个batch → 输出
[9]           → 凑不满 → 但已跳过2个 → 输出

输出: [[6,7,8], [9]]
```

### DataLoader 最终封装

```python
loader = DataLoader(
    train_ds,
    batch_sampler=batch_sampler,  # SkipBatchSampler控制顺序
    num_workers=args.num_workers,
    pin_memory=True
)
```

**DataLoader 只做**：

| 功能 | 说明 |
|------|------|
| 按 batch_sampler 取 batch | 顺序由 SkipBatchSampler 控制 |
| 多进程加载 | `num_workers` |
| CPU→GPU 加速传输 | `pin_memory` |

**注意**：用了 `batch_sampler` 后，DataLoader 不再负责 shuffle 或 batch_size

---

## 5. 总体流程图

```
        DDP?
          ↓
 train_sampler.set_epoch(epoch)
          ↓
   ┌────────────────────┐
   │                    │
   │  DDP          Single GPU
   │  ↓                ↓
   │ DistributedSampler randperm(indices)
   └──────────┬─────────┘
              ↓
        train_sampler or indices
              ↓
     SkipBatchSampler(batch_size, skip)
              ↓
          batch_sampler
              ↓
        DataLoader(loader)
              ↓
            model
```

---

## 6. 一句话终极总结

**这段代码通过 DDP sampler + 随机索引 + skip 控制，构造了一个支持分布式 shuffle 和断点续训的 batch-level 数据流，最终由 DataLoader 负责并行加载。**
