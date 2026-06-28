# 第 14 课：学习率、梯度累积与混合精度

这一课只解决一个问题：MiniMind 的训练循环如何把多个 micro-batch 的 loss 变成一次稳定的参数更新。

## 目录

- [0. 本节主线](#l14-mainline)
- [1. 原理讲解](#l14-principle)
- [2. 源码阅读顺序图](#l14-reading-order)
- [3. MiniMind 源码走读](#l14-source-walkthrough)
- [4. 本节必须会写 / 暂时不要求](#l14-must-write)
- [5. 手写模块](#l14-handwrite)
- [6. 对齐测试](#l14-alignment-test)
- [7. 阶段组装](#l14-stage-assembly)
- [8. 本节检查](#l14-check)
- [9. 下一课](#l14-next)

<a id="l14-mainline"></a>
## 0. 本节主线

MiniMind 的一个训练 micro-step 是：

```text
取一个 batch
-> 把 input_ids / labels 放到 device
-> 按当前全局 step 设置 learning rate
-> autocast forward 得到 lm loss 和 aux loss
-> loss 除以 accumulation_steps
-> backward，把梯度累积到参数上
-> 每 accumulation_steps 个 micro-step 做一次 optimizer step
-> step 前 unscale + clip grad
-> step 后 scaler.update + zero_grad
-> epoch 结束时处理不足 accumulation_steps 的尾巴
```

这一课的核心不是“调用 `loss.backward()`”，而是看懂这四件事如何配合：

```text
learning rate schedule
gradient accumulation
mixed precision / GradScaler
gradient clipping + optimizer.step
```

<a id="l14-principle"></a>
## 1. 原理讲解

### 1.1 一个训练 step 到底更新了什么

模型训练的目标是让参数 $\theta$ 沿着 loss 下降的方向移动。

如果一个 batch 的训练损失是 $L(\theta)$，反向传播计算的是：

$$
g = \nabla_{\theta} L(\theta)
$$

优化器做的事可以粗略理解成：

$$
\theta_{\text{new}} = \theta_{\text{old}} - \eta \cdot \mathrm{Update}(g)
$$

这里：

| 符号 | 形状 | 含义 |
|---|---|---|
| $\theta$ | 每个参数自己的形状 | 模型参数 |
| $L$ | 标量 | 当前 batch 的训练损失 |
| $g$ | 和参数同形状 | 反向传播得到的梯度 |
| $\eta$ | 标量 | learning rate |

对 MiniMind 来说，训练 loss 通常是：

$$
L_{\text{train}} = L_{\text{lm}} + L_{\text{aux}}
$$

其中 $L_{\text{lm}}$ 是 next-token loss，$L_{\text{aux}}$ 是 MoE router 的辅助损失。非 MoE 模型里，`aux_loss` 通常就是 0。

### 1.2 学习率不是固定不变的

Learning rate 控制每次参数更新走多大一步。太大容易震荡，太小训练太慢。

MiniMind 的 `get_lr` 用的是余弦衰减，但不是衰减到 0，而是从初始学习率衰减到 10% 初始学习率。

公式是：

$$
\eta_s
= \eta_0 \left(0.1 + 0.45\left(1 + \cos\left(\frac{\pi s}{T}\right)\right)\right)
$$

其中：

| 符号 | 形状 | 含义 |
|---|---|---|
| $s$ | 标量 | 当前全局 micro-step |
| $T$ | 标量 | 总 micro-step 数 |
| $\eta_0$ | 标量 | 命令行传入的初始学习率 |
| $\eta_s$ | 标量 | 当前 step 实际使用的学习率 |

代入两个边界：

$$
\begin{aligned}
\eta_0^{\text{actual}}
&= \eta_0 \left(0.1 + 0.45(1 + 1)\right)
= \eta_0 \\
\eta_T
&= \eta_0 \left(0.1 + 0.45(1 - 1)\right)
= 0.1\eta_0
\end{aligned}
$$

也就是说，它从 `learning_rate` 平滑降到 `0.1 * learning_rate`。

### 1.3 梯度累积解决什么问题

显存放不下很大的 batch 时，可以用多个小 batch 模拟一个大 batch。

假设：

| 符号 | 形状 | 含义 |
|---|---|---|
| $B$ | 标量 | 单个 micro-batch 的 batch size |
| $K$ | 标量 | `accumulation_steps` |
| $B_{\text{eff}}$ | 标量 | 等效 batch size |

单卡非 DDP 时：

$$
B_{\text{eff}} = B \times K
$$

如果是 DDP，多卡训练还要乘以进程数 $W$：

$$
B_{\text{eff}} = B \times K \times W
$$

为什么要把 loss 除以 $K$？

如果第 $i$ 个 micro-batch 的原始 loss 是 $L_i$，MiniMind 用的是：

$$
\tilde{L}_i = \frac{L_i}{K}
$$

连续反向传播 $K$ 次后，参数上累积的梯度是：

$$
\begin{aligned}
g
&= \sum_{i=1}^{K}\nabla_{\theta}\tilde{L}_i \\
&= \sum_{i=1}^{K}\nabla_{\theta}\left(\frac{L_i}{K}\right) \\
&= \frac{1}{K}\sum_{i=1}^{K}\nabla_{\theta}L_i
\end{aligned}
$$

这相当于取了 $K$ 个 micro-batch 梯度的平均值，而不是把它们直接相加。否则 `accumulation_steps` 越大，梯度尺度越大，学习率也会被隐式放大。

### 1.4 混合精度做了什么

混合精度的目标是减少显存和加速计算。

MiniMind 的 forward 放在 `autocast_ctx` 里：

```text
with autocast_ctx:
    res = model(...)
    loss = ...
```

这表示某些矩阵乘法可以用 `float16` 或 `bfloat16` 执行，而不是全部用 `float32`。

但 `float16` 动态范围小，梯度可能 underflow。`GradScaler` 的思想是先把 loss 放大：

$$
L_{\text{scaled}} = c \cdot L
$$

反向传播得到放大后的梯度：

$$
g_{\text{scaled}} = c \cdot g
$$

真正更新前再除回来：

$$
g = \frac{g_{\text{scaled}}}{c}
$$

MiniMind 只在 `dtype == float16` 时启用 `GradScaler`。如果是 `bfloat16`，代码仍然调用 `scaler.scale(...)`，但 scaler 是 disabled 状态，本质上是 no-op。

### 1.5 为什么要先 unscale 再 clip grad

梯度裁剪限制梯度范数，避免一次异常梯度把参数推太远。

如果最大范数是 $C$，裁剪可以理解成：

$$
g_{\text{clipped}}
= g \cdot \min\left(1, \frac{C}{\lVert g \rVert_2}\right)
$$

如果用了 `GradScaler`，参数上的梯度可能还是放大后的 $g_{\text{scaled}}$。这时直接 clip，裁剪的就不是原始梯度。

所以 MiniMind 的顺序是：

```text
scaler.unscale_(optimizer)
-> clip_grad_norm_
-> scaler.step(optimizer)
-> scaler.update()
```

先 unscale，再 clip，最后 step。

### 1.6 变量形状

训练循环里最重要的张量形状是：

| 变量 | 形状 | 含义 |
|---|---|---|
| `input_ids` | `[B, S]` | token id |
| `labels` | `[B, S]` | 训练目标，忽略位置是 `-100` |
| `logits` | `[B, S, V]` | 每个位置对词表的预测 |
| `res.loss` | 标量 | causal LM loss |
| `res.aux_loss` | 标量 | MoE aux loss，非 MoE 时通常是 0 |
| `loss` | 标量 | `(res.loss + res.aux_loss) / accumulation_steps` |
| `param.grad` | 和参数同形状 | 当前累计梯度 |

注意：`loss` 是标量，不是 `[B, S]`。前面第 10 课已经讲过，cross entropy 会把有效 token 的 loss 聚合成一个标量。

<a id="l14-reading-order"></a>
## 2. 源码阅读顺序图

这节源码按这个顺序读：

```text
trainer_utils.get_lr
-> train_pretrain.py 参数
-> autocast / GradScaler 初始化
-> train_epoch 前半段：lr + forward + loss / accumulation_steps
-> train_epoch 中段：backward + unscale + clip + step + zero_grad
-> train_epoch 末尾：不足 accumulation_steps 的尾批处理
```

对应文件：

```text
trainer/trainer_utils.py
trainer/train_pretrain.py
```

先看 `get_lr`，是为了知道每个 step 用什么学习率。  
再看 `train_epoch`，是为了知道 loss、梯度、optimizer step 如何串起来。  
最后看初始化部分，是为了知道 `autocast_ctx` 和 `GradScaler` 是怎么决定的。

<a id="l14-source-walkthrough"></a>
## 3. MiniMind 源码走读

### 第 1 步：学习率公式

File: `trainer/trainer_utils.py:40-41`

Read this to understand: MiniMind 的 learning rate schedule 不是 PyTorch scheduler，而是每个 step 手动计算。

Code/config/template excerpt:

```python
def get_lr(current_step, total_steps, lr):
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))
```

This code shows:

- `current_step` 是当前训练进度。
- `total_steps` 是总训练步数。
- 返回值直接写入 optimizer 的 param group。
- 它是 cosine decay，终点是 `0.1 * lr`，不是 0。

### 第 2 步：训练参数决定 batch、lr、累积步数和裁剪阈值

File: `trainer/train_pretrain.py:87-95`

Read this to understand: 训练机制的关键超参都来自命令行参数。

Code/config/template excerpt:

```python
parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
```

This code shows:

- `batch_size` 是单个 micro-batch 的大小。
- `accumulation_steps` 决定多少个 micro-batch 做一次 optimizer update。
- `learning_rate` 是 schedule 的初始值。
- `dtype` 决定 autocast 用 `float16` 还是 `bfloat16`。
- `grad_clip` 决定梯度裁剪上限。

### 第 3 步：初始化 autocast 和 GradScaler

File: `trainer/train_pretrain.py:119-138`

Read this to understand: 混合精度由 `autocast_ctx` 和 `GradScaler` 两部分组成。

Code/config/template excerpt:

```python
device_type = "cuda" if "cuda" in args.device else "cpu"
dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
...
scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
```

This code shows:

- CPU 上不用 autocast，直接是 `nullcontext()`。
- CUDA 上会进入 `torch.cuda.amp.autocast(dtype=dtype)`。
- `bfloat16` 只用 autocast，不启用 GradScaler。
- `float16` 同时使用 autocast 和 GradScaler。
- optimizer 是 AdamW。

### 第 4 步：每个 micro-step 设置学习率

File: `trainer/train_pretrain.py:24-34`

Read this to understand: 学习率不是 epoch 级别更新，而是每个 micro-step 更新。

Code/config/template excerpt:

```python
for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
    input_ids = input_ids.to(args.device)
    labels = labels.to(args.device)
    last_step = step
    lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```

This code shows:

- `step` 从 `start_step + 1` 开始，支持断点续训。
- `epoch * iters + step` 是跨 epoch 的全局 step。
- 每个 param group 都会被写入当前 lr。
- `input_ids` 和 `labels` 先搬到训练 device。

### 第 5 步：forward、loss 和梯度累积缩放

File: `trainer/train_pretrain.py:35-40`

Read this to understand: MiniMind 是先求完整 loss，再除以 `accumulation_steps`。

Code/config/template excerpt:

```python
with autocast_ctx:
    res = model(input_ids, labels=labels)
    loss = res.loss + res.aux_loss
    loss = loss / args.accumulation_steps

scaler.scale(loss).backward()
```

This code shows:

- `res.loss` 是语言模型主 loss。
- `res.aux_loss` 是 MoE 辅助 loss。
- `loss / accumulation_steps` 是为了让累积梯度保持平均尺度。
- `scaler.scale(loss).backward()` 兼容 float16 AMP；如果 scaler disabled，就等价于普通 backward。

### 第 6 步：到达累积步数后才更新参数

File: `trainer/train_pretrain.py:42-50`

Read this to understand: 不是每个 micro-batch 都调用 optimizer step。

Code/config/template excerpt:

```python
if step % args.accumulation_steps == 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

    scaler.step(optimizer)
    scaler.update()

    optimizer.zero_grad(set_to_none=True)
```

This code shows:

- 只有 `step % accumulation_steps == 0` 时才更新参数。
- `unscale_` 必须在 `clip_grad_norm_` 前面。
- `scaler.step(optimizer)` 内部会在 scaler enabled 时检查溢出。
- `optimizer.zero_grad(set_to_none=True)` 清掉已使用的梯度，为下一组累积做准备。

### 第 7 步：日志里的 loss 要乘回去

File: `trainer/train_pretrain.py:51-59`

Read this to understand: 训练内部用的是缩放后的 loss，但日志要展示原始尺度。

Code/config/template excerpt:

```python
current_loss = loss.item() * args.accumulation_steps
current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
current_logits_loss = current_loss - current_aux_loss
current_lr = optimizer.param_groups[-1]['lr']
```

This code shows:

- `loss` 已经除过 `accumulation_steps`。
- 打日志时乘回 `accumulation_steps`，得到原始训练 loss 尺度。
- `current_logits_loss` 是总 loss 减掉 aux loss。
- `current_lr` 从 optimizer 里读当前实际学习率。

### 第 8 步：epoch 结束时处理尾巴

File: `trainer/train_pretrain.py:75-80`

Read this to understand: 如果最后剩下不足 `accumulation_steps` 的梯度，也要做一次 optimizer step。

Code/config/template excerpt:

```python
if last_step > start_step and last_step % args.accumulation_steps != 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

This code shows:

- 如果 epoch 里真的跑过 batch，`last_step > start_step`。
- 如果最后一个 step 不是累积边界，说明还有没 step 的梯度。
- 这时需要 flush tail，否则最后几个 micro-batch 的梯度会被丢掉。

<a id="l14-must-write"></a>
## 4. 本节必须会写 / 暂时不要求

必须会写：

1. MiniMind 的 cosine lr：

$$
\eta_s
= \eta_0 \left(0.1 + 0.45\left(1 + \cos\left(\frac{\pi s}{T}\right)\right)\right)
$$

2. 梯度累积的 loss 缩放：

$$
\tilde{L} = \frac{L}{K}
$$

3. optimizer step 判断：

$$
\text{should\_step}(s, K) =
\begin{cases}
\text{True}, & s \bmod K = 0 \\
\text{False}, & s \bmod K \ne 0
\end{cases}
$$

4. 尾批 flush 判断：

$$
\text{flush\_tail} =
(s_{\text{last}} > s_{\text{start}})
\land
(s_{\text{last}} \bmod K \ne 0)
$$

5. AMP update 顺序：

```text
scale loss
-> backward
-> unscale grad
-> clip grad
-> optimizer step
-> scaler update
-> zero grad
```

暂时不要求：

```text
1. DDP 梯度同步细节
2. checkpoint / resume 的完整状态恢复
3. torch.compile 对训练循环的影响
4. wandb / swanlab 日志工程
5. PPO / GRPO 里的 scheduler 版本
```

这一课只要求你能手写 pretrain/SFT 这类监督训练最小循环。

<a id="l14-handwrite"></a>
## 5. 手写模块

本节你要补的是：

```text
course/impl/core/train_loop.py
```

### 5.1 补 `cosine_lr`

接口：

```python
def cosine_lr(current_step: int, total_steps: int, base_lr: float) -> float:
    ...
```

对齐源码：

```text
trainer/trainer_utils.py:40-41
```

你要实现的行为：

- 输入 `current_step`、`total_steps`、`base_lr`。
- 输出和 MiniMind `get_lr` 完全一致的 float。
- 不要调用原项目的 `get_lr`，要自己写公式。

### 5.2 补 `scale_loss_for_accumulation`

接口：

```python
def scale_loss_for_accumulation(loss: torch.Tensor, accumulation_steps: int) -> torch.Tensor:
    ...
```

对齐源码：

```text
trainer/train_pretrain.py:35-40
```

你要实现的行为：

$$
\tilde{L} = \frac{L}{K}
$$

其中 $K$ 就是 `accumulation_steps`。

### 5.3 补 `should_step_optimizer`

接口：

```python
def should_step_optimizer(step: int, accumulation_steps: int) -> bool:
    ...
```

对齐源码：

```text
trainer/train_pretrain.py:42
```

你要实现的行为：

```text
step=1, K=4 -> False
step=2, K=4 -> False
step=3, K=4 -> False
step=4, K=4 -> True
```

### 5.4 补 `should_flush_tail`

接口：

```python
def should_flush_tail(last_step: int, start_step: int, accumulation_steps: int) -> bool:
    ...
```

对齐源码：

```text
trainer/train_pretrain.py:75
```

你要实现的行为：

```text
last_step=10, start_step=0, K=4 -> True
last_step=8,  start_step=0, K=4 -> False
last_step=0,  start_step=0, K=4 -> False
```

### 5.5 补 `train_one_epoch`

接口已经在骨架里。第一版只需要对齐监督训练主循环，不要写 DDP、wandb、checkpoint。

对齐源码：

```text
trainer/train_pretrain.py:24-80
```

建议实现顺序：

```text
1. 遍历 loader，得到 input_ids / labels。
2. 搬到 device。
3. 用 cosine_lr 设置 optimizer lr。
4. 进入 autocast_ctx。
5. 调 model(input_ids, labels=labels)。
6. loss = outputs.loss + outputs.aux_loss。
7. loss = scale_loss_for_accumulation(loss, accumulation_steps)。
8. scaler.scale(loss).backward()。
9. 如果 should_step_optimizer 为 True：
   unscale -> clip -> step -> update -> zero_grad。
10. epoch 结束后，如果 should_flush_tail 为 True：
    再执行一次 unscale -> clip -> step -> update -> zero_grad。
```

注意：

- 如果 `scaler is None`，你可以创建 `torch.cuda.amp.GradScaler(enabled=False)`，让 CPU 路径也能跑。
- 如果 `autocast_ctx is None`，可以用 `contextlib.nullcontext()`。
- `outputs.aux_loss` 可能是 0 标量，不要特殊跳过。

<a id="l14-alignment-test"></a>
## 6. 对齐测试

本节新增对齐测试：

```text
course/impl/tests/test_train_loop_mechanics.py
```

运行命令：

```bash
cd /home/sun/minimind
python course/impl/tests/test_train_loop_mechanics.py
```

现在还没有实现时，这个测试会因为 `NotImplementedError` 失败。等你补完前四个小函数后，它应该打印类似：

```text
lr_step_0_diff=0.000000000000
lr_step_1_diff=0.000000000000
...
scaled_loss=2.000000
optimizer_step_micro_steps=[4, 8]
train_loop_mechanics=passed
```

这个测试先不检查完整 `train_one_epoch`，因为完整训练循环需要模型、数据和 optimizer 共同参与。先把机制函数对齐，再进入阶段组装更稳。

另外可以观察原项目 tiny 训练 step：

```bash
cd /home/sun/minimind
python course/labs/trace_pretrain_step.py --batch_size 2 --max_length 48 --hidden_size 64 --num_hidden_layers 2
```

这个实验会打印：

```text
input_ids.shape
labels.shape
logits.shape
loss
aux_loss
lr
grad_norm_before_clip
max_first_param_delta_after_step
```

<a id="l14-stage-assembly"></a>
## 7. 阶段组装

本节完成后，Pretrain 阶段会多出训练循环核心能力：

```text
course/impl/core/train_loop.py::cosine_lr
course/impl/core/train_loop.py::scale_loss_for_accumulation
course/impl/core/train_loop.py::should_step_optimizer
course/impl/core/train_loop.py::should_flush_tail
course/impl/core/train_loop.py::train_one_epoch
```

它们以后会接到：

```text
course/impl/train_pretrain_impl.py
```

阶段目标是：

```text
dataset
-> DataLoader
-> CourseMiniMindForCausalLM
-> train_one_epoch
-> save course_pretrain checkpoint
```

当前 Pretrain 阶段还缺：

```text
1. 教学版 CausalLM 完整 forward/loss
2. 教学版 pretrain dataset 或 dataset 适配
3. checkpoint 保存和加载
4. tiny pretrain 脚本组装
```

第 14 课只先把训练循环的机械部分写清楚。

<a id="l14-check"></a>
## 8. 本节检查

1. MiniMind 的 `get_lr` 起点和终点分别是多少？
2. 为什么梯度累积时要把 loss 除以 `accumulation_steps`？
3. 如果 `batch_size=4`，`accumulation_steps=8`，单卡训练的等效 batch size 是多少？
4. `autocast` 和 `GradScaler` 分别解决什么问题？
5. 为什么 `clip_grad_norm_` 前要先 `scaler.unscale_(optimizer)`？
6. `step % accumulation_steps == 0` 控制的是什么？
7. 为什么 epoch 结束时还要处理不足 `accumulation_steps` 的尾巴？
8. 写出 `input_ids`、`labels`、`logits`、`loss`、`param.grad` 的形状。

<a id="l14-next"></a>
## 9. 下一课

第 15 课讲 checkpoint 与断点续训。

下一课要解决：

```text
保存哪些状态；
为什么只保存 model 不够；
optimizer / scaler / epoch / step 如何恢复；
SkipBatchSampler 如何跳过已经训练过的 batch；
普通权重文件和 resume checkpoint 有什么区别。
```
