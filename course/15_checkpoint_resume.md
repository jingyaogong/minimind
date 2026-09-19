# 第 15 课：Checkpoint 与断点续训

这一课只解决一个问题：训练中断后，MiniMind 如何从上次的位置继续训练，而不是从头再来。

## 目录

- [0. 本节主线](#l15-mainline)
- [1. 原理讲解](#l15-principle)
- [2. 源码阅读顺序图](#l15-reading-order)
- [3. MiniMind 源码走读](#l15-source-walkthrough)
- [4. 本节必须会写 / 暂时不要求](#l15-must-write)
- [5. 手写模块](#l15-handwrite)
- [6. 对齐测试](#l15-alignment-test)
- [7. 阶段组装](#l15-stage-assembly)
- [8. 本节检查](#l15-check)
- [9. 下一课](#l15-next)

<a id="l15-mainline"></a>
## 0. 本节主线

MiniMind 的断点续训是：

```text
训练到某个 step
-> 保存普通权重文件到 out/
-> 保存 resume checkpoint 到 checkpoints/
-> checkpoint 里包含 model / optimizer / scaler / epoch / step
-> 下次启动时先查 resume checkpoint
-> 恢复 model / optimizer / scaler
-> 恢复 start_epoch / start_step
-> 用 SkipBatchSampler 跳过当前 epoch 已训练过的 batch
-> 从下一个 batch 继续训练
```

一句话：

```text
普通权重只够“推理或继续初始化”；resume checkpoint 才够“无缝续训”。
```

<a id="l15-principle"></a>
## 1. 原理讲解

### 1.1 为什么只保存 model 不够

如果只保存模型参数：

```text
model.state_dict()
```

你只能恢复“模型当前会怎么预测”。但训练状态不止模型参数。

一次训练能否无缝继续，至少依赖：

| 状态 | 形状/类型 | 为什么要保存 |
|---|---|---|
| model | 参数名到 tensor 的 dict | 恢复模型参数 |
| optimizer | dict | 恢复 AdamW 的动量、二阶矩等内部状态 |
| scaler | dict | 恢复 float16 AMP 的动态 loss scale |
| epoch | int | 知道从第几个 epoch 继续 |
| step | int | 知道当前 epoch 已训练到第几个 batch |
| world_size | int | DDP world size 变化时修正 step |
| wandb_id | str 或 None | 日志平台续接同一个 run |

对 AdamW 来说，optimizer 里不只是 learning rate，还有每个参数的动量估计。粗略写成：

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1)g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2)g_t^2
\end{aligned}
$$

如果恢复训练时只加载 model，不加载 optimizer，那么 $m_t$ 和 $v_t$ 会丢失。训练还能跑，但不是从完全相同的训练状态继续。

### 1.2 MiniMind 保存两类文件

MiniMind 训练时会保存两类文件。

第一类是普通权重文件：

```text
../out/pretrain_768.pth
../out/full_sft_768.pth
```

它主要用于：

```text
推理；
作为下一阶段初始化权重；
手动加载 model state_dict。
```

第二类是 resume checkpoint：

```text
../checkpoints/pretrain_768_resume.pth
../checkpoints/full_sft_768_resume.pth
```

它主要用于：

```text
中断后继续训练；
恢复 optimizer / scaler / epoch / step；
跳过已经训练过的 batch。
```

路径命名规则可以写成：

$$
\begin{aligned}
\text{normal\_path}
&= \text{save\_dir}/(\text{weight}\_\text{hidden\_size}\_\text{moe?}.pth) \\
\text{resume\_path}
&= \text{save\_dir}/(\text{weight}\_\text{hidden\_size}\_\text{moe?}\_\text{resume}.pth)
\end{aligned}
$$

其中 `use_moe=True` 时，中间会多一个 `_moe`。

### 1.3 保存 checkpoint 时到底保存什么

MiniMind 的 resume checkpoint 是一个 dict，可以理解成：

```text
resume_data = {
    "model": model_state_dict,
    "optimizer": optimizer_state_dict,
    "epoch": epoch,
    "step": step,
    "world_size": world_size,
    "wandb_id": wandb_id,
    "scaler": scaler_state_dict,
}
```

变量含义：

| key | 类型 | 含义 |
|---|---|---|
| `model` | dict[str, Tensor] | 模型参数 |
| `optimizer` | dict | 优化器状态 |
| `epoch` | int | 保存时所在 epoch |
| `step` | int | 保存时当前 epoch 的 batch step |
| `world_size` | int | 保存时 DDP 进程数 |
| `wandb_id` | str 或 None | 日志 run id |
| `scaler` | dict | AMP GradScaler 状态 |

这里的 `step` 不是全局 token 数，也不是 optimizer update 次数，而是当前 epoch 里的 dataloader step。它后面会用于跳过已经训练过的 batch。

### 1.4 为什么保存前要 unwrap model

训练时模型可能被包过：

```text
DistributedDataParallel(model)
torch.compile(model)
```

这时真正的模型可能藏在包装对象内部。

MiniMind 保存前会做：

```text
raw_model = model.module if isinstance(model, DistributedDataParallel) else model
raw_model = getattr(raw_model, '_orig_mod', raw_model)
state_dict = raw_model.state_dict()
```

意思是：

```text
如果是 DDP，取 model.module；
如果是 torch.compile，取 _orig_mod；
最后再取 state_dict。
```

教学版第一阶段可以先不支持 DDP 和 compile，但要知道原源码为什么这么写。

### 1.5 断点恢复时为什么要跳过 batch

假设第 0 个 epoch 训练到 `step=3` 后保存了 checkpoint。

这个 epoch 已经训练过：

```text
batch 1
batch 2
batch 3
```

恢复时如果直接重新从 dataloader 开头跑，就会重复训练这 3 个 batch。

MiniMind 的做法是：

```text
start_epoch = ckp_data["epoch"]
start_step = ckp_data["step"]
skip = start_step if epoch == start_epoch else 0
SkipBatchSampler(..., skip_batches=skip)
```

这样当前 epoch 会跳过前 `start_step` 个 batch，从下一个 batch 继续。

### 1.6 world_size 改变时为什么要修正 step

DDP 里每个进程看到的是全局数据的一部分。如果保存时 GPU 数和恢复时 GPU 数不同，同一个 `step` 对应的全局样本进度会变。

MiniMind 用一个简单修正：

$$
s_{\text{new}}
= \left\lfloor \frac{s_{\text{saved}} \cdot W_{\text{saved}}}{W_{\text{current}}} \right\rfloor
$$

其中：

| 符号 | 类型 | 含义 |
|---|---|---|
| $s_{\text{saved}}$ | int | checkpoint 里保存的 step |
| $W_{\text{saved}}$ | int | 保存时的 world size |
| $W_{\text{current}}$ | int | 当前恢复训练的 world size |
| $s_{\text{new}}$ | int | 修正后的 start_step |

这不是完美的数据级恢复，但能让恢复后的进度大致对齐。

教学版第一阶段可以不实现 world_size 修正；先实现单机单进程的恢复。

### 1.7 保存时为什么用 tmp 文件再 replace

MiniMind 的 `lm_checkpoint` 保存时会先写临时文件：

```text
xxx.pth.tmp
```

写完后再：

```text
os.replace(tmp_path, final_path)
```

这样做是为了避免保存过程中程序中断，留下一个半写入的 checkpoint。`os.replace` 在同一个文件系统里通常是原子替换：要么还是旧文件，要么变成新文件。

教学版可以复用这个习惯。

<a id="l15-reading-order"></a>
## 2. 源码阅读顺序图

这节源码按这个顺序读：

```text
train_pretrain.py 保存入口
-> trainer_utils.lm_checkpoint 保存模式
-> trainer_utils.lm_checkpoint 加载模式
-> train_pretrain.py 恢复 model / optimizer / scaler / epoch / step
-> train_pretrain.py 构造 skip
-> trainer_utils.SkipBatchSampler
```

对应文件：

```text
trainer/train_pretrain.py
trainer/trainer_utils.py
```

先看保存入口，是为了知道 checkpoint 在训练循环的哪个时机触发。  
再看 `lm_checkpoint`，是为了知道普通权重和 resume checkpoint 分别保存什么。  
最后看 `SkipBatchSampler`，是为了知道恢复后怎么避免重复训练已经处理过的 batch。

<a id="l15-source-walkthrough"></a>
## 3. MiniMind 源码走读

### 第 1 步：训练循环触发保存

File: `trainer/train_pretrain.py:61-70`

Read this to understand: MiniMind 会在固定间隔或 epoch 结束时保存权重和 resume checkpoint。

Code/config/template excerpt:

```python
if (step % args.save_interval == 0 or step == iters) and is_main_process():
    model.eval()
    moe_suffix = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model = getattr(raw_model, '_orig_mod', raw_model)
    state_dict = raw_model.state_dict()
    torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
    lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
    model.train()
```

This code shows:

- 只有主进程保存，避免 DDP 多进程同时写同一个文件。
- 普通权重保存到 `args.save_dir`，默认是 `../out`。
- `lm_checkpoint` 额外保存 resume checkpoint，默认到 `../checkpoints`。
- 保存前切到 eval，保存后切回 train。

### 第 2 步：`lm_checkpoint` 构造路径

File: `trainer/trainer_utils.py:63-67`

Read this to understand: checkpoint 文件名如何由阶段名、hidden size 和 MoE 后缀组成。

Code/config/template excerpt:

```python
def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'
```

This code shows:

- `weight` 是阶段名，比如 `pretrain`、`full_sft`、`dpo`。
- `hidden_size` 是模型规模标识。
- `use_moe=True` 时文件名带 `_moe`。
- `resume_path` 比普通路径多 `_resume`。

### 第 3 步：保存模式写两个文件

File: `trainer/trainer_utils.py:69-106`

Read this to understand: `model is not None` 时，`lm_checkpoint` 进入保存模式。

Code/config/template excerpt:

```python
if model is not None:
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model = getattr(raw_model, '_orig_mod', raw_model)
    state_dict = raw_model.state_dict()
    state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
    ckp_tmp = ckp_path + '.tmp'
    torch.save(state_dict, ckp_tmp)
    os.replace(ckp_tmp, ckp_path)
    ...
    resume_data = {
        'model': state_dict,
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'world_size': dist.get_world_size() if dist.is_initialized() else 1,
        'wandb_id': wandb_id
    }
```

This code shows:

- 普通权重文件保存的是半精度 CPU state dict。
- resume checkpoint 也包含同一份 `model`。
- resume checkpoint 还包含 optimizer、epoch、step、world_size、wandb_id。
- `scaler` 是通过 `**kwargs` 传进来的额外状态。

### 第 4 步：额外状态通过 `kwargs` 存入 resume dict

File: `trainer/trainer_utils.py:93-104`

Read this to understand: 为什么 `scaler=scaler` 能被保存进 checkpoint。

Code/config/template excerpt:

```python
for key, value in kwargs.items():
    if value is not None:
        if hasattr(value, 'state_dict'):
            raw_value = value.module if isinstance(value, DistributedDataParallel) else value
            raw_value = getattr(raw_value, '_orig_mod', raw_value)
            resume_data[key] = raw_value.state_dict()
        else:
            resume_data[key] = value

resume_tmp = resume_path + '.tmp'
torch.save(resume_data, resume_tmp)
os.replace(resume_tmp, resume_path)
```

This code shows:

- `scaler` 有 `state_dict()`，所以会保存为 `resume_data["scaler"]`。
- PPO/GRPO 里 scheduler、critic model 等也可以通过同样方式保存。
- resume checkpoint 也先写 tmp，再原子替换。

### 第 5 步：加载模式只查 resume checkpoint

File: `trainer/trainer_utils.py:107-116`

Read this to understand: `model is None` 时，`lm_checkpoint` 进入加载模式。

Code/config/template excerpt:

```python
else:
    if os.path.exists(resume_path):
        ckp_data = torch.load(resume_path, map_location='cpu')
        saved_ws = ckp_data.get('world_size', 1)
        current_ws = dist.get_world_size() if dist.is_initialized() else 1
        if saved_ws != current_ws:
            ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
            Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
        return ckp_data
    return None
```

This code shows:

- 自动续训只读 `_resume.pth`。
- 找不到 resume checkpoint 时返回 `None`。
- 如果 world size 变化，会修正 `step`。
- 加载位置是 CPU，后面再由 `load_state_dict` 放回模型/优化器。

### 第 6 步：训练脚本恢复状态

File: `trainer/train_pretrain.py:140-147`

Read this to understand: checkpoint 读出来以后，具体恢复哪些对象。

Code/config/template excerpt:

```python
start_epoch, start_step = 0, 0
if ckp_data:
    model.load_state_dict(ckp_data['model'])
    optimizer.load_state_dict(ckp_data['optimizer'])
    scaler.load_state_dict(ckp_data['scaler'])
    start_epoch = ckp_data['epoch']
    start_step = ckp_data.get('step', 0)
```

This code shows:

- 没有 checkpoint 时从 `epoch=0, step=0` 开始。
- 有 checkpoint 时恢复 model、optimizer、scaler。
- `start_epoch` 决定从哪个 epoch 开始循环。
- `start_step` 决定当前 epoch 跳过多少 batch。

### 第 7 步：恢复时跳过已训练 batch

File: `trainer/train_pretrain.py:156-167`

Read this to understand: `start_step` 如何进入 dataloader。

Code/config/template excerpt:

```python
for epoch in range(start_epoch, args.epochs):
    train_sampler and train_sampler.set_epoch(epoch)
    setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
    skip = start_step if (epoch == start_epoch and start_step > 0) else 0
    batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
    loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
    if skip > 0:
        train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
    else:
        train_epoch(epoch, loader, len(loader), 0, wandb)
```

This code shows:

- 只有恢复后的第一个 epoch 需要 skip。
- 后续 epoch 从头正常训练。
- `len(loader) + skip` 传给 `train_epoch`，是为了让日志里的总 step 保持原 epoch 视角。
- `start_step` 传给 `train_epoch`，让 enumerate 从 `start_step + 1` 开始。

### 第 8 步：`SkipBatchSampler` 如何跳过 batch

File: `trainer/trainer_utils.py:134-157`

Read this to understand: 跳过的是 batch，不是单条样本。

Code/config/template excerpt:

```python
class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch
```

This code shows:

- `sampler` 提供样本索引。
- 每攒够 `batch_size` 个索引，形成一个 batch。
- 前 `skip_batches` 个完整 batch 会被丢弃。
- 不足一个 batch 的尾巴也会 yield，前提是已经跳过够了。

<a id="l15-must-write"></a>
## 4. 本节必须会写 / 暂时不要求

必须会写：

1. checkpoint 路径命名：

$$
\begin{aligned}
\text{suffix}
&=
\begin{cases}
\text{"\_moe"}, & \text{use\_moe=True} \\
\text{""}, & \text{use\_moe=False}
\end{cases} \\
\text{ckp\_path}
&= \text{save\_dir}/(\text{weight}\_\text{hidden\_size}\text{suffix}.pth) \\
\text{resume\_path}
&= \text{save\_dir}/(\text{weight}\_\text{hidden\_size}\text{suffix}\_\text{resume}.pth)
\end{aligned}
$$

2. 保存普通权重：

```text
unwrap model
-> state_dict
-> half
-> cpu
-> torch.save
```

3. 保存 resume checkpoint：

```text
model state_dict
optimizer state_dict
scaler state_dict
epoch
step
extra_state
```

4. 加载 resume checkpoint：

```text
如果 resume_path 存在 -> torch.load(..., map_location="cpu")
否则 -> None
```

5. `SkipBatchSampler`：

```text
按 batch_size 聚合索引
跳过前 skip_batches 个 batch
yield 后面的 batch
```

暂时不要求：

```text
1. DDP world_size 修正
2. wandb_id 恢复
3. torch.compile unwrap
4. 多模型 checkpoint，例如 PPO 的 actor/critic
5. 跨机器存储和远程对象存储
```

<a id="l15-handwrite"></a>
## 5. 手写模块

本节你要补的是：

```text
course/impl/core/train_loop.py
```

### 5.1 补 `checkpoint_paths`

接口：

```python
def checkpoint_paths(
    save_dir: str | Path,
    weight: str,
    hidden_size: int,
    use_moe: bool = False,
) -> tuple[Path, Path]:
    ...
```

对齐源码：

```text
trainer/trainer_utils.py:63-67
```

你要实现的行为：

```text
checkpoint_paths("checkpoints", "pretrain", 768, False)
-> checkpoints/pretrain_768.pth
-> checkpoints/pretrain_768_resume.pth

checkpoint_paths("checkpoints", "pretrain", 768, True)
-> checkpoints/pretrain_768_moe.pth
-> checkpoints/pretrain_768_moe_resume.pth
```

### 5.2 补 `CourseSkipBatchSampler`

接口：

```python
class CourseSkipBatchSampler(Sampler[list[int]]):
    def __iter__(self):
        ...

    def __len__(self) -> int:
        ...
```

对齐源码：

```text
trainer/trainer_utils.py:134-157
```

你要实现的行为：

```text
sampler = range(10)
batch_size = 3
skip_batches = 2

原 batch:
[0, 1, 2]
[3, 4, 5]
[6, 7, 8]
[9]

输出:
[6, 7, 8]
[9]
```

### 5.3 补 `save_course_checkpoint`

接口已经在骨架里：

```python
def save_course_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    save_dir: str | Path,
    weight: str,
    hidden_size: int,
    use_moe: bool,
    epoch: int,
    step: int,
    scaler: torch.cuda.amp.GradScaler | None = None,
    extra_state: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    ...
```

对齐源码：

```text
trainer/trainer_utils.py:69-106
```

你要实现的行为：

- 创建 `save_dir`。
- 调用 `checkpoint_paths` 得到 `ckp_path` 和 `resume_path`。
- 保存普通权重到 `ckp_path`。
- 保存 resume dict 到 `resume_path`。
- resume dict 至少包含 `model`、`optimizer`、`epoch`、`step`。
- 如果传了 `scaler`，保存 `scaler.state_dict()`。
- 如果传了 `extra_state`，把里面的 key 合并进 resume dict。
- 返回 `(ckp_path, resume_path)`。

第一版可以先不写 tmp + replace；写完基础功能后再加。

### 5.4 补 `load_course_checkpoint`

接口：

```python
def load_course_checkpoint(
    save_dir: str | Path,
    weight: str,
    hidden_size: int,
    use_moe: bool = False,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any] | None:
    ...
```

对齐源码：

```text
trainer/trainer_utils.py:107-116
```

你要实现的行为：

- 调用 `checkpoint_paths` 得到 `resume_path`。
- 如果文件不存在，返回 `None`。
- 如果文件存在，返回 `torch.load(resume_path, map_location=map_location)`。

<a id="l15-alignment-test"></a>
## 6. 对齐测试

本节新增对齐测试：

```text
course/impl/tests/test_checkpoint_resume.py
```

运行命令：

```bash
cd /home/sun/minimind
python course/impl/tests/test_checkpoint_resume.py
```

现在还没有实现时，这个测试会因为 `NotImplementedError` 失败。等你补完后，它应该打印类似：

```text
ckp_path=checkpoints/pretrain_768.pth
resume_path=checkpoints/pretrain_768_resume.pth
skipped_batches=[[6, 7, 8], [9]]
saved_ckp_exists=True
saved_resume_exists=True
checkpoint_resume=passed
```

如果这个测试通过，说明你已经掌握了：

```text
文件名规则
resume dict 的核心字段
跳过已训练 batch
保存后再加载 checkpoint
```

<a id="l15-stage-assembly"></a>
## 7. 阶段组装

本节完成后，Pretrain 阶段会多出 checkpoint/resume 能力：

```text
course/impl/core/train_loop.py::checkpoint_paths
course/impl/core/train_loop.py::save_course_checkpoint
course/impl/core/train_loop.py::load_course_checkpoint
course/impl/core/train_loop.py::CourseSkipBatchSampler
```

之后 `course/impl/train_pretrain_impl.py` 可以按这个流程组装：

```text
启动脚本
-> load_course_checkpoint(...)
-> 如果有 checkpoint，恢复 model / optimizer / scaler / epoch / step
-> 构造 CourseSkipBatchSampler
-> train_one_epoch
-> save_course_checkpoint
```

当前 Pretrain 阶段还缺：

```text
1. 教学版 CausalLM 完整 forward/loss
2. 教学版 pretrain dataset 或 dataset 适配
3. tiny pretrain 脚本组装
```

第 15 课先补齐训练状态的保存和恢复。

<a id="l15-check"></a>
## 8. 本节检查

1. 普通权重文件和 resume checkpoint 的用途有什么区别？
2. 为什么断点续训不能只保存 `model.state_dict()`？
3. AdamW 的 optimizer state 里为什么有必要保存动量状态？
4. `epoch` 和 `step` 分别表示什么？
5. `SkipBatchSampler` 跳过的是样本还是 batch？
6. 为什么恢复后的第一个 epoch 要 skip，后续 epoch 不需要 skip？
7. `scaler.state_dict()` 是为了解决什么恢复问题？
8. 为什么保存 checkpoint 时建议先写 `.tmp` 再 `os.replace`？

<a id="l15-next"></a>
## 9. 下一课

第 16 课进入 Full SFT 训练脚本总览。

下一课要解决：

```text
SFTDataset 如何产出 input_ids / labels；
train_full_sft.py 和 train_pretrain.py 哪些地方相同；
从 pretrain 权重继续训练时 from_weight 做了什么；
SFT 的 loss 仍然如何走 causal LM forward；
教学版 SFT 脚本应该如何复用 train_loop 和 checkpoint。
```
