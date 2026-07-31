# 02-实战 / 01 - 预训练 Pretrain

> **TL;DR**：跑通 `trainer/train_pretrain.py`，让一个 64M 模型从随机权重学会"高质量词语接龙"，
> 单卡 4090 约 1.2 小时。这是 minimind 全链路的起点。
>
> **心理学技巧**：**即时反馈** —— 每 100 步打印一次 loss，每 1000 步存一次权重，让你立刻看见曲线在下降，
> 多巴胺驱动你跑完整轮训练。

## ✅ 你将能

- 说出预训练在做什么、与 SFT 的本质区别
- 跑通 `train_pretrain.py`，产出 `out/pretrain_768.pth`
- 解释 cosine schedule / AMP / 梯度累积 / DDP / 断点续训每个机制
- 用 `eval_llm.py --weight pretrain` 验证模型已学会"接龙"
- 看到 swanlab 实时训练曲线

---

## 一、预训练在做什么

一句话：**让模型从大量文本中学会"高质量词语接龙"**。

```
输入: "清晨的阳光透过窗帘"
目标: "清晨的阳光透过窗帘洒进房间"
          ↑ 模型要在每个位置预测下一个 token
```

minimind 的预训练数据是 `pretrain_t2t_mini.jsonl`（1.2GB，纯文本，无对话结构）。每条样本被切成：

```
[bos] + tokens + [eos] + [pad] [pad] [pad] ...
  ↑      ↑        ↑      ← 不在 pad 上算 loss（labels = -100）
```

代码见 `dataset/lm_dataset.py:37-55`（`PretrainDataset`）：
- 第 49 行：切 token，留 2 位给 bos/eos
- 第 50 行：`[bos] + tokens + [eos]`
- 第 54 行：`labels[input_ids == pad] = -100`

**预训练 vs SFT 的本质区别**：
| | Pretrain | SFT |
|---|---|---|
| 数据 | 纯文本 | 对话（user/assistant） |
| 算 loss 的位置 | 整段（除 pad） | 只在 assistant 回复部分 |
| 学到的是 | 语言知识、世界知识 | 对话格式、指令遵循 |

> 预训练后模型只会"接着写"，还不会"回答问题"。比如问"天空为什么是蓝色的"，它可能续写成"...天空为什么是蓝色的呢？这是很多人好奇的问题。"—— 不会停止也不会"回答"你。**这很正常，SFT 之后才像样**。

---

## 步骤 1：确认数据已下载

```bash
cd /path/to/minimind
ls -lh dataset/pretrain_t2t_mini.jsonl
```

✅ **验证**：文件存在，约 1.2GB。

如果没有，回 [00-起步/03-跑通现成模型](../00-起步/03-跑通现成模型.md) 步骤 7 下载。

## 步骤 2：单卡训练（最简命令）

```bash
cd /path/to/minimind/trainer
python train_pretrain.py
```

> ⚠️ **必须在 `trainer/` 目录下执行**，脚本里默认路径都是相对路径（`../dataset/`、`../out/`）。

默认参数（见 `trainer/train_pretrain.py:84-107`）：

| 参数 | 默认 | 含义 |
|---|---|---|
| `--epochs` | 2 | 训练轮数 |
| `--batch_size` | 32 | 单步 batch 大小 |
| `--accumulation_steps` | 8 | 梯度累积步数（有效 batch = 32×8 = 256） |
| `--learning_rate` | 5e-4 | 峰值学习率 |
| `--max_seq_len` | 340 | 单样本最大长度（中文 ≈ 510 字符） |
| `--hidden_size` | 768 | 模型宽度 |
| `--num_hidden_layers` | 8 | 层数 |
| `--dtype` | bfloat16 | 混合精度类型 |
| `--use_moe` | 0 | 是否用 MoE 架构 |
| `--data_path` | ../dataset/pretrain_t2t_mini.jsonl | 数据 |
| `--save_dir` | ../out | 权重保存目录 |
| `--save_weight` | pretrain | 权重名前缀 |

启动后看到类似输出：
```
Model Params: 64.31M
Trainable Params: 64.309M
Epoch:[1/2](100/5569), loss: 7.4521, logits_loss: 7.4521, aux_loss: 0.0000, lr: 0.00047..., epoch_time: 32.1min
```

✅ **验证**：每 100 步打印一行日志（`--log_interval 100`，见 `train_pretrain.py:95`），loss 从约 10 下降到约 2-3。

## 步骤 3：开 swanlab 看曲线（强烈推荐）

```bash
cd /path/to/minimind/trainer
python train_pretrain.py --use_wandb
```

> minimind 用 `swanlab` 作为 wandb 替代（见 `train_pretrain.py:127` `import swanlab as wandb`），国内免梯子。
> 没注册过的话先 `swanlab login`。

浏览器打开 swanlab 看板，能看到 4 条曲线：
- `loss`：总 loss = logits_loss + aux_loss
- `logits_loss`：纯语言建模 loss
- `aux_loss`：MoE 负载均衡 loss（dense 模型为 0）
- `learning_rate`：cosine 衰减曲线

✅ **验证**：曲线呈下降趋势，lr 呈 cosine 形状（从 5e-4 平滑降到 ~5e-5）。

## 步骤 4：训练完成后查看产物

```bash
ls -lh /path/to/minimind/out/
ls -lh /path/to/minimind/checkpoints/
```

应该看到：
```
out/pretrain_768.pth              ← 训练产出权重（half + cpu）
checkpoints/pretrain_768_resume.pth   ← 断点续训检查点
```

权重保存逻辑见 `train_pretrain.py:61-71`：每 `--save_interval`（默认 1000）步保存一次，**先写 `.tmp` 再 `os.replace` 原子替换**，避免训练中断时写出半截文件。

## 步骤 5：验证模型已学会"接龙"

```bash
cd /path/to/minimind
python eval_llm.py --load_from ./model --weight pretrain
```

> `--load_from ./model`：用 `model/` 下的代码 + `out/pretrain_768.pth` 推理
> `--weight pretrain`：指定权重名前缀（拼接成 `out/pretrain_768.pth`）

启动后选 `[1] 手动输入`，输入 `为什么天空是蓝色的`（不带问号、不带问句语气，更像续写）。

✅ **验证**：模型会**续写**而不是"回答"。例如可能输出：
```
🧠: 之所以呈现蓝色，是因为大气中的气体...
```
或类似续写文本。**它不会停下，会一直写到 `--max_new_tokens`**（默认 8192）或碰到 eos。

> 这是预训练模型的正常表现 —— 它只会"接着写"，要它"回答+停止"需要 SFT。

---

## 二、关键机制详解

### 2.1 Cosine 学习率调度（get_lr）

代码：`trainer/trainer_utils.py:40`

```python
def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))
```

调用点：`train_pretrain.py:31`，每步动态改 `optimizer.param_groups['lr']`。

```
lr
 ↑  ─────╮
0.0005    │ ╲
          │   ╲
          │     ╲
0.0001    │       ╲─────
          └──────────────────→ step
          0                  total
```

**为什么这么设计？**
- 起步：从 `lr × 0.55`（≈ 2.75e-4）开始，**不是直接 5e-4**，避免初始炸
- 中段：cosine 平滑衰减
- 末段：降到 `lr × 0.1`（≈ 5e-5），让 loss 在低 lr 下精细收敛
- 公式恒 > `lr × 0.1`，保证末端不会学到 0 失去更新能力

### 2.2 混合精度 AMP（bfloat16）

代码：`train_pretrain.py:119-122` + `137`

```python
dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
autocast_ctx = torch.cuda.amp.autocast(dtype=dtype)   # 自动混合精度
scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
```

训练循环里（`train_pretrain.py:35-49`）：
```python
with autocast_ctx:                          # 前向在 bf16 下
    res = model(input_ids, labels=labels)
    loss = res.loss + res.aux_loss
    loss = loss / args.accumulation_steps
scaler.scale(loss).backward()              # 反向在 fp32 缩放下
if step % args.accumulation_steps == 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

**为什么用 bfloat16 而不是 float16？**
- bf16：指数位和 fp32 一样多，**数值范围大**，几乎不会溢出，**不需要 GradScaler**
- fp16：精度高但范围小，**需要 scaler 防止梯度下溢**
- minimind 默认 bf16，`GradScaler(enabled=False)` 只是个 no-op 兼容垫片

> A100 / 4090 / H100 都原生支持 bf16。老卡（V100 之前）只有 fp16，要改 `--dtype float16`。

### 2.3 梯度累积（accumulation_steps）

代码：`train_pretrain.py:38, 42-49`

```python
loss = loss / args.accumulation_steps      # ← loss 除以累积步数
scaler.scale(loss).backward()             # 每步都反向，梯度累加
if step % args.accumulation_steps == 0:   # 凑够 N 步才更新
    scaler.step(optimizer)
    optimizer.zero_grad(set_to_none=True)
```

**为什么需要？**
- 显存不够跑 batch_size=256，但 batch_size=32 + accumulation=8 **数学上等价**
- 梯度在 8 步内累加，第 8 步才更新参数

```
step 1: loss/8 → backward (grad += ...)
step 2: loss/8 → backward (grad += ...)
...
step 8: loss/8 → backward (grad += ...)
        optimizer.step() + zero_grad()    ← 真正更新一次
```

> 训练循环末尾的 `train_pretrain.py:75-80` 处理一个边界情况：最后一个 batch 凑不齐 8 步时也要更新，避免丢梯度。

### 2.4 DDP 多卡训练

代码：`trainer/trainer_utils.py:44` `init_distributed_mode` + `train_pretrain.py:153-154`

启动多卡用 `torchrun`：

```bash
cd /path/to/minimind/trainer
torchrun --nproc_per_node 2 train_pretrain.py
# 或 4 卡
torchrun --nproc_per_node 4 train_pretrain.py
```

`torchrun` 会注入 `RANK` / `LOCAL_RANK` / `WORLD_SIZE` 环境变量，`init_distributed_mode()` 检测到后：
1. 初始化 NCCL 进程组（`trainer_utils.py:48`）
2. 设置当前进程绑哪张卡（`torch.cuda.set_device(local_rank)`）
3. `train_pretrain.py:111`：把 `args.device` 改成 `cuda:local_rank`
4. `train_pretrain.py:154`：用 `DistributedDataParallel` 包装模型，梯度自动 all-reduce 同步

**关键约定**：
- `is_main_process()`（`trainer_utils.py:31`）：只有 rank=0 才存权重、打印日志、初始化 wandb，避免 N 个进程重复写
- `DistributedSampler`：自动切数据，每张卡只看 1/N 样本
- `train_pretrain.py:158`：`train_sampler.set_epoch(epoch)` 每轮改种子，避免每 epoch 数据顺序一样

> 切换 GPU 数量后续训？`lm_checkpoint` 会自动换算 step（见下文）。

### 2.5 断点续训（lm_checkpoint）

代码：`trainer/trainer_utils.py:63-116`

启用：`--from_resume 1`（`train_pretrain.py:103`）

**机制**：
```
训练中（model is not None）:
    1. 存两个文件：
       - {save_dir}/pretrain_768.pth              ← 纯模型权重（half+cpu，给推理用）
       - {save_dir}/pretrain_768_resume.pth       ← 完整训练状态（model+optimizer+scaler+epoch+step+wandb_id）
    2. 用 .tmp + os.replace 原子写，防止中断时写出半截

下次启动（model is None, 加载模式）:
    if os.path.exists(resume_path):
        ckp_data = torch.load(resume_path)
        # 自动换算 GPU 数量变化时的 step
        if saved_ws != current_ws:
            ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
        return ckp_data
```

恢复状态在 `train_pretrain.py:140-148`：
```python
if ckp_data:
    model.load_state_dict(ckp_data['model'])
    optimizer.load_state_dict(ckp_data['optimizer'])
    scaler.load_state_dict(ckp_data['scaler'])
    start_epoch = ckp_data['epoch']
    start_step = ckp_data.get('step', 0)
```

**SkipBatchSampler**（`trainer_utils.py:134`）：跳过已训过的 batch，避免重复。
```
启动: --from_resume 1
日志: "Epoch [2/2]: 跳过前3456个step，从step 3457开始"
```

**完整续训命令**：
```bash
cd /path/to/minimind/trainer
python train_pretrain.py --from_resume 1
```

✅ **验证**：日志看到 `跳过前XXX个step，从step XXX+1开始` 即续训成功。

> **OOM、断电、改参数都能续**。但要改 `--hidden_size` 等结构性参数的话，权重不匹配，续不了。

### 2.6 MoE 架构（可选）

启用：`--use_moe 1`

```bash
cd /path/to/minimind/trainer
python train_pretrain.py --use_moe 1
```

效果：
- 权重名变成 `pretrain_768_moe.pth`（见 `train_pretrain.py:63` `moe_suffix = '_moe'`）
- 总参数 198M，激活参数 64M（每 token 只激活 top-1 专家）
- 训练日志多出 `aux_loss`（负载均衡损失，见 `model_minimind.py:171-173`）

MoE 实现：`model/model_minimind.py:148-176` `MOEFeedForward`
- 4 个专家，每 token top-1 routing
- `aux_loss = (load * scores.mean(0)).sum() * num_experts * router_aux_loss_coef`
- 系数 `router_aux_loss_coef = 5e-4`（`model_minimind.py:45`）

> 详见 [01-理论/02-Transformer核心架构](../01-理论/02-Transformer核心架构.md) 的 MoE 部分。

---

## 三、训练曲线怎么看

正常 pretrain 曲线（dense 模型）：

| 阶段 | loss | 说明 |
|---|---|---|
| step 0 | ~10 | 随机权重，cross_entropy ≈ ln(6400) ≈ 8.76 + 一点 |
| step 100 | ~7-8 | 学到 token 频率分布 |
| step 1000 | ~5-6 | 学到基本语法 |
| step 5000 | ~3-4 | 学到语义连贯 |
| 2 epochs 后 | ~2-3 | 收敛，不会再大幅下降 |

MoE 模型还会看到 `aux_loss` 在 0.001-0.01 之间波动，目标是让各专家负载均衡。

**异常信号**：
- loss 不下降 → 学习率太低 / 数据有问题
- loss 突然飙升 → 梯度爆炸，检查 `grad_clip`（默认 1.0，`train_pretrain.py:94`）
- loss 长期 NaN → AMP 数据溢出，换 `--dtype bfloat16`

---

## 🧯 踩坑提示

### Q1：`CUDA out of memory`
- 减小 `--batch_size`（32 → 8）
- 增大 `--accumulation_steps`（8 → 32）保持有效 batch 不变
- 缩短 `--max_seq_len`（340 → 256）

### Q2：训练慢（4090 上一轮超过 1.5h）
- 检查是否真的用了 GPU：日志里 `args.device` 应是 `cuda:0`
- `nvidia-smi` 看 GPU 利用率，应在 90%+
- `--num_workers 8` 默认，CPU 不够可调小

### Q3：`--from_resume 1` 但没续上
检查 `checkpoints/pretrain_768_resume.pth` 是否存在。被中断在 `save_interval` 间隔内的话，会丢最近一段进度。

### Q4：DDP 多卡训练出现 hang
- 确认所有卡都能互相通信（NCCL 版本一致）
- 不要用 `python train_pretrain.py` 启动多卡，必须 `torchrun --nproc_per_node N`
- 加环境变量 `NCCL_P2P_DISABLE=1` 排查 NVLink 问题

### Q5：日志里 `aux_loss: 0.0000` 是不是 bug
不是。`aux_loss` 只有 `--use_moe 1` 时才有非零值（`model_minimind.py:175` dense 模型返回 0）。

### Q6：训练时 `torch.compile` 报错
先关掉 `--use_compile 0`（默认就是 0）。compile 在某些 PyTorch 版本对自定义模型有 bug，能快 10-20% 但不是必需。

### Q7：4090 时间预算
- dense（64M）：约 1.2h / 2 epochs
- MoE（198M-A64M）：约 1.5h / 2 epochs
- 2 卡 DDP：约 0.7h
- 完整数据集 `pretrain_t2t.jsonl`（10GB）：约 10x 时间

---

## ✅ 本篇完成自检

<details>
<summary>点开自检（先想 30 秒）</summary>

1. 预训练模型问"天空为什么是蓝色的"，它会回答吗？
   - 答：不会。预训练只学会了"接着写"，会续写一段相关文字，但不会主动回答+停止。要"回答"行为需要 SFT。
2. 为什么 `loss = res.loss + res.aux_loss`？dense 模型时 aux_loss 是多少？
   - dense 模型 aux_loss = 0（`model_minimind.py:175`）；MoE 时是负载均衡损失。两者相加是为了统一损失接口。
3. `--batch_size 32 --accumulation_steps 8` 和 `--batch_size 256 --accumulation_steps 1` 数学等价吗？
   - 大致等价（梯度均值），但 BN 类层有差异；Transformer 没 BN，所以严格等价。前者显存占用低。
4. cosine schedule 起始 lr 是多少？为什么不是 5e-4？
   - 起始 `lr × 0.55 ≈ 2.75e-4`。避免冷启动梯度爆炸。
5. `lm_checkpoint` 为什么要先写 `.tmp` 再 `os.replace`？
   - 原子替换。训练中断时可能写出半截文件，下次加载就崩；先写 tmp 再 rename 保证文件要么完整要么不存在。
6. 续训时从 4 卡换成 2 卡会怎样？
   - `lm_checkpoint` 自动换算 step（`trainer_utils.py:113` `step = step * saved_ws // current_ws`），因为每张卡每个 step 处理的 batch 数变了。
7. `bfloat16` 比 `float16` 好在哪？为什么 minimind 默认 bf16 但还保留 GradScaler？
   - bf16 范围大、不易溢出、不需要 scaler。保留 `GradScaler(enabled=(dtype=='float16'))` 是兼容 fp16 用户，bf16 时是 no-op。

</details>

下一篇：[02-指令微调SFT](./02-指令微调SFT.md) —— 让只会"接着写"的模型学会"回答+停止"。
