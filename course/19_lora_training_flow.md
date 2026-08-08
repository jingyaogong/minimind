# 第 19 课：LoRA 训练流程

这一课只解决一个问题：MiniMind 的 `train_lora.py` 如何把第 18 课的 LoRA 核心模块接进 SFT 训练，只更新低秩 adapter，并把 adapter 单独保存下来。

## 目录

- [0. 本节主线](#l19-mainline)
- [1. 原理讲解](#l19-principle)
- [2. 源码阅读顺序图](#l19-reading-order)
- [3. MiniMind 源码走读](#l19-source-walkthrough)
- [4. 本节必须会写 / 暂时不要求](#l19-must-write)
- [5. 手写模块](#l19-handwrite)
- [6. 对齐测试](#l19-alignment-test)
- [7. 阶段组装](#l19-stage-assembly)
- [8. 本节检查](#l19-check)
- [9. 下一课](#l19-next)

<a id="l19-mainline"></a>
## 0. 本节主线

LoRA 训练流程是：

```text
加载 full_sft 基座模型
-> 给目标 Linear 注入 LoRA 分支
-> 统计总参数和 LoRA 参数
-> 冻结 base 参数，只保留 LoRA A/B 可训练
-> 继续使用 SFTDataset 构造 input_ids/labels
-> optimizer 只接收 lora_params
-> forward/loss/backward 仍走普通 SFT 训练循环
-> 保存时只保存 LoRA adapter 权重
-> 推理时先加载 base，再加载 LoRA adapter
```

第 18 课解决的是：

```text
一个 Linear 旁边怎么加 BA 低秩分支
```

第 19 课解决的是：

```text
整棵模型什么时候注入 LoRA，怎么只训练 LoRA，怎么保存和推理
```

一句话：

```text
LoRA 训练不是换一种 loss，而是换一种可训练参数集合。
```

SFT 的数据、prompt、labels、causal LM loss 都沿用第 16-17 课；变化只在模型参数和保存方式。

<a id="l19-principle"></a>
## 1. 原理讲解

### 1.1 LoRA 训练为什么从 full_sft 开始

LoRA 微调通常不是从随机初始化或纯 pretrain 权重开始。它要在一个已经会对话的基座上学习少量新行为，例如身份、领域知识、格式习惯。

在 MiniMind 里，LoRA 脚本默认：

```text
from_weight = full_sft
```

这表示：

```text
先加载已经 SFT 过的完整模型
再注入 LoRA
然后只训练 LoRA adapter
```

如果基座还不会按 chat template 回复，LoRA adapter 需要同时学会基础对话和新任务，这就不再是轻量适配。

### 1.2 为什么要先注入 LoRA，再冻结参数

冻结参数的目标是：

```text
base 权重不更新
LoRA A/B 更新
```

所以顺序很重要：

```text
先 apply_lora(model)
再遍历 model.named_parameters()
根据参数名或模块类型决定 requires_grad
```

如果先冻结，再注入 LoRA，后注入的 LoRA 参数默认仍可能是可训练的，但统计和 optimizer 收集容易漏掉；如果先建 optimizer，再注入 LoRA，optimizer 根本拿不到新参数。

正确心智模型是：

```text
模型结构最终长什么样
-> 哪些参数可训练
-> optimizer 管哪些参数
```

### 1.3 LoRA 训练仍然是 SFT 训练

LoRA 不是新的训练目标。MiniMind 的 LoRA 训练仍然读取 SFT 数据：

```text
conversations
-> chat template
-> input_ids
-> assistant labels
-> causal LM loss
```

所以第 5、16、17 课讲过的 SFT labels 规则继续成立：

```text
user/system/tool prompt 区域 label = -100
assistant 回复区域 label = token id
```

LoRA 改变的是：

```text
哪些参数会被 optimizer 更新
```

不是：

```text
数据格式
loss 公式
forward 输出
```

### 1.4 为什么 optimizer 只接收 LoRA 参数

即使 base 参数 `requires_grad=False`，更清晰的做法仍然是：

```text
optimizer = AdamW(lora_params, lr=...)
```

这让训练边界很明确：

```text
backward 可以经过整棵模型
optimizer.step 只更新 LoRA A/B
```

梯度裁剪也只对 `lora_params` 做：

```text
clip_grad_norm_(lora_params, grad_clip)
```

这样参数统计、显存、保存和调试都围绕 adapter 展开。

### 1.5 为什么 LoRA 只保存 adapter

LoRA 训练的产物不是完整模型，而是低秩增量。

保存完整模型会浪费空间，也会模糊基座和 adapter 的关系。MiniMind 的做法是：

```text
base 权重：full_sft_768.pth
LoRA 权重：lora_medical_768.pth
```

推理时组合：

```text
加载 base
-> 注入同样结构的 LoRA 分支
-> 加载 adapter 权重
```

所以 LoRA 文件不能单独推理。它只是一组增量参数，必须挂到同结构的基座模型上。

### 1.6 为什么原脚本关闭 torch.compile

MiniMind 原版 `apply_lora` 是 monkey-patch 写法：它把某些 `nn.Linear` 的 `forward` 函数替换成 `forward_with_lora`。

这种运行时改 `forward` 的方式直观，但和 `torch.compile` 的图捕获不稳定，所以原训练脚本检测到 `use_compile=1` 时会关闭它。

教学版建议用包装模块：

```text
nn.Linear -> LoRALinear(base_linear)
```

这种方式比 monkey-patch 更容易测试，也更适合课程里做模块替换练习。理解原项目时要知道：原项目是给 Linear 挂 `module.lora` 并改 forward；教学版是把目标 Linear 换成 `LoRALinear`。

<a id="l19-reading-order"></a>
## 2. 源码阅读顺序图

先读第 18 课的 LoRA 核心，再读训练脚本：

```text
model/model_lora.py
  LoRA
  apply_lora
  save_lora
  load_lora

trainer/train_lora.py
  argparse 默认参数
  init_model 加载 full_sft
  apply_lora 注入 adapter
  参数统计
  冻结非 LoRA 参数
  SFTDataset / optimizer
  train_epoch
  save_lora

eval_llm.py
  apply_lora
  load_lora
```

建议按这个顺序读：

```text
1. train_lora.py:79-101       看 LoRA 训练默认参数
2. trainer_utils.py:119-131   看 from_weight 如何加载基座
3. train_lora.py:128-146      看注入、统计、冻结
4. train_lora.py:148-152      看数据集和 optimizer
5. train_lora.py:25-76        看训练循环和保存
6. model_lora.py:21-32        看 apply_lora 如何改模型结构
7. model_lora.py:45-53        看 save_lora 只保存 adapter
8. eval_llm.py:24-26          看推理如何加载 LoRA
```

<a id="l19-source-walkthrough"></a>
## 3. MiniMind 源码走读

### 3.1 LoRA 训练脚本默认从 full_sft 加载

#### 源码证据 A：LoRA 训练参数

文件：`trainer/train_lora.py:79-101`

看它是为了理解：LoRA 训练的默认基座、数据、学习率和保存名。

代码摘录：

```python
parser.add_argument("--lora_name", type=str, default="lora_medical")
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--data_path", type=str, default="../dataset/lora_medical.jsonl")
parser.add_argument('--from_weight', default='full_sft', type=str)
```

这段代码说明：

- LoRA 默认产物名是 `lora_medical`。
- LoRA 默认学习率比 full SFT 的 `1e-5` 更大，是 `1e-4`。
- 数据仍然是 JSONL 对话数据。
- 默认基座权重是 `full_sft`，不是 `pretrain`。

#### 源码证据 B：init_model 加载基座权重

文件：`trainer/trainer_utils.py:119-131`

看它是为了理解：`from_weight='full_sft'` 最终如何变成一个本地权重文件。

代码摘录：

```python
def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model', save_dir='../out', device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)

    if from_weight!= 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)
```

这段代码说明：

- LoRA 脚本不是从 Transformers 格式模型开始，而是构建 MiniMind 原生模型。
- `from_weight` 决定加载 `../out/{from_weight}_{hidden_size}.pth`。
- dense 默认会找 `../out/full_sft_768.pth`。
- MoE 会多一个 `_moe` 后缀。

理解到这一步就够：

```text
LoRA 训练之前，模型已经有一套完整 base 权重。
```

暂时不要看：

```text
AutoTokenizer 内部加载细节
strict=False 对每个 key 的兼容细节
```

### 3.2 apply_lora 必须在冻结和 optimizer 之前

#### 源码证据 A：训练脚本中的调用顺序

文件：`trainer/train_lora.py:128-146`

看它是为了理解：模型结构、可训练参数和 optimizer 参数集合的顺序关系。

代码摘录：

```python
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
apply_lora(model)

total_params = sum(p.numel() for p in model.parameters())
lora_params_count = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)

lora_params = []
for name, param in model.named_parameters():
    if 'lora' in name:
        param.requires_grad = True
        lora_params.append(param)
    else:
        param.requires_grad = False
```

这段代码说明：

- `apply_lora(model)` 发生在参数统计之前。
- 统计 LoRA 参数依赖参数名里出现 `lora`。
- 冻结规则也依赖参数名里出现 `lora`。
- `lora_params` 是后面 optimizer 和 grad clip 的唯一参数集合。

#### 源码证据 B：原项目如何注入 LoRA

文件：`model/model_lora.py:21-32`

看它是为了理解：MiniMind 原项目的 `apply_lora` 实际改变了什么。

代码摘录：

```python
def apply_lora(model, rank=16):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.in_features == module.out_features:
            lora = LoRA(module.in_features, module.out_features, rank=rank).to(model.device)
            setattr(module, "lora", lora)
            original_forward = module.forward

            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora
```

这段代码说明：

- MiniMind 只给方阵 `Linear` 注入 LoRA。
- 每个命中的 Linear 会得到一个 `module.lora` 子模块。
- 原始 forward 被保留下来，新的 forward 返回 `base(x) + lora(x)`。
- 因为 `LoRA.B` 初始化为 0，刚注入时模型输出不应该改变。

教学版和原项目的差异：

```text
原项目：给 Linear 挂 module.lora，并 monkey-patch forward。
教学版：把目标 Linear 替换成 LoRALinear(base)。
```

两者要对齐的行为是：

```text
刚注入时输出不变；
训练时只有 A/B 更新；
保存时只保存 A/B。
```

### 3.3 LoRA 数据仍然走 SFTDataset

#### 源码证据 A：LoRA 训练复用 SFTDataset

文件：`trainer/train_lora.py:148-152`

看它是为了理解：LoRA 没有专门的数据集或 loss。

代码摘录：

```python
train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
```

这段代码说明：

- LoRA 训练数据仍然交给 `SFTDataset`。
- `SFTDataset` 负责 prompt 和 labels。
- optimizer 只接收 `lora_params`。
- scaler 和混合精度机制沿用普通训练脚本。

#### 源码证据 B：SFTDataset 的 labels 规则

文件：`dataset/lm_dataset.py:88-119`

看它是为了理解：LoRA 训练为什么还是 assistant-only loss。

代码摘录：

```python
labels = [-100] * len(input_ids)
...
if input_ids[i:i + len(self.bos_id)] == self.bos_id:
    start = i + len(self.bos_id)
    ...
    for j in range(start, min(end + len(self.eos_id), self.max_length)):
        labels[j] = input_ids[j]
...
return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
```

这段代码说明：

- 初始 labels 全部是 `-100`。
- 只有 assistant 回复段被改成真实 token id。
- cross entropy 会忽略 `-100` 区域。
- LoRA 训练目标仍然是让 assistant 回复 token 更可能出现。

理解到这一步就够：

```text
LoRA 的“参数高效”来自参数冻结，不来自改 SFT labels。
```

### 3.4 train_epoch 和普通 SFT 训练几乎一样

#### 源码证据 A：forward/loss/backward 没变

文件：`trainer/train_lora.py:25-49`

看它是为了理解：LoRA 训练循环仍然是 causal LM 训练。

代码摘录：

```python
res = model(input_ids, labels=labels)
loss = res.loss + res.aux_loss
loss = loss / args.accumulation_steps

scaler.scale(loss).backward()

if step % args.accumulation_steps == 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

这段代码说明：

- forward 仍然调用 `model(input_ids, labels=labels)`。
- loss 仍然是 logits loss 加 MoE aux loss。
- 梯度累积仍然要除以 `accumulation_steps`。
- 梯度裁剪只裁剪 `lora_params`。
- optimizer step 只更新 LoRA 参数，因为 optimizer 只管理 LoRA 参数。

#### 源码证据 B：epoch 末尾 flush 残余梯度

文件：`trainer/train_lora.py:71-76`

看它是为了理解：LoRA 训练也继承了第 14 课的 gradient accumulation 尾批处理。

代码摘录：

```python
if last_step > start_step and last_step % args.accumulation_steps != 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

这段代码说明：

- 如果最后一组 micro-batch 不满 `accumulation_steps`，也要 step 一次。
- 这个逻辑和普通 pretrain/full_sft 一致。
- 区别仍然只是裁剪和更新的参数集合是 `lora_params`。

### 3.5 LoRA 保存和 resume 是两条线

#### 源码证据 A：训练中保存 adapter

文件：`trainer/train_lora.py:60-67`

看它是为了理解：普通保存和 LoRA adapter 保存有什么不同。

代码摘录：

```python
lora_save_path = f'{args.save_dir}/{args.lora_name}_{lm_config.hidden_size}{moe_suffix}.pth'
save_lora(model, lora_save_path)
lm_checkpoint(lm_config, weight=args.lora_name, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
```

这段代码说明：

- `save_lora(...)` 保存给推理使用的 LoRA adapter。
- `lm_checkpoint(...)` 保存断点续训需要的模型、optimizer、scaler、epoch、step。
- 推理主要用 `../out/lora_name_768.pth`。
- 断点续训主要用 `../checkpoints/lora_name_768_resume.pth`。

#### 源码证据 B：save_lora 只收集 module.lora

文件：`model/model_lora.py:45-53`

看它是为了理解：LoRA 权重文件为什么很小。

代码摘录：

```python
for name, module in raw_model.named_modules():
    if hasattr(module, 'lora'):
        clean_name = name[7:] if name.startswith("module.") else name
        lora_state = {f'{clean_name}.lora.{k}': v.cpu().half() for k, v in module.lora.state_dict().items()}
        state_dict.update(lora_state)
torch.save(state_dict, path)
```

这段代码说明：

- 只有带 `module.lora` 的模块会被保存。
- 保存内容来自 `module.lora.state_dict()`。
- base 模型权重不会进入 LoRA 文件。
- DDP 下可能出现的 `module.` 前缀会被处理。

教学版的 `lora_state_dict` key 可以不完全照原项目的 `.lora.` 命名，但必须满足：

```text
只保存 LoRALinear 里的 A.weight 和 B.weight
不保存 base.weight
```

### 3.6 推理时必须 base + adapter 一起加载

#### 源码证据：eval_llm 加载 LoRA

文件：`eval_llm.py:12-30`

看它是为了理解：训练保存的 LoRA 文件如何在推理入口被使用。

代码摘录：

```python
model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
if args.lora_weight != 'None':
    apply_lora(model)
    load_lora(model, f'./{args.save_dir}/{args.lora_weight}_{args.hidden_size}.pth')
...
return model.half().eval().to(args.device), tokenizer
```

这段代码说明：

- 推理先加载 base 权重，例如 `full_sft_768.pth`。
- 如果传了 `--lora_weight`，再注入 LoRA 结构。
- LoRA adapter 文件只负责填充注入后的 A/B。
- 最后模型进入 half、eval，并移动到设备。

因此推理命令的心智模型是：

```bash
python eval_llm.py --weight full_sft --lora_weight lora_medical
```

含义不是“只加载 lora_medical”，而是：

```text
加载 full_sft base
再叠加 lora_medical adapter
```

<a id="l19-must-write"></a>
## 4. 本节必须会写 / 暂时不要求

必须会写：

```text
1. 说明 LoRA 训练和 Full SFT 的共同点：
   - 都用 SFTDataset
   - 都用 causal LM loss
   - 都有 lr / grad accumulation / scaler / grad clip

2. 说明 LoRA 训练和 Full SFT 的不同点：
   - 加载 full_sft 作为 base
   - 训练前注入 LoRA
   - 只让 LoRA A/B requires_grad=True
   - optimizer 只接收 lora_params
   - 保存 adapter，而不是保存完整模型

3. 手写 apply_lora_to_linear_layers：
   - 遍历模型子模块
   - 找到目标 nn.Linear
   - 用 LoRALinear 包装或替换它
   - 返回注入数量

4. 手写最小 train_lora_impl.py 的主流程：
   - parse args
   - load tokenizer / model
   - apply LoRA
   - mark only LoRA trainable
   - build SFT dataset
   - build AdamW(lora_params)
   - run short train loop
   - save lora_state_dict
```

暂时不要求：

```text
1. 真实医疗数据训练效果
2. 多 target module 配置
3. LoRA 权重合并导出
4. HuggingFace / llama.cpp / ollama 格式转换
5. DDP 下的完整恢复细节
6. rank / alpha / dropout 系统调参
```

<a id="l19-handwrite"></a>
## 5. 手写模块

本节涉及两个文件：

```text
course/impl/core/lora.py
course/impl/train_lora_impl.py
```

### 5.1 补 `apply_lora_to_linear_layers`

接口：

```python
def apply_lora_to_linear_layers(
    model: nn.Module,
    rank: int = 16,
    alpha: float | None = None,
    dropout: float = 0.0,
    square_only: bool = True,
) -> int:
    ...
```

它要对齐的源码：

```text
model/model_lora.py::apply_lora
```

教学版建议行为：

```text
遍历 model.named_children()
-> 如果 child 是 nn.Linear 且满足筛选条件
-> 用 LoRALinear(child, rank, alpha, dropout) 替换这个 child
-> 如果 child 不是目标 Linear，则递归处理它的子模块
-> 返回被替换的 Linear 数量
```

筛选规则：

```text
square_only=True:
  只替换 in_features == out_features 的 Linear

square_only=False:
  替换所有 Linear
```

`alpha` 的默认值建议：

```text
alpha = rank
```

这样教学版默认与 MiniMind 原源码一致：

```text
base(x) + B(A(x))
```

而不是：

```text
base(x) + alpha/r * B(A(x))   # alpha != rank 时才有额外缩放
```

注意：

```text
不要直接遍历 named_modules 后 setattr 当前 module 自己。
```

原因是 `named_modules()` 会递归返回所有后代模块，但替换子模块需要知道它的父模块。教学版更适合用 `named_children()` 递归，因为 `setattr(parent, child_name, wrapped_child)` 更清楚。

### 5.2 保留第 18 课的冻结函数

接口：

```python
def mark_only_lora_as_trainable(model: nn.Module) -> list[nn.Parameter]:
    ...
```

它在第 19 课中的位置是：

```text
apply_lora_to_linear_layers(model)
-> mark_only_lora_as_trainable(model)
-> optimizer = AdamW(lora_params, lr=...)
```

验收点：

```text
model.named_parameters() 里只有 A.weight / B.weight requires_grad=True
optimizer 的参数就是这两个集合
base.weight 不会被 optimizer 更新
```

### 5.3 保留第 18 课的保存函数

接口：

```python
def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    ...
```

它在第 19 课中的位置是：

```text
训练结束或 save_interval 到达
-> state = lora_state_dict(model)
-> torch.save(state, adapter_path)
```

验收点：

```text
state_dict 只包含 LoRA A/B
不包含 base.weight
保存出来的文件可以和 base 模型重新组合
```

### 5.4 组装 `train_lora_impl.py`

`course/impl/train_lora_impl.py` 是教学版脚本。它不需要复制原项目所有工程能力，只要跑通 LoRA SFT 的最小闭环。

建议参数：

```text
--tokenizer_path model
--data_path course/labs/tiny_sft.jsonl
--from_weight course_sft
--save_weight course_lora
--save_dir course/impl/out
--hidden_size 64
--num_hidden_layers 2
--max_seq_len 128
--rank 4
--alpha 4
--epochs 1
--batch_size 1
--learning_rate 1e-4
--device cpu
```

教学版主流程：

```text
1. 读取 args
2. 加载 tokenizer
3. 构造 CourseMiniMindForCausalLM 或复用 MiniMindForCausalLM
4. 加载 from_weight 对应的 base 权重
5. apply_lora_to_linear_layers(model, rank, alpha)
6. lora_params = mark_only_lora_as_trainable(model)
7. train_ds = CourseSFTDataset 或 SFTDataset
8. optimizer = AdamW(lora_params, lr)
9. 复用 train loop 跑少量 step
10. torch.save(lora_state_dict(model), adapter_path)
```

如果前面课程的教学版 CausalLM / train loop 还没补完，可以先只完成第 19 课的轻量注入测试，不强行跑完整 `train_lora_impl.py`。

<a id="l19-alignment-test"></a>
## 6. 对齐测试

本节新增测试：

```text
course/impl/tests/test_lora_injection.py
```

运行命令：

```bash
cd /home/sun/minimind
python course/impl/tests/test_lora_core.py
python course/impl/tests/test_lora_injection.py
```

第一个测试来自第 18 课，检查单个 `LoRALinear`、冻结函数和保存函数。

第二个测试检查第 19 课新增能力：

```text
1. apply_lora_to_linear_layers 能把目标 Linear 替换成 LoRALinear。
2. square_only=True 时只替换方阵 Linear。
3. 因为 B 初始化为 0，注入前后输出一致。
4. mark_only_lora_as_trainable 后只有 A/B 可训练。
5. lora_state_dict 只导出 A/B。
```

还没实现 `apply_lora_to_linear_layers` 时，测试会提示：

```text
TODO not implemented yet: Implement in the LoRA lesson.
```

补完后应该看到类似：

```text
injected_lora_layers=2
initial_output_max_abs_diff=0.000000000000
trainable_names=['0.A.weight', '0.B.weight', '2.0.A.weight', '2.0.B.weight']
lora_state_keys=['0.A.weight', '0.B.weight', '2.0.A.weight', '2.0.B.weight']
lora_injection=passed
```

### 6.1 真实训练命令不是当前必跑验收

如果本地已经有：

```text
out/full_sft_768.pth
dataset/lora_medical.jsonl
```

可以跑原项目 LoRA 训练：

```bash
cd /home/sun/minimind/trainer
python train_lora.py \
  --from_weight full_sft \
  --lora_name lora_medical \
  --data_path ../dataset/lora_medical.jsonl \
  --epochs 1
```

当前课程环境更适合把这个命令当作“有数据和权重后的实战命令”，不是第 19 课的必跑验收。

原因：

```text
1. 默认 full_sft 权重可能不存在。
2. 默认 lora_medical 数据可能不存在。
3. CPU 环境跑完整训练没有学习必要。
4. 第 19 课的核心是参数注入、冻结和保存边界。
```

<a id="l19-stage-assembly"></a>
## 7. 阶段组装

LoRA 阶段现在有三层产物：

```text
第 18 课：
  LoRALinear
  mark_only_lora_as_trainable
  lora_state_dict

第 19 课：
  apply_lora_to_linear_layers
  train_lora_impl.py 最小流程

原项目能力：
  trainer/train_lora.py
  model/model_lora.py
  eval_llm.py --lora_weight
```

### 7.1 教学版和原项目差异

| 位置 | 原项目 | 教学版 |
|---|---|---|
| 注入方式 | 给 Linear 挂 `module.lora` 并替换 `forward` | 用 `LoRALinear(base)` 替换目标 Linear |
| 缩放 | 没有显式 `alpha/r` | 有 `alpha/r`，默认 `alpha=rank` |
| 目标模块 | 只处理方阵 Linear | 默认同样只处理方阵 Linear |
| 保存 key | `name.lora.A.weight` / `name.lora.B.weight` | `name.A.weight` / `name.B.weight` |
| 完整训练 | 支持 DDP、resume、swanlab | 先跑通 tiny/CPU 最小闭环 |
| compile | monkey-patch 下关闭 | 包装模块更容易测试，仍可先不讲 compile |

这些差异不是错误。课程目标是让你能解释核心机制，而不是复制所有工程外围。

### 7.2 阶段验收顺序

建议按这个顺序验收：

```bash
cd /home/sun/minimind
python course/impl/tests/test_lora_core.py
python course/impl/tests/test_lora_injection.py
```

如果前面教学版 SFT 阶段也已经完成，再跑：

```bash
python course/impl/train_lora_impl.py \
  --tokenizer_path model \
  --data_path course/labs/tiny_sft.jsonl \
  --from_weight course_sft \
  --save_weight course_lora \
  --save_dir course/impl/out \
  --hidden_size 64 \
  --num_hidden_layers 2 \
  --max_seq_len 128 \
  --rank 4 \
  --alpha 4 \
  --epochs 1 \
  --batch_size 1 \
  --learning_rate 1e-4 \
  --device cpu
```

如果原项目真实权重和数据齐备，再跑：

```bash
cd /home/sun/minimind/trainer
python train_lora.py --from_weight full_sft --lora_name lora_medical --epochs 1
```

### 7.3 Portfolio 记录

完成本节后，可以在 `course/portfolio/implementation.md` 记录：

```text
实现了教学版 LoRA 注入流程：
- 用 LoRALinear 包装目标 Linear。
- 支持 square_only 筛选。
- 验证 B=0 初始化使注入前后输出一致。
- 冻结 base 参数，只训练 LoRA A/B。
- 只导出 adapter state_dict。
```

在 `course/notes/mistakes.md` 记录容易错的点：

```text
1. 先建 optimizer 再注入 LoRA，导致 optimizer 没有 LoRA 参数。
2. 遍历 named_modules 时不知道父模块，无法正确 setattr。
3. 忘记 B 初始化为 0，导致注入后输出改变。
4. 保存 state_dict 时把 base.weight 也保存了。
5. 推理时只加载 LoRA 文件，忘记先加载 base。
```

<a id="l19-check"></a>
## 8. 本节检查

1. LoRA 训练为什么默认从 `full_sft` 权重开始，而不是从 `pretrain` 或 `none` 开始？
2. 为什么必须在构建 optimizer 之前调用 `apply_lora`？
3. LoRA 训练和 Full SFT 的数据、labels、loss 有什么相同点？
4. `clip_grad_norm_` 为什么只传入 `lora_params`？
5. `save_lora` 为什么不能保存完整 base 权重？
6. 推理时 `--weight full_sft --lora_weight lora_medical` 分别代表什么？
7. 教学版用 `LoRALinear(base)` 替换 Linear，和原项目 monkey-patch forward 的共同目标是什么？

<a id="l19-next"></a>
## 9. 下一课

第 20 课进入模型蒸馏训练链路。

下一课要解决：

```text
teacher/student 模型如何同时参与训练；
distillation loss 和普通 causal LM loss 如何组合；
teacher logits 为什么通常不需要反向传播；
MiniMind 的 train_distillation.py 如何组织数据、模型和 optimizer。
```
