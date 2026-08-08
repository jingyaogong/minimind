# 第 18 课：LoRA 原理和手写实现

这一课只解决一个问题：LoRA 如何在不全量更新模型参数的情况下，让模型学到一个新的低秩增量。

## 目录

- [0. 本节主线](#l18-mainline)
- [1. 原理讲解](#l18-principle)
- [2. 源码阅读顺序图](#l18-reading-order)
- [3. MiniMind 源码走读](#l18-source-walkthrough)
- [4. 本节必须会写 / 暂时不要求](#l18-must-write)
- [5. 手写模块](#l18-handwrite)
- [6. 对齐测试](#l18-alignment-test)
- [7. 阶段组装](#l18-stage-assembly)
- [8. 本节检查](#l18-check)
- [9. 下一课](#l18-next)

<a id="l18-mainline"></a>
## 0. 本节主线

LoRA 的核心是：

```text
冻结原来的 Linear 权重 W
-> 旁边加一个低秩分支 BA
-> forward 时输出 W x + BA x
-> 训练时只更新 A 和 B
-> 保存时只保存 A 和 B
```

数学上可以写成：

$$
\begin{aligned}
y_{\text{base}} &= Wx \\
\Delta y &= BAx \\
y &= y_{\text{base}} + \Delta y
\end{aligned}
$$

教学版常见写法会加缩放：

$$
y = Wx + \frac{\alpha}{r}BAx
$$

MiniMind 原源码没有显式写 `alpha / r`，等价于教学版里设置：

$$
\alpha = r
$$

这一课先写 LoRA 核心模块。第 19 课再讲完整 `train_lora.py` 训练流程。

<a id="l18-principle"></a>
## 1. 原理讲解

### 1.1 LoRA 为什么能减少训练参数

普通全参数微调会更新原始线性层权重：

$$
W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}
$$

参数量是：

$$
d_{\text{out}} \cdot d_{\text{in}}
$$

LoRA 不直接训练完整的 $\Delta W$，而是假设需要学习的增量可以用两个低秩矩阵近似：

$$
\Delta W = BA
$$

其中：

| 符号 | 形状 | 含义 |
|---|---|---|
| $W$ | `[out_features, in_features]` | 原始 Linear 权重，冻结 |
| $A$ | `[rank, in_features]` | LoRA 下投影，训练 |
| $B$ | `[out_features, rank]` | LoRA 上投影，训练 |
| $r$ | 标量 | rank，远小于输入/输出维度 |
| $\Delta W$ | `[out_features, in_features]` | LoRA 形成的权重增量 |

LoRA 参数量是：

$$
r \cdot d_{\text{in}} + d_{\text{out}} \cdot r
= r(d_{\text{in}} + d_{\text{out}})
$$

当 $r \ll d_{\text{in}}, d_{\text{out}}$ 时，训练参数会少很多。

例如一个方阵 Linear：

$$
W \in \mathbb{R}^{768 \times 768}
$$

全参数量：

$$
768 \times 768 = 589824
$$

如果 LoRA rank 是 16：

$$
16 \times 768 + 768 \times 16 = 24576
$$

只相当于原层参数量的约 4.17%。

### 1.2 LoRA forward 做了什么

普通 Linear 是：

$$
y = Wx
$$

LoRA Linear 是：

$$
y = Wx + \frac{\alpha}{r}BAx
$$

变量形状：

| 变量 | 形状 | 含义 |
|---|---|---|
| $x$ | `[B, S, d_in]` 或 `[N, d_in]` | 输入 hidden |
| $Wx$ | `[B, S, d_out]` 或 `[N, d_out]` | 原始 Linear 输出 |
| $Ax$ | `[B, S, r]` 或 `[N, r]` | 低秩中间表示 |
| $BAx$ | `[B, S, d_out]` 或 `[N, d_out]` | LoRA 增量输出 |
| $y$ | `[B, S, d_out]` 或 `[N, d_out]` | 最终输出 |

注意：LoRA 不改变 Linear 的输入输出 shape。它只在数值上加一个可训练增量。

### 1.3 为什么 B 通常初始化为 0

MiniMind 的 LoRA 初始化是：

```text
A: normal(0, 0.02)
B: zero
```

这样一开始：

$$
B = 0
$$

所以：

$$
\Delta y = BAx = 0
$$

也就是说，刚注入 LoRA 时，模型输出不变：

$$
y = Wx
$$

这很重要。它保证你把 LoRA 插进去后，不会在训练开始前突然改变基座模型行为。训练过程中，$B$ 逐渐学到非零值，LoRA 分支才开始改变输出。

### 1.4 MiniMind 把 LoRA 插到哪些 Linear 上

MiniMind 的 `apply_lora` 只给满足这个条件的 Linear 加 LoRA：

```text
isinstance(module, nn.Linear)
and module.in_features == module.out_features
```

也就是只处理输入输出维度相同的 Linear。

这是一种简化策略。它通常会覆盖一部分方阵投影，例如 attention 或主干里的某些投影，但不会覆盖所有 Linear。

教学版也先照这个范围做：

```text
square_only=True
```

后面如果想扩展，可以再支持指定目标模块名，比如：

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

但第 18 课先不做复杂 target module 配置。

### 1.5 为什么训练时只更新 LoRA 参数

LoRA 的参数高效来自“冻结基座，只训练低秩分支”。

训练时要设置：

```text
base model 参数: requires_grad = False
LoRA A/B 参数:  requires_grad = True
```

优化器也只接收 LoRA 参数：

```text
optimizer = AdamW(lora_params, lr=...)
```

这样反向传播虽然会经过整个模型，但 optimizer step 只会更新 LoRA 参数。

可以把训练目标写成：

$$
\min_{A,B} L\left(W + \frac{\alpha}{r}BA\right)
$$

这里 $W$ 固定不动，只有 $A$ 和 $B$ 参与优化。

### 1.6 LoRA 保存的不是完整模型

Full SFT 保存的是完整模型权重：

```text
full_sft_768.pth
```

LoRA 保存的只是 adapter 权重：

```text
lora_medical_768.pth
```

也就是只保存：

```text
*.lora.A.weight
*.lora.B.weight
```

推理时需要：

```text
基础模型权重 + LoRA 权重
```

例如：

```bash
python eval_llm.py --weight full_sft --lora_weight lora_medical
```

意思是：先加载 `full_sft` 基座，再叠加 `lora_medical` adapter。

<a id="l18-reading-order"></a>
## 2. 源码阅读顺序图

这节源码按这个顺序读：

```text
model/model_lora.py::LoRA
-> model/model_lora.py::apply_lora
-> trainer/train_lora.py 冻结非 LoRA 参数
-> trainer/train_lora.py optimizer 只接 LoRA 参数
-> model/model_lora.py::save_lora
-> model/model_lora.py::load_lora
-> model/model_lora.py::merge_lora
```

对应文件：

```text
model/model_lora.py
trainer/train_lora.py
eval_llm.py
```

本节重点看 `model_lora.py` 的核心实现。第 19 课再完整走 `train_lora.py` 的训练流程。

<a id="l18-source-walkthrough"></a>
## 3. MiniMind 源码走读

### 第 1 步：LoRA 模块就是两个 Linear

File: `model/model_lora.py:5-18`

Read this to understand: MiniMind 的 LoRA 分支如何构成低秩增量。

Code/config/template excerpt:

```python
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))
```

This code shows:

- `A` 把 `in_features` 降到 `rank`。
- `B` 把 `rank` 升回 `out_features`。
- `B` 初始为 0，所以刚注入时 LoRA 分支输出为 0。
- MiniMind 源码里的 LoRA forward 没有显式 `alpha / rank` 缩放。

对应公式是：

$$
\Delta y = BAx
$$

### 第 2 步：apply_lora 给方阵 Linear 注入 LoRA

File: `model/model_lora.py:21-32`

Read this to understand: MiniMind 没有替换整个模型结构，而是 monkey-patch Linear 的 forward。

Code/config/template excerpt:

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

This code shows:

- 只处理 `in_features == out_features` 的 Linear。
- 每个目标 Linear 多了一个 `module.lora` 子模块。
- `original_forward(x)` 是原始输出 $Wx$。
- `layer2(x)` 是 LoRA 增量 $BAx$。
- 最终输出是两者相加。

对应公式是：

$$
y = Wx + BAx
$$

### 第 3 步：训练时冻结非 LoRA 参数

File: `trainer/train_lora.py:139-147`

Read this to understand: LoRA 训练为什么只更新 adapter 参数。

Code/config/template excerpt:

```python
lora_params = []
for name, param in model.named_parameters():
    if 'lora' in name:
        param.requires_grad = True
        lora_params.append(param)
    else:
        param.requires_grad = False
```

This code shows:

- 参数名里包含 `lora` 的参数会被训练。
- 其它参数全部冻结。
- `lora_params` 会传给 optimizer。

### 第 4 步：optimizer 只接 LoRA 参数

File: `trainer/train_lora.py:148-152`

Read this to understand: 冻结参数之外，optimizer 也只管理 LoRA 参数。

Code/config/template excerpt:

```python
train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
```

This code shows:

- LoRA 训练仍然使用 SFTDataset。
- optimizer 不是 `model.parameters()`，而是 `lora_params`。
- 这保证 optimizer step 只更新 LoRA A/B。

### 第 5 步：保存时只保存 LoRA 权重

File: `model/model_lora.py:45-53`

Read this to understand: LoRA 权重文件为什么很小。

Code/config/template excerpt:

```python
def save_lora(model, path):
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {f'{clean_name}.lora.{k}': v.cpu().half() for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)
```

This code shows:

- 只遍历带 `lora` 的模块。
- 只保存 `module.lora.state_dict()`。
- 不保存基座模型参数。
- 保存到独立的 LoRA 权重文件。

### 第 6 步：加载时按模块名把 LoRA 权重装回去

File: `model/model_lora.py:35-42`

Read this to understand: 加载 LoRA 前必须先对模型调用 `apply_lora`。

Code/config/template excerpt:

```python
def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device)
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)
```

This code shows:

- LoRA 权重文件按模块名保存。
- 加载时要找到对应模块的 `module.lora`。
- 如果没有先 `apply_lora(model)`，模型里就没有 `module.lora` 可加载。

### 第 7 步：合并 LoRA 到完整权重

File: `model/model_lora.py:56-65`

Read this to understand: LoRA 可以在部署前合并回 base weight。

Code/config/template excerpt:

```python
if isinstance(module, nn.Linear) and '.lora.' not in name:
    state_dict[f'{name}.weight'] = module.weight.data.clone().cpu().half()
    if hasattr(module, 'lora'):
        state_dict[f'{name}.weight'] += (module.lora.B.weight.data @ module.lora.A.weight.data).cpu().half()
```

This code shows:

- 合并时把 $BA$ 加到原始 $W$ 上。
- 合并后的权重不再需要 LoRA 分支。
- MiniMind 源码合并时也没有显式 `alpha / rank`。

对应公式是：

$$
W_{\text{merged}} = W + BA
$$

<a id="l18-must-write"></a>
## 4. 本节必须会写 / 暂时不要求

必须会写：

1. LoRA 低秩分支：

$$
\Delta y = BAx
$$

2. LoRA Linear 输出：

$$
y = Wx + \frac{\alpha}{r}BAx
$$

3. MiniMind 源码等价形式：

$$
y = Wx + BAx
$$

4. 参数量对比：

$$
\begin{aligned}
N_{\text{full}}
&= d_{\text{out}}d_{\text{in}} \\
N_{\text{lora}}
&= r(d_{\text{in}} + d_{\text{out}})
\end{aligned}
$$

5. 冻结规则：

```text
base 参数 requires_grad=False
LoRA A/B 参数 requires_grad=True
optimizer 只接收 LoRA 参数
```

暂时不要求：

```text
1. LoRA 训练完整流程
2. LoRA 数据集选择
3. LoRA rank/alpha 搜参
4. 多 target module 配置
5. 合并后导出 HuggingFace 格式
```

<a id="l18-handwrite"></a>
## 5. 手写模块

本节你要补的是：

```text
course/impl/core/lora.py
```

### 5.1 补 `LoRALinear`

接口：

```python
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

你要实现的行为：

- 保存原始 `base` Linear。
- 创建 `A = nn.Linear(base.in_features, rank, bias=False)`。
- 创建 `B = nn.Linear(rank, base.out_features, bias=False)`。
- `A.weight` 用 normal 初始化。
- `B.weight` 用 0 初始化。
- `scaling = alpha / rank`。
- forward 返回：

$$
y = \mathrm{base}(x) + \frac{\alpha}{r}B(A(x))
$$

为了和 MiniMind 源码数值对齐，测试里会用：

$$
\alpha = r
$$

这样：

$$
\frac{\alpha}{r} = 1
$$

### 5.2 补 `mark_only_lora_as_trainable`

接口：

```python
def mark_only_lora_as_trainable(model: nn.Module) -> list[nn.Parameter]:
    ...
```

你要实现的行为：

- 遍历 `model.named_parameters()`。
- 只有名称中属于 LoRA 的 `A.weight` 和 `B.weight` 保持 `requires_grad=True`。
- 其它参数设为 `False`。
- 返回 LoRA 参数列表，供 optimizer 使用。

### 5.3 补 `lora_state_dict`

接口：

```python
def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    ...
```

你要实现的行为：

- 只返回 LoRA 参数。
- 不返回 base weight。
- key 保留模块路径，例如：

```text
0.A.weight
0.B.weight
```

这对应 MiniMind `save_lora` 的思想：LoRA 文件只保存 adapter 权重。

### 5.4 暂时不强制实现 `apply_lora_to_linear_layers`

这个接口已经放到骨架里：

```python
def apply_lora_to_linear_layers(...):
    ...
```

它用于后续把模型里的目标 Linear 批量换成 `LoRALinear`。第 18 课可以先理解和设计，第 19 课训练流程里再完整接入。

<a id="l18-alignment-test"></a>
## 6. 对齐测试

本节新增对齐测试：

```text
course/impl/tests/test_lora_core.py
```

运行命令：

```bash
cd /home/sun/minimind
python course/impl/tests/test_lora_core.py
```

现在还没有实现时，这个测试会因为 `NotImplementedError` 或缺少 `A/B` 失败。等你补完后，它应该打印：

```text
lora_linear_forward_diff=0.000000000000
trainable_names=['0.A.weight', '0.B.weight']
lora_state_keys=['0.A.weight', '0.B.weight']
lora_core=passed
```

这个测试检查三件事：

```text
1. LoRALinear 的 forward 是否等于 base(x) + B(A(x))。
2. 冻结函数是否只让 A/B 可训练。
3. lora_state_dict 是否只导出 A/B。
```

<a id="l18-stage-assembly"></a>
## 7. 阶段组装

本节完成后，LoRA 阶段会多出核心模块：

```text
course/impl/core/lora.py::LoRALinear
course/impl/core/lora.py::apply_lora_to_linear_layers
course/impl/core/lora.py::mark_only_lora_as_trainable
course/impl/core/lora.py::lora_state_dict
```

第 19 课会把这些接到：

```text
course/impl/train_lora_impl.py
```

完整 LoRA SFT 阶段应该是：

```text
加载 full_sft / course_sft 基座
-> 注入 LoRA
-> 冻结 base 参数
-> SFTDataset
-> optimizer 只接 LoRA 参数
-> 训练
-> 只保存 LoRA adapter 权重
```

<a id="l18-check"></a>
## 8. 本节检查

1. LoRA 中 $A$、$B$、$W$ 的形状分别是什么？
2. 为什么 LoRA 的参数量比完整 Linear 小？
3. 为什么 MiniMind 把 `B.weight` 初始化为 0？
4. `apply_lora` 为什么只处理 `in_features == out_features` 的 Linear？
5. LoRA forward 为什么不改变输入输出 shape？
6. 为什么训练时 optimizer 只接收 LoRA 参数？
7. LoRA 权重文件为什么不能单独用于推理？
8. 合并 LoRA 到 base weight 的公式是什么？

<a id="l18-next"></a>
## 9. 下一课

第 19 课讲 LoRA 训练流程。

下一课要解决：

```text
train_lora.py 如何加载 full_sft 基座；
apply_lora 在训练脚本里什么时候调用；
如何统计 LoRA 参数占比；
为什么 torch.compile 被关闭；
LoRA 权重如何和 eval_llm.py 的 --lora_weight 对上。
```
