# 第 17 课：从 Pretrain 权重到 Full SFT 权重

这一课只解决一个问题：MiniMind 如何把预训练阶段产出的权重，接到 Full SFT 阶段继续训练，并最终被推理脚本加载。

## 目录

- [0. 本节主线](#l17-mainline)
- [1. 原理讲解](#l17-principle)
- [2. 源码阅读顺序图](#l17-reading-order)
- [3. MiniMind 源码走读](#l17-source-walkthrough)
- [4. 本节必须会写 / 暂时不要求](#l17-must-write)
- [5. 手写模块](#l17-handwrite)
- [6. 对齐测试](#l17-alignment-test)
- [7. 阶段组装](#l17-stage-assembly)
- [8. 本节检查](#l17-check)
- [9. 下一课](#l17-next)

<a id="l17-mainline"></a>
## 0. 本节主线

MiniMind 原项目的阶段切换是：

```text
train_pretrain.py
-> 保存 out/pretrain_768.pth
-> train_full_sft.py --from_weight pretrain
-> init_model 加载 out/pretrain_768.pth
-> 用 SFTDataset 继续训练
-> 保存 out/full_sft_768.pth
-> eval_llm.py --weight full_sft
-> 加载 out/full_sft_768.pth 推理
```

教学版实现的阶段切换应该是：

```text
train_pretrain_impl.py
-> 保存 out/course_pretrain_*.pth
-> train_sft_impl.py --from_weight course_pretrain
-> 加载 course_pretrain 权重
-> 用 CourseSFTDataset 继续训练
-> 保存 out/course_sft_*.pth
-> 用同一套 tokenizer + 模型结构做推理验证
```

一句话：

```text
from_weight 决定“从哪个已有权重开始训练”，save_weight 决定“这一阶段训练完保存成什么权重”。
```

<a id="l17-principle"></a>
## 1. 原理讲解

### 1.1 阶段训练不是重新开始

Pretrain 阶段学到的是基础语言建模能力：

```text
普通文本
-> next-token prediction
-> 学语言统计、知识片段、基本续写能力
```

SFT 阶段不是把模型随机初始化后重新学，而是在 pretrain 参数基础上继续优化：

```text
pretrain 参数
-> 对话格式数据
-> assistant 区域监督
-> 学会按用户指令回答
```

参数变化可以写成：

$$
\begin{aligned}
\theta_{\text{pretrain}}
&= \mathrm{Train}_{\text{pretrain}}(\theta_0, D_{\text{pretrain}}) \\
\theta_{\text{sft}}
&= \mathrm{Train}_{\text{sft}}(\theta_{\text{pretrain}}, D_{\text{sft}})
\end{aligned}
$$

其中：

| 符号 | 类型 | 含义 |
|---|---|---|
| $\theta_0$ | 参数集合 | 随机初始化参数 |
| $\theta_{\text{pretrain}}$ | 参数集合 | 预训练后参数 |
| $\theta_{\text{sft}}$ | 参数集合 | SFT 后参数 |
| $D_{\text{pretrain}}$ | 数据集 | 普通文本预训练数据 |
| $D_{\text{sft}}$ | 数据集 | 对话监督微调数据 |

这就是为什么 `train_full_sft.py` 默认 `from_weight='pretrain'`。

### 1.2 `from_weight` 和 `from_resume` 不是一回事

这两个参数很容易混。

`from_weight` 解决的是“阶段初始化”：

```text
从 out/pretrain_768.pth 初始化模型参数；
然后开始一个新的 SFT 阶段。
```

`from_resume` 解决的是“同一阶段中断后继续”：

```text
从 checkpoints/full_sft_768_resume.pth 恢复；
继续未完成的 Full SFT 阶段。
```

对比：

| 参数 | 读取文件 | 恢复内容 | 使用场景 |
|---|---|---|---|
| `from_weight='pretrain'` | `out/pretrain_768.pth` | 只恢复模型权重 | 从 pretrain 进入 SFT |
| `from_resume=1` | `checkpoints/full_sft_768_resume.pth` | model / optimizer / scaler / epoch / step | SFT 中断后继续 |

所以阶段切换时，通常是：

```text
from_weight='pretrain'
from_resume=0
```

如果 SFT 训练中断，再恢复时才用：

```text
from_resume=1
```

### 1.3 权重文件名是阶段接口

MiniMind 用权重名前缀表达阶段：

```text
pretrain_768.pth
full_sft_768.pth
dpo_768.pth
grpo_768.pth
```

文件名规则是：

$$
\begin{aligned}
\text{weight\_path}
&= \text{save\_dir}/(\text{weight}\_\text{hidden\_size}\text{moe\_suffix}.pth) \\
\text{moe\_suffix}
&=
\begin{cases}
\text{"\_moe"}, & \text{use\_moe=True} \\
\text{""}, & \text{use\_moe=False}
\end{cases}
\end{aligned}
$$

例子：

| 阶段 | `weight` | `hidden_size` | `use_moe` | 文件 |
|---|---|---:|---|---|
| Pretrain | `pretrain` | 768 | False | `out/pretrain_768.pth` |
| Full SFT | `full_sft` | 768 | False | `out/full_sft_768.pth` |
| MoE Pretrain | `pretrain` | 768 | True | `out/pretrain_768_moe.pth` |
| MoE SFT | `full_sft` | 768 | True | `out/full_sft_768_moe.pth` |

这一套命名让训练脚本和推理脚本能通过同一个 `weight` 参数找到对应权重。

### 1.4 SFT 为什么需要更小学习率

Pretrain 是从随机初始化开始，学习率可以大一些。

SFT 是在已有模型上微调，如果学习率太大，可能破坏 pretrain 学到的语言能力。MiniMind 默认：

```text
pretrain learning_rate = 5e-4
full_sft learning_rate = 1e-5
```

可以理解成：

$$
\eta_{\text{sft}} \ll \eta_{\text{pretrain}}
$$

这里 $\eta$ 表示学习率。

SFT 的目标不是让模型从零学语言，而是让参数在已有能力附近移动到“更会对话”的区域。

### 1.5 推理时为什么 `pretrain` 和 `full_sft` prompt 不同

`eval_llm.py` 里有一个分支：

```text
如果 weight 名里包含 pretrain：
    输入 = bos_token + 用户原文
否则：
    输入 = chat_template(conversation, add_generation_prompt=True)
```

原因是：

```text
pretrain 权重主要学普通文本续写，不一定学会 chat template；
full_sft 权重学过对话格式，推理时应该喂同样的 chat template。
```

这体现了一个重要原则：

```text
训练时用什么格式，推理时就尽量用什么格式。
```

### 1.6 阶段切换的变量流

从 pretrain 到 full_sft，关键变量是：

| 变量 | 类型/形状 | 阶段 | 含义 |
|---|---|---|---|
| `from_weight` | str | SFT 启动 | 要加载的旧阶段权重名前缀 |
| `save_weight` | str | SFT 保存 | 当前阶段保存的新权重名前缀 |
| `weight_path` | str | 加载 | `out/{from_weight}_{hidden_size}.pth` |
| `ckp` | str | 保存/推理 | `out/{save_weight}_{hidden_size}.pth` |
| `input_ids` | `[B, S]` | 训练 | SFTDataset 输出 |
| `labels` | `[B, S]` | 训练 | assistant 区域 label |
| `logits` | `[B, S, V]` | forward | 模型输出 |
| `loss` | 标量 | backward | SFT 训练损失 |

你要把这条线记住：

```text
from_weight 只影响训练开始；
save_weight 影响训练结束和后续推理。
```

<a id="l17-reading-order"></a>
## 2. 源码阅读顺序图

这节源码按这个顺序读：

```text
README 训练命令
-> train_pretrain.py 保存 pretrain 权重
-> train_full_sft.py 默认 from_weight / save_weight
-> trainer_utils.init_model 加载 from_weight
-> train_full_sft.py 保存 full_sft 权重
-> eval_llm.py 根据 --weight 加载推理权重
-> eval_llm.py 根据 weight 选择 prompt 格式
```

对应文件：

```text
README.md
trainer/train_pretrain.py
trainer/train_full_sft.py
trainer/trainer_utils.py
eval_llm.py
```

这节不要再陷进 `SFTDataset` 内部，第 16 课已经讲过。现在重点是阶段之间的权重如何流动。

<a id="l17-source-walkthrough"></a>
## 3. MiniMind 源码走读

### 第 1 步：README 给出的阶段顺序

File: `README.md:322-340`

Read this to understand: 原项目推荐的最小训练顺序就是先 pretrain，再 full_sft，再 eval。

Code/config/template excerpt:

```bash
cd trainer && python train_pretrain.py
cd trainer && python train_full_sft.py
python eval_llm.py --weight full_sft
```

This shows:

- `train_pretrain.py` 先产出 pretrain 权重。
- `train_full_sft.py` 再基于 pretrain 权重训练。
- `eval_llm.py --weight full_sft` 用 SFT 后的权重推理。

### 第 2 步：Pretrain 默认保存成 `pretrain`

File: `trainer/train_pretrain.py:84-103`

Read this to understand: pretrain 阶段的默认保存名前缀。

Code/config/template excerpt:

```python
parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")
```

This code shows:

- Pretrain 默认从头开始，`from_weight='none'`。
- Pretrain 默认保存名前缀是 `pretrain`。
- 默认会写出 `out/pretrain_768.pth`。

### 第 3 步：SFT 默认加载 `pretrain`，保存 `full_sft`

File: `trainer/train_full_sft.py:84-104`

Read this to understand: SFT 阶段如何接上 pretrain 阶段。

Code/config/template excerpt:

```python
parser.add_argument('--save_weight', default='full_sft', type=str, help="保存权重的前缀名")
parser.add_argument("--learning_rate", type=float, default=1e-5, help="初始学习率")
parser.add_argument("--data_path", type=str, default="../dataset/sft_t2t_mini.jsonl", help="训练数据路径")
parser.add_argument('--from_weight', default='pretrain', type=str, help="基于哪个权重训练，为none则不基于任何权重训练")
parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
```

This code shows:

- SFT 默认从 `pretrain` 权重初始化。
- SFT 默认保存成 `full_sft`。
- SFT 默认学习率是 `1e-5`。
- `from_resume` 是同一阶段续训，不是阶段初始化。

### 第 4 步：`init_model` 把 `from_weight` 变成文件路径

File: `trainer/trainer_utils.py:119-131`

Read this to understand: `from_weight='pretrain'` 如何定位到真实 `.pth` 文件。

Code/config/template excerpt:

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

This code shows:

- `from_weight` 只是前缀，不是完整路径。
- `save_dir` 默认是 `../out`。
- `hidden_size` 和 MoE 后缀会参与文件名。
- 训练脚本里的 `from_weight='pretrain'` 会加载 `../out/pretrain_768.pth`。

### 第 5 步：SFT 保存的是 `full_sft`

File: `trainer/train_full_sft.py:61-70`

Read this to understand: SFT 训练后会写出普通权重和 resume checkpoint。

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
    lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scaler=scaler)
```

This code shows:

- `ckp` 使用 `args.save_weight`。
- 默认 `args.save_weight='full_sft'`。
- 普通推理权重会保存到 `out/full_sft_768.pth`。
- resume checkpoint 会保存到 `checkpoints/full_sft_768_resume.pth`。

### 第 6 步：`eval_llm.py` 加载 native torch 权重

File: `eval_llm.py:12-30`

Read this to understand: 推理时 `--weight full_sft` 如何加载权重。

Code/config/template excerpt:

```python
tokenizer = AutoTokenizer.from_pretrained(args.load_from)
if 'model' in args.load_from:
    model = MiniMindForCausalLM(MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        inference_rope_scaling=args.inference_rope_scaling
    ))
    moe_suffix = '_moe' if args.use_moe else ''
    ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
    model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
else:
    model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
```

This code shows:

- `--load_from ./model` 会走 native torch 权重加载。
- `--weight full_sft` 会参与构造 `./out/full_sft_768.pth`。
- `hidden_size`、`num_hidden_layers`、`use_moe` 必须和权重对应。
- `--load_from ./minimind-3` 这类路径会走 HuggingFace transformers 格式。

### 第 7 步：推理 prompt 根据权重阶段变化

File: `eval_llm.py:71-78`

Read this to understand: pretrain 权重和 SFT 权重推理输入格式不同。

Code/config/template excerpt:

```python
conversation.append({"role": "user", "content": prompt})
if 'pretrain' in args.weight:
    inputs = tokenizer.bos_token + prompt
else:
    inputs = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, open_thinking=bool(args.open_thinking))

inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)
```

This code shows:

- `pretrain` 权重用普通文本续写式输入。
- `full_sft` 权重用 chat template。
- SFT 后的模型推理格式要和 SFT 训练格式保持一致。

<a id="l17-must-write"></a>
## 4. 本节必须会写 / 暂时不要求

必须会写：

1. 阶段权重路径：

$$
\begin{aligned}
\text{pretrain\_path}
&= \text{out}/\text{pretrain}_{H}.pth \\
\text{sft\_path}
&= \text{out}/\text{full\_sft}_{H}.pth
\end{aligned}
$$

2. 阶段初始化关系：

$$
\theta_{\text{sft,start}} = \theta_{\text{pretrain}}
$$

3. 阶段训练更新：

$$
\theta_{\text{sft,end}}
= \mathrm{Train}_{\text{sft}}(\theta_{\text{sft,start}}, D_{\text{sft}})
$$

4. 推理加载关系：

```text
eval_llm.py --load_from ./model --weight full_sft
-> ./out/full_sft_768.pth
```

5. 区分：

```text
from_weight: 阶段开始时加载哪个普通权重
save_weight: 阶段结束时保存成哪个普通权重
from_resume: 当前阶段中断后恢复 optimizer/scaler/epoch/step
```

暂时不要求：

```text
1. 真正训练出高质量 full_sft 权重
2. HuggingFace 格式转换
3. WebUI / OpenAI API 部署
4. LoRA 合并
5. 多阶段自动流水线脚本
```

<a id="l17-handwrite"></a>
## 5. 手写模块

本节你要看的是：

```text
course/impl/train_sft_impl.py
```

这节已经给它补了阶段脚本参数骨架。你要理解每个默认值为什么这样设置：

```python
parser.add_argument("--data_path", default="course/labs/tiny_sft.jsonl")
parser.add_argument("--from_weight", default="course_pretrain")
parser.add_argument("--save_weight", default="course_sft")
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--accumulation_steps", type=int, default=1)
```

教学版不用直接复用 `pretrain/full_sft` 这两个名字，而是用：

```text
course_pretrain
course_sft
```

这样可以避免和原项目训练出来的正式权重混在一起。

### 5.1 本节你要补的不是训练逻辑

完整 `train_sft_impl.py` 还需要：

```text
CourseMiniMindForCausalLM
CourseSFTDataset
train_one_epoch
save_course_checkpoint
load_course_checkpoint
```

这些模块有些还没全部实现。所以第 17 课先只要求你把阶段参数理解清楚。

真正要你能写出来的是：

```text
from_weight 默认是 course_pretrain
save_weight 默认是 course_sft
data_path 默认是 tiny_sft.jsonl
learning_rate 默认比 pretrain 小
```

### 5.2 后续组装时的伪流程

后续实现 `main()` 时应该长这样：

```text
args = parse_args()
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
model = CourseMiniMindForCausalLM(...)
load weights from args.from_weight
dataset = CourseSFTDataset(args.data_path, tokenizer, args.max_seq_len)
loader = DataLoader(dataset, ...)
optimizer = AdamW(model.parameters(), lr=args.learning_rate)
train_one_epoch(...)
save_course_checkpoint(..., weight=args.save_weight)
```

注意：这里是流程说明，不是让你现在把所有代码一次写完。

<a id="l17-alignment-test"></a>
## 6. 对齐测试

本节新增一个轻量测试：

```text
course/impl/tests/test_sft_stage_args.py
```

运行命令：

```bash
cd /home/sun/minimind
python course/impl/tests/test_sft_stage_args.py
```

这个测试不训练模型，只检查 SFT 阶段脚本默认参数是否符合课程路线：

```text
data_path=course/labs/tiny_sft.jsonl
from_weight=course_pretrain
save_weight=course_sft
learning_rate=1e-05
sft_stage_args=passed
```

它的意义是：先把阶段接口固定，再逐步把数据、模型、训练循环接进去。

<a id="l17-stage-assembly"></a>
## 7. 阶段组装

这一课结束后，你应该能画出两条阶段链。

原项目链路：

```text
trainer/train_pretrain.py
-> out/pretrain_768.pth
-> trainer/train_full_sft.py --from_weight pretrain
-> out/full_sft_768.pth
-> eval_llm.py --load_from ./model --weight full_sft
```

教学版链路：

```text
course/impl/train_pretrain_impl.py
-> out/course_pretrain_*.pth
-> course/impl/train_sft_impl.py --from_weight course_pretrain
-> out/course_sft_*.pth
-> 教学版推理或原生加载验证
```

当前还缺的实现：

```text
1. CourseMiniMindForCausalLM 完整组装
2. CoursePretrainDataset
3. CourseSFTDataset
4. train_one_epoch 完整实现
5. save_course_checkpoint / load_course_checkpoint 完整实现
```

这几个补齐后，SFT 阶段就能真正跑起来。

<a id="l17-check"></a>
## 8. 本节检查

1. `train_pretrain.py` 默认保存成什么权重名前缀？
2. `train_full_sft.py` 默认从哪个权重名前缀加载？
3. `train_full_sft.py` 默认保存成什么权重名前缀？
4. `from_weight` 和 `from_resume` 的区别是什么？
5. `eval_llm.py --load_from ./model --weight full_sft` 会加载哪个文件？
6. 为什么 `pretrain` 权重推理时不用 chat template，而 `full_sft` 权重推理时要用？
7. 教学版为什么用 `course_pretrain` 和 `course_sft`，而不是直接覆盖 `pretrain/full_sft`？
8. 如果 `hidden_size=512` 且 `use_moe=True`，`save_weight='full_sft'` 对应的权重文件名是什么？

<a id="l17-next"></a>
## 9. 下一课

第 18 课进入 LoRA 原理和手写实现。

下一课要解决：

```text
为什么 LoRA 可以只训练低秩增量；
LoRA 如何插到 Linear 层；
哪些参数冻结，哪些参数训练；
LoRA 权重如何保存和加载；
教学版 core/lora.py 应该怎么实现。
```
