# 第 7 课：Pretrain 训练入口如何串起来

这一课只解决一个问题：`train_pretrain.py` 是如何把 tokenizer、模型、数据、loss、反向传播和保存权重串成一次预训练的。

## 目录

- [0. 本节主线](#l07-mainline)
- [1. 本节要懂的 6 个原理](#l07-principles)
- [2. 变量流转](#l07-flow)
- [3. 原理一：训练脚本先把超参数变成 `args`](#l07-args)
- [4. 原理二：`MiniMindConfig` 决定模型骨架](#l07-config)
- [5. 原理三：`init_model` 同时准备 tokenizer 和 model](#l07-init-model)
- [6. 原理四：`PretrainDataset` 决定训练样本长什么样](#l07-pretrain-dataset)
- [7. 原理五：`train_epoch` 是真正训练一步的地方](#l07-train-epoch)
- [8. 原理六：checkpoint 分两类](#l07-checkpoint)
- [9. 实验验证](#l07-experiment)
- [10. 本节检查](#l07-check)
- [11. 下一课](#l07-next)

<a id="l07-mainline"></a>
## 0. 本节主线

Pretrain 训练入口的本质是：

```text
读取命令行参数
-> 创建 MiniMindConfig
-> init_model 得到 model 和 tokenizer
-> PretrainDataset 产出 input_ids / labels
-> DataLoader 组 batch
-> model(input_ids, labels=labels) 计算 loss
-> loss.backward()
-> optimizer.step()
-> 定期保存 .pth 和 resume checkpoint
```

所以本节的核心不是“模型内部怎么计算 attention”，而是先看清楚训练脚本把哪些对象接在一起。

<a id="l07-principles"></a>
## 1. 本节要懂的 6 个原理

| 原理 | 要理解什么 | 源码证据 |
|---|---|---|
| 训练脚本先把超参数变成 `args` | 命令行参数决定模型尺寸、数据路径、训练轮数、保存位置 | `trainer/train_pretrain.py:83-107` |
| `MiniMindConfig` 决定模型骨架 | `hidden_size`、`num_hidden_layers`、`use_moe` 会进入模型构造 | `trainer/train_pretrain.py:114-117` |
| `init_model` 同时准备 tokenizer 和 model | tokenizer 负责数据编码，model 负责 forward/loss | `trainer/train_pretrain.py:133-138`, `trainer/trainer_utils.py:119-131` |
| `PretrainDataset` 决定训练样本长什么样 | 每条样本返回 `(input_ids, labels)` | `dataset/lm_dataset.py:37-55` |
| `train_epoch` 是真正训练一步的地方 | forward、loss、backward、clip、step 都在这里发生 | `trainer/train_pretrain.py:24-49` |
| checkpoint 分两类 | `out/*.pth` 给后续加载权重，`checkpoints/*_resume.pth` 给断点续训 | `trainer/train_pretrain.py:61-70`, `trainer/trainer_utils.py:73-116` |

学完本节，你应该能把预训练脚本读成一条链路，而不是一堆分散的配置项。

<a id="l07-flow"></a>
## 2. 变量流转

从命令行到一次参数更新：

```text
args.hidden_size / args.num_hidden_layers
-> lm_config
-> init_model(lm_config, args.from_weight)
-> model, tokenizer

args.data_path / args.max_seq_len
-> PretrainDataset(...)
-> DataLoader(...)
-> input_ids, labels

input_ids, labels
-> model(input_ids, labels=labels)
-> res.loss
-> loss.backward()
-> optimizer.step()
```

要特别注意这三个变量：

```text
input_ids: [batch_size, max_seq_len]
labels:    [batch_size, max_seq_len]
logits:    [batch_size, max_seq_len, vocab_size]
```

<a id="l07-args"></a>
## 3. 原理一：训练脚本先把超参数变成 `args`

### 原理讲解

训练脚本不是一上来就训练模型，而是先把所有会影响训练行为的参数集中放到 `args` 里。

这里的参数可以分成四类：

```text
模型尺寸：hidden_size, num_hidden_layers, use_moe
数据形状：data_path, max_seq_len, batch_size
优化行为：learning_rate, epochs, accumulation_steps, grad_clip
保存/恢复：save_dir, save_weight, from_weight, from_resume
```

读这一段时，不需要背每个默认值。你要理解的是：后面的 `lm_config`、`PretrainDataset`、`optimizer`、checkpoint 路径，都来自这些参数。

### 源码证据 A：命令行参数定义

文件：`trainer/train_pretrain.py:83-107`

看它是为了理解：训练脚本暴露了哪些旋钮，以及这些旋钮后面会控制什么。

源码：

```python
parser = argparse.ArgumentParser(description="MiniMind Pretraining")
parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")
parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
args = parser.parse_args()
```

这段代码说明：

- `hidden_size` 和 `num_hidden_layers` 后面会进入模型配置。
- `data_path` 和 `max_seq_len` 后面会进入数据集。
- `from_weight='none'` 表示预训练默认从随机初始化开始。
- `from_resume` 不是加载普通权重，而是恢复训练状态。

### 理解到这一步就够

你应该能看到一个参数，并判断它大概会影响哪一部分：

```text
--max_seq_len 影响 Dataset 产出的序列长度；
--hidden_size 影响模型参数量；
--from_weight 影响是否从已有 .pth 加载模型权重。
```

暂时不用看：

- `wandb_project` 怎么上报日志。
- `use_compile` 的性能优化细节。
- DDP 多卡启动参数。

<a id="l07-config"></a>
## 4. 原理二：`MiniMindConfig` 决定模型骨架

### 原理讲解

`MiniMindConfig` 可以理解为模型的结构说明书。

它不是权重本身，而是告诉模型：

```text
hidden_size 多宽
num_hidden_layers 有多少层
use_moe 是否用 MoE
vocab_size 输出词表多大
num_attention_heads 有多少个 attention head
```

在 `train_pretrain.py` 里，脚本只显式传了 `hidden_size`、`num_hidden_layers`、`use_moe`。其他配置使用 `MiniMindConfig` 的默认值。也就是说，训练脚本没有把所有模型细节都写在自己这里，而是把大部分结构默认值交给配置类。

### 源码证据 A：训练脚本创建配置

文件：`trainer/train_pretrain.py:114-117`

看它是为了理解：命令行参数如何变成模型配置对象。

源码：

```python
os.makedirs(args.save_dir, exist_ok=True)
lm_config = MiniMindConfig(
    hidden_size=args.hidden_size,
    num_hidden_layers=args.num_hidden_layers,
    use_moe=bool(args.use_moe)
)
ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
```

这段代码说明：

- `lm_config` 是后面创建模型和 checkpoint 命名的共同依据。
- `hidden_size`、`num_hidden_layers`、`use_moe` 是这一节需要先盯住的三个结构参数。
- 如果 `from_resume==1`，脚本会先尝试读取 resume checkpoint。

### 源码证据 B：配置类的默认值

文件：`model/model_minimind.py:10-27`

看它是为了理解：训练脚本没有传的结构参数从哪里来。

源码：

```python
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
```

这段代码说明：

- 默认词表大小是 `6400`，要和 tokenizer 的词表规模匹配。
- 默认 `head_dim = hidden_size // num_attention_heads`。
- `intermediate_size` 是由 `hidden_size` 推出来的 FFN 中间维度。

### 理解到这一步就够

你应该能说清楚：

```text
args 只是命令行参数；
lm_config 才是模型构造真正使用的结构配置。
```

暂时不用看：

- RoPE/YaRN 的具体数学。
- MoE 的 router 和 expert 细节。
- 每一层参数量的精确计算。

<a id="l07-init-model"></a>
## 5. 原理三：`init_model` 同时准备 tokenizer 和 model

### 原理讲解

训练时需要两个东西同时存在：

```text
tokenizer: 把文本变成 input_ids
model: 接收 input_ids，输出 logits/loss
```

`init_model` 把这两个对象一起返回，是因为 Dataset 要用 tokenizer，而训练 step 要用 model。

这里还有一个关键分支：

```text
from_weight == 'none'    -> 从随机初始化开始
from_weight != 'none'    -> 从 out/{from_weight}_{hidden_size}.pth 加载权重
```

预训练默认 `from_weight='none'`，所以是从头学语言规律。SFT 阶段通常会从预训练权重继续。

### 源码证据 A：训练入口调用 `init_model`

文件：`trainer/train_pretrain.py:133-138`

看它是为了理解：模型、tokenizer、dataset、optimizer 在哪里被创建。

源码：

```python
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
```

这段代码说明：

- `init_model` 先给训练准备模型和 tokenizer。
- `PretrainDataset` 依赖 tokenizer，因为它要把文本编码成 token id。
- `optimizer` 直接绑定 `model.parameters()`，也就是会更新模型所有可训练参数。

### 源码证据 B：`init_model` 内部做了什么

文件：`trainer/trainer_utils.py:119-131`

看它是为了理解：加载 tokenizer、创建模型、可选加载权重这三步的顺序。

源码：

```python
def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model', save_dir='../out', device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)

    if from_weight!= 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer
```

这段代码说明：

- tokenizer 默认从 `../model` 目录加载。
- 模型类是 `MiniMindForCausalLM`。
- `from_weight` 决定是否去 `../out` 找 `.pth` 权重。
- 返回前会打印总参数量和可训练参数量。

### 理解到这一步就够

你应该能回答：

```text
为什么本地缺少 out/pretrain_768.pth 时，from_weight='pretrain' 会失败？
因为 init_model 会拼出这个路径并 torch.load。
```

但在 `train_pretrain.py` 默认参数里：

```text
from_weight='none'
```

所以预训练默认不会加载已有权重。

暂时不用看：

- HuggingFace `AutoTokenizer` 的内部加载细节。
- `strict=False` 的所有兼容场景。

<a id="l07-pretrain-dataset"></a>
## 6. 原理四：`PretrainDataset` 决定训练样本长什么样

### 原理讲解

训练脚本本身不关心原始 jsonl 里文本怎么切。它只要求 DataLoader 每次给它：

```text
input_ids
labels
```

这两个张量都来自 `PretrainDataset`。

对预训练来说，样本逻辑很直接：

```text
读取 text
-> tokenizer 编码
-> 手动加 BOS/EOS
-> pad 到 max_length
-> labels = input_ids.clone()
-> padding label 改成 -100
```

所以预训练阶段学的是：给定前面的 token，预测下一个 token。普通文本区域基本都参与 loss，padding 不参与。

### 源码证据 A：`PretrainDataset.__getitem__`

文件：`dataset/lm_dataset.py:47-55`

看它是为了理解：一条 jsonl 文本如何变成训练脚本需要的 `(input_ids, labels)`。

源码：

```python
def __getitem__(self, index):
    sample = self.samples[index]
    tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
    tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
    input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    labels = input_ids.clone()
    labels[input_ids == self.tokenizer.pad_token_id] = -100
    return input_ids, labels
```

这段代码说明：

- `input_ids` 和 `labels` 长度一样。
- `labels = input_ids.clone()` 是预训练目标的核心。
- `-100` 只用于忽略 padding token。
- Dataset 返回的就是训练循环里 `for step, (input_ids, labels)` 接到的东西。

### 理解到这一步就够

你应该能说清楚：

```text
train_pretrain.py 没有直接读取 jsonl 内容；
它通过 PretrainDataset 间接得到已经编码好的 input_ids / labels。
```

暂时不用看：

- `datasets.load_dataset` 的缓存实现。
- 大数据集的分片和流式加载。

<a id="l07-train-epoch"></a>
## 7. 原理五：`train_epoch` 是真正训练一步的地方

### 原理讲解

训练循环里最重要的动作只有五个：

```text
1. 从 loader 取 input_ids / labels
2. 把 batch 放到 device
3. forward 得到 loss
4. backward 计算梯度
5. optimizer.step 更新参数
```

你读训练循环时，不要先陷入日志、保存、wandb。先找这五个动作。

这里还有两个训练技巧：

```text
loss = loss / accumulation_steps
```

表示梯度累积时，每个小 batch 的 loss 先缩小，避免累积后梯度过大。

```text
clip_grad_norm_
```

表示更新前裁剪梯度，防止梯度爆炸。

### 源码证据 A：forward、loss、backward、step

文件：`trainer/train_pretrain.py:24-49`

看它是为了理解：一次训练 step 到底发生在哪几行。

源码：

```python
def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
```

这段代码说明：

- `input_ids` 和 `labels` 来自 DataLoader。
- `model(input_ids, labels=labels)` 会触发模型内部的 causal LM loss。
- `loss.backward()` 不是直接调用，而是包在 `scaler.scale(loss).backward()` 里，兼容混合精度。
- `optimizer.step()` 也通过 `scaler.step(optimizer)` 执行。

### 源码证据 B：模型 forward 计算 causal LM loss

文件：`model/model_minimind.py:245-253`

看它是为了理解：训练循环里的 `res.loss` 从哪里来。

源码：

```python
def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
    hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :])
    loss = None
    if labels is not None:
        x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
        loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
    return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
```

这段代码说明：

- `labels is not None` 时才会计算 loss。
- `logits[..., :-1, :]` 对齐 `labels[..., 1:]`，这就是 next-token prediction。
- `ignore_index=-100` 会忽略 Dataset 标出来的不训练位置。

### 理解到这一步就够

你应该能把一次训练 step 讲成：

```text
batch 进模型
-> logits 和 loss 出来
-> loss 反传成梯度
-> optimizer 根据梯度改参数
```

暂时不用看：

- AMP 的数值缩放细节。
- `aux_loss` 在 MoE 里的具体来源。
- `get_lr` 的完整调度曲线。

<a id="l07-checkpoint"></a>
## 8. 原理六：checkpoint 分两类

### 原理讲解

训练脚本会保存两类文件，它们用途不同：

```text
out/pretrain_768.pth
```

这是普通模型权重，后面 SFT 或推理可以拿来加载。

```text
checkpoints/pretrain_768_resume.pth
```

这是断点续训文件，除了模型权重，还保存 optimizer、epoch、step、scaler 等训练状态。

初学时最容易混淆的是：

```text
from_weight 加载 out 里的普通权重
from_resume 加载 checkpoints 里的续训状态
```

这两个不是一回事。

### 源码证据 A：训练过程中保存普通权重和 resume checkpoint

文件：`trainer/train_pretrain.py:61-70`

看它是为了理解：一次保存动作会写出哪两类文件。

源码：

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

这段代码说明：

- `torch.save(..., ckp)` 保存普通 `.pth` 权重。
- `lm_checkpoint(...)` 另外保存断点续训信息。
- 保存路径里的 `768` 来自 `lm_config.hidden_size`。

### 源码证据 B：resume checkpoint 的内容

文件：`trainer/trainer_utils.py:73-116`

看它是为了理解：为什么 resume 文件不只是模型权重。

源码：

```python
resume_data = {
    'model': state_dict,
    'optimizer': optimizer.state_dict(),
    'epoch': epoch,
    'step': step,
    'world_size': dist.get_world_size() if dist.is_initialized() else 1,
    'wandb_id': wandb_id
}
...
torch.save(resume_data, resume_tmp)
os.replace(resume_tmp, resume_path)
...
if os.path.exists(resume_path):
    ckp_data = torch.load(resume_path, map_location='cpu')
    return ckp_data
return None
```

这段代码说明：

- resume 文件保存了 `optimizer`，所以能恢复优化器状态。
- resume 文件保存了 `epoch` 和 `step`，所以能从中断位置继续。
- 普通 `.pth` 通常只够加载模型参数，不够精确恢复训练现场。

### 理解到这一步就够

你应该能区分：

```text
我要继续训练中断任务：看 from_resume / checkpoints。
我要基于已有模型开始新阶段：看 from_weight / out。
```

暂时不用看：

- 多卡 world_size 变化时 step 如何换算。
- `.tmp` 文件和 `os.replace` 的原子保存细节。

<a id="l07-experiment"></a>
## 9. 实验验证

### 实验 A：CPU 跑一个 tiny pretrain step

这个实验验证：

```text
MiniMindConfig
-> MiniMindForCausalLM
-> PretrainDataset
-> DataLoader batch
-> forward loss
-> backward
-> optimizer.step
```

它不会训练出有效模型，只用来确认一次训练 step 的对象流转。

运行：

```bash
cd /home/sun/minimind
PYTHONDONTWRITEBYTECODE=1 python course/labs/trace_pretrain_step.py
```

记录：

```text
input_ids.shape =
labels.shape =
logits.shape =
loss =
grad_norm_before_clip =
max_first_param_delta_after_step =
```

你应该看到：

```text
input_ids.shape = (2, 48)
labels.shape = (2, 48)
logits.shape = (2, 48, 6400)
loss 是一个正数
max_first_param_delta_after_step 大于 0
```

最后一项大于 0，说明 optimizer 确实更新了模型参数。

### 实验 B：改变模型尺寸看参数量变化

运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python course/labs/trace_pretrain_step.py --hidden_size 96 --num_hidden_layers 3
```

记录：

```text
params =
logits.shape =
loss =
```

你应该看到：

- `params` 变大。
- `logits.shape` 的最后一维仍然是 `vocab_size`。
- `hidden_size` 影响模型内部宽度，但最终输出仍要预测词表里的 token。

<a id="l07-check"></a>
## 10. 本节检查

如果你真懂了本节，应该能不看答案说清楚：

1. `args`、`lm_config`、`model` 三者分别是什么关系。
2. `train_pretrain.py` 默认为什么是从头预训练，而不是加载已有权重。
3. `PretrainDataset` 返回的 `(input_ids, labels)` 在训练循环哪一行被接住。
4. `res.loss` 是训练脚本自己算的，还是模型 forward 里算的。
5. `from_weight` 和 `from_resume` 分别会去找哪类文件。
6. 为什么 `out/pretrain_768.pth` 和 `checkpoints/pretrain_768_resume.pth` 不能简单当成同一个东西。

<a id="l07-next"></a>
## 11. 下一课

第 8 课进入 `MiniMindConfig` 与参数规模：我们会把 `hidden_size`、`num_hidden_layers`、`vocab_size`、`num_attention_heads` 和参数量关系讲清楚。
