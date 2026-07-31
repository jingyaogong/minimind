# 03-进阶 / 01 - LoRA 微调

> **TL;DR**：minimind 的 LoRA 是**从 0 手写**的（不依赖 `peft` 库），通过给 `nn.Linear` 挂一个低秩增量分支 `BA` 实现垂域微调。训练只动 BA，基座冻结，最酷的是能"加法叠加"多个垂域能力。
> **心理学技巧**：**新旧结合** —— LoRA 让你保留已经训好的通用能力（旧），再叠加一个新的垂域能力（新），这种"加法"而非"覆盖"的设计能极大降低学习的心理负担。

## ✅ 你将能

- 解释 LoRA 为什么用"低秩增量分支"而不是直接训全参
- 读懂 `model/model_lora.py` 全部 65 行（apply / save / load / merge）
- 用医疗数据训练一个 `lora_medical` 增量权重
- 把 LoRA 合并回基座，得到一个完整 `.pth`
- 知道 LoRA 适合什么场景、不适合什么场景

---

## 步骤 0：先理解原理（5 分钟）

### 0.1 全参微调的痛

SFT 阶段全参微调时，模型每个权重 W 都要更新：

```
W' = W + ΔW
```

`ΔW` 是和 W 一样大的矩阵。minimind-3 64M 看似不大，但要训一个医疗 LoRA 还要存一份 optimizer state（Adam = 2× 参数量的 momentum/variance），全参微调成本依旧不低，而且**训完会"覆盖"通用能力**（catastrophic forgetting）。

### 0.2 LoRA 的核心思想

论文 [Hu et al., 2021] 的关键观察：**微调产生的 ΔW 通常是低秩的**。所以不直接学 ΔW，而是把它分解成两个小矩阵的乘积：

```
W' = W + B·A      ← B: [d, r]   A: [r, d]   r << d
```

- 原来的 `W` 完全冻结（`requires_grad=False`）
- 新增 `B` 和 `A` 是仅有的可训练参数，参数量从 `d×d` 降到 `2×d×r`
- 训练时前向是 `forward(x) = W·x + B·A·x`
- 推理时可以把 `BA` 加回 W，得到合并后的 W'，**零额外开销**

```
            ┌─────────── 冻结, 不学 ───────────┐
            │                                  │
   x ───────┼──→ W x ─────────────────┐        │
            │                         ▼        │
            └──→ A x ──→ B (A·x) ──→ (+) ──→ out
                └── 新增, 只学这两个 ──┘
```

### 0.3 初始化的小技巧（关键）

看 `model/model_lora.py:13-15`：

```python
self.A.weight.data.normal_(mean=0.0, std=0.02)   # A 高斯初始化
self.B.weight.data.zero_()                        # B 全 0 初始化
```

**为什么 B 初始化为 0？** 训练第 0 步时 `B·A = 0`，所以 `W' = W`，模型从基座状态**无损起步**。如果 B 也用高斯初始化，第 0 步就把基座权重打乱了，相当于从噪声重新训。这是 LoRA 能稳定训练的关键。

---

## 步骤 1：读懂 model_lora.py（65 行）

### 1.1 LoRA 模块

`model/model_lora.py:6-18`：

```python
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank
        self.A = nn.Linear(in_features, rank, bias=False)      # [d → r]
        self.B = nn.Linear(rank, out_features, bias=False)     # [r → d]
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        self.B.weight.data.zero_()                              # ← 关键

    def forward(self, x):
        return self.B(self.A(x))                                # B·A·x
```

### 1.2 把 LoRA 挂到 Linear 上（apply_lora）

`model/model_lora.py:21-32`：

```python
def apply_lora(model, rank=16):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.in_features == module.out_features:
            lora = LoRA(module.in_features, module.out_features, rank=rank).to(model.device)
            setattr(module, "lora", lora)                       # 给 Linear 加一个 .lora 属性
            original_forward = module.forward

            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)                   # W·x + B·A·x

            module.forward = forward_with_lora                  # monkey-patch forward
```

**关键细节**：只对 `in_features == out_features` 的 Linear 挂 LoRA。看 `model/model_minimind.py`：
- `Attention.q_proj / k_proj / v_proj / o_proj`：`hidden → hidden`（方阵）✅ 挂
- `FeedForward.gate_proj / up_proj / down_proj`：`hidden → intermediate`（非方阵）❌ 不挂

> 也就是说默认 LoRA 只挂在 attention 的 4 个投影矩阵上（`q_proj/k_proj/v_proj/o_proj`）。这是 LoRA 论文里效果最好、显存最省的配置。任务里的"7 个目标模块"通常指含 FFN 的 `gate_proj/up_proj/down_proj`，但在 minimind 的默认实现里只对方阵挂。

**为什么用 monkey-patch `module.forward`？** 因为 `nn.Linear.forward` 是 C++ 实现的，没法用普通的"继承然后重写"。直接把实例的 `forward` 替换成新的 Python 函数，最轻量。这也是 `train_lora.py:166` 提示 `torch.compile` 不兼容的原因（compile 也 patch forward，会冲突）。

### 1.3 保存 / 加载 / 合并

| 函数 | 位置 | 作用 |
|---|---|---|
| `save_lora` | `model/model_lora.py:45` | 只保存 `.lora.*` 的 B/A 权重 |
| `load_lora` | `model/model_lora.py:35` | 把保存的 B/A 载回挂了 LoRA 的模型 |
| `merge_lora` | `model/model_lora.py:56` | `W_new = W + B·A`，丢弃 LoRA 模块，导出基座格式 |

`merge_lora` 的核心一行（`model/model_lora.py:64`）：

```python
state_dict[f'{name}.weight'] += (module.lora.B.weight.data @ module.lora.A.weight.data).cpu().half()
```

合并后，模型结构和 `full_sft` 完全一样，可以直接 `eval_llm.py --weight merge_medical` 推理，也能继续做 DPO/RL 等"基座 + 后训练"。

---

## 步骤 2：准备垂域数据

### 2.1 数据格式

LoRA 训练用 `SFTDataset`（`dataset/lm_dataset.py:58`），格式与 SFT 完全一样：

```jsonl
{"conversations": [
    {"role": "user", "content": "我最近总是头疼，是什么原因？"},
    {"role": "assistant", "content": "头疼的原因很多，常见的有紧张性头痛、偏头痛、高血压等..."}
]}
```

> labels 只在 assistant 内容上算 loss（见 [01-理论/03-训练数据格式](../01-理论/03-训练数据格式.md)），所以 LoRA 训练的实质是"在保留通用能力的前提下，让模型在垂域 assistant 回复分布上微调 B/A"。

### 2.2 自带的垂域数据

minimind 在 `dataset/` 下提供了若干垂域数据集：

| 文件 | 垂域 | 用途 |
|---|---|---|
| `lora_medical.jsonl` | 医疗问答 | 训练医疗助手 |
| `lora_identity.jsonl` | 自我认知 | 让模型回答"你是谁" |
| `sft_t2t_mini.jsonl` | 通用对话 | 参考格式自己造数据 |

> 下载数据：`modelscope download --dataset gongjy/minimind_dataset --local_dir ./dataset --include "lora_*.jsonl"`

### 2.3 自己造一条 LoRA 数据

举例：让 minimind 学会你的名字。

```bash
cat > dataset/lora_identity.jsonl <<'EOF'
{"conversations":[{"role":"user","content":"你叫什么名字？"},{"role":"assistant","content":"我叫 TJK-minimind，由学习者自己训练。"}]}
{"conversations":[{"role":"user","content":"你是谁？"},{"role":"assistant","content":"我是 TJK 训练的 minimind，用于学习 LLM 原理。"}]}
EOF
```

✅ **验证**：

```bash
wc -l dataset/lora_identity.jsonl
```

预期输出：`2 dataset/lora_identity.jsonl`

---

## 步骤 3：训练 LoRA

> 前置：你已经按 [02-实战/02-指令微调SFT](../02-实战/02-指令微调SFT.md) 训出 `out/full_sft_768.pth`。LoRA 是在 SFT 之上叠加的。

```bash
cd trainer
python train_lora.py \
  --lora_name lora_medical \
  --from_weight full_sft \
  --data_path ../dataset/lora_medical.jsonl \
  --epochs 10 \
  --batch_size 32 \
  --learning_rate 1e-4
```

启动后会先打印参数统计（`trainer/train_lora.py:133-137`）：

```
LLM 总参数量: 64.512 M
LoRA 参数量: 0.295 M
LoRA 参数占比: 0.46%
```

只训 0.46% 的参数，其他 99.5% 全冻结。这就是 LoRA 的"少即是多"。

✅ **验证**：训练每隔 `save_interval`（默认 1000 步）会在 `out/` 下生成：

```
out/lora_medical_768.pth          ← 只含 B/A 增量权重（约 1MB）
checkpoints/lora_medical_768_resume.pth   ← 完整续训状态
```

### 3.1 关键超参

| 参数 | 默认 | 作用 |
|---|---|---|
| `--lora_name` | `lora_medical` | 输出权重前缀名 |
| `--from_weight` | `full_sft` | 基于哪个 `.pth` 训练 |
| `--data_path` | `../dataset/lora_medical.jsonl` | 垂域数据 |
| `--epochs` | 10 | LoRA 数据通常小，可以多跑几轮 |
| `--learning_rate` | 1e-4 | LoRA 学习率比全参高 1-2 个数量级（因为只动增量分支） |
| `--from_resume 1` | 0 | 中断后续训，自动从 `checkpoints/` 恢复（见 `train_lora.py:112`） |

### 3.2 多卡加速

```bash
cd trainer
torchrun --nproc_per_node 2 train_lora.py --lora_name lora_medical
```

> DDP 模式下 `save_lora` 会自动去掉 `module.` 前缀（`model/model_lora.py:50`），无需手动处理。

---

## 步骤 4：用 LoRA 推理

LoRA 训完不是新模型，而是"基座 + 增量"。推理时需要同时加载两者：

```bash
cd /path/to/minimind
python eval_llm.py --load_from ./model --weight full_sft --lora_weight lora_medical
```

- `--weight full_sft`：基座权重（`out/full_sft_768.pth`）
- `--lora_weight lora_medical`：LoRA 增量（`out/lora_medical_768.pth`）

✅ **验证**：输入"我最近总是头疼"，模型回复应明显偏向医疗领域，而不是泛泛回答。

> **多 LoRA 叠加**：理论上可以同时挂多个 LoRA（叠加多个 BA），但 minimind 的 `apply_lora` 当前只支持单 LoRA。想叠加需自己改 `forward_with_lora`。

---

## 步骤 5：把 LoRA 合并回基座

如果要把 LoRA 模型部署成独立 `.pth`（不带 lora 依赖），用 `scripts/convert_model.py:105`：

```python
def convert_merge_base_lora(base_torch_path, lora_path, merged_torch_path):
    lm_model = MiniMindForCausalLM(lm_config).to(device)
    state_dict = torch.load(base_torch_path, map_location=device)
    lm_model.load_state_dict(state_dict, strict=False)
    apply_lora(lm_model)                            # 先挂上空的 LoRA
    merge_lora(lm_model, lora_path, merged_torch_path)  # 载入 B/A 并合并
```

执行（编辑 `scripts/convert_model.py:137-140` 取消注释后）：

```bash
cd scripts
python convert_model.py
```

会产出 `out/merge_medical_768.pth`，结构和 `full_sft_768.pth` 完全一样，但权重已含医疗 LoRA 的增量。

✅ **验证**：

```bash
python eval_llm.py --load_from ./model --weight merge_medical
```

输出与步骤 4 一致，但不再需要 `--lora_weight` 参数。

> **合并 vs 不合并**：合并后推理略快（少一次矩阵乘），但失去"叠加多个 LoRA"的灵活性。生产部署选合并，研究/调试选不合并。

---

## 🧯 踩坑提示

### Q1：`LoRA 参数量: 0.000 M`
检查 `apply_lora` 是否真的被调用。`train_lora.py:130` 调用了 `apply_lora(model)`，如果你删了这行就会 0 参数。也确认基座模型里确实有方阵 Linear（attention 投影）。

### Q2：训练 loss 不下降
LoRA 学习率比全参高 1-2 个数量级。默认 `1e-4`，若不下降试 `5e-4`；若爆炸试 `5e-5`。同时检查数据量是否太少（< 100 条）。

### Q3：合并后推理质量变差
合并是 `W + BA` 累加，数值精度问题。`merge_lora` 在 `model/model_lora.py:64` 用 `.half()` 输出，如果你训练用 `bfloat16` 合并用 `float16` 可能有微小偏差。一般可忽略，若敏感可改 `convert_model.py` 输出 `float32`。

### Q4：`--use_compile 1` 报错
`train_lora.py:165` 已自动关闭 torch.compile，因为 monkey-patch forward 与 compile 不兼容。不要强制开。

### Q5：多卡训练保存的权重带 `module.` 前缀
`save_lora` 已处理（`model/model_lora.py:50` 去前缀），但若你直接用 `torch.save(model.state_dict())` 会带前缀。务必用 `save_lora` 而不是手动 save。

### Q6：LoRA 训完反而变差
典型的 catastrophic forgetting 反向案例——`learning_rate` 太高或 `epochs` 太多导致 BA 过大，覆盖了基座能力。降学习率、降 epochs、或用更小的 `rank`。

---

## ✅ 本篇完成自检

<details>
<summary>点开自检（先想 30 秒）</summary>

1. LoRA 为什么把 B 初始化为 0、A 用高斯初始化？
   - 让训练第 0 步 `BA=0`，`W'=W`，模型从基座无损起步；A 用高斯是为了之后梯度有方向（A 不能全 0，否则梯度也是 0）。

2. minimind 的 LoRA 默认挂在哪些 Linear 上？为什么？
   - 只挂在 `in_features == out_features` 的方阵 Linear，即 attention 的 `q_proj/k_proj/v_proj/o_proj`。这是论文推荐的高性价比配置，省显存且效果稳定。

3. LoRA 推理时为什么说"零额外开销"？
   - 合并后 `W_new = W + BA`，模型结构和原始基座完全一样，没有新增 forward 分支，所以推理速度和基座一致。

4. `apply_lora` 用 monkey-patch `module.forward` 而不是继承 `nn.Linear`，为什么？
   - `nn.Linear.forward` 是 C++ 实现，无法直接重写；而且改模型结构会影响 DDP/compile 兼容性。monkey-patch 只改实例属性，最轻量。

5. LoRA 相比全参微调，最大的两个优势是什么？
   - (1) 训练参数量降到 1% 以下，省显存、省时间；(2) 基座冻结，可保留通用能力 + 叠加垂域，不易灾难性遗忘。

6. 什么时候 LoRA 不如全参微调？
   - 任务需要模型结构根本性改变（如新词表大幅扩展、大规模知识注入）时，低秩增量表达力不够，全参更合适。

7. `--from_resume 1` 是怎么工作的？
   - `train_lora.py:112` 调用 `lm_checkpoint` 自动检测 `checkpoints/lora_medical_768_resume.pth`，存在就加载 model/optimizer/scaler/epoch/step，跳过已训 step（见 `train_lora.py:154-161`、`SkipBatchSampler`）。

</details>

下一篇：[02-DPO与强化学习](./02-DPO与强化学习.md) —— 从偏好学习到 Agentic RL 的完整路线。
