# 模型训练

本页面介绍如何从 0 开始训练 MiniMind 语言模型。

## 📊 数据准备

### 1. 下载数据集

从 [ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files) 或 [HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind_dataset) 下载数据集。

创建 `./dataset` 目录并放入数据文件：

```bash
./dataset/
├── pretrain_hq.jsonl (1.6GB, ✨推荐)
├── sft_mini_512.jsonl (1.2GB, ✨推荐)
├── sft_512.jsonl (7.5GB)
├── sft_1024.jsonl (5.6GB)
├── sft_2048.jsonl (9GB)
├── dpo.jsonl (909MB)
├── r1_mix_1024.jsonl (340MB)
└── lora_*.jsonl
```

!!! tip "推荐组合"
    最快速度复现：`pretrain_hq.jsonl` + `sft_mini_512.jsonl`
    
    **单卡 3090 仅需 2 小时 + 3 块钱！**

### 2. 数据格式

**预训练数据** (`pretrain_hq.jsonl`):
```json
{"text": "如何才能摆脱拖延症？治愈拖延症并不容易..."}
```

**SFT 数据** (`sft_*.jsonl`):
```json
{
  "conversations": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！"}
  ]
}
```

## 🎯 训练流程

所有训练脚本位于 `./trainer` 目录。

### 1. 预训练（Pretrain）

预训练阶段让模型学习基础知识，目标是**学会词语接龙**。

```bash
cd trainer
python train_pretrain.py

# 多卡训练
torchrun --nproc_per_node 2 train_pretrain.py
```

输出权重：`./out/pretrain_*.pth`

!!! info "训练时长"
    - MiniMind2-Small (26M): ~1.1h (单卡 3090)
    - MiniMind2 (104M): ~3.9h (单卡 3090)

### 2. 监督微调（SFT）

SFT 阶段让模型学习对话方式，适应聊天模板。

```bash
python train_full_sft.py

# 多卡训练
torchrun --nproc_per_node 2 train_full_sft.py
```

输出权重：`./out/full_sft_*.pth`

!!! info "训练时长"
    - MiniMind2-Small: ~1h (使用 sft_mini_512)
    - MiniMind2: ~3.3h (使用 sft_mini_512)

### 3. LoRA 微调（可选）

LoRA 是一种参数高效的微调方法，适合领域适配。

```bash
python train_lora.py
```

**应用场景**：
- 医疗问答：使用 `lora_medical.jsonl`
- 自我认知：使用 `lora_identity.jsonl`

输出权重：`./out/lora/lora_*.pth`

### 4. DPO 强化学习（可选）

DPO 用于优化模型回复质量，使其更符合人类偏好。

```bash
python train_dpo.py
```

输出权重：`./out/rlhf_*.pth`

### 5. 推理模型蒸馏（可选）

蒸馏 DeepSeek-R1 的推理能力。

```bash
python train_distill_reason.py
```

输出权重：`./out/reason_*.pth`

## 📈 模型结构

MiniMind 使用 Transformer Decoder-Only 结构（类似 Llama3）：

![structure](images/LLM-structure.png)

### 模型参数配置

| Model Name | params | d_model | n_layers | kv_heads | q_heads |
|------------|--------|---------|----------|----------|---------|
| MiniMind2-Small | 26M | 512 | 8 | 2 | 8 |
| MiniMind2-MoE | 145M | 640 | 8 | 2 | 8 |
| MiniMind2 | 104M | 768 | 16 | 2 | 8 |

## 🧪 测试模型

```bash
# model_mode: 0=pretrain, 1=sft, 2=rlhf, 3=reason
python eval_model.py --model_mode 1

# 测试 LoRA 模型
python eval_model.py --lora_name 'lora_medical' --model_mode 2
```

## 🔧 多卡训练

### DDP 方式

```bash
torchrun --nproc_per_node N train_xxx.py
```

### DeepSpeed 方式

```bash
deepspeed --master_port 29500 --num_gpus=N train_xxx.py
```

### Wandb 监控

```bash
# 需要先登录
wandb login

# 启用 wandb
torchrun --nproc_per_node N train_xxx.py --use_wandb
```

## 💰 训练成本

基于单卡 NVIDIA 3090：

| 数据集组合 | 时长 | 成本 | 效果 |
|-----------|------|------|------|
| pretrain_hq + sft_mini_512 | 2.1h | ≈2.73￥ | 😊😊 基础对话 |
| 完整数据集 (MiniMind2-Small) | 38h | ≈49.61￥ | 😊😊😊😊😊😊 完整能力 |
| 完整数据集 (MiniMind2) | 122h | ≈158.6￥ | 😊😊😊😊😊😊😊😊 最强性能 |

!!! success "极速复现"
    使用 `pretrain_hq` + `sft_mini_512`，单卡 3090 仅需 **2 小时 + 3 块钱**即可训练出能对话的 ChatBot！

## 📝 常见问题

- **显存不足**：减小 `batch_size` 或使用 DeepSpeed
- **训练不收敛**：调整学习率或检查数据质量
- **多卡训练报错**：确保所有卡都可见且 CUDA 版本一致

