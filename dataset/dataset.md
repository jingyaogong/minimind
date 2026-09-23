# MiniMind 数据集指南 / Dataset Guide

[中文](#user-content-chinese) | [English](#user-content-english)

<a id="chinese"></a>

## 中文

本目录存放 MiniMind 各训练阶段使用的最终 JSONL 文件。官方发布的数据已经完成收集、蒸馏、清洗、去重、长度控制与格式统一，可直接由 `dataset/lm_dataset.py` 中的数据加载器读取。

### 下载与最小复现

数据集可从 [ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files) 或 [Hugging Face](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main) 按文件下载，无需克隆全部数据。

将文件放在当前目录：

```text
dataset/
├── pretrain_t2t_mini.jsonl
├── sft_t2t_mini.jsonl
├── pretrain_t2t.jsonl
├── sft_t2t.jsonl
├── dpo.jsonl
├── rlaif.jsonl
├── agent_rl.jsonl
└── agent_rl_math.jsonl
```

只想快速跑通从预训练到对话模型的流程时，下载以下两个文件即可：

- `pretrain_t2t_mini.jsonl`
- `sft_t2t_mini.jsonl`

然后从仓库根目录运行：

```bash
cd trainer
python train_pretrain.py
python train_full_sft.py
```

### 文件与训练脚本对应关系

| 文件 | 用途 | 默认使用者 |
| --- | --- | --- |
| `pretrain_t2t_mini.jsonl` | 快速预训练 | `train_pretrain.py` |
| `pretrain_t2t.jsonl` | 完整 MiniMind-3 预训练 | 通过 `train_pretrain.py --data_path ...` 指定 |
| `sft_t2t_mini.jsonl` | 快速 SFT、蒸馏及 LoRA 自定义数据格式参考 | `train_full_sft.py`、`train_distillation.py` |
| `sft_t2t.jsonl` | 完整 MiniMind-3 SFT | 通过 `train_full_sft.py --data_path ...` 指定 |
| `dpo.jsonl` | 偏好优化 | `train_dpo.py` |
| `rlaif.jsonl` | PPO、GRPO 与 CISPO | `train_ppo.py`、`train_grpo.py` |
| `agent_rl.jsonl` | 多轮 Tool-Use Agentic RL | `train_agent.py` |
| `agent_rl_math.jsonl` | 带可验证最终答案的数学 Agentic RL | 通过 `train_agent.py --data_path ...` 指定 |

上表中的训练脚本都支持 `--data_path`，因此也可直接指向自己的 JSONL 文件。

### 最终数据格式

预训练数据每行包含一个 `text` 字段：

```jsonl
{"text": "Transformer 通过自注意力机制建模上下文关系。"}
```

SFT、RLAIF 与常规对话数据使用 `conversations` 数组：

```json
{
  "conversations": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！"}
  ]
}
```

DPO 数据分别提供偏好回复与非偏好回复：

```json
{
  "chosen": [
    {"role": "user", "content": "Q"},
    {"role": "assistant", "content": "good answer"}
  ],
  "rejected": [
    {"role": "user", "content": "Q"},
    {"role": "assistant", "content": "bad answer"}
  ]
}
```

Agentic RL 数据还可在 system 消息中包含序列化的 `tools`，并用顶层 `gt` 字段提供可验证目标。具体解析逻辑以 [`lm_dataset.py`](./lm_dataset.py) 为准。

### 来源、处理与自定义数据

当前公开版本使用的主要社区来源包括：

- [匠数大模型数据集](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data)
- [Magpie-Align](https://www.modelscope.cn/organization/Magpie-Align)
- [R1-Distill-SFT](https://www.modelscope.cn/datasets/AI-ModelScope/R1-Distill-SFT)
- [COIG](https://huggingface.co/datasets/BAAI/COIG)
- [Step-3.5-Flash-SFT](https://huggingface.co/datasets/stepfun-ai/Step-3.5-Flash-SFT)
- [DPO-En-Zh-20k](https://huggingface.co/datasets/llamafactory/DPO-En-Zh-20k)

本仓库目前提供训练所需的最终数据、格式说明和加载器；原始数据的完整收集、蒸馏与清洗流水线并未作为可执行脚本包含在本目录中。准备自定义数据时应：

1. 确认每个来源允许当前用途与再分发方式，并保留来源和许可证记录。
2. 完成去重、质量过滤、隐私信息清理与长度控制。
3. 转换为上面的最终 JSONL 格式。
4. 先用小样本运行对应训练脚本，确认 schema 和 chat template 正确，再开始完整训练。

数据来源、推荐组合、Tool Calling 格式与 token 长度说明以仓库根目录的[中文 README](../README.md#-数据介绍)为准。

<a id="english"></a>

## English

This directory stores the final JSONL files consumed by each MiniMind training stage. The published datasets have already gone through collection, distillation, cleaning, deduplication, length control, and format normalization, and can be read directly by the loaders in `dataset/lm_dataset.py`.

### Download and minimal reproduction

Download individual files from [ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files) or [Hugging Face](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main). You do not need to clone the full dataset repository.

Place the downloaded files in this directory:

```text
dataset/
├── pretrain_t2t_mini.jsonl
├── sft_t2t_mini.jsonl
├── pretrain_t2t.jsonl
├── sft_t2t.jsonl
├── dpo.jsonl
├── rlaif.jsonl
├── agent_rl.jsonl
└── agent_rl_math.jsonl
```

For the quickest pretraining-to-chat-model reproduction, only these two files are required:

- `pretrain_t2t_mini.jsonl`
- `sft_t2t_mini.jsonl`

Then run from the repository root:

```bash
cd trainer
python train_pretrain.py
python train_full_sft.py
```

### Dataset-to-trainer map

| File | Purpose | Default consumer |
| --- | --- | --- |
| `pretrain_t2t_mini.jsonl` | Quick pretraining | `train_pretrain.py` |
| `pretrain_t2t.jsonl` | Full MiniMind-3 pretraining | Pass with `train_pretrain.py --data_path ...` |
| `sft_t2t_mini.jsonl` | Quick SFT, distillation, and the reference schema for custom LoRA data | `train_full_sft.py`, `train_distillation.py` |
| `sft_t2t.jsonl` | Full MiniMind-3 SFT | Pass with `train_full_sft.py --data_path ...` |
| `dpo.jsonl` | Preference optimization | `train_dpo.py` |
| `rlaif.jsonl` | PPO, GRPO, and CISPO | `train_ppo.py`, `train_grpo.py` |
| `agent_rl.jsonl` | Multi-turn Tool-Use Agentic RL | `train_agent.py` |
| `agent_rl_math.jsonl` | Math Agentic RL with verifiable final answers | Pass with `train_agent.py --data_path ...` |

Every trainer listed above accepts `--data_path`, so it can also read a custom JSONL file directly.

### Final data schemas

Each pretraining line contains a `text` field:

```jsonl
{"text": "Transformers model context with self-attention."}
```

SFT, RLAIF, and regular conversation data use a `conversations` array:

```json
{
  "conversations": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hello!"}
  ]
}
```

DPO data provides preferred and rejected conversations separately:

```json
{
  "chosen": [
    {"role": "user", "content": "Q"},
    {"role": "assistant", "content": "good answer"}
  ],
  "rejected": [
    {"role": "user", "content": "Q"},
    {"role": "assistant", "content": "bad answer"}
  ]
}
```

Agentic RL records may also contain serialized `tools` on the system message and a top-level `gt` field for verifiable targets. See [`lm_dataset.py`](./lm_dataset.py) for the authoritative parsing behavior.

### Sources, processing, and custom data

Major community sources used by the current published datasets include:

- [Craftsman LLM Dataset](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data)
- [Magpie-Align](https://www.modelscope.cn/organization/Magpie-Align)
- [R1-Distill-SFT](https://www.modelscope.cn/datasets/AI-ModelScope/R1-Distill-SFT)
- [COIG](https://huggingface.co/datasets/BAAI/COIG)
- [Step-3.5-Flash-SFT](https://huggingface.co/datasets/stepfun-ai/Step-3.5-Flash-SFT)
- [DPO-En-Zh-20k](https://huggingface.co/datasets/llamafactory/DPO-En-Zh-20k)

This repository currently provides the final training datasets, their schemas, and the loaders. The complete raw-data collection, distillation, and cleaning pipeline is not included here as an executable script. When preparing custom data:

1. Confirm that each source permits the intended use and redistribution, and retain provenance and license records.
2. Deduplicate, quality-filter, remove private information, and control sample lengths.
3. Convert records to the final JSONL schemas above.
4. Run the matching trainer on a small sample to validate the schema and chat template before starting a full training run.

For source details, recommended dataset combinations, Tool Calling schemas, and token-length guidance, see the [English README](../README_en.md#-data-introduction).
