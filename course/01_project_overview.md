# 第 1 课：项目地图与学习路线

## 目录

- [1. 本节目标](#l01-goals)
- [2. 背景概念](#l01-background)
- [3. 源码地图](#l01-source-map)
- [4. 主调用链](#l01-main-chain)
- [5. 当前本地状态](#l01-local-status)
- [6. 最小实验](#l01-experiment)
- [7. 常见坑](#l01-pitfalls)
- [8. 课后任务](#l01-homework)
- [9. 检查题](#l01-check)
- [10. 下一课预告](#l01-next)

<a id="l01-goals"></a>
## 1. 本节目标

学完本节，你应该能回答：

- MiniMind 这个项目到底包含哪些 LLM 训练阶段。
- 从数据到模型推理的主调用链是什么。
- 哪些文件是后续课程必须反复阅读的核心文件。
- 当前本地仓库还缺什么资源，为什么现在不能直接完整训练或推理。

<a id="l01-background"></a>
## 2. 背景概念

MiniMind 是一个小型 LLM 全流程教学项目。它不是只演示调用大模型 API，而是把模型结构、数据集、训练脚本、推理脚本、后训练算法都放在一个相对小的仓库里。

这个项目的主线可以分成五层：

```text
数据层：dataset/*.jsonl
编码层：model/tokenizer.json + chat template
模型层：model/model_minimind.py
训练层：trainer/train_*.py
推理部署层：eval_llm.py + scripts/*.py
```

本课程的学习方式是“源码驱动”：每节课都会围绕一条调用链读源码，然后做一个最小实验。

<a id="l01-source-map"></a>
## 3. 源码地图

本节先建立地图，不深入细节。

| 路径 | 作用 |
|---|---|
| `README.md` | 项目说明、数据下载、训练命令、模型发布信息 |
| `requirements.txt` | 依赖列表 |
| `model/model_minimind.py` | MiniMind 模型结构，后续最核心的源码 |
| `model/model_lora.py` | 手写 LoRA 注入、保存、加载、合并 |
| `model/tokenizer.json` | tokenizer 主文件 |
| `model/tokenizer_config.json` | chat template 和 special tokens 配置 |
| `dataset/lm_dataset.py` | Pretrain/SFT/DPO/RLAIF/Agent 数据集 |
| `trainer/train_pretrain.py` | 预训练入口 |
| `trainer/train_full_sft.py` | 全参数 SFT 入口 |
| `trainer/train_lora.py` | LoRA 训练入口 |
| `trainer/train_dpo.py` | DPO 偏好学习入口 |
| `trainer/train_ppo.py` | PPO 强化学习入口 |
| `trainer/train_grpo.py` | GRPO/CISPO 强化学习入口 |
| `trainer/train_agent.py` | Agentic RL / Tool Use 训练入口 |
| `trainer/trainer_utils.py` | 通用训练工具：lr、seed、checkpoint、模型加载 |
| `eval_llm.py` | CLI 推理入口 |
| `scripts/serve_openai_api.py` | OpenAI 兼容 API 服务 |
| `scripts/web_demo.py` | Streamlit WebUI |
| `scripts/convert_model.py` | PyTorch / Transformers 模型转换 |

<a id="l01-main-chain"></a>
## 4. 主调用链

### 4.1 预训练链路

```text
dataset/pretrain_t2t_mini.jsonl
-> PretrainDataset.__getitem__
-> tokenizer(text)
-> input_ids, labels
-> MiniMindForCausalLM.forward(input_ids, labels)
-> cross_entropy loss
-> AdamW
-> out/pretrain_768.pth
```

预训练的本质：普通文本的 next-token prediction。

### 4.2 SFT 链路

```text
dataset/sft_t2t_mini.jsonl
-> SFTDataset.__getitem__
-> tokenizer.apply_chat_template
-> generate_labels
-> input_ids, labels
-> MiniMindForCausalLM.forward(input_ids, labels)
-> 只对 assistant 回复计算 loss
-> out/full_sft_768.pth
```

SFT 的本质：把对话样本转成模型能学习的 next-token prediction，但只训练 assistant 回复部分。

### 4.3 推理链路

```text
用户输入
-> conversation
-> tokenizer.apply_chat_template
-> tokenizer(...)
-> model.generate(...)
-> tokenizer.decode(...)
-> 文本回复
```

推理时没有 labels，也没有 optimizer，只做自回归生成。

<a id="l01-local-status"></a>
## 5. 当前本地状态

这个仓库当前只有源码、tokenizer 和图片资源，尚未下载：

- 训练数据：例如 `pretrain_t2t_mini.jsonl`、`sft_t2t_mini.jsonl`。
- 模型权重：例如 `full_sft_768.pth` 或 transformers 格式的 `minimind-3` 目录。

因此现在直接运行默认训练或推理，大概率会遇到文件不存在：

```text
../dataset/pretrain_t2t_mini.jsonl
../dataset/sft_t2t_mini.jsonl
./out/full_sft_768.pth
./minimind-3
```

课程前期会用 tiny 数据和小模型参数学习流程，避免一开始依赖大数据和大权重。

<a id="l01-experiment"></a>
## 6. 最小实验

本节先做两个只读检查。

### 实验 A：列出核心源码

```bash
cd /home/sun/minimind
find model dataset trainer scripts -maxdepth 1 -type f | sort
```

你应该看到模型、数据集、训练、脚本四类文件。

### 实验 B：确认核心模型能实例化

```bash
cd /home/sun/minimind
PYTHONDONTWRITEBYTECODE=1 python course/labs/check_minimind_import.py
```

如果成功，会打印 torch/transformers/datasets 版本，以及一个 tiny MiniMind 的参数量。

<a id="l01-pitfalls"></a>
## 7. 常见坑

- README 里的默认命令默认需要数据或权重，源码仓库刚 clone 下来不等于能直接训练/推理。
- `SFT` 容易被写成 `STF`，后续统一用 `SFT`。
- 默认 hidden size 是 768，直接 CPU 训练会很慢；学习阶段用 128 或更小。
- `--use_wandb` 实际导入的是 `swanlab`，当前环境没装时不要打开。
- WebUI 需要 `streamlit`，当前环境未必已安装。

<a id="l01-homework"></a>
## 8. 课后任务

1. 用自己的话写出 MiniMind 的五层结构：数据层、编码层、模型层、训练层、推理部署层。
2. 解释 pretrain、SFT、推理三条链路中，哪一步开始出现差异。
3. 找到 `train_pretrain.py` 和 `train_full_sft.py` 中 parser 参数的不同点。

<a id="l01-check"></a>
## 9. 检查题

这些题只检查你是否建立了项目地图，不要求理解训练细节。

1. MiniMind 项目里，哪个文件负责模型结构？
2. 哪个目录主要放训练脚本？
3. 哪个文件是 CLI 推理入口？
4. 当前本地仓库为什么还不能直接完整训练或推理？
5. pretrain、SFT、推理分别大致对应哪三个入口文件？

后续课程会再回答这些更深入的问题：

- 为什么 SFT 仍然可以使用 causal LM 的 next-token loss？
- 为什么 SFT 需要把 user token 的 label 设成 `-100`？
- `eval_llm.py --load_from ./model` 和 `eval_llm.py --load_from ./minimind-3` 的加载路径有什么区别？
- 如果本地没有 `out/full_sft_768.pth`，默认推理会在哪一步失败？

<a id="l01-next"></a>
## 10. 下一课预告

第 2 课会处理环境、依赖、数据目录、权重目录和最小运行检查。目标是明确“哪些命令当前能跑，哪些命令必须先准备资源”。
