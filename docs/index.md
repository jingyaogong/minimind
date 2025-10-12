# <strong>Welcome to MiniMind!</strong>


<figure markdown>
  ![logo](images/logo.png)
</figure>

## 📌 项目简介

MiniMind 是一个完全从 0 开始训练的超小语言模型项目，**仅需 3 块钱成本 + 2 小时**即可训练出仅为 **26M** 的语言模型！

- **MiniMind** 系列极其轻量，最小版本体积是 GPT-3 的 **1/7000**
- 项目开源了大模型的极简结构，包含：
  - 混合专家模型（MoE）
  - 数据集清洗
  - 预训练（Pretrain）
  - 监督微调（SFT）
  - LoRA 微调
  - 直接偏好优化（DPO）
  - 模型蒸馏
- 所有核心算法代码均从 0 使用 PyTorch 原生重构，不依赖第三方抽象接口
- 这不仅是大语言模型的全阶段开源复现，也是一个入门 LLM 的教程

!!! note "训练成本"
    "2小时" 基于 NVIDIA 3090 硬件设备（单卡）测试，"3块钱" 指 GPU 服务器租用成本

## ✨ 主要特点

- **超低成本**：单卡 3090，2 小时，3 块钱即可从 0 训练 ChatBot
- **完整流程**：涵盖 Tokenizer、预训练、SFT、LoRA、DPO、蒸馏全流程
- **教育友好**：代码简洁，适合学习 LLM 原理
- **生态兼容**：支持 `transformers`、`llama.cpp`、`vllm`、`ollama` 等主流框架

## 📊 模型列表

| 模型 (大小) | 推理占用 (约) | Release |
|------------|----------|---------|
| MiniMind2-small (26M) | 0.5 GB | 2025.04.26 |
| MiniMind2-MoE (145M) | 1.0 GB | 2025.04.26 |
| MiniMind2 (104M) | 1.0 GB | 2025.04.26 |

## 🚀 快速导航

- [快速开始](quickstart.md) - 环境安装、模型下载、快速测试
- [模型训练](training.md) - 预训练、SFT、LoRA、DPO 等训练流程

## 🔗 相关链接

- **GitHub**: [https://github.com/jingyaogong/minimind](https://github.com/jingyaogong/minimind)
- **HuggingFace**: [MiniMind Collection](https://huggingface.co/collections/jingyaogong/minimind-66caf8d999f5c7fa64f399e5)
- **ModelScope**: [MiniMind 模型](https://www.modelscope.cn/profile/gongjy)
- **在线体验**: [ModelScope 创空间](https://www.modelscope.cn/studios/gongjy/MiniMind)

