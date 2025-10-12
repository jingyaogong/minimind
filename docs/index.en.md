# <strong>Welcome to MiniMind!</strong>

<figure markdown>
  ![logo](images/logo.png)
  <figcaption><strong>"Simplicity is the ultimate sophistication"</strong></figcaption>
</figure>

## 📌 Introduction

MiniMind is a super-small language model project trained completely from scratch, requiring **only $0.5 + 2 hours** to train a **26M** language model!

- **MiniMind** series is extremely lightweight, the smallest version is **1/7000** the size of GPT-3
- The project open-sources the minimalist structure of large models, including:
  - Mixture of Experts (MoE)
  - Dataset cleaning
  - Pretraining
  - Supervised Fine-Tuning (SFT)
  - LoRA fine-tuning
  - Direct Preference Optimization (DPO)
  - Model distillation
- All core algorithm code is reconstructed from scratch using native PyTorch, without relying on third-party abstract interfaces
- This is not only a full-stage open-source reproduction of large language models, but also a tutorial for getting started with LLMs

!!! note "Training Cost"
    "2 hours" is based on NVIDIA 3090 hardware (single card) testing, "$0.5" refers to GPU server rental cost

## ✨ Key Features

- **Ultra-low cost**: Single 3090, 2 hours, $0.5 to train a ChatBot from scratch
- **Complete pipeline**: Covers Tokenizer, pretraining, SFT, LoRA, DPO, distillation full process
- **Education-friendly**: Clean code, suitable for learning LLM principles
- **Ecosystem compatible**: Supports `transformers`, `llama.cpp`, `vllm`, `ollama` and other mainstream frameworks

## 📊 Model List

| Model (Size) | Inference Memory (Approx.) | Release |
|------------|----------|---------|
| MiniMind2-small (26M) | 0.5 GB | 2025.04.26 |
| MiniMind2-MoE (145M) | 1.0 GB | 2025.04.26 |
| MiniMind2 (104M) | 1.0 GB | 2025.04.26 |

## 🚀 Quick Navigation

- [Quick Start](quickstart.en.md) - Environment setup, model download, quick testing
- [Model Training](training.en.md) - Pretraining, SFT, LoRA, DPO training process

## 🔗 Related Links

- **GitHub**: [https://github.com/jingyaogong/minimind](https://github.com/jingyaogong/minimind)
- **HuggingFace**: [MiniMind Collection](https://huggingface.co/collections/jingyaogong/minimind-66caf8d999f5c7fa64f399e5)
- **ModelScope**: [MiniMind Models](https://www.modelscope.cn/profile/gongjy)
- **Online Demo**: [ModelScope Studio](https://www.modelscope.cn/studios/gongjy/MiniMind)

