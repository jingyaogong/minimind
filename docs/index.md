# Welcome to MiniMind!

<figure markdown>
  ![logo](images/logo.png)
  <figcaption><strong>"Simplicity is the ultimate sophistication"</strong></figcaption>
</figure>

## 📌 Introduction

**MiniMind** is a complete, open-source project for training ultra-small language models from scratch with minimal cost. Train a **64M** ChatBot in just **2 hours** with only **$3** on a single 3090 GPU!

- **MiniMind** series is extremely lightweight, the smallest version is **1/2700** the size of GPT-3
- Complete implementation covering:
  - **Tokenizer training** with custom vocabulary
  - **Pretraining** (knowledge learning)
  - **Supervised Fine-Tuning (SFT)** (conversation patterns)
  - **LoRA fine-tuning** (parameter-efficient adaptation)
  - **Direct Preference Optimization (DPO)** (human preference alignment)
  - **RLAIF algorithms** (PPO/GRPO/CISPO - reinforcement learning)
  - **Agentic RL** (multi-turn Tool-Use with delayed rewards)
  - **Knowledge distillation** (black-box & white-box)
  - **Tool Calling & Adaptive Thinking** (native template support)
  - **YaRN algorithm** (context length extrapolation)
- **Pure PyTorch implementation**: All core algorithms are implemented from scratch using native PyTorch, without relying on third-party abstract interfaces
- **Educational value**: This is not only a full-stage open-source reproduction of large language models, but also a comprehensive tutorial for getting started with LLMs
- **Extended capabilities**: MiniMind now supports [MiniMind-V](https://github.com/jingyaogong/minimind-v) for vision multimodal tasks

!!! note "Training Cost & Time"
    "2 hours" is based on **NVIDIA 3090** hardware (single card) testing
    
    "$3" refers to GPU server rental cost
    
    With 8× RTX 4090 GPUs, training time can be compressed to **under 10 minutes**

## ✨ Key Highlights

- **Ultra-low cost**: Single 3090, 2 hours, $3 to train a fully functional ChatBot from scratch
- **Complete pipeline**: Tokenizer → Pretraining → SFT → LoRA → DPO → PPO/GRPO/CISPO → Agentic RL
- **Latest algorithms**: Implements cutting-edge techniques including GRPO, CISPO, Agentic RL, and YaRN
- **Education-friendly**: Clean, well-documented code suitable for learning LLM principles
- **Ecosystem compatible**: Seamless support for `transformers`, `llama.cpp`, `vllm`, `ollama`, `SGLang`, and `MNN`
- **Full capabilities**: Supports multi-GPU training (DDP/DeepSpeed), model visualization (Wandb/SwanLab), and dynamic checkpoint management
- **Production-ready**: OpenAI API with Tool Calling & Adaptive Thinking for easy integration with FastGPT, Open-WebUI, Dify, etc.
- **Multimodal extension**: Extended to vision with [MiniMind-V](https://github.com/jingyaogong/minimind-v)

## 📊 Model Series

### MiniMind-3 Series (Latest - 2026.03.20)

| Model | Parameters | Vocabulary | Layers | Hidden Dim | Context | Inference Memory |
|-------|-----------|------------|--------|-----------|---------|-----------------|
| MiniMind-3 | 64M | 6,400 | 8 | 768 | 2K | ~0.5 GB |
| MiniMind-3-MoE | 198M / A64M | 6,400 | 8 | 768 | 2K | ~1.0 GB |

## 📅 Latest Updates (2026-03-20)

🔥 **MiniMind-3 Release**: Architecture aligned with Qwen3/Qwen3-MoE, Dense ~64M, MoE ~198M/A64M

- **Agentic RL**: New `train_agent.py` for multi-turn Tool-Use RL with GRPO/CISPO
- **RLAIF rollout engine**: Decoupled training/inference for flexible backends (SGLang, etc.)
- **Tool Calling & Adaptive Thinking**: Native template support with `open_thinking` switch
- **OpenAI API**: `serve_openai_api.py` supports `reasoning_content`, `tool_calls`, `open_thinking`
- **Tokenizer**: Updated BPE + ByteLevel with tool call & thinking special tokens

## 🎯 Project Contents

- Complete MiniMind-LLM architecture code (Dense + MoE models)
- Detailed Tokenizer training code
- Full training pipeline: Pretrain → SFT → LoRA → DPO → PPO/GRPO/CISPO → Agentic RL
- Tool Calling & Adaptive Thinking (native chat template support)
- High-quality, curated and deduplicated datasets at all stages
- Native PyTorch implementation of key algorithms, minimal third-party dependencies
- Multi-GPU training support (single-machine multi-card DDP, DeepSpeed, distributed clusters)
- Visualization with SwanLab
- Model evaluation on third-party benchmarks (C-Eval, CMMLU, OpenBookQA, etc.)
- YaRN algorithm for RoPE context length extrapolation
- OpenAI API server with reasoning_content / tool_calls / open_thinking
- Streamlit web UI for chat
- Full compatibility with community tools: llama.cpp, vllm, ollama, SGLang, MNN

## 🚀 Quick Navigation

- **[Quick Start](quickstart.md)** - Environment setup, model download, quick testing
- **[Model Training](training.md)** - Pretraining, SFT, LoRA, RLHF, RLAIF, and reasoning training

## 🔗 Links & Resources

**Project Repositories**:
- **GitHub**: [https://github.com/jingyaogong/minimind](https://github.com/jingyaogong/minimind)
- **HuggingFace**: [MiniMind Collection](https://huggingface.co/collections/jingyaogong/minimind-66caf8d999f5c7fa64f399e5)
- **ModelScope**: [MiniMind Profile](https://www.modelscope.cn/profile/gongjy)

**Online Demos**:
- [ModelScope Studio - Standard Chat](https://www.modelscope.cn/studios/gongjy/MiniMind)
- [ModelScope Studio - Reasoning Model](https://www.modelscope.cn/studios/gongjy/MiniMind-Reasoning)
- [Bilibili Video Introduction](https://www.bilibili.com/video/BV12dHPeqE72/)

**Vision Extension**:
- [MiniMind-V](https://github.com/jingyaogong/minimind-v) - Multimodal vision language models

## 💡 Why MiniMind?

The AI community is flooded with high-cost, complex frameworks that abstract away the fundamentals. MiniMind aims to democratize LLM learning by:

1. **Lowering the barrier**: No need for expensive GPUs or cloud services
2. **Understanding, not just using**: Learn every detail from tokenization to inference
3. **End-to-end learning**: Train from scratch, not just fine-tune existing models
4. **Code clarity**: Pure PyTorch implementations you can read and understand
5. **Practical results**: Get a working ChatBot with minimal resources

As we say: **"Building a Lego airplane is far more exciting than flying first class!"**

---

Next: [Get Started →](quickstart.md)

