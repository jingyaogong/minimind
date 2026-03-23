# Quick Start

Get MiniMind up and running in minutes!

## 📋 Requirements

### Hardware

- **GPU Memory**: 8GB minimum (24GB recommended for comfortable development)
- **Recommended GPU**: NVIDIA RTX 3090 (24GB)

### Software

- **Python**: 3.10+
- **PyTorch**: 2.0+ (with CUDA 12.2+ for GPU support)
- **CUDA**: 12.2+ (optional, for GPU acceleration)

!!! tip "Hardware Configuration Reference"
    - **CPU**: Intel i9-10980XE @ 3.00GHz
    - **RAM**: 128 GB
    - **GPU**: NVIDIA GeForce RTX 3090 (24GB) × 8
    - **OS**: Ubuntu 20.04
    - **CUDA**: 12.2
    - **Python**: 3.10.16

## 🚀 Step 0: Clone the Repository

```bash
git clone https://github.com/jingyaogong/minimind.git
cd minimind
```

## 🎯 Section I: Testing Existing Models

### 1. Environment Setup

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

!!! warning "Verify CUDA Support"
    After installation, verify PyTorch can access CUDA:
    ```python
    import torch
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    ```
    If `False`, download the correct PyTorch version from [PyTorch Official](https://download.pytorch.org/whl/torch_stable.html)

### 2. Download Pretrained Models

Choose one option:

**From HuggingFace** (recommended for international users):
```bash
git clone https://huggingface.co/jingyaogong/minimind-3
```

**From ModelScope** (recommended for China users):
```bash
git clone https://www.modelscope.cn/models/gongjy/minimind-3.git
```

### 3. Command-Line Chat

```bash
# Use transformers format model
python eval_llm.py --load_from ./minimind-3
```

**Weight Options** (`--weight` parameter):
- `pretrain`: Pretrain model (word continuation)
- `full_sft`: SFT Chat model (conversation)
- `dpo`: DPO model (preference optimization)
- `ppo_actor`, `grpo`: RLAIF models (reinforcement learning trained)
- `agent`: Agentic RL model (multi-turn Tool-Use)

**Example Session**:
```text
👶: Hello, please introduce yourself.
🤖️: I am MiniMind, an AI assistant developed by Jingyao Gong.
    I use natural language processing and machine learning algorithms to interact with users.

👶: What's the capital of France?
🤖️: The capital of France is Paris, which is located in the northern central part of France.
    It is the largest city in France and serves as its political, economic, and cultural center.
```

### 4. Web UI Demo (Optional)

```bash
# Requires Python >= 3.10
pip install streamlit

cd scripts
streamlit run web_demo.py
```

Visit `http://localhost:8501` to use the interactive web interface.

### 5. Rope Length Extrapolation with YaRN

Extend context length beyond training with RoPE extrapolation:

```bash
python eval_llm.py --weight full_sft --inference_rope_scaling
```

This enables the YaRN algorithm to handle sequences longer than the 2K training context, useful for processing documents and long conversations.

## 🔧 Third-Party Inference Frameworks

MiniMind is compatible with popular inference engines:

### Ollama (Easiest)

```bash
ollama run jingyaogong/minimind-3
```

### vLLM (Fastest)

```bash
vllm serve ./minimind-3/ --model-impl transformers --served-model-name "minimind" --port 8998

# Test with curl
curl http://localhost:8998/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minimind",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

### llama.cpp (CPU-Friendly)

```bash
# Convert to GGUF format (in llama.cpp directory)
python convert_hf_to_gguf.py /path/to/minimind-3

# Quantize for size reduction
./build/bin/llama-quantize /path/to/model/xxxx.gguf /path/to/model/xxxx.q8.gguf Q8_0

# Run inference
./build/bin/llama-cli -m /path/to/model/xxxx.gguf
```

## 🔌 OpenAI API Server (For Integration)

Run MiniMind as an OpenAI API-compatible service:

```bash
cd scripts && python serve_openai_api.py
```

Test the API:

```bash
cd scripts && python chat_api.py
```

**cURL Example**:
```bash
curl http://localhost:8998/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minimind",
    "messages": [
      {"role": "user", "content": "Explain machine learning in one sentence."}
    ],
    "temperature": 0.7,
    "max_tokens": 256,
    "stream": true,
    "open_thinking": true
  }'
```

This enables integration with:
- [FastGPT](https://fastgpt.run/)
- [Open-WebUI](https://github.com/open-webui/open-webui)
- [Dify](https://dify.ai/)
- Any OpenAI API-compatible client

## 📊 Model Selection Guide

| Use Case | Recommended Model | Memory | Speed |
|----------|------------------|--------|-------|
| Learning/Testing | MiniMind-3 (64M) | ~0.5 GB | Fast |
| Expert System (MoE) | MiniMind-3-MoE (198M/A64M) | ~1.0 GB | Dynamic |
| Tool-Use / Agent | MiniMind-3 Agent (64M) | ~0.5 GB | Fast |

## ⚡ Quick Test Results

**Model**: MiniMind-3 (64M parameters)

```text
Q: What is photosynthesis?
A: Photosynthesis is a process in which plants convert light energy from the sun 
   into chemical energy to produce glucose. This process occurs mainly in leaves 
   and is essential for plant growth and survival.

Q: Write a Python function to calculate Fibonacci numbers.
A: def fibonacci(n):
       if n <= 1:
           return n
       return fibonacci(n-1) + fibonacci(n-2)
   
   # For better performance, use dynamic programming:
   def fibonacci_dp(n):
       dp = [0] * (n + 1)
       for i in range(2, n + 1):
           dp[i] = dp[i-1] + dp[i-2]
       return dp[n]

Q: 世界上最高的山峰是什么?  (What is the highest mountain?)
A: 珠穆朗玛峰（Mount Everest）是世界上最高的山峰，位于喜马拉雅山脉...
   (Mount Everest is the world's highest mountain, located in the Himalayas...)
```

## 🆘 Troubleshooting

### Issue: CUDA Out of Memory

**Solution**:
```bash
# Reduce batch size
python eval_llm.py --batch_size 1

# Or use CPU (slow but works)
python eval_llm.py --device cpu
```

### Issue: Slow Inference

**Solutions**:
- Use vLLM or llama.cpp for faster inference
- Enable quantization (4-bit, 8-bit)
- Use GPU instead of CPU
- Reduce `max_tokens` parameter

### Issue: Model Responses Are Poor Quality

**Possible Causes**:
- Using pretrain model (`--weight pretrain`) instead of SFT (`--weight full_sft`)
- Model is undertrained - download the full checkpoint instead
- Input prompt is too short - provide more context

### Issue: Python/PyTorch Version Mismatch

**Solution**:
```bash
# Use conda for clean environment
conda create -n minimind python=3.10
conda activate minimind
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu122
pip install -r requirements.txt
```

## 📖 Next Steps

- **[Model Training Guide](training.md)** - Train your own MiniMind from scratch
- **[Source Code](https://github.com/jingyaogong/minimind)** - Explore and learn LLM implementation
- **[Inference Benchmarks](https://huggingface.co/collections/jingyaogong/minimind-66caf8d999f5c7fa64f399e5)** - See model performance comparisons

## 💡 Pro Tips

1. **GPU Memory Optimization**: Use `torch.cuda.empty_cache()` periodically
2. **Batch Processing**: For efficiency, process multiple prompts in batches
3. **Temperature Tuning**: Lower (0.3-0.7) = more consistent, Higher (0.8-1.0) = more creative
4. **Prompt Engineering**: Better prompts → better results, even for small models
5. **Model Quantization**: Use 4-bit quantization to run on smaller GPUs

---

Done! Now you're ready to use MiniMind. Start with the Quick Start, then move to [Model Training](training.md) to learn how to train your own models.

