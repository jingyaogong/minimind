# Quick Start

This page will help you quickly get started with the MiniMind project.

## 📋 Requirements

- **Python**: 3.10+
- **PyTorch**: 1.12+
- **CUDA**: 12.2+ (optional, for GPU acceleration)
- **VRAM**: At least 8GB (24GB recommended)

!!! tip "Hardware Configuration Reference"
    - CPU: Intel i9-10980XE @ 3.00GHz
    - RAM: 128 GB
    - GPU: NVIDIA GeForce RTX 3090 (24GB)

## 🚀 Testing Existing Models

### 1. Clone the Project

```bash
git clone https://github.com/jingyaogong/minimind.git
cd minimind
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

!!! warning "Torch CUDA Check"
    After installation, test if Torch can use CUDA:
    ```python
    import torch
    print(torch.cuda.is_available())
    ```

### 3. Download Model

Download pretrained models from HuggingFace or ModelScope:

```bash
# From HuggingFace
git clone https://huggingface.co/jingyaogong/MiniMind2

# Or from ModelScope
git clone https://www.modelscope.cn/models/gongjy/MiniMind2.git
```

### 4. Command Line Q&A

```bash
# load=0: load PyTorch model, load=1: load Transformers model
python eval_model.py --load 1 --model_mode 2
```

### 5. Start WebUI (Optional)

```bash
# Requires Python >= 3.10
pip install streamlit
cd scripts
streamlit run web_demo.py
```

Visit `http://localhost:8501` to use the web interface.

## 🔧 Third-party Inference Frameworks

MiniMind supports multiple mainstream inference frameworks:

### Ollama

```bash
ollama run jingyaogong/minimind2
```

### vLLM

```bash
vllm serve ./MiniMind2/ --served-model-name "minimind"
```

### llama.cpp

```bash
# Convert model
python convert_hf_to_gguf.py ./MiniMind2/

# Quantize model
./build/bin/llama-quantize ./MiniMind2/MiniMind2-109M-F16.gguf ./Q4-MiniMind2.gguf Q4_K_M

# Inference
./build/bin/llama-cli -m ./Q4-MiniMind2.gguf --chat-template chatml
```

## 📝 Effect Testing

```text
👶: Hello, please introduce yourself.
🤖️: Hello! I'm MiniMind, an AI assistant developed by Jingyao Gong.
    I interact with users through natural language processing and algorithm training.

👶: What is the highest mountain in the world?
🤖️: Mount Everest is the highest mountain in the world, located in the Himalayas,
    with an elevation of 8,848.86 meters (29,031.7 feet).
```

## 🎯 Next Steps

- Check [Model Training](training.md) to learn how to train your own model from scratch
- Read the source code to understand LLM implementation principles

