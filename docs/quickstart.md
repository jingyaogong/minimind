# 快速开始

本页面将帮助你快速上手 MiniMind 项目。

## 📋 环境要求

- **Python**: 3.10+
- **PyTorch**: 1.12+
- **CUDA**: 12.2+（可选，用于 GPU 加速）
- **显存**: 至少 8GB（推荐 24GB）

!!! tip "硬件配置参考"
    - CPU: Intel i9-10980XE @ 3.00GHz
    - RAM: 128 GB
    - GPU: NVIDIA GeForce RTX 3090 (24GB)

## 🚀 测试已有模型

### 1. 克隆项目

```bash
git clone https://github.com/jingyaogong/minimind.git
cd minimind
```

### 2. 安装依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

!!! warning "Torch CUDA 检查"
    安装后请测试 Torch 是否可用 CUDA：
    ```python
    import torch
    print(torch.cuda.is_available())
    ```

### 3. 下载模型

从 HuggingFace 或 ModelScope 下载预训练模型：

```bash
# 从 HuggingFace 下载
git clone https://huggingface.co/jingyaogong/MiniMind2

# 或从 ModelScope 下载
git clone https://www.modelscope.cn/models/gongjy/MiniMind2.git
```

### 4. 命令行问答

```bash
# load=0: 加载 PyTorch 模型, load=1: 加载 Transformers 模型
python eval_model.py --load 1 --model_mode 2
```

### 5. 启动 WebUI（可选）

```bash
# 需要 Python >= 3.10
pip install streamlit
cd scripts
streamlit run web_demo.py
```

访问 `http://localhost:8501` 即可使用 Web 界面。

## 🔧 第三方推理框架

MiniMind 支持多种主流推理框架：

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
# 转换模型
python convert_hf_to_gguf.py ./MiniMind2/

# 量化模型
./build/bin/llama-quantize ./MiniMind2/MiniMind2-109M-F16.gguf ./Q4-MiniMind2.gguf Q4_K_M

# 推理
./build/bin/llama-cli -m ./Q4-MiniMind2.gguf --chat-template chatml
```

## 📝 效果测试

```text
👶: 你好，请介绍一下自己。
🤖️: 你好！我是 MiniMind，一个由 Jingyao Gong 开发的人工智能助手。
    我通过自然语言处理和算法训练来与用户进行交互。

👶: 世界上最高的山峰是什么？
🤖️: 珠穆朗玛峰是世界上最高的山峰，位于喜马拉雅山脉，
    海拔 8,848.86 米（29,031.7 英尺）。
```

## 🎯 下一步

- 查看 [模型训练](training.md) 了解如何从 0 开始训练自己的模型
- 阅读源码了解 LLM 的实现原理

