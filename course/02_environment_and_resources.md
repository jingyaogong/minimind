# 第 2 课：环境、依赖、数据与权重

## 目录

- [1. 本节目标](#l02-goals)
- [2. 四类运行条件](#l02-runtime-requirements)
- [3. 目录约定](#l02-directory-conventions)
- [4. 哪些命令依赖什么资源](#l02-command-resources)
- [5. 本节最小实验](#l02-experiment)
- [6. 如何判断“现在能不能跑”](#l02-readiness)
- [7. 常见坑](#l02-pitfalls)
- [8. 课后任务](#l02-homework)
- [9. 检查题](#l02-check)
- [10. 下一课预告](#l02-next)

<a id="l02-goals"></a>
## 1. 本节目标

学完本节，你应该能回答：

- 这个项目运行时依赖哪些资源：源码、Python 包、数据、权重。
- 为什么刚 clone 下来的仓库通常不能直接训练或推理。
- `dataset/`、`out/`、`checkpoints/`、transformers 模型目录分别放什么。
- 如何判断一个命令现在能不能跑。

本节不要求理解模型结构、loss、SFT labels，这些后面单独讲。

<a id="l02-runtime-requirements"></a>
## 2. 四类运行条件

MiniMind 要跑起来，通常需要四类东西。

| 类型 | 例子 | 用途 |
|---|---|---|
| 源码 | `model/`, `dataset/`, `trainer/`, `scripts/` | 定义模型、数据集、训练和推理逻辑 |
| Python 包 | `torch`, `transformers`, `datasets` | 执行模型、分词、数据加载 |
| 训练数据 | `pretrain_t2t_mini.jsonl`, `sft_t2t_mini.jsonl` | 训练时读取 |
| 模型权重 | `out/full_sft_768.pth` 或 `minimind-3/` | 推理或继续训练时加载 |

源码仓库只保证第一类资源存在。数据和权重通常要单独下载。

<a id="l02-directory-conventions"></a>
## 3. 目录约定

MiniMind 的常见目录含义如下。

```text
/home/sun/minimind/
├── model/              # tokenizer 和 MiniMind 模型源码
├── dataset/            # 训练数据应该放这里
├── trainer/            # 训练脚本
├── scripts/            # API、WebUI、模型转换等脚本
├── out/                # 训练得到的普通 pth 权重，当前可能不存在
├── checkpoints/        # 断点续训 checkpoint，当前可能不存在
├── minimind-3/         # transformers 格式模型目录，需单独下载
└── course/             # 本课程材料
```

三个目录最容易混：

- `model/`：当前仓库自带，主要是源码和 tokenizer，不等于已训练模型权重。
- `out/`：放 `pretrain_768.pth`、`full_sft_768.pth` 这类 PyTorch 权重。
- `minimind-3/`：放 Hugging Face / ModelScope 下载的 transformers 格式模型。

<a id="l02-command-resources"></a>
## 4. 哪些命令依赖什么资源

### 4.1 只读检查

这些命令通常只需要源码和已安装的基础包：

```bash
cd /home/sun/minimind
PYTHONDONTWRITEBYTECODE=1 python course/labs/check_minimind_import.py
PYTHONDONTWRITEBYTECODE=1 python course/labs/check_project_readiness.py
```

### 4.2 默认预训练

```bash
cd /home/sun/minimind/trainer
python train_pretrain.py
```

默认会找：

```text
../dataset/pretrain_t2t_mini.jsonl
```

如果这个文件不存在，会在加载数据阶段失败。

### 4.3 默认 SFT

```bash
cd /home/sun/minimind/trainer
python train_full_sft.py
```

默认会找：

```text
../out/pretrain_768.pth
../dataset/sft_t2t_mini.jsonl
```

也就是说，默认 SFT 假设你已经有预训练权重。

### 4.4 默认 CLI 推理

```bash
cd /home/sun/minimind
python eval_llm.py
```

默认参数相当于：

```bash
python eval_llm.py --load_from model --weight full_sft --save_dir out
```

默认会使用仓库里的 tokenizer，但还会找：

```text
./out/full_sft_768.pth
```

如果本地没有这个权重，默认推理不能跑。

### 4.5 transformers 格式推理

```bash
cd /home/sun/minimind
python eval_llm.py --load_from ./minimind-3
```

这种方式要求 `./minimind-3` 是完整的 transformers 模型目录，里面通常有：

```text
config.json
tokenizer.json
tokenizer_config.json
model.safetensors 或 pytorch_model.bin
```

<a id="l02-experiment"></a>
## 5. 本节最小实验

运行项目 readiness 检查：

```bash
cd /home/sun/minimind
PYTHONDONTWRITEBYTECODE=1 python course/labs/check_project_readiness.py
```

它会检查：

- Python 版本。
- 关键包是否能导入。
- CUDA 是否可用。
- 默认数据文件是否存在。
- 默认 PyTorch 权重是否存在。
- transformers 模型目录是否存在。
- WebUI 和训练记录相关的可选依赖是否存在。

这个脚本不会下载任何东西，也不会开始训练。

<a id="l02-readiness"></a>
## 6. 如何判断“现在能不能跑”

用下面这个规则即可。

```text
只读源码实验：需要源码 + 基础依赖
默认 pretrain：需要 pretrain jsonl
默认 SFT：需要 pretrain pth + SFT jsonl
默认 eval：需要 full_sft pth
transformers eval：需要完整 transformers 模型目录
WebUI：还需要 streamlit
--use_wandb：还需要 swanlab
```

如果缺资源，不要先改代码。先确认是缺数据、缺权重，还是缺 Python 包。

<a id="l02-pitfalls"></a>
## 7. 常见坑

- `model/` 目录不是已训练模型目录，它主要是源码和 tokenizer。
- `checkpoints/` 是断点续训用的，不等同于最终推理权重。
- `out/` 是训练输出目录，刚 clone 下来一般没有。
- `python eval_llm.py --load_from ./model` 会走原生 PyTorch 权重加载。
- `python eval_llm.py --load_from ./minimind-3` 会走 transformers 模型加载。
- 当前课程前期使用 tiny 数据，不急着下载 GB 级训练集。

<a id="l02-homework"></a>
## 8. 课后任务

1. 运行 `check_project_readiness.py`，记录哪些是 `OK`，哪些是 `MISSING` 或 `OPTIONAL-MISSING`。
2. 找出 `train_pretrain.py` 默认的数据路径。
3. 找出 `train_full_sft.py` 默认的 `from_weight` 是什么。
4. 用一句话解释 `out/` 和 `checkpoints/` 的区别。

<a id="l02-check"></a>
## 9. 检查题

1. 为什么源码仓库存在，不代表默认训练能跑？
2. 默认 pretrain 需要哪个数据文件？
3. 默认 SFT 至少需要哪两个资源？
4. 默认 `eval_llm.py` 会尝试加载哪个权重文件？
5. 如果想用下载好的 transformers 模型推理，应该使用哪个参数？

<a id="l02-next"></a>
## 10. 下一课预告

第 3 课会讲 CLI 推理流程。我们会只看推理调用链：用户输入如何变成 prompt，prompt 如何变成 token，模型如何生成，再如何 decode 回文本。
