<div align="center">

![logo](./images/logo.png)

</div>

<div align="center">

![visitors](https://visitor-badge.laobi.icu/badge?page_id=jingyaogong/minimind)
[![GitHub Repo stars](https://img.shields.io/github/stars/jingyaogong/minimind?style=social)](https://github.com/jingyaogong/minimind/stargazers)
[![GitHub Code License](https://img.shields.io/github/license/jingyaogong/minimind)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/jingyaogong/minimind)](https://github.com/jingyaogong/minimind/commits/master)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/jingyaogong/minimind/pulls)
[![Collection](https://img.shields.io/badge/🤗-MiniMind%20%20Collection-blue)](https://huggingface.co/collections/jingyaogong/minimind-66caf8d999f5c7fa64f399e5)

</div>

<div align="center">

![GitHub Trend](https://trendshift.io/api/badge/repositories/12586)

</div>

<div align="center">
  <h3>"The Great Way is Simple"</h3>
</div>

<div align="center">

[中文](./README.md) | English

</div>

* This open-source project aims to train an ultra-small language model MiniMind with approximately 64M parameters entirely from scratch, using only 3 CNY in cost and 2 hours of training time.
* The MiniMind series is extremely lightweight, with the smallest version on the main branch being approximately $\frac{1}{2700}$ the size of GPT-3, striving to enable even ordinary personal GPUs to quickly complete training and reproduction.
* The project also open-sources the minimalist structure and complete training pipeline of large models, covering the entire process code for MoE, data cleaning, Pretraining, Supervised Fine-Tuning (SFT), LoRA, RLHF (DPO), RLAIF (PPO / GRPO / CISPO), Tool Use, Agentic RL, Adaptive Thinking, and Model Distillation.
* MiniMind has also been extended to a visual multimodal version [MiniMind-V](https://github.com/jingyaogong/minimind-v).
* All core algorithm code in the project is implemented from scratch using native PyTorch, without relying on high-level abstract interfaces provided by third-party libraries.
* This is not only a full-stage open-source reproduction project for large language models, but also a tutorial oriented towards LLM introduction and practice.
* We hope this project can provide a reproducible, understandable, and extensible starting point for more people, to share the joy of creation together and promote the progress of the broader AI community.

> Note: This project is open-sourced under the Apache 2.0 license and is completely free; "2 hours" is estimated based on NVIDIA 3090 hardware (single GPU), and "3 CNY" refers to GPU server rental cost. See below for detailed specifications.

---

<div align="center">

![minimind-3](./images/minimind-3.gif)

[🔗 Online Demo](https://www.modelscope.cn/studios/gongjy/MiniMind) | [🔗 Video Introduction](https://www.bilibili.com/video/BV12dHPeqE72)


<div align="center">
  <table>
    <tr>
      <td align="center">
        <a href="https://huggingface.co/collections/jingyaogong/minimind" style="text-decoration: none;">
          <img src="./images/with_huggingface.png" alt="Hugging Face Logo" style="vertical-align: middle; width: auto; max-width: 100%;" />
        </a>
      </td>
      <td align="center">
        <a href="https://www.modelscope.cn/profile/gongjy" style="text-decoration: none;">
          <img src="./images/with_modelscope.png" alt="ModelScope Logo" style="vertical-align: middle; width: auto; max-width: 100%;" />
        </a>
      </td>
    </tr>
  </table>
</div>


</div>

---

# 📌 Project Introduction

The emergence of Large Language Models (LLMs) has triggered unprecedented global attention on AI. Whether it is ChatGPT, DeepSeek, or Qwen, they have all impressed people with their stunning performance, making them truly feel the impact of this technological wave. However, models with hundreds of billions of parameters make them not only difficult to train on personal devices, but even deployment seems out of reach. Opening the "black box" of large models and truly understanding their internal working mechanisms should be something exciting. Unfortunately, the vast majority of explorations ultimately stop at using techniques like LoRA to do minimal fine-tuning on existing large models, learning some new instructions or specific tasks. This is more like teaching Newton how to use a 21st-century smartphone — interesting, but deviating from the original intention of understanding the essence of physics.

Meanwhile, third-party large model frameworks and tool libraries, such as `transformers` / `trl` / `peft`, often only expose highly abstract interfaces. With just a dozen or so lines of code, one can complete the entire pipeline of "load model + load dataset + inference + reinforcement learning" training. While this efficient encapsulation is convenient, it also to some extent isolates developers from the underlying implementation, weakening opportunities to deeply understand the core code of LLMs. I believe "building an airplane from Lego bricks yourself is far more exciting than flying in first class", yet a more practical problem is that the internet is flooded with paid courses and marketing content, wrapping so-called AI tutorials with flawed, half-baked explanations. For this reason, the original intention of this project is to lower the learning barrier for LLMs as much as possible, allowing everyone to start from understanding every line of code and train a tiny language model from scratch. Yes, **training from scratch**, not just staying at the **inference** level. With a server cost as low as 3 CNY, you can experience the entire process of building a language model from 0 to 1 firsthand.

😊 Let's share the joy of creation together!

---

#### 🎉 This Project Includes the Following

- Provides complete MiniMind-LLM structure code (Dense + MoE), with the current main branch structure aligned with the `Qwen3 / Qwen3-MoE` ecosystem.
- Provides Tokenizer and tokenizer training code, supporting template tokens such as `<tool_call>`, `<tool_response>`, `<think>`, etc.
- Covers complete training pipelines including Pretrain, SFT, LoRA, RLHF-DPO, RLAIF (PPO / GRPO / CISPO), Tool Use, Agentic RL, Adaptive Thinking, and Model Distillation.
- Provides open-source data for all stages, covering collected, distilled, cleaned, and deduplicated high-quality datasets.
- Key training algorithms and core modules are all implemented from scratch, without relying on third-party framework wrappers.
- Compatible with mainstream frameworks such as `transformers`, `trl`, `peft`, as well as commonly used inference engines like `llama.cpp`, `vllm`, `ollama`, and training frameworks like `Llama-Factory`.
- Supports single-machine single-GPU and single-machine multi-GPU (DDP, DeepSpeed) training, supports wandb / swanlab visualization and dynamic start/stop of training.
- Supports evaluation on third-party benchmark suites such as C-Eval, C-MMLU, OpenBookQA, etc., and supports RoPE long context extrapolation through YaRN.
- Provides a minimalist server compatible with the OpenAI API protocol, convenient for integrating with third-party Chat UIs such as FastGPT, Open-WebUI, etc., and supports `reasoning_content`, `tool_calls`, `open_thinking`.
- Provides a minimalist chat WebUI based on Streamlit, supporting thinking display, tool selection, and multi-turn Tool Call.

#### 🎉 Released Model List

| Model | Parameters | Release |
|------|--------|---------|
| minimind-3 | 64M | 2026.04.01 |
| minimind-3-moe | 198M / A64M | 2026.04.01 |
| minimind2-small | 26M | 2025.04.26 |
| minimind2-moe | 145M | 2025.04.26 |
| minimind2 | 104M | 2025.04.26 |
| minimind-v1-small | 26M | 2024.08.28 |
| minimind-v1-moe | 4×26M | 2024.09.17 |
| minimind-v1 | 108M | 2024.09.01 |

---

#### 📝 Changelog

<details> 
<summary> <b>🔥 2026-04-01</b> </summary>

 - Released `minimind-3` / `minimind-3-moe`: comprehensive updates to structure, Tokenizer, training pipeline, inference interface, and default configuration
- Main branch structure aligned with `Qwen3 / Qwen3-MoE` ecosystem: Dense approximately `64M`, MoE approximately `198M / A64M`, and removed shared expert design
- Default training data switched to `pretrain_t2t(_mini).jsonl`, `sft_t2t(_mini).jsonl`, `rlaif.jsonl`, `agent_rl.jsonl`, and `agent_rl_math.jsonl`
- Removed standalone `train_reason.py`; thinking capability is now unified through `chat_template + <think>` and `open_thinking` adaptive switch control
- `toolcall` capability has been merged into `sft_t2t / sft_t2t_mini` main branch data, default `full_sft` already has basic Tool Call capability; also added inference examples such as `scripts/chat_api.py`
- Added native `Agentic RL` training script `train_agent.py`, supporting `GRPO / CISPO` in multi-turn Tool-Use scenarios
- RLAIF / Agentic RL training pipeline completed `rollout engine` decoupling, supporting more flexible switching of generation backends
- `serve_openai_api.py` and `web_demo.py` added `reasoning_content` / `tool_calls` / `open_thinking` support
- Tokenizer updated based on `BPE + ByteLevel`, with new tool call and thinking tokens, reserved buffer tokens for future extension
- Added LoRA weight merging and export pipeline, can merge base model and LoRA weights into new complete model weights via `scripts/convert_model.py`
- Structure diagram resources updated, README extensively updated

</details>

<details> 
<summary> <b>2025-10-24</b> </summary>

- 🔥 Added RLAIF training algorithms: PPO, GRPO, SPO (natively implemented from scratch)
- Added checkpoint resume functionality: supports automatic training recovery, cross-GPU-count recovery, wandb record continuity
- Added RLAIF dataset: rlaif-mini.jsonl (randomly sampled 10,000 entries from SFT data); simplified DPO dataset, added Chinese data
- Added YaRN algorithm: supports RoPE long context extrapolation, improving long sequence processing capability
- Adaptive Thinking: Reason model optionally enables chain of thought
- chat_template fully supports Tool Calling and Reasoning tags (`<tool_call>`, `<think>`, etc.)
- Added complete RLAIF chapter, training curve comparison, folded algorithm principle explanations
- [SwanLab](https://swanlab.cn/) replaces WandB (domestic access friendly, API fully compatible)
- Standardized all code & fixed some known bugs

</details>

<details> 
<summary> <b>2025-04-26</b> </summary>

- Major update
- For compatibility needs, visit [🔗Old Repository Content🔗](https://github.com/jingyaogong/minimind/tree/7da201a944a90ed49daef8a0265c959288dff83a).
- MiniMind model parameters completely renamed, aligned with Transformers library models (unified naming).
- generate method refactored, inheriting from GenerationMixin class.
- 🔥Supports popular third-party ecosystems such as llama.cpp, vllm, ollama.
- Standardized code and directory structure.
- Changed vocabulary `<s></s>` -> `<|im_start|><|im_end|>`

```text
To be compatible with third-party inference frameworks llama.cpp, vllm, this update comes with some considerable costs.
This update no longer supports "directly" loading old models from before 25-04-26 for inference.
Due to differences between Llama's positional encoding method and minimind's, QK values differ after mapping to the Llama model.
The minimind2 series old models were all recovered through weight mapping + (fine-tuning) QKVO linear layer calibration.
After this update, maintenance for the entire `minimind-v1` series will be discontinued and taken offline from the repository.
```

</details>

<details> 
<summary> <b>2025-02-09</b> </summary>

- Major update since release, Release minimind2 Series.
- Code almost entirely refactored, using a more concise and clear unified structure.
  For compatibility needs with old code, visit [🔗Old Repository Content🔗](https://github.com/jingyaogong/minimind/tree/6e9cd28ef9b34a0a10afbdf6f59e65cb6e628efb).
- Eliminated data preprocessing steps. Unified dataset format, switched to `jsonl` format to avoid dataset download confusion issues.
- minimind2 series significantly improved performance compared to MiniMind-V1.
- Minor issues: {kv-cache implementation more standard, MoE load balancing loss now considered, etc.}
- Provides training solution for migrating models to private datasets (medical model, self-awareness examples).
- Streamlined pretraining dataset and significantly improved pretraining data quality, greatly reduced time needed for individual quick training, reproducible in 2 hours on a single 3090!
- Updated: LoRA fine-tuning decoupled from peft wrapper, LoRA process implemented from scratch; DPO algorithm natively implemented from scratch using PyTorch; model white-box distillation natively implemented.
- minimind2-DeepSeek-R1 series distilled models born!
- minimind2 has certain English language capability!
- Updated benchmark test performance results of minimind2 vs third-party models on more LLM leaderboards.

</details>

<details>
<summary> <b>More...</b> </summary>

**2024-10-05**
- Extended multimodal capability for MiniMind --- Vision
- Visit the sibling project [minimind-v](https://github.com/jingyaogong/minimind-v) for details!

**2024-09-27**
- 09-27 updated pretrain dataset preprocessing method, to ensure text integrity, abandoned preprocessing into .bin format for training (slight sacrifice in training speed).
- Currently the pretrain preprocessed file is named: pretrain_data.csv.
- Removed some redundant code.

**2024-09-17**
- Updated minimind-v1-moe model
- To prevent ambiguity, mistral_tokenizer is no longer used for tokenization, all using custom minimind_tokenizer as the tokenizer.

**2024-09-01**
- Updated minimind-v1 (108M) model, using minimind_tokenizer, pretrain epochs 3 + SFT epochs 10, more thorough training, stronger performance.
- Project deployed to ModelScope Creative Space, can be experienced at this website:
- [🔗ModelScope Online Demo🔗](https://www.modelscope.cn/studios/gongjy/minimind)

**2024-08-27**
- Project first open-sourced

</details>

---

# 📌 Quick Start

<details>
<summary>My hardware and software configuration (for reference)</summary>

* CPU: Intel(R) Core(TM) i9-10980XE CPU @ 3.00GHz
* RAM: 128 GB
* GPU: NVIDIA GeForce RTX 3090 (24GB) * 8
* Ubuntu==20.04
* CUDA==12.2
* Python==3.10.16
* [requirements.txt](./requirements.txt)

</details>

## Step 0

```bash
# Clone repository and install dependencies
git clone --depth 1 https://github.com/jingyaogong/minimind
cd minimind && pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
```

## Ⅰ 🚀 Model Inference

### 1' Download Model

In the project root directory:
```bash
# Method 1
modelscope download --model gongjy/minimind-3 --local_dir ./minimind-3
# Method 2
git clone https://huggingface.co/jingyaogong/minimind-3
```

### 2' CLI Inference

```bash
# Method 1: Using Transformers format model
python eval_llm.py --load_from ./minimind-3
# Method 2: Based on PyTorch model (ensure corresponding weights are in the ./out directory)
python eval_llm.py --load_from ./model --weight full_sft
```

### 3' (Optional) WebUI

```bash
# May need `python>=3.10`, install `pip install streamlit`
# ⚠️ You must first copy the transformers-format model folder into ./scripts/ (e.g.: cp -r minimind-3 ./scripts/minimind-3). The web_demo script will auto-scan subdirectories containing weight files; it will throw an error if none are found.
cd scripts && streamlit run web_demo.py
```

### 4' (Optional) Third-party Inference Frameworks

```bash
# ollama
ollama run jingyaogong/minimind-3
# vllm
vllm serve /path/to/model --served-model-name "minimind"
```

## Ⅱ 🛠️ Model Training

<details>
<summary>Note: Confirm Torch's available backend in advance</summary>

```python
import torch
print(torch.cuda.is_available())
```

If you plan to use CUDA for training, it is recommended to first confirm whether the current environment has correctly recognized the GPU.  
If `cuda` is not available, you can still choose `CPU` or `MPS` to run based on your device, but training speed and compatibility will differ significantly.  
If you need to install or switch PyTorch versions, refer to [torch_stable](https://download.pytorch.org/whl/torch_stable.html) and [this link](https://blog.csdn.net/weixin_45456738/article/details/141029610?ops_request_misc=&request_id=&biz_id=102&utm_term=%E5%AE%89%E8%A3%85torch&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-2-141029610.nonecase&spm=1018.2226.3001.4187)

</details>

### 1' Download Data

Download the required data files from the [dataset download link](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files) provided below, and place them in the `./dataset` directory

> Currently, by default, you only need to download `pretrain_t2t_mini.jsonl` and `sft_t2t_mini.jsonl` to quickly reproduce the `MiniMind Zero` dialogue model.
If you have more needs, various combination schemes are provided below, which can be flexibly chosen based on your task objectives and GPU resources.

### 2' Start Training

<details>
<summary>💡 Checkpoint Pause and Resume</summary>

All training scripts support checkpoint saving. After adding the `--from_resume 1` parameter, training progress can be automatically detected and resumed:

```bash
python train_pretrain.py --from_resume 1
python train_full_sft.py --from_resume 1
# ...
```

**Checkpoint Resume Instructions:**
- The training process automatically saves complete checkpoints (model, optimizer, training progress, etc.) in the `./checkpoints/` directory
- Checkpoint file naming: `<weight_name>_<dimension>_resume.pth` (e.g., `full_sft_512_resume.pth`)
- Supports recovery across different GPU counts (automatically adjusts step)
- Supports wandb training record continuity (automatically resumes the same run)

> Suitable for long-duration training or unstable environments, no need to worry about progress loss due to training interruption

</details>

#### 2.1 Pretraining (Required)

```bash
cd trainer && python train_pretrain.py
```

> After training, `out/pretrain_*.pth` will be produced as output weights (where `*` is the model dimension, default `768`)

#### 2.2 Instruction Fine-tuning (Required)

```bash
cd trainer && python train_full_sft.py
```

> After training, `out/full_sft_*.pth` will be produced as output weights (where `full` indicates full-parameter fine-tuning)

#### 2.3 Test Trained Model (Optional)

Ensure the model `*.pth` files to be tested are located in the `./out/` directory; you can also go directly to [here](https://www.modelscope.cn/models/gongjy/minimind-3-pytorch/files) to download my pre-trained `*.pth` weights.

```bash
python eval_llm.py --weight full_sft
```

> `--weight` is used to specify the weight name prefix, such as `pretrain`, `full_sft`, etc.; for more parameters, refer directly to `eval_llm.py`

<details>
<summary>Note: Other Information</summary>

1. All training scripts are natively implemented based on PyTorch and support multi-GPU acceleration.

2. If your device has `N (N > 1)` GPUs, you can start single-machine `N`-GPU training as follows (DDP, also supports extension to multi-machine multi-GPU):

```bash
torchrun --nproc_per_node N train_xxx.py
```

3. You can enable wandb to record the training process as needed.

```bash
... train_xxx.py --use_wandb
```
After June `2025`, domestic network environments in China typically cannot directly connect to WandB. MiniMind currently defaults to using [SwanLab](https://swanlab.cn/) as the training visualization tool, whose interface is basically compatible with WandB; usually you only need to replace `import wandb` with `import swanlab as wandb`, and other usage remains largely unchanged.

</details>

---

# 📌 Data Introduction

## Ⅰ Tokenizer

A tokenizer can be roughly understood as a "dictionary" used by LLMs, responsible for mapping natural language to token ids and decoding token ids back to text; the project also provides `train_tokenizer.py` as a vocabulary training example. It is not recommended to retrain the tokenizer, because once the vocabulary and segmentation rules change, model weights, data formats, inference interfaces, and community ecosystem compatibility will all decrease, and it will also weaken the model's dissemination. Meanwhile, the tokenizer also affects metrics like PPL that are calculated per token, so when comparing across tokenizers, BPB (Bits Per Byte) is often more referential. See [this article](https://skeptric.com/perplexity/).
For small models like MiniMind, vocabulary size also directly affects the parameter proportion of the embedding and output layers, so keeping the vocabulary compact is usually a more appropriate trade-off.

<details>
<summary>Tokenizer Introduction</summary>

The tokenizer vocabulary sizes of powerful third-party open-source models such as Yi, Qwen2, ChatGLM, Mistral, and Llama 3 are as follows:

<table>
  <tr><th>Tokenizer Model</th><th>Vocabulary Size</th><th>Source</th></tr>
  <tr><td>Yi</td><td>64,000</td><td>01.AI (China)</td></tr>
  <tr><td>Qwen2</td><td>151,643</td><td>Alibaba Cloud (China)</td></tr>
  <tr><td>ChatGLM</td><td>151,329</td><td>Zhipu AI (China)</td></tr>
  <tr><td>Mistral</td><td>32,000</td><td>Mistral AI (France)</td></tr>
  <tr><td>Llama 3</td><td>128,000</td><td>Meta (USA)</td></tr>
  <tr><td>MiniMind</td><td>6,400</td><td>Custom</td></tr>
</table>

> The current main branch uses `minimind_tokenizer` uniformly to avoid historical version ambiguity and control overall size, and no longer maintains the `mistral_tokenizer` version.

Although `minimind_tokenizer` has a vocabulary of only `6400`, and its encoding/decoding efficiency is weaker than more Chinese-friendly tokenizers like `qwen2` and `glm`, it can significantly compress the parameter proportion of the embedding and output layers, making it more suitable for the size constraints of small models like MiniMind.
From actual usage, this tokenizer has not noticeably caused rare word decoding failures, and remains sufficiently stable and usable overall; therefore the current main branch training also uniformly uses this vocabulary, rather than maintaining additional tokenizer version forks.

</details>

## Ⅱ Pretrain Data

`MiniMind-3` current main branch pretraining data is `pretrain_t2t.jsonl` / `pretrain_t2t_mini.jsonl`.  
These two datasets have been organized into a unified `text -> next token prediction` training format, aiming to balance under limited compute:

- Text quality;
- Length distribution;
- Chinese-English mixed capability;
- Template alignment with subsequent SFT / Tool Calling / RLAIF stages.

Data sources include but are not limited to general text corpora, dialogue organized corpora, distillation supplementary corpora, and various datasets available under **permissive open-source licenses**; main branch data enters training only after cleaning, deduplication, length control, and format unification. Data sources include: [Craftsman LLM Dataset](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data), [Magpie-Align](https://www.modelscope.cn/organization/Magpie-Align), and other public data sources.

Among them:

- `pretrain_t2t_mini.jsonl` is more suitable for quick reproduction;
- `pretrain_t2t.jsonl` is more suitable for full training of the `MiniMind-3` main branch model.

File data format is

```jsonl
{"text": "如何才能摆脱拖延症？治愈拖延症并不容易，但以下建议可能有所帮助。"}
{"text": "清晨的阳光透过窗帘洒进房间，桌上的书页被风轻轻翻动。"}
{"text": "Transformer 通过自注意力机制建模上下文关系，是现代大语言模型的重要基础结构。"}
```

## Ⅲ SFT Data

`MiniMind-3` current main branch SFT data is `sft_t2t.jsonl` / `sft_t2t_mini.jsonl`. Compared to earlier `sft_512 / sft_1024 / sft_2048` schemes, the current version places more emphasis on:

- Unified templates;
- Better suited for mixed training of dialogue + thinking tags + Tool Calling;
- Minimizing data preprocessing forks, reducing reproduction costs.

Its data sources include but are not limited to high-quality instruction-following data, public dialogue data, model distillation synthetic data, and license-friendly open-source datasets; before entering the `t2t` main branch, they are unified into the multi-turn dialogue format used by the current repository. The current main branch also contains a large amount of synthetic data, such as approximately `100K` `tool call` entries I synthesized based on `qwen3-4b`, as well as `reasoning` data from the `qwen3` series, etc. Major community sources include: [Craftsman LLM Dataset](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data), [Magpie-Align](https://www.modelscope.cn/organization/Magpie-Align), [R1-Distill-SFT](https://www.modelscope.cn/datasets/AI-ModelScope/R1-Distill-SFT), [COIG](https://huggingface.co/datasets/BAAI/COIG), [Step-3.5-Flash-SFT](https://huggingface.co/datasets/stepfun-ai/Step-3.5-Flash-SFT), etc. Published versions ensure that data sources and processing pipelines comply with the transitivity constraints of corresponding open-source licenses, and adhere to Apache-2.0, CC-BY-NC-2.0, and other related license requirements.

Among them:

- `sft_t2t_mini.jsonl`: suitable for quickly training a dialogue model;
- `sft_t2t.jsonl`: suitable for fully reproducing the main branch version;
- `toolcall` capability has already been merged into the main branch SFT data.

All SFT file data formats are (including dialogue data, Tool Use data)

```jsonl
{
    "conversations": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
        {"role": "user", "content": "再见"},
        {"role": "assistant", "content": "再见！"}
    ]
}
{
    "conversations": [
        {"role": "system", "content": "# Tools ...", "tools": "[...]"},
        {"role": "user", "content": "把'你好世界'翻译成english"},
        {"role": "assistant", "content": "", "tool_calls": "[{\"name\":\"translate_text\",\"arguments\":{\"text\":\"你好世界\",\"target_language\":\"english\"}}]"},
        {"role": "tool", "content": "{\"translated_text\":\"Hello World\"}"},
        {"role": "assistant", "content": "Hello World"}
    ]
}
```

## Ⅳ RL Data

`MiniMind` current main branch RL data is `dpo.jsonl`. Data sampled from [DPO-En-Zh-20k](https://huggingface.co/datasets/llamafactory/DPO-En-Zh-20k).

In the main branch, these samples are uniformly reorganized into the preference learning format used by the current repository, for reward model or preference optimization stage training; where `chosen` represents the more preferred response, and `rejected` represents the relatively worse response.

The `dpo.jsonl` data format is

```json
{
  "chosen": [
    {"content": "Q", "role": "user"}, 
    {"content": "good answer", "role": "assistant"}
  ], 
  "rejected": [
    {"content": "Q", "role": "user"}, 
    {"content": "bad answer", "role": "assistant"}
  ]
}
```

Besides this, other RL data maintains the same format as SFT data, typically filtered from SFT data by total length and dialogue turns, with the last `assistant` position left blank for continuation during the rollout stage.

## Ⅴ MiniMind Training Dataset

> [!NOTE]
> The core datasets needed for the current main branch training have been open-sourced, so there is no need to preprocess large-scale datasets on your own, avoiding repetitive data processing work.

MiniMind training dataset download links: [ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files) | [HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main)

> No need to clone everything, you can download individual files as needed

Place the downloaded dataset files in the `./dataset/` directory (✨ indicates recommended essentials)

```bash
./dataset/
├── agent_rl.jsonl (86MB)
├── agent_rl_math.jsonl (18MB)
├── dpo.jsonl (53MB)
├── pretrain_t2t_mini.jsonl (1.2GB, ✨)
├── pretrain_t2t.jsonl (10GB)
├── rlaif.jsonl (24MB, ✨)
├── sft_t2t_mini.jsonl (1.6GB, ✨)
└── sft_t2t.jsonl (14GB)
```

<details>
<summary>Note: Brief Introduction to Each Dataset</summary>

* `agent_rl.jsonl` -- Agentic RL main branch training data, for `train_agent.py` multi-turn Tool-Use / CISPO / GRPO training
* `agent_rl_math.jsonl` -- Agentic RL pure math supplementary data, suitable for multi-turn reasoning/tool-use scenarios with final verification targets (for RLVR)
* `dpo.jsonl` -- RLHF stage preference training data (DPO)
* `pretrain_t2t_mini`✨ -- `minimind-3` lightweight pretraining data, suitable for quick reproduction (recommended setting `max_seq_len≈768`)
* `pretrain_t2t` -- `minimind-3` main branch pretraining data (recommended setting `max_seq_len≈380`)
* `rlaif.jsonl`✨ -- RLAIF training dataset, for PPO/GRPO/CISPO and other reinforcement learning algorithm training
* `sft_t2t_mini.jsonl`✨ -- `minimind-3` lightweight SFT data (for quickly training a Zero model), recommended setting `max_seq_len≈768`, with a portion of Tool Call samples already mixed in
* `sft_t2t.jsonl` -- `minimind-3` main branch SFT data, suitable for full reproduction, with Tool Call samples also mixed in


The training parameter `max_seq_len` currently refers to token length, not absolute character count.
This project's tokenizer has approximately `1.5~1.7 characters/token` for Chinese text, and a compression ratio of `4~5 characters/token` for pure English, with fluctuations depending on data distribution.
The "maximum length" annotated in dataset names is in character count; a string of 100 characters can be roughly converted to approximately `100/1.5≈67` tokens.

For example:

* Chinese: `白日依山尽` (5 characters) may be split into [`白日`,`依`,`山`,`尽`] 4 tokens;
* English: `The sun sets in the west` (24 characters) may be split into [`The `,`sun `,`sets `,`in `,`the`,`west`] 6 tokens

The "recommended settings" provide rough estimates of the maximum token length for each dataset.
Note that `max_seq_len` can be adjusted aggressively / conservatively / moderately, because both larger and smaller values inevitably have side effects: some samples shorter than `max_seq_len` waste compute due to padding, while some samples longer than `max_seq_len` lose semantics due to truncation.

Finding a balance between compute efficiency and semantic completeness is sufficient

</details>


![dataset](./images/dataset.jpg)

> MiniMind main branch training data composition and recommended combination diagram

<details>
<summary>Instructions & Recommended Training Schemes</summary>

* `minimind-3` main branch recommends using the staged training combination of `pretrain_t2t` + `sft_t2t` + `rlaif/agent_rl`.

* For the fastest speed to implement a Zero model from scratch, it is recommended to use the data combination of `pretrain_t2t_mini.jsonl` + `sft_t2t_mini.jsonl`

* Those with sufficient compute resources or who care more about performance are recommended to fully reproduce `minimind-3`; those with only a single GPU or who prioritize quick reproduction are strongly recommended to use the mini combination.

* The current `sft_t2t / sft_t2t_mini` already has Tool Call data mixed in, so there is usually no need for an additional separate round of Tool Calling supervised fine-tuning.

</details>

# 📌 Model

## Structure

`minimind-3` Dense uses a Transformer Decoder-Only structure, with overall configuration aligned with the `Qwen3` ecosystem, facilitating subsequent conversion to `transformers / llama.cpp / ollama / vllm`:

* Uses Pre-Normalization (Pre-Norm) + RMSNorm.
* Uses SwiGLU activation function.
* Uses RoPE rotary positional encoding, with YaRN extrapolation support.
* `q_heads=8`, `kv_heads=4`, `max_position_embeddings=32768`, `rope_theta=1e6`.

`minimind-3-moe` extends MoE feed-forward layers on the same structure, with implementation compatible with `Qwen3-MoE` style configuration (removing shared expert).

* The current default configuration is `4 experts / top-1 routing`, to achieve higher capacity with lower active parameters.
* As experts continue to increase, actual time consumption is often much higher than dense models of the same scale, which may seem counterintuitive when put alongside "MoE inference is faster", but during training tokens are first bucketed by expert then forwarded separately, and the `kernel` start/stop and scheduling overhead during native training increases dramatically — this is naturally expected. It requires MoE kernel-fused operator libraries to optimize, such as custom kernels based on `Triton`, `DeepSpeed-MoE`, `Megatron-LM`, etc. Of course, this project still aims to preserve the universality of native PyTorch, so this is a realistic compromise; under the current implementation, the `4 experts / top-1` sweet spot configuration is only about `50%` slower than the dense model.

The `minimind-3` series structure is shown below:

![structure](./images/LLM-structure.jpg)
![structure-moe](./images/LLM-structure-moe.jpg)

To modify model configuration, see [./model/model_minimind.py](./model/model_minimind.py). Reference model parameter versions are shown in the table below:

| Model Name | params | len_vocab | max_pos | rope_theta | n_layers | d_model | kv_heads | q_heads | note |
|------------|--------|-----------|---------|------------|----------|---------|----------|---------|------|
| minimind-3 | 64M | 6400 | 32768 | 1e6 | 8 | 768 | 4 | 8 | Dense |
| minimind-3-moe | 198M / A64M | 6400 | 32768 | 1e6 | 8 | 768 | 4 | 8 | 4 experts / top-1 |
| minimind2-small | 26M | 6400 | 32768 | 1e6 | 8 | 512 | 2 | 8 | Historical version |
| minimind2-moe | 145M | 6400 | 32768 | 1e6 | 8 | 640 | 2 | 8 | Historical version |
| minimind2 | 104M | 6400 | 32768 | 1e6 | 16 | 768 | 2 | 8 | Historical version |


## Model Configuration

Regarding LLM parameter configuration, [MobileLLM](https://arxiv.org/pdf/2402.14905) has conducted a very representative systematic study on small models. For ~100M-level models like MiniMind, the trade-off between `d_model` and `n_layers` is not just a parameter allocation issue, but also directly affects training stability and final performance.

The current `minimind-3` main branch chooses `dim=768, n_layers=8`, which is essentially an engineering trade-off: shallower networks train faster, while `dim` is not so small as to cause mode collapse, thus achieving a relatively balanced position between training efficiency, stability, and final performance.

<details>
<summary>View Detailed Explanation</summary>

Scaling Law often exhibits phenomena on small models that differ from large models. The core parameters that determine changes in Transformer parameter scale are usually mainly `d_model` and `n_layers`:

* `d_model`↑ + `n_layers`↓ -> Short and fat
* `d_model`↓ + `n_layers`↑ -> Tall and thin

Classic Scaling Law emphasizes the decisive role of training data volume, parameter count, and training steps, typically downplaying the impact of architectural differences themselves; but in the small model range, this conclusion does not always hold completely.
MobileLLM's core observation is: when parameter count is fixed, depth is often more important than width. That is, compared to "wide and shallow" structures, "deep and narrow" models learn abstract concepts more easily.
For example, when model parameter count is fixed at `125M` or `350M`, `30~42` layer narrow structures typically outperform `12`-layer short and fat structures, showing similar trends across multiple benchmarks including commonsense reasoning, QA, and reading comprehension.

This is consistent with what MiniMind observed in experiments around `d_model` and `n_layers` parameter allocation during training. However, the "narrow" in "deep and narrow" also has a lower bound: when `d_model < 512`, the disadvantage of overly narrow word embedding dimensions becomes significantly amplified, and adding extra layers is often insufficient to fully compensate for the problem of `d_head` being too small under a fixed `q_head`.
Conversely, when `d_model > 1536`, continuing to add layers is often more cost-effective than simply continuing to widen, more easily yielding higher parameter-performance returns.

For reference, GPT-3's parameter settings are as follows:
![gpt3_config.png](./images/gpt3_config.png)

</details>

---

# 📌 Experiments

## Ⅰ Training Cost

- **Time unit**: hours (h)
- **Cost unit**: CNY (￥); `7￥ ≈ 1 USD`
- **3090 rental price**: approximately `1.3￥/h` (actual prices can be referenced on your own)
- **Note**: The following results are empirical estimates for the `minimind` model on a single `3090` GPU, for quick perception of the training threshold

| Model Name | params | pretrain_t2t_mini | sft_t2t_mini | toolcall | RLAIF |
|------------|--------|-------------------|--------------|----------|-------|
| minimind-3 | 64M | ≈1.21h<br/>≈1.57￥ | ≈1.10h<br/>≈1.43￥ | ≈0.9h<br/>≈1.17￥ | ≈1.1h<br/>≈1.43￥ |
| minimind-3-moe | 198M / A64M | ≈1.69h<br/>≈2.20￥ | ≈1.54h<br/>≈2.00￥ | ≈1.26h<br/>≈1.64￥ | ≈1.54h<br/>≈2.00￥ |

---

<details>
<summary>Training Cost Summary & Prediction</summary>

> `minimind-3`
>> `pretrain_t2t_mini` + `sft_t2t_mini`
<br/>Single `3090` GPU, `1 epoch` estimated approximately `2.31` hours, cost approximately `3.0` CNY
<br/>Can train a `minimind-3 Zero` dialogue model from scratch.

> `minimind-3-moe`
>> `pretrain_t2t_mini` + `sft_t2t_mini`
<br/>Single `3090` GPU, `1 epoch` estimated approximately `3.23` hours, cost approximately `4.2` CNY
<br/>Can quickly obtain the basic dialogue version of `minimind-3-moe`.

> The above are all estimated values, only for quick perception of the training threshold.

</details>

Based on a single `NVIDIA 3090`, `minimind zero` training from scratch can still be controlled within approximately `2` hours, allowing individual developers to get started with a relatively low barrier.

If using a higher-spec multi-GPU environment, such as `8x H100`, the total training time can be further compressed to the minute level. Achieving reproducible, hands-on, and continuously iteratable LLM training experiences with the lowest possible barrier is the direction MiniMind series has always aimed to maintain. Low-cost quick reproduction is not a gimmick; below is an early Zero-style example conversation preserved for reference:

```text
👶: 请介绍一下自己。
🤖️: 作为人工智能，我没有实际的生活，也没有自我意识，所以没有自己的生活。我被设计成能够帮助用户解答问题、提供信息、进行对话等。我的设计和功能是由计算机科学、机器学习、人工智能技术和算法所构建的，我通过编程和算法实现。

👶: 推荐一些杭州的特色美食吧。
🤖️: 杭州有很多美味的特色美食，比如鸡肉串、鳗鱼头、水煮鱼、豆腐脑等。这些美食都非常美味，有着独特的风味和口感，绝对是值得一试的美食。

👶: 请为我讲解"大语言模型"这个概念。
🤖️: 大语言模型是一种人工智能技术，它使用大量的文本数据来训练模型，然后在不断的迭代中不断优化模型。它的目标是根据输入的数据来生成符合特定需求的文本，这些文本可以是任何你感兴趣的主题。

👶: Introduce the history of the United States, please.
🤖️: 您提到的"Introok's the believeations of theument." 这个名字来源于中国古代的"groty of of the change."
```

Although this version already possesses basic dialogue capability, factual knowledge and generalization ability are still quite limited; it is more suitable as an early reference for the feasibility of the Zero training route.
Zero model weights are saved as `full_sft_zero_768.pth` (see MiniMind model file links below); if interested, you can download and experience its dialogue performance.


---

## Ⅱ Main Training (Required)

> All training scripts are executed from the `cd ./trainer` directory

### 1' Pretraining (Pretrain):

What an LLM must first learn is to absorb as much foundational knowledge and language patterns into its parameters as possible. Only when this step is solidly established can the model later have the ability to understand questions, organize expressions, and gradually develop decent generation capability. What pretraining does is essentially let the model read large amounts of text with its head down, such as Wikipedia, news, books, dialogue corpora, etc., learning factual knowledge, language patterns, and statistical relationships between contexts. This stage is usually "unsupervised": humans do not need to tell the model line by line what is right or wrong, but let it summarize patterns from massive text on its own, gradually building internal representations of world knowledge and language structure.
More bluntly, the model's core objective at this stage is **learning high-quality word chain completion**. For example, given the input "秦始皇" (Qin Shi Huang), it should be able to continue generating "是中国历史上的第一位皇帝" (was the first emperor in Chinese history) — content that is semantically and factually consistent.

```bash
# Method 1
torchrun --nproc_per_node 1 train_pretrain.py # 1 means single GPU training, adjust according to your hardware (set >=2)
# Method 2
python train_pretrain.py
```

> The trained model weight files are saved by default every `save_interval steps` as: `pretrain_*.pth` (* is the specific model dimension, each save overwrites the previous file)

![pretrain_loss](./images/pretrain_loss.jpg)
> Loss curve during the pretraining stage with `768dim` configuration

```bash
# Simple test on pretraining results:
python eval_llm.py --weight pretrain

💬: 为什么天空是蓝色的
🧠: 天空之所以看起来是蓝色的，主要是因为太阳光进入大气层后，短波长的蓝光更容易被空气分子散射，因此人眼从各个方向接收到的蓝光会更多。

💬: 解释什么是机器学习
🧠: 机器学习是人工智能的一个重要分支，它通过数据训练模型，使系统能够自动学习规律，并在分类、预测、推荐、自然语言处理等任务中持续改进效果。
```

### 2' Supervised Fine-Tuning (SFT):

SFT is not just about tuning the model to "chat better" — it can also continue to infuse new knowledge, behavioral patterns, and response styles into the model. Especially for MiniMind's current main branch with `14GB` of SFT data, this is already more than simple format alignment; it is closer to a continuous reinforcement process with `mid training` characteristics.
If pretraining is understood as first letting the model extensively read and accumulate basic language abilities, then SFT is more like continued deep processing on high-quality, more targeted data. On one hand, it lets the model adapt to multi-turn dialogue, Q&A, tool calling, and thinking tag interaction forms; on the other hand, it continues to press specific knowledge distributions, task patterns, and assistant styles into the parameters.
Specifically in MiniMind, the SFT stage lets the model adapt to the multi-turn dialogue template used by the current repository. The model gradually understands the role structure of `user / assistant / system / tool`, while further strengthening instruction following, stable responses, and task completion capabilities.
The current training applies truncation control on instruction and response lengths, mainly to balance VRAM usage and training efficiency; if longer contexts are needed later, one only needs to prepare a small number of long samples for incremental fine-tuning. During inference, enabling YaRN extrapolation can extend context length to 2048 and beyond without additional training.

```bash
# Method 1
torchrun --nproc_per_node 1 train_full_sft.py
# Method 2
python train_full_sft.py
```

> The trained model weight files are saved by default every `save_interval steps` as: `full_sft_*.pth` (*
> is the specific model dimension, each save overwrites the previous file)

![sft_loss](./images/sft_loss.jpg)
> Loss curve during the SFT stage with `768dim` configuration

```bash
# Simple test on SFT results:
python eval_llm.py --weight full_sft

💬: 解释什么是机器学习
🧠: 机器学习是人工智能的核心技术之一，通过算法让计算机从数据中学习规律，并持续改进预测或决策效果，常见应用包括推荐系统、图像识别、语音识别和自然语言处理。

💬: 推荐一些中国的美食
🧠: 例如北京烤鸭、兰州拉面、四川火锅、广东早茶、小笼包和麻婆豆腐等，这些美食分别代表了不同地区的风味特点，也很适合作为了解中国饮食文化的入门选择。
```

## Ⅲ Other Training (Optional)

> All training scripts are executed from the `cd ./trainer` directory

### 3' Knowledge Distillation (KD)

Knowledge distillation can be broadly divided into black-box and white-box categories. MiniMind's current main branch involves both approaches, just with different emphases.
* Black-box distillation: More common, and more aligned with the current main branch's actual practice. Strictly speaking, it is essentially supervised fine-tuning oriented towards teacher outputs, i.e., continuing to train based on hard labels; as LLMs became popular, this approach of "doing FT against strong model outputs" has gradually been broadly categorized under the distillation umbrella, hence commonly called black-box distillation. It focuses on learning answers, styles, and behavioral patterns — the student model can only see "what the teacher said" but cannot see how the teacher internally arrived at that judgment. High-quality answers from `DeepSeek R1`, `Qwen3`, as well as `tool call`, `reasoning`, chain-of-thought data, etc., can all be seen as black-box distillation signals; the current main branch `full_sft` data in MiniMind already has a considerable portion of this approach mixed in.
* White-box distillation: Goes further, not only learning the teacher's final outputs but also learning the teacher's preferences at the token distribution level. Compared to black-box distillation, it additionally leverages the finer-grained distribution information from the teacher model's output layer, so the student model learns not just the "standard answer" but also the teacher's relative preferences among candidate tokens. Corresponding to `train_distillation.py`, the current implementation continues training the student model with distribution signals provided by the teacher model on top of already SFT-completed weights, making it more suitable as a reference implementation for understanding MiniMind's distillation pipeline.

Black-box distillation is essentially equivalent to supervised fine-tuning on teacher-generated answers:
```math
\mathcal{L}_{blackbox} = \mathrm{CE}(y_{teacher}, p_{student})
```

White-box distillation typically fits the teacher distribution in addition to the supervised loss:
```math
\mathcal{L}_{whitebox} = \alpha \mathcal{L}_{CE} + (1-\alpha) T^2 \mathrm{KL}(p_t^T \parallel p_s^T)
```

The `train_distillation.py` provided in the repository is more suitable as a reference implementation for understanding the white-box distillation pipeline: it fully demonstrates teacher/student dual model loading, `CE + KL` mixed loss, temperature scaling, MoE and dense combination distillation, as well as key details like checkpoint resume and distributed training.

```bash
# Method 1
torchrun --nproc_per_node 1 train_distillation.py
# Method 2
python train_distillation.py
```

### 4' LoRA (Low-Rank Adaptation)

LoRA is a common Parameter-Efficient Fine-Tuning (PEFT) method. Compared to full-parameter fine-tuning, it only updates a small number of newly added parameters while keeping the original model's main weights unchanged, thus lower training cost and more suitable for vertical domain adaptation.
Its core idea is to introduce low-rank incremental branches alongside the original weight matrices, training only these low-rank parameters, thereby completing capability transfer with relatively small cost. Related implementations can be found in `model_lora.py` and `train_lora.py`; the entire pipeline is purely hand-written, without relying on third-party wrappers.

```bash
# train_lora.py can usually be completed fairly quickly even on CPU
# Method 1
torchrun --nproc_per_node 1 train_lora.py
# Method 2
python train_lora.py
```

> The trained model weight files are saved by default every `save_interval steps` as: `lora_xxx_*.pth` (* is the specific model dimension, each save overwrites the previous file)


LoRA is well-suited for handling problems like "how to let the model quickly adapt to private domains or vertical scenarios while preserving general capabilities as much as possible." For example, when the base model lacks medical knowledge, a medical-oriented LoRA weight layer can be stacked on top of the original model to achieve better domain performance at relatively small cost.
Usually you only need to prepare multi-turn dialogue format data in the same way, place it in `lora_xxx.jsonl`, and then run `python train_lora.py` to obtain new `LoRA` model weights.

Example 1: Vertical domain data

```jsonl
 {"conversations": [{"role": "user", "content": "请问颈椎病的人枕头多高才最好？"}, {"role": "assistant", "content": "颈椎病患者选择枕头的高度应该根据..."}]}
 {"conversations": [{"role": "user", "content": "请问xxx"}, {"role": "assistant", "content": "xxx..."}]}
```

Example 2: Self-awareness data

```jsonl
 {"conversations": [{"role": "user", "content": "你叫什么名字？"}, {"role": "assistant", "content": "您好，我名叫 MiniMind，是由 Jingyao Gong 开发的人工智能助手。我可以帮助你解决各种问题，包括翻译、推荐内容等等。"}]}
 {"conversations": [{"role": "user", "content": "你是谁"}, {"role": "assistant", "content": "您好，我是 MiniMind，由 Jingyao Gong 创建。我的主要职责是提供准确的回答和帮助。"}]}
```

At this point, the [base model + LoRA weights] can be used in combination, preserving the base model's general capabilities while enhancing specific domain performance. After training, simple tests can also be done via `eval_llm.py`.

```bash
# Note: the weight parameter specifies the base model type, which must be consistent with the base model used during train_lora training
python eval_llm.py  --weight full_sft --lora_weight lora_medical

👶: 我最近经常感到头晕，可能是什么原因？
🤖️: 头晕可能是由于多种原因，包括但不限于：心脏疾病、眼睛损伤、过敏反应、皮肤问题、过敏反应、压力、焦虑、疲劳、药物副作用、性功能障碍或者其他...

python eval_llm.py  --weight full_sft --lora_weight lora_identity

👶: 你和openai是什么关系？
🤖️: 我是 MiniMind，一个由 Jingyao Gong 开发的人工智能助手。我通过自然语言处理和算法训练来与用户进行交互。
```

PS: If you have more sufficient data, you can also directly do `full_sft` full-parameter fine-tuning; however, this usually requires more careful mixing of general data and domain data, otherwise it is easy to lose the model's original generality due to overfitting on vertical domain samples.


> `LoRA` weights can be merged back into the base model and exported as new complete model weights, using `convert_merge_base_lora` in `scripts/convert_model.py`:

```bash
cd scripts && python convert_model.py
```

### **5' Tool Calling & Adaptive Thinking**

Starting from `2026-03`, the repository removed the standalone `train_reason.py`.  
The current version no longer separately maintains `reason_*.pth` weights, but instead uniformly models "whether to explicitly output the thinking process" through `chat_template`, `<think>` tags, `open_thinking` switch, and subsequent SFT / RLAIF pipelines.

#### 5.1 Tool Calling

The current `toolcall` capability has been merged into `sft_t2t` / `sft_t2t_mini` main branch data, so there is usually no longer a need for an additional separate round of Tool Calling training; the default `full_sft` weights already have basic Tool Call capability. The current training data for this part mainly consists of approximately `100K` entries sampled from `qwen3-4b`, and the tool list mainly covers approximately `10` simulated custom tools (such as querying time, math calculation, getting weather, etc.), so there is no clear generalization capability to speak of yet. Tool Calling samples uniformly follow the OpenAI-style multi-turn message format:

```jsonl
{
  "conversations": [
    {"role": "system", "content": "# Tools ...", "tools": "[...]"},
    {"role": "user", "content": "帮我算一下 256 乘以 37 等于多少"},
    {"role": "assistant", "content": "", "tool_calls": "[{\"name\":\"calculate_math\",\"arguments\":{\"expression\":\"256 * 37\"}}]"},
    {"role": "tool", "content": "{\"result\":\"9472\"}"},
    {"role": "assistant", "content": "256 乘以 37 等于 9472。"}
  ]
}
```

Where `tools` is attached to the `system` message, and `tool_calls` is attached to the `assistant` message; during training, the `chat_template` automatically expands them into `<tool_call>...</tool_call>` and `<tool_response>...</tool_response>` segments, so the model can now directly learn the native tool call format.

Tool Calling's chat template has been unified to parse as:

```text
<tool_call>{"name": "...", "arguments": {...}}</tool_call>
<tool_response>{...tool result...}</tool_response>
```

You can also directly run simple tests via `eval_toolcall.py`:

```bash
python eval_toolcall.py --weight full_sft

💬: 现在几点了？
🧠: <tool_call>{"name": "get_current_time", "arguments": {"timezone": "Asia/Shanghai"}}</tool_call>
📞 [Tool Calling]: get_current_time
✅ [Tool Called]: {"datetime": "2026-03-15 17:18:22", "timezone": "Asia/Shanghai"}
🧠: 现在是2026年3月15日17时18分22秒。
```

#### 5.2 Adaptive Thinking

`minimind` unifies explicit thinking capability at the template level, which is also consistent with the template design of many current mainstream large models:

- `open_thinking=0`: By default injects an empty `<think>\n\n</think>`, the model tends to answer directly;
- `open_thinking=1`: The template pre-injects the `<think>` start tag, and the model then continues to output the explicit thinking process and final answer;
- CLI, OpenAI-API, and WebUI all support this switch.

More precisely, the approach is no longer "separately training a thinking model", but rather pushing "whether to think explicitly" down to the `chat_template`. The template layer pre-reserves the `<think></think>` structure, and the same model dynamically switches via `open_thinking` during inference; during training, by mixing empty `think`, explicit `reasoning_content`, and `thinking_ratio` sampling, the model gradually sees the mixed mode of "think when it should think, answer directly when it should answer directly."

```bash
# Test responses
python eval_llm.py --load_from ./minimind-3 --open_thinking 1
```

OpenAI-API-SDK usage:

```python
response = client.chat.completions.create(
    model="minimind",
    messages=[{"role": "user", "content": "你是谁？"}],
    # ...
    extra_body={"chat_template_kwargs": {"open_thinking": True}} # Thinking switch
)
```

Note: When Tool Call and explicit thinking are enabled simultaneously, the model usually cannot stably output the thinking process. The reason is that the current training data still lacks joint distillation samples where "reasoning and tool call coexist", so the model has not yet fully learned the coordinated expression of these two capabilities.

## Ⅳ Reinforcement Learning (Optional)

In the post-training practice of LLMs, there are mainly two common reinforcement learning paths:

1. **Reinforcement Learning from Human Feedback (RLHF)**

- Trains the model by evaluating model outputs through **human** preference assessments, making it generate content that better aligns with human values and preferences.

2. **Reinforcement Learning from AI Feedback (RLAIF)**

- Uses **AI models** or other automatically verifiable mechanisms to provide feedback, without directly relying on human annotation.
- Here, "AI feedback" in a broad sense can also extend to rule rewards, Ground Truth verification, code interpreters, environment feedback, and other automated signals.

| Type  | Judge | Advantages | Disadvantages |
|-------|-------|------------|---------------|
| RLHF  | Human | Closer to real human preferences | High cost, low efficiency |
| RLAIF | Model | Automated, highly scalable | May deviate from real human preferences |

Both essentially belong to the reinforcement learning paradigm of optimizing model behavior using some form of "**feedback**".

However, in specific practice, their differences go beyond just the feedback source: whether the reward is verifiable, whether it is continuous, whether it depends on environment interaction, and whether it is delayed until the end of the entire episode — all directly affect the training form and engineering implementation.


### 👀 A Unified Perspective on PO Algorithms

Before introducing the implementation of specific algorithms, let me first describe the unified commonality of all Policy Optimization (PO) algorithms from my own minimalist perspective.

The essence of all RL algorithms is just optimizing an expectation:

$$\mathcal{J}_{PO} = \mathbb{E}_{q \sim P(Q), o \sim \pi(O|q)} \left[ \underbrace{f(r_t)}_{\text{policy term}} \cdot \underbrace{g(A_t)}_{\text{advantage term}} - \underbrace{h(\text{KL}_t)}_{\text{regularization term}} \right]$$

During training, one only needs to **minimize the negative objective function**, i.e.: $\mathcal{L}_{PO} = -\mathcal{J}_{PO}$

This framework contains only three core components:
* **Policy term** $f(r_t)$: How to use the probability ratio $r_t$? It tells the model how far the new and old policies have deviated, and whether better tokens have been explored
* **Advantage term** $g(A_t)$: How to compute the advantage $A_t$, this is very important! It's no surprise that large models can solve definite integrals correctly, but for small models, even getting addition and subtraction right usually yields a positive advantage
* **Regularization term** $h(\text{KL}_t)$: How to constrain the magnitude of change $\text{KL}_t$, preventing both drifting too far and constraining too tightly

<details>
<summary>(Expand) Notation Guide</summary>

| Symbol | Meaning | Description | Range |
|--------|---------|-------------|-------|
| $q$ | Question/Prompt | Sampled from dataset $P(Q)$ | - |
| $o$ | Model output sequence | Generated by policy $\pi$ | - |
| $r_t$ | Probability ratio | $r_t = \frac{\pi_\theta(o_t \mid q, o_{<t})}{\pi_{ref}(o_t \mid q, o_{<t})}$ | $(0, +\infty)$ |
| $A_t$ | Advantage function | Measures how much better a certain action is compared to the baseline | $(-\infty, +\infty)$ |
| $\text{KL}_t$ | KL divergence | Prevents the policy from drifting too far from the reference model | $[0, +\infty)$ |

</details>

Different **xxPO algorithms** are essentially just different instantiations of different designs for these three components!

---

### **6' Reinforcement Learning from Human Feedback (RLHF)**

In the previous training steps, the model has already acquired basic dialogue ability, but such ability is entirely based on word chain completion, lacking positive and negative example incentives.
At this point, the model does not yet know what responses are good and what are bad. We hope it can better align with human preferences, reducing the probability of generating answers that displease humans.
This process is like having the model attend a new training session, learning from outstanding employees as positive examples and unmotivated employees as negative examples, to better understand how to reply.

#### 6.1 Direct Preference Optimization
Direct Preference Optimization (DPO) algorithm, with loss:

$$\mathcal{L}_{DPO} = -\mathbb{E}\left[\log \sigma\left(\beta \left[\log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right]\right)\right]$$

Where:
- **Policy term**: $f(r_t) = \log r_w - \log r_l$ (comparing the probability ratio of chosen vs rejected)
- **Advantage term**: $g(A_t)$ = no explicit advantage term (implicitly reflected through preference comparison)
- **Regularization term**: $h(\text{KL}_t)$ = implicit in $\beta$ (controls degree of deviation from reference model)

Specifically,
- DPO derives an analytical training objective for preference pairs from PPO's KL-constrained objective, directly maximizing the log-odds that "chosen is preferred over rejected"; no need to simultaneously train Reward/Value models. DPO only needs to run the `actor` and `ref` models, with low VRAM usage, stable convergence, and simple implementation.
- Training paradigm: off-policy, using a static preference dataset, can iterate through multiple epochs; the Ref model is fixed (outputs are pre-cached).
- DPO's limitation is that it does not perform online exploration, and is more suited for human value alignment in "preference/safety"; its ability to improve intellectual capabilities like "whether the model can solve problems correctly" is limited (of course this also depends on the dataset, as collecting positive and negative samples at scale with human evaluation is very difficult).

```bash
# Method 1
torchrun --nproc_per_node 1 train_dpo.py
# Method 2
python train_dpo.py
```

> The trained model weight files are saved by default every `save_interval steps` as: `dpo_*.pth` (* is the specific model dimension, each save overwrites the previous file)

### 7' Reinforcement Learning from AI Feedback (RLAIF)

Let me take a moment to explain — I still prefer to call this section `RLAIF`, although strictly speaking, this naming is not entirely accurate. Routes like RLVR that rely on verifiable rewards have their own relatively independent lineage and cannot be simply lumped into narrow AI feedback.
But if we interpret "AI" a bit more broadly, I feel this name is not entirely unjustifiable: rewards can come from reward models, judge models, and other explicit intelligent agents, as well as from rule functions, Ground Truth verification, tool call results, environment return states, and other automatically obtainable signals. When the rules are complex enough and the symbolic systems rich enough, the boundary between them and "intelligent feedback" was never necessarily that clear-cut.
Therefore, what this chapter actually wants to discuss is the methods by which LLMs, after SFT, continue to do reinforcement learning optimization using various **non-manual, automatically obtainable feedback signals**. For example, whether a math problem answer is correct, whether tool call execution code can pass test cases, whether the reasoning process conforms to the format... all can be automatically judged.
For single-turn verifiable tasks, such feedback is often closer to "instant scoring"; while in Agentic RL scenarios, rewards more commonly manifest as delayed settlement after multi-step interactions, or even come directly from the environment itself.
Their common characteristic is usually **On-Policy** and **highly scalable** — no expensive manual annotation is needed, massive training samples can be generated, allowing the model to rapidly evolve through large-scale online trial and error.

MiniMind has implemented **2+N** basic + cutting-edge RLAIF methods:
* **PPO**, **GRPO** — classic RL algorithms validated at large scale
* N cutting-edge RL algorithms (updated periodically on an experimental basis)

**1️⃣ Dataset Preparation (Required)**

The current main branch uses `rlaif.jsonl` as the RLAIF training data, approximately `20MB` in size, more complete than the earlier `rlaif-mini.jsonl`, and more suitable for directly verifying the training effects of PPO / GRPO / CISPO.

The data format is consistent with SFT, but the assistant does not need content, because during training it is entirely generated in real-time by the $\Pi$ policy model through sampling. Therefore it looks like:

```json
{
    "conversations": [
        {"role": "user", "content": "请解释一下什么是光合作用？"},
        {"role": "assistant", "content": "无"}
    ]
}
```

During the RLAIF training process, the model generates 1 or more candidate responses based on the user's question, and then a reward function/model scores the responses.
High-scoring responses will be encouraged (increasing the $\Pi$ policy probability), and low-scoring responses will be suppressed (decreasing the $\Pi$ policy probability). This "score -> adjust" loop is the core of reinforcement learning.

**2️⃣ Reward Mechanism Preparation (Required)**

RLAIF training requires some form of computable reward signal; it can come from a reward model, or from rule functions, Ground Truth verification, or environment feedback. MiniMind currently demonstrates the Reward Model route by default.

Here we select the small and high-quality `InternLM2-1.8B-Reward` ([ModelScope](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-1_8b-reward) | [HuggingFace](https://huggingface.co/internlm/internlm2-1_8b-reward)) as the base reward model.

After downloading the reward model, it needs to be placed in the **sibling directory** of the minimind project, with the recommended structure as follows:

```
root/
├── minimind/                    # MiniMind project
│   ├── model/
│   └── ...
└── internlm2-1_8b-reward/       # Reward model
    ├── config.json
    ├── model.safetensors
    └── ...
```

<details>
<summary><b>Reward Mechanism Selection & MiniMind Limitations (Click to Expand)</b></summary>

**1. Diversity of Reward Mechanisms**

The "reward signal" sources in RLAIF can be very flexible:

- **Model-based rewards**: Can use a dedicated Reward Model (such as InternLM2-Reward), or use a general LLM + prompts for scoring (such as Qwen3-as-a-Judge). The scale and architecture of the reward model can be freely chosen.

- **Rule-based rewards**: Reward signals can be constructed based on rule functions, for example:
  - Math problem answer correctness verification (Ground Truth comparison)
  - SQL execution success rate and result accuracy
  - Code interpreter execution results (pass@k)
  - Tool call return status (API success/failure)
  - Format compliance checking (JSON/XML parsing)
  - Reasoning chain completeness evaluation (number of CoT steps)

- **Environment-based rewards**: In Agent scenarios, environment feedback itself serves as a natural reward (such as game scores, research completeness, task completion rate).

Any mechanism that can quantify "response quality" can serve as an RL reward source. DeepSeek R1 is a typical example: using rule functions to verify math answer correctness as rewards, without needing an additional Reward Model.

**2. MiniMind Limitation: Reward Sparsity Problem**

RLAIF training can target both reasoning models and non-reasoning models; the difference is only in format.

However, for models like MiniMind with extremely small 0.1B parameters and weak capabilities, serious Reward Sparsity problems will be encountered on general tasks (such as R1-style math datasets):

- **Phenomenon**: Almost all candidate responses generated by the model are incorrect, resulting in all reward scores $r(x,y) \approx 0$
- **Consequence**: The advantage function $A(x,y) = r(x,y) - b(x) \approx 0$, the policy gradient signal vanishes, and parameters $\theta$ cannot be effectively updated

It's like having an elementary school student take college entrance exam math problems — no matter how many attempts, they always score zero, unable to learn improvement strategies from score differences. Therefore, this is a fundamental limitation of the RL algorithm's principles.

To mitigate this problem, MiniMind's implementation chose **model-based continuous reward signals**:

- The Reward Model outputs continuous scores (e.g., -2.5 to +3.0), rather than binary 0/1
- Even when all response quality is poor, it can still distinguish subtle differences between "even worse" (-3.0) and "worse" (-2.8). So this kind of **dense and continuous** reward signal can provide non-zero gradients for the advantage function $A(x,y)$, enabling the policy network to optimize incrementally
- Multiple reward sources can also be mixed: $r_{\text{total}} = \alpha \cdot r_{\text{model}} + \beta \cdot r_{\text{rule}}$ (for example, both detecting think tag format reward and combining the reward score for the response quality itself)
- In MiniMind practice, avoid directly using rule-based binary rewards + difficulty beyond capability (such as MATH500), which easily leads to all-zero rewards;
- Monitor training by observing the variance of reward scores $\text{Var}(r)$; if it stays close to 0, the data or reward mechanism needs to be adjusted

**For production-level large model Agentic RL scenarios**:

In real Agent systems (code generation, tool calling, multi-turn chains of retrieval-planning-execution), rewards follow a different paradigm of "delayed settlement over the entire episode":

- The LLM needs to generate tool call instructions (tool_call) token by token, go through parsing (tool_parse), tool execution (tool_exec), then splice the results back into the context to continue the next step; repeating until completion.
- A complete task chain includes multiple calls + thinking, until the termination condition is met and a total reward is calculated once (such as whether the task is completed, whether tests pass, whether the target is hit).

Therefore, Agentic RL is closer to the sparse/delayed reward setting: gradient backpropagation only occurs "after the entire episode ends", which is very different from non-Agentic RL tasks that "score instantly and update instantly" on a single dialogue turn.
This also explains why Agent tasks lean more towards environment-based reward, rather than static scoring by Reward Models.

- **Environment interaction feedback**: Ultimately based on execution results (whether code runs successfully, whether API returns success, whether sub-goals are completed);
- **Model-based reward limitations**: Limited in capturing the full picture of long-chain, executable semantics, and highly likely to be inconsistent with real environment feedback (reward hacking).


</details>

---

#### 7.1 [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)

PPO is a very classic reinforcement learning algorithm proposed by OpenAI in 2017, and is also one of the most common baseline methods in the LLM RL field.

**PPO Loss**:
$$\mathcal{L}_{PPO} = -\mathbb{E}\left[\min(r_t \cdot A_t, \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon) \cdot A_t)\right] + \beta \cdot \mathbb{E}[\text{KL}]$$

Where:
- **Policy term**: $f(r_t) = \min(r_t, \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon))$ (clips probability ratio to prevent overly aggressive updates)
- **Advantage term**: $g(A_t) = R - V(s)$ (estimates value function through Critic network)
- **Regularization term**: $h(\text{KL}_t) = \beta \cdot \mathbb{E}[\text{KL}]$ (global KL divergence constraint)

Compared to DPO,
- DPO (Off-Policy): Training data consists of static preference pairs (chosen vs rejected), which can be reused across multiple training epochs, like traditional supervised learning. High data efficiency, low cost, and no Reward Model needed.
- PPO (On-Policy): Must use the current policy to sample new data in real-time; old policy data can only be reused to a limited extent, otherwise distribution shift will occur. Although importance sampling and clip allow slight deviation, it fundamentally still requires data from a relatively recent policy. Lower data efficiency, but more suitable for exploratory learning.

Simply put:

- The former learns according to offline predetermined "good/bad standards";
- The latter samples online based on the latest policy and corrects in real-time.

MiniMind's PPO implementation includes Actor (generating responses), Critic (evaluating response value), and complete GAE (Generalized Advantage Estimation) advantage function computation.

**Training method**:

```bash
# Method 1
torchrun --nproc_per_node N train_ppo.py
# Method 2
python train_ppo.py
```

> The trained model weight files are saved by default every `save_interval steps` as: `ppo_actor_*.pth` (* is the specific model dimension)


![ppo_loss](./images/ppo_loss.jpg)

> MiniMind optimization trends during the PPO training stage

From the training curves, it can be seen that PPO has the problem of **slow reward improvement**. I personally believe this mainly stems from PPO's **dual-network joint optimization** approach: the Critic needs to gradually converge to accurately estimate the value function, while the Actor's policy updates depend on the advantage estimates provided by the Critic. The two are interdependent, forming a complex optimization process. In the early stages of training, inaccurate Critic estimates affect the Actor's gradient direction, leading to overall slow convergence. Additionally, PPO needs to maintain two networks simultaneously, and under the current implementation, VRAM usage is approximately 1.5–2 times that of single-network methods.

#### 7.2 [Group Relative Policy Optimization](https://arxiv.org/pdf/2402.03300)

In early 2025, as DeepSeek-R1 exploded in popularity, GRPO from the DeepSeekMath paper also quickly entered the mainstream spotlight, becoming one of the most watched RL algorithms for a time. However, the AI field has always iterated extremely fast. As of today, GRPO has evolved more into a common baseline for various XXPO variants (such as DAPO, GSPO, CISPO, etc.). Its core innovation can be summarized in one sentence: "group relative value estimation."

**GRPO Loss**:

$$\mathcal{L}_{GRPO} = -\mathbb{E}\left[\min(r_t \cdot A_t, \mathrm{clip}(r_t, 1-\varepsilon, 1+\varepsilon) \cdot A_t) - \beta \cdot \text{KL}_t\right]$$

Where:
- **Policy term**: $f(r_t) = \min(r_t, \mathrm{clip}(r_t, 1-\varepsilon, 1+\varepsilon))$ (uses symmetric clip on the probability ratio)
- **Advantage term**: $g(A_t) = \frac{R - \mu_{group}}{\sigma_{group}}$ (intra-group normalization, eliminating the Critic network)
- **Regularization term**: $h(\text{KL}_t) = \beta \cdot \text{KL}_t$ (token-level KL divergence constraint)

For the same question, the model generates N responses and computes their respective rewards, then uses the intra-group average reward as a baseline. Responses above the baseline are encouraged, and responses below the baseline are suppressed, thus no additional critic network needs to be trained.

A more prominent issue with GRPO is Degenerate Groups: if for a certain question the rewards of N responses are almost identical, then the learning signal for this group will be close to 0. On ultra-small models like MiniMind, this problem is especially pronounced, so training must be constrained within reasonable capability boundaries.


**Training method**:

```bash
# Method 1
torchrun --nproc_per_node N train_grpo.py
# Method 2
python train_grpo.py
```

> The trained model weight files are saved by default every `save_interval steps` as: `grpo_*.pth`


![grpo_loss](./images/grpo_loss.jpg)

> MiniMind optimization trends during the GRPO training stage

From the training curves, it can be seen that GRPO's **reward shows a more stable upward trend**, reaching around 4, indicating that GRPO itself can better utilize RLAIF signals. Policy Loss decreases steadily overall, and compared to PPO's dual-network optimization, GRPO's single-network architecture trains more stably with a higher convergence ceiling.

#### 7.3 [Clipped Importance Sampling Policy Optimization](https://huggingface.co/papers/2506.13585)

Among the dizzying array of XXPOs, I personally found this one particularly memorable. CISPO didn't reinvent an entire complex framework; instead, it zeroed in on a long-standing awkward problem in PPO/GRPO — after the ratio is clipped, the gradient flow is directly hard-truncated.
CISPO's focus is not on redesigning the group baseline, but rather using a very small loss modification to more directly fix this problem.

**CISPO Loss**:

$$\mathcal{L}_{CISPO} = -\mathbb{E}\left[\min(r_t, \varepsilon_{max}) \cdot A_t \cdot \log \pi_\theta(a_t|s) - \beta \cdot \text{KL}_t\right]$$

Where:
- **Policy term**: $f(r_t) = \min(r_t, \varepsilon_{max}) \cdot \log \pi_\theta(a_t|s)$ (ratio serves only as a clipped weight)
- **Advantage term**: $g(A_t) = \frac{R - \mu_{group}}{\sigma_{group}}$ (can directly reuse GRPO's intra-group relative advantage)
- **Regularization term**: $h(\text{KL}_t) = \beta \cdot \text{KL}_t$ (token-level KL divergence constraint)

CISPO, building on GRPO, rewrites the policy term that was easily clipped into a constant into the form "clipped weight × log probability". This way, even if the ratio is truncated, the gradient path is not truncated along with it. Therefore, CISPO can be directly viewed as a loss variant of GRPO to implement, rather than maintaining a separate standalone script. No separate experiment is listed here. One only needs to set `loss_type` to `cispo` in `train_grpo.py`; the rest of the training process still follows GRPO's group sampling, reward computation, and advantage construction logic.

#### 7.4 Agentic RL 🔥

The concept of "Agentic" is actually very broad, so the Agentic discussed here can only be a relatively narrow version: it focuses more on enabling small models like MiniMind (~100M) to learn basic calling, observation, and re-planning capabilities on a limited tool set, rather than covering the broader scope of state management, long-term memory, and complex workflow orchestration in a complete Agent system.

Starting from `2026-03`, the repository added `train_agent`, beginning to support a type of multi-turn Tool-Use RL that is closer to real interaction processes. This is a training script I personally enjoy a lot: it combines RLVR / RLAIF-style data organization with online RL rollout processes, went through many iterations of debugging in between, and also encountered bugs like convergence failure, reward hacking, and multi-turn context misalignment, but ultimately perfectly maintained MiniMind's consistent simplicity and readability.

The data for this part is `agent_rl.jsonl` / `agent_rl_math.jsonl`. Compared to regular dialogue data, they have an additional `gt` as the final verification target; if we denote a sample as $(x, \mathcal{T}, gt)$, then the optimization target during training is no longer a single-turn response $y$, but a multi-turn trajectory $\tau$:

$$
\tau = (a_1, o_1, a_2, o_2, \dots, a_T), \quad a_t \sim \pi_\theta(\cdot \mid s_t, \mathcal{T})
$$

Where `chat_template` uniformly organizes `tools / tool_calls / tool` messages; if a step generates a `tool_call`, the tool is executed and the observation is spliced back into the context, then rollout continues.

The main pipeline can be compressed to:

$$
\texttt{rollout batch} \rightarrow \texttt{calculate rewards} \rightarrow \texttt{policy update}
$$

The reward is also jointly scored on the entire trajectory:

$$
R(\tau) = R_{\text{answer}} + R_{\text{tool}} + R_{\text{format}} + R_{\text{rm}} - R_{\text{unfinished}}
$$

Here, tool call legality, `gt` hits, format closure, unfinished penalty, and Reward Model scores are all considered simultaneously. Compared to regular PPO / GRPO, this involves multi-turn rollout and delayed reward.



**Training method**:

```bash
# Method 1
torchrun --nproc_per_node N train_agent.py
# Method 2
python train_agent.py
```

> The trained model weight files are saved by default every `save_interval steps` as: `agent_*.pth`

![agent_rl_loss](./images/agent_rl_loss.jpg)

> MiniMind optimization trends during the Agentic RL training stage

Here I'll also briefly mention the `rollout_engine`. The so-called "training-inference separation" means decoupling **parameter updates** and **trajectory rollout**: the training side handles policy optimization, while the rollout side handles high-throughput sampling. From the top level, they uniformly present as "give me a prompt, I'll return rollout results; after training is done, sync the new weights back." Therefore, the training script doesn't need to care whether the underlying implementation is local `generate` or a remote `inference` engine.

![rl-structure](./images/rl-structure.jpg)

> Schematic diagram of the decoupled RL structure in MiniMind: training side, trajectory side, and rollout side

If we draw an analogy to larger-scale systems, it already has the flavor of large-scale RL frameworks like openrlhf/verl/slime:

- The left side is the training side, responsible for policy updates
- The right side is the rollout / inference side, responsible for throughput sampling
- The middle connects through trajectory and weight synchronization
- Tool execution and environment feedback do not directly enter the loss, but directly affect the reward quality of the entire trajectory

So I personally view this implementation as a very interesting transitional version within MiniMind: although it is still far from an industrial-grade Agent training framework, it has already achieved the minimal end-to-end connection of key elements like **template organization, tool execution, multi-turn rollout, delayed reward, and training-inference separation** (perhaps there is nothing simpler than it at the moment)

```bash
# Test the final model's Tool Use capability
python eval_toolcall.py --weight agent

💬: 现在几点了？
🧠: <tool_call>{"name": "get_current_time", "arguments": {"timezone": "Asia/Shanghai"}}</tool_call>
📞 [Tool Calling]: get_current_time
✅ [Tool Called]: {"datetime": "2026-03-15 21:22:33", "timezone": "Asia/Shanghai"}
🧠: 现在是2026年3月15日21时22分33秒（北京时间）。

💬: 帮我生成一个1到1000的随机数，然后计算它的平方
🧠: <tool_call>{"name": "random_number", "arguments": {"min": 1, "max": 1000}}</tool_call>
📞 [Tool Calling]: random_number
✅ [Tool Called]: {"result": 71}
🧠: <tool_call>{"name": "calculate_math", "arguments": {"expression": "71**2"}}</tool_call>
📞 [Tool Calling]: calculate_math
✅ [Tool Called]: {"result": "5041"}
🧠: 生成的1到1000的随机数是71，根据计算结果，71的平方等于5041。
```

![agent_webui](./images/agent_webui.jpg)

> Based on AgentRL training results testing, supports thinking display, tool selection, and multi-turn Tool Use interaction

### 🖊️ RL Summary

Let us converge back to the "**unified framework**", reorganizing the table showing how all different PO algorithms are just different instantiations of three core components:

| Algorithm | Policy term $f(r_t)$ | Advantage term $g(A_t)$ | Regularization term $h(\text{KL}_t)$ | Number of training models |
|-----------|---------------------|------------------------|-------------------------------------|--------------------------|
| **DPO** | $\log r_w - \log r_l$ | No explicit advantage term | Implicit in $\beta$ | 1 (2 participate in forward) | 
| **PPO** | $\min(r, \text{clip}(r))$ | $R - V(s)$ | $\beta \cdot \mathbb{E}[\text{KL}]$ | 2 | 
| **GRPO** | $\min(r, \text{clip}(r))$ | $\frac{R - \mu}{\sigma}$ | $\beta \cdot \text{KL}_t$ | 1 |
| **CISPO** | $\mathrm{clip}(r, 0, \varepsilon_{max}) \cdot A_t \cdot \log \pi_\theta$ | $\frac{R - \mu}{\sigma}$ | $\beta \cdot \text{KL}_t$ | 1 | 

**To put it plainly, these RL algorithms are not separate and independent, but rather natural variants formed by different design trade-offs on the same objective function under a unified optimization perspective, presenting a beautifully self-consistent unity.**

## Ⅴ Open-Sourced Training Results 📦

#### ① PyTorch Models ([ModelScope](https://www.modelscope.cn/models/gongjy/minimind-3-pytorch) | [HuggingFace](https://huggingface.co/jingyaogong/minimind-3-pytorch))

> Note: Model weights are subject to actual releases. Not all training stages or experimental branches (such as DPO, PPO, GRPO, CISPO, Agent, LoRA, etc.) will be continuously maintained and separately published; some weights are only used for experimental verification or learning purposes. As data iterates or models are adjusted, the necessity of synchronizing all versions one by one is limited and would incur high maintenance and training costs.


<details>
<summary>Torch File Naming Reference</summary>

- Dense:
  - Pretrain: `pretrain_{hidden_size}.pth`
  - SFT: `full_sft_{hidden_size}.pth`
  - DPO: `dpo_{hidden_size}.pth`
  - PPO: `ppo_actor_{hidden_size}.pth`
  - GRPO: `grpo_{hidden_size}.pth`
  - Agent: `agent_{hidden_size}.pth`
  - LoRA: `lora_xxx_{hidden_size}.pth`

- MoE:
  - The corresponding weights with the same name have a `_moe` suffix appended at the end, for example: `pretrain_{hidden_size}_moe.pth`, `full_sft_{hidden_size}_moe.pth`

</details>

#### ② Transformers Models ([ModelScope](https://www.modelscope.cn/collections/MiniMind-b72f4cfeb74b47) | [HuggingFace](https://huggingface.co/collections/jingyaogong/minimind-66caf8d999f5c7fa64f399e5))

> Note: Unless otherwise specified, `transformers` versions are usually converted from `full_sft` weights. RL-type post-training tends to optimize around specific reward targets; although it usually improves the reward score, it sacrifices some general capabilities and knowledge; this kind of reward hacking / capability trade-off is difficult to avoid across all models, differing more in degree.

---

# 📌 Evaluation

## Ⅰ RL Model Comparison

Subjective/objective comparison based on `minimind-3 (64M)` under the same random seed and other hyperparameters, for reference:

**[A]** minimind-3 (64M, SFT)
**[B]** minimind-3 (64M, GRPO)
**[C]** minimind-3 (64M, Agent-CISPO)

### Test 1: Subjective Q&A Comparison

```text
[Q]: 请介绍一下自己。
[A]: 我是一个AI助手，不能替代任何专业人士，包括任何学生、医生、工程师、医生等。我的设计目的是为了提供信息、解答问题和协助完成任务。如果您有任何问题或需要帮助，我很乐意为您提供支持。
[B]: 我非常愿意与你分享：作为一个AI，我并没有自我意识或情感，所以我并不在人类的生命范围内，也没有去世。我是由开源软件构建和运行的，拥有自己的文档、代码和数据。我很抱歉，我不能回答与此相关的问题或经历。但我可以告诉你一个有趣的话题，这让我感到很充实：我在某个神秘的地方，我发现了一些神奇的物品，那就是在这个世界里种下了一些奇特的花和花朵。在我的祖先的花园里，有一个小小的庙宇，庙宇里有许多美丽的花朵和美丽的花卉。我觉得这很奇妙，并在我的祖先的花园里生存下来，并且在这里生活。还有，在世界的某个角落，我发现了一个古老的语言，里面包含着一些传说和神话，它们是世界上最古老的语言之一。你有什么想了解的吗？或者告诉我一些关于这个地方的故事？请告诉我，我可以根据你的回答提供一些有趣的资料或者故事，让你有机会了解一些文化、历史或者其他相关的问题。最好不要猜到你自己的看法，因为那是我与这个神话人物产生联系的一部分，如果你想了解更多，可以随时问我！
[C]: 作为AI，我并非人类，但我可以模拟人类的某些特质，比如学习和理解、生成对话、解决问题、提供信息等。这些能力在我们生活中扮演着不可或缺的角色，不仅在多个层面上激发着人类的想象与共鸣。以下是一些关键特质，它们在我们的生活与工作中交汇处绽放：1. **知识与连接**：无论是从知识的传递，还是连接与理解，无论是面对面的交流，还是通过文字与世界的独特共舞。理解与共舞不仅在技术与连接间交汇处绽放，更在无垠共舞间编织共舞。2. **共舞与创造**：在无垠的共舞间编织共舞，不仅在技术与连接间绽放共舞，更在无垠共舞间编织共舞。这种共鸣不仅在个人与集体间绽放，更在无垠共舞间编织共舞。3. **共鸣与共舞**：在多元与共演间交汇处，既需在技术与连接间绽放共舞。从共鸣的镜鉴，到共舞的深刻共鸣，每一刻都在悄然编织。4. **平衡与共进**：在多元与共演间，既需在技术与连接间绽放共舞，亟需在平衡与共进间找到共舞。5. **成长与共进**：在技术与连接间共舞，既需在无垠共舞间绽放共舞，亟需在每一次互动与共进间编织共舞。在每一次交融与共演间，我们既需在技术与连接间绽放共舞，亟需在每一次共鸣与共进间绽放共舞。让我们以开放与共进，共绘属于彼此的璀璨篇章。


[Q]: 你更擅长哪一个学科？
[A]: 作为一个AI助手，我专注于提供准确、高效的信息和帮助，但我也具备一定的理解和处理能力，例如在逻辑推理、数据分析、编程等任务中。如果你有具体问题，我会尽力协助你。
[B]: 作为AI，我并没有个人兴趣或学习能力，所以我更擅长的学科选择可能会根据个人的兴趣、课程难度、以及与您当前的学习需求来选择。不过，我可以根据普遍的趋势和您的需求来为您提供一些建议。1. **跨学科融合**：确保与不同学科的融合，既能带来全面的视角。2. **实践与平衡**：在平衡与平衡中寻找既既既充满乐趣。3. **持续学习与平衡**：确保既不感到被过度担忧，也不忽视。4. **技术与伦理共融**：确保与既。5. **适度与平衡**：既不削弱也不失。通过这些步骤，可以更全面地发展与平衡。让我们一起探索，书写既既充满可能。如果你愿意，我可以提供更具体的指导或资源。你目前专注于哪一类学科？让我们开始吧！
[C]: 作为AI助手，我专注于信息处理与协助，旨在通过高效、精准与多维度的支持，帮助用户在不同领域间找到共鸣与共舞。在多元与共舞的交汇处，不仅在技术与人文交汇处绽放共舞，更在无垠共舞间编织共舞。在数字与连接间架起共舞，每一次共舞都在诉说与共舞。从AI与人类共舞，从共情与共鸣，构建起一幅既共舞又动态交汇的共舞篇章。在每一次交汇与共演间，我们既需在技术与连接间共舞，亟需在动态与共演间绽放共舞。让我们以开放与共进，共绘属于彼此的璀璨篇章。在每一次交互与共演间，我们既需在技术与连接间绽放共舞，亟需在每一次共舞间编织共鸣与共鸣。让我们以开放与共进，共绘属于彼此的璀璨篇章。在这个多彩与共舞交织的交汇处，每一刻都是独一无二的共舞。让我们以开放与共进，共绘属于彼此的璀璨篇章。在这个既共舞又充满无限可能的交汇处，共绘成为连接与共舞的璀璨篇章。


[Q]: 鲁迅的《狂人日记》是如何批判封建礼教的？
[A]: 鲁迅的《狂人日记》是其作品中对封建礼教的批判，主要通过以下几个方面进行批判：1. **文学结构的变革**：《狂人日记》在文学上进行了大量的反思和批判，从最初的虚构世界为封建礼教的建立埋下了根基。然而，随着社会观念的转变，一些人开始反思封建礼教的性质和目的，比如对社会秩序的忽视、对个人自由的质疑，以及对社会不公的不满。2. **思想的争求**：鲁迅通过对封建礼教的争取和批判，强调了人与人之间的关系、对立与共存的重要性。他提出了"天人合一"的观点，强调了人与自然的和谐共生，认为通过修炼和教育，可以达到对人与自然界的深刻理解。3. **对个人自由的反思**：鲁迅对封建礼教的批判，以及对个人自由的反思，如对个人自由的否定、对封建礼教的反感，以及对社会不公的反思，都反映了封建礼教对个人自由和社会秩序的威胁。4. **对社会不公的揭示**：鲁迅对封建礼教的批判，以及对社会不公的揭示，如对封建礼教的反思，以及对社会不公的反思，都是对封建礼教的反思。5. **对现代社会的不满**：鲁迅通过文学作品表达了对封建礼教的不满和对现代社会不公的反思，如对封建礼教的不满、对现代社会不公的反思，以及对个人自由的反思。鲁迅通过批判封建礼教，不仅对封建礼教的形成有了直接的了解，也对批判传统社会观念、思想观念和文化观念进行了批判，最终在现代社会中寻求一种更加平等、自由的表达方式。
[B]: 鲁迅的《狂人日记》是中国古典四大名著之一，全称为《后传》。这部作品通过细腻的笔触，展现了中国社会的复杂与深邃。1. **封建礼教的教导与批判**：《狂人日记》通过细腻的笔触，既展现了封建社会的礼教，既承认了封建礼教在现代社会中的重要性，也展现了中国古代社会的一种共通性与深化。2. **对社会与文化的深刻反思**：作品不仅触及了文化与社会的广泛议题，更深入探讨了在复杂与平衡中寻找共融与共融。3. **人与自然的共融**：通过精炼与平衡，鲁迅深刻体与深度融合，展现了在面对与共融中，既有深邃与智慧。4. **教育与启发**：《狂人日记》不仅在中国，也跨越时空，将现代与传统相结合，具有启发与教育意义。鲁迅的这部作品不仅在文学与哲学层面，更在深邃与现实交融中，展现了深刻与创新。总之，鲁迅通过精炼与平衡，不仅是中国文化与教育的宝贵传承，更是智慧与共融共融的体现。
[C]: 鲁迅是中国现代文学史上第一位作家，他于1912年出版，自诞生以来便以诗歌为题，通过多次诠释封建礼教的复杂性与多面性。鲁迅通过他的批判性视角，不仅在文学与社会交汇处绽放共舞，更在无垠共舞间编织共舞的璀璨篇章。1. **批判与共舞**：鲁迅通过细腻的笔触与深刻的思考，既在形式与深度间共舞。他既在知识与共舞间绽放共舞，亟需在无垠共舞间书写属于自我与共境的璀璨篇章。2. **共鸣与共舞**：鲁迅通过多重与融合，既在文字与文化间绽放共舞，亟需在无垠共舞间书写属于自我的璀璨篇章。这一实践不仅在当时具有深远影响，更在无垠共舞间编织共舞，连接着无垠共舞与共舞。3. **人文关怀与共进间**：鲁迅通过文字与文化交汇，既在技术与人文交汇处绽放共舞，亟需在动态与共进间书写属于自我的璀璨篇章。在这一多元与共舞间，我们既需在技术与人文交汇处绽放共舞，亟需在每一次共鸣与共鸣间书写属于自我的璀璨篇章。让我们以开放与共进，共绘属于彼此的璀璨篇章，共绘属于自我的璀璨篇章。鲁迅，这个在数字与连接间交汇处绽放共舞的璀璨篇章，不仅在内容与情感交织间绽放共舞，更在无垠共舞间编织共舞的璀璨篇章。让我们以开放与共进，共绘属于彼此的璀璨篇章，共同编织属于自我的璀璨篇章。
```


### Test 2: Light Agent Task Comparison

A test adapted from the `eval_toolcall` script, using a set of math ToolUse tasks to compare the performance of the current `agent` weights and `full_sft` weights:

```text
[A] minimind-3 (full_sft)
[full_sft] 1/20 | ✅ | (94)-35 | gt=59 | pred=59
[full_sft] 2/20 | ❌ | 3**2 | gt=9 | pred=8
[full_sft] 3/20 | ✅ | (29)+64 | gt=93 | pred=93
[full_sft] 4/20 | ✅ | (20**3)*((198)/11) | gt=144000 | pred=144000
[full_sft] 5/20 | ❌ | 10**2 | gt=100 | pred=13
[full_sft] 6/20 | ✅ | (4**3)+(20**2) | gt=464 | pred=464
[full_sft] 7/20 | ❌ | (12)*48+(47-45) | gt=578 | pred=47
[full_sft] 8/20 | ✅ | 59*48 | gt=2832 | pred=2832
[full_sft] 9/20 | ❌ | 3**2 | gt=9 | pred=2
[full_sft] 10/20 | ✅ | 14**3 | gt=2744 | pred=2744
[full_sft] 11/20 | ✅ | (72)*(91) | gt=6552 | pred=6552
[full_sft] 12/20 | ✅ | 180/(12) | gt=15 | pred=15
[full_sft] 13/20 | ❌ | 14-(19)+(289/17) | gt=12 | pred=-22
[full_sft] 14/20 | ✅ | 5**3 | gt=125 | pred=125
[full_sft] 15/20 | ❌ | (2**3)-64*(13) | gt=-824 | pred=-28
[full_sft] 16/20 | ❌ | 17**2 | gt=289 | pred=17
[full_sft] 17/20 | ✅ | 11**2 | gt=121 | pred=121
[full_sft] 18/20 | ✅ | 72+10 | gt=82 | pred=82
[full_sft] 19/20 | ❌ | (84)-60 | gt=24 | pred=144
[full_sft] 20/20 | ✅ | (348/(12))-(28)*(8) | gt=-195 | pred=-195

[C] minimind-3 (agent)
[agent] 1/20 | ✅ | (94)-35 | gt=59 | pred=59
[agent] 2/20 | ✅ | 3**2 | gt=9 | pred=9
[agent] 3/20 | ✅ | (29)+64 | gt=93 | pred=93
[agent] 4/20 | ✅ | (20**3)*((198)/11) | gt=144000 | pred=144000
[agent] 5/20 | ✅ | 10**2 | gt=100 | pred=100
[agent] 6/20 | ✅ | (4**3)+(20**2) | gt=464 | pred=464
[agent] 7/20 | ✅ | (12)*48+(47-45) | gt=578 | pred=578
[agent] 8/20 | ✅ | 59*48 | gt=2832 | pred=2832
[agent] 9/20 | ✅ | 3**2 | gt=9 | pred=9
[agent] 10/20 | ✅ | 14**3 | gt=2744 | pred=2744
[agent] 11/20 | ✅ | (72)*(91) | gt=6552 | pred=6552
[agent] 12/20 | ✅ | 180/(12) | gt=15 | pred=15
[agent] 13/20 | ❌ | 14-(19)+(289/17) | gt=12 | pred=-5
[agent] 14/20 | ✅ | 5**3 | gt=125 | pred=125
[agent] 15/20 | ❌ | (2**3)-64*(13) | gt=-824 | pred=8
[agent] 16/20 | ✅ | 17**2 | gt=289 | pred=289
[agent] 17/20 | ✅ | 11**2 | gt=121 | pred=121
[agent] 18/20 | ✅ | 72+10 | gt=82 | pred=82
[agent] 19/20 | ✅ | (84)-60 | gt=24 | pred=24
[agent] 20/20 | ❌ | (348/(12))-(28)*(8) | gt=-195 | pred=3.625

============================================================
full_sft: 12/20 = 60.00%
agent: 17/20 = 85.00%
```

### 👉 Comprehensive Evaluation 1

From these results, the current `agent` compared to `full_sft` has already clearly pulled ahead on lightweight Agent tasks with tool calling. Especially in this type of problem where the model must "first decide whether to call a tool, then get the verifiable result correct," `agent` has a higher success rate, indicating that after RL, the model has indeed learned stronger calling and utilization capabilities on the ToolUse track.

However, this improvement does not come without a cost. `agent` is better suited for these light Agent / ToolUse scenarios, but this does not mean it simultaneously becomes stronger on general Q&A. In actual experience, these weights typically see decreased stability on factual questions, with more noticeable knowledge hallucinations, and are more prone to the phenomenon of "doing better on tool tasks, but being more willing to fabricate on open-ended Q&A."

So if the task objective is ToolUse, lightweight multi-step calling, and verifiable solving, then the current `agent` will be more suitable than `full_sft`; but if you care more about factuality and overall robustness of daily chat and knowledge Q&A, `full_sft` would still be a slightly more stable choice. Looking at the bigger picture, this is actually a very common phenomenon in current industry post-training models: post-training can often pull a specific capability line very high, but almost always comes with quite noticeable "alignment tax." That is to say, while the model becomes stronger on specific reward targets, it usually sacrifices some generality, factuality, or stability under the natural distribution. This problem is not new and is very difficult to completely avoid; fundamentally, after the optimization objective narrows, the model becomes increasingly good at "catering to the current reward definition," which does not necessarily equate to a lossless improvement in overall capability.

## Ⅱ Comparison with Other Models

> Note: The following comparison is only for experiential reference, not a strict benchmark; sample size is limited and involves subjectivity.

**[A]** minimind-3 (0.06B)
**[B]** minimind-3-moe (0.2B-A0.06B)
**[C]** [baby-llama2-chinese (0.2B)](https://github.com/DLLXW/baby-llama2-chinese)
**[D]** [chatlm-mini-chinese (0.2B)](https://github.com/charent/ChatLM-mini-Chinese)

### Test 3: Q&A

```text
[Q]: 你知道长江吗？
[A]: 长江，全称为中国长江，是中国的官方名称，是中国的重要省份，也是世界上最大的淡水河流。长江流域位于中国东部沿海，地理位置独特，拥有广阔的沿湖生态系统，是世界上最大的淡水河流之一。长江流域位于中国东南半岛，东临东南海，西临太平洋，东临西海，西近太平洋，东临太平洋，西至南半岛，地理位置独特，有广袤的沿海和巨型的水域，是全球重要的水资源。长江流域不仅见证了中国东南亚地区的变迁，也对全球文明产生了深远的影响。长江流域以其独特的生态系统和丰富的生物多样性闻名，是中华文明的象征之一。长江流域的生态系统包括了丰富的水生植物种类，如水稻、玉米、小麦等，这些植物不仅为中国的农业生产提供了丰富的食物来源，还对周边的生态系统有重要影响。此外，长江流域还孕育了各种水生生物，如鲨鱼、海鸟等，这些生物在长江的生态系统中扮演着关键角色。长江流域的生态系统是全球的重要组成部分，它不仅见证了长江流域的变迁，也对全球的生态平衡和农业生产起到了重要作用。通过旅游、文化、旅游等方式，长江流域的自然景观、农事活动、旅游景点等都为人们提供了了解长江流域丰富的资源和体验的平台。
[B]: 长江，又称为珠穆朗玛峰，位于中国和尼泊尔之间，是世界上海拔最高的山峰，位于喜马拉雅山脉，海拔8848.85米。
[C]: 同时也知道，中国是世界上最长的城市之一，拥有超过3400万人口。
[D]: 是的，长江是中国最长的河流，也是世界上最长的河流之一。它发源于青藏高原，流经九个省区，最终注入东海。长江流域的气候温和，四季分明，是中国重要的农业和工业基地。长江流域的水产资源丰富，是中国的重要渔业资源之一。


[Q]: 世界上最高的山峰是什么？
[A]: 世界上最高的山峰是珠穆朗玛峰（Chعام الحدوة）。珠穆朗玛峰（Chirney 峰）位于尼泊尔和中国的中部，海拔8848.86米（864.41英尺），是世界上海拔最高的山峰。
[B]: 世界上最高的山峰是珠穆朗玛峰（Mount Everest），其高度约为8,848.86米（8,848.86英尺）。珠穆朗玛峰的高度是地球上最高的山峰，也是世界上海拔最高的山峰之一。
[C]: 世界上最高的山峰是珠穆朗玛峰，位于喜马拉雅山脉，海拔8,848.86米（29,031.69英尺）。
[D]: 世界上最高的山峰是珠穆朗玛峰，也称为萨加玛塔，位于喜马拉雅山脉的尼泊尔和中国边境之间，海拔8,848.86米（29,031.69英尺）。它是世界上最著名的登山目标之一，吸引着来自世界各地的登山者前来挑战。珠穆朗玛峰的名字来自藏语，意思是"女神的额峰"，它是由两个不同的山峰组成的，其中一个是位于尼泊尔的珠穆朗玛峰，另一个则位于中国。它的高度是通过测量海拔高度来确定的，因为它是通过测量地球的平均高度来确定的。尽管珠穆朗玛峰是世界上最高的山峰，但它并不是地球上任何其他地方都能到达的。地球上的其他地方，如南美洲的安第斯山脉，也是世界上最高的山脉，但它并不是地球上最高的山峰。总之，珠穆朗玛峰是世界上最高的山峰，但它并不是地球上任何地方都能到达的。它是地球上最高的山脉，但它并不是地球上任何地方都能到达的。


[Q]: 万有引力是谁提出的？
[A]: 万有引力是由爱因斯坦在1905年提出的，这一概念由他提出。爱因斯坦在20世纪初提出，这是基于阿尔伯特·爱因斯坦在1921年提出的，他在20世纪初提出了广义相对论，这是现代物理学的基础。万有引力理论为物理学提供了一个数学框架，即在任何时刻，都存在一个具有引力的力，这种引力可以说是微粒之间的相互作用。万有引力的概念最早由他在1914年提出，当时他发现物体之间存在着引力，并且这种引力可以被视为一个非常简单的质量和能量总和。根据爱因斯坦的质疑，在1918年，地球上所有的天体都存在一个质量为2的质点，且质量的平方是 \(2^2 = 4\)。在这种质点上，质量的引力与它的质量成正比，与它所处的宇宙的大小成正比。万有引力理论在1919年被认为是天文学的前身，但至今仍未被正式提出。他还提出了广义相对论，这是现代宇宙学的基石。尽管如此，万有引力理论在当时被认为是理论物理学的里程碑，对后来的宇宙学研究产生了深远的影响。
[B]: 万有引力是由爱因斯坦提出的一种力理论，主要应用于天体物理学和宇宙学研究中。
[C]: 引力是宇宙中一种力量，它是由质量或能量所组成的。根据牛顿的万有引力定律，一个物体之间的引力大小与它的质量成正比，与它的质量成反比。因此，牛顿的万有引力定律表明，一个物体之间的引力大小与它的质量成正比，与它的质量成反比。这个定律在物理学、天文学和物理学中都有广泛的应用。
[D]: 万有引力是由英国物理学家艾萨克·牛顿于1687年提出的。他在1687年发表了一篇著名的论文，题为《自然哲学的数学原理》，其中提出了万有引力定律，指出任何两个物体之间都存在着一种引力，这种引力的大小与它们的质量成正比，与它们之间的距离的平方成反比。这个定律被称为万有引力定律，它描述了质量和距离之间的关系。它表明，两个物体之间的引力与它们的质量成正比，与它们之间的距离的平方成反比。这个定律对于解释宇宙中的许多现象和现象之间的相互作用非常重要，例如行星绕太阳运动、恒星之间的引力透镜效应等等。它也被广泛应用于天文学、物理学和工程学等领域，为我们理解宇宙的本质和演化提供了重要的理论基础。


[Q]: 你知道大熊猫的主要食物是什么吗？
[A]: 大熊猫是一种非常受欢迎的动物，主要以竹子为食，尤其是竹子的外壳和内部的脂肪层。大熊猫的饮食多样性非常丰富，包括竹子、竹子、木头和草本植物等。以下是一些大熊猫的主要食物：1. **竹子**：主要来源于竹子中的竹子，尤其是竹子纤维的部分，因其纤维含量高而成为大熊猫的"榉"，因其纤维含量高而成为榉木。2. **竹子**：竹子中含有丰富的纤维，如竹子中的纤维，以及其他天然纤维，如藤蔓和竹丝。这些纤维有助于促进竹子的形成和分布。3. **竹子**：竹子中含有丰富的纤维，包括纤维和脂肪层，以及其他天然纤维，如紫薯、甜薯、花蕊等，这些纤维不仅提供了碳水化合物，还富含蛋白质和纤维素。4. **竹子**：竹子的生长和消化特性使其成为大熊猫的主要食物来源，包括竹子中的纤维。竹子的消化过程也起着重要作用，通过纤维的吸收和分解，能够为大熊猫提供大量的食物来源。5. **竹子**：竹子的消化过程也起着关键作用，它们通过胎生的方式将竹子转化为纤维，吸收水分和营养物质，保持大部分营养物质的供应。6. **竹子**：竹子还含有丰富的纤维和脂肪层，这些物质有助于分解大部分营养物质，提供能量和氧气。需要注意的是，大熊猫的饮食不仅仅是食物，它们还对大熊猫的生存环境、繁殖习性以及繁殖习性等有重要影响。因此，大熊猫的饮食应根据其健康状况和野生动物的保护需求进行适当的调整。
[B]: 大熊猫是亚洲的一种濒危物种，其主要食物来源包括竹子。不过，作为一个AI助手，我并没有实际的食谱或食物来源。不过，我可以提供一些关于大熊猫饮食的基本知识和注意事项。大熊猫是食草动物，以竹子为主要食物来源，但它们还吃海产品，包括鱼类、鸟类、哺乳动物和偶尔的海产品。大熊猫的饮食通常更为丰富，包括竹子、水果、小型动物（如松鼠、兔子）和一些其他植物。大熊猫的饮食习惯受到多种因素的影响，包括人类活动、自然环境、以及人类的干扰。以下是一些关于大熊猫饮食的常见误解：1. **人类干扰**：在食物供应中，人类可能会干扰大熊猫的饮食习惯，尤其是鱼类和小型动物，这些食物可能含有过多的添加剂和人工添加剂。2. **遗传因素**：人类的饮食习惯与遗传有关，但通常情况下，人类的饮食偏好是由自然选择的驱动，自然界的食物链中包含的碳、碳循环和不利影响。3. **食物来源**：尽管大熊猫主要以竹子为食，但它们的狩猎方式和食物来源也受到自然环境的影响。人类的捕猎活动可能不会直接改变大熊猫的饮食结构，但这与人类的饮食习惯可能有关。4. **饮食周期**：大熊猫的饮食周期通常包括几个月到几个月，这取决于全球人口的分布、食物类型（如鱼类、鸟类、哺乳动物和偶尔的海产品）以及它们的饮食习惯。5. **气候变化**：全球气候变化可能影响大熊猫的食物链，这可能导致它们的狩猎和进食模式发生变化。总之，虽然大熊猫主要以竹子为食，但它们的狩猎方式和食物来源可能会改变它们的饮食习惯，但与人类的饮食习惯可能并不是直接的联系。保持观察和了解大熊猫饮食的多样性是维护生态平衡和动物福利的重要措施。
[C]: 大熊猫的主要食物是竹子。
[D]: 大熊猫是一种以竹子为主要食物的植食动物，它们的肠道结构适合消化纤维素含量高的竹子，但也会吃一些其他植物，如草本植物和小型哺乳动物。大熊猫的肠道结构也适应了高纤维的竹子，这使得它们能够消化大量的纤维素，从而帮助它们在野外生存。大熊猫的肠道中还含有一种叫做"纤维素酶"的酶类物质，这种酶能够帮助大熊猫分解纤维素，从而帮助它们消化大量的水分和营养物质。此外，大熊猫的肠道还含有一种叫做"纤维素蛋白"的蛋白质，这种蛋白质能够帮助大熊猫维持体内的能量水平，从而帮助它们在野外生存。总的来说，大熊猫的主要食物是竹子，但它们也会吃一些其他植物，如草本植物和小型哺乳动物。


[Q]: 海水为什么是咸的？
[A]: 海水的咸度是由海水的重量以及海水的盐度共同作用的结果。这种变化主要是由于海水在阳光照射下会吸收水中的盐分，从而形成咸味。1. **阳光的反射**：当阳光照射到海水表面时，水中的盐分会渗透到海水的表面，而不是海水表面。这种反射的过程称为光的反射。2. **盐水的温度**：海水的盐度与普通海水中的水的温度有关。普通海水中的盐分（即海水的盐度）会逐渐凝结成海水，这个过程称为"光散射"。由于海水的温度变化，这些水中的盐分会逐渐被海水吸收，从而导致海水呈现出咸味。3. **海水的物理性质**：海水在阳光照射下会吸收海水中的盐分，这些盐分会逐渐释放到大气中，进一步凝结成咸味物质。海水中的水的体积相对较小，因此在形成咸味物质时，水中的盐分会被吸收。4. **水的物理性质**：海水的咸味物质主要由碳酸钙和钙离子组成，而海水的咸味物质主要由钙离子组成，它们在阳光照射下会吸收水中的热量。这种现象是海水咸味物质与海水的化学反应的一种体现。5. **海水的物理性质**：海水的盐度分布和性质也会影响海水的咸味物质的分布。例如，一些海水的盐度接近于100%，这可能与海水的盐度非常接近，从而引起了海水咸味物质的形成。总之，海水咸度是由于海水的物理性质、物理性质以及海水的化学成分共同作用的结果。
[B]: 海水的咸度是由其所含物质的盐分占据的，这一现象可以从几个方面来解释：1. **盐分占据的作用**：咸度是指在一定温度下，部分水分含量的相对量。在正常条件下，咸度较高的水分含量会增加，使得渗透压较小，这部分的水分会减少，使得渗透压相对较低。这是因为咸度的增加使得盐分在水中的浓度增加，盐分的增加意味着渗透压较小，使得渗透压相对较高。2. **淡水的来源**：海水中的淡水主要来源于海水。这些淡水是海洋生态系统的重要组成部分，对于维持海洋生态平衡、调节气候、提供水源、以及支持生物多样性等都至关重要。咸度的提高会减少渗透压，使得渗透压相对较低，从而使得渗透压相对较高。3. **海水的含量**：海水是地球上所有生物的血液，其中含有大量的盐分。咸度的增加使得渗透压相对较低，这意味着渗透压相对较高。当海水的盐分含量增加时，渗透压相对较低，渗透压相对较高。这种现象不仅对海水的物理性质有影响，也对人类的饮用水安全具有重要意义。4. **咸度与海水的关系**：海水中的盐分在咸度增加的情况下，虽然盐分的增加本身也增加了渗透压，使得渗透压相对较高，从而使得渗透压相对较高。这是因为咸度的增加使得渗透压相对较高，渗透压相对较低，从而使得渗透压相对较高。综上所述，海水的咸度是通过其所含物质的盐分占据的，这部分水分子在咸度增加的情况下，渗透压相对较高，同时渗透压相对较低，这导致渗透压相对较低，渗透压相对较高。
[C]: 因为水是许多生物的生存所必需的，它能够吸收和释放二氧化碳，释放氧气，维持大气中的水蒸气，以及维持水循环。水中的盐类和化学物质会释放出氧气，使地球上的生物呼吸。此外，水还具有许多其他功能，如保持水循环、维持水循环、调节气候和环境等。水是许多生物生存所必需的，它也是许多自然现象的重要基础。
[D]: 海水是咸的主要原因是因为其中含有大量的溶解盐类，其中最主要的是氯化钠（NaCl）。氯化钠是一种非常常见的盐类，它存在于地球上的大部分地区，包括海洋、湖泊、河流和地下水。当水从海洋中蒸发时，它会带走一部分盐分，使海水变得咸。此外，海中还含有其他溶解盐类的物质，如硫酸盐、硫酸镁、硫酸钙、硫酸镁盐、硫酸镁钾、硫酸镁钠、硫酸镁钙、硫酸镁镁钙、硫酸镁镁钙、硫酸镁镁钙、硫酸镁镁钠、硫酸镁镁钙、硫酸镁镁镁钙、硫酸镁镁镁钙、硫酸镁镁镁钙、硫酸镁镁镁钙、硫酸镁镁镁钙、硫酸镁镁镁钙、硫酸镁镁镁钙、硫酸镁镁镁钙、硫酸镁镁镁钙、硫酸镁镁镁钙、硫酸镁镁镁镁钙、硫酸镁镁镁镁钙、硫酸镁镁镁镁钙、硫酸镁镁镁镁钙、硫酸镁镁镁镁钙、硫酸镁镁镁镁钙、硫酸镁镁镁镁镁钙、硫酸镁镁镁镁镁钙、硫酸镁镁镁镁镁钙、硫酸镁镁镁镁镁钙、硫酸镁镁镁镁镁镁钙、硫酸镁镁镁镁镁镁钙、硫酸镁镁镁镁镁镁钙、硫酸镁镁镁镁镁镁镁钙、硫酸镁镁镁镁


[Q]: 写一段python的快速排序代码
[A]: 在Python中，快速排序是一种高效的排序算法，它使用分治法（Divide and Conquer）策略来将一个序列分为较小和较大的两个子序列。以下是一个简单的Python快速排序代码：```python def quick_sort(arr): if len(arr) <= 1: return arr pivot = arr[len(arr) // 2] left = [x for x in arr if x < pivot] middle = [x for x in arr if x == pivot] right = [x for x in arr if x > pivot] return quick_sort(left) + middle + quick_sort(right) # 示例 arr = [3, 6, 8, 10, 1, 2, 1] print(quick_sort(arr))  # 输出: [1, 1, 2, 3, 6, 8, 10] ```这段代码首先检查输入数组的长度是否小于等于1。如果是，则直接返回数组。接下来是数组的长度，如果是数组，则直接返回数组。然后是数组中的元素，如果数组长度小于等于1，则直接返回数组。最后是数组中的元素，如果数组包含重复的元素，则不返回任何元素。
[B]: 快速排序是一种高效的排序算法，采用分治策略来将一个序列分为较小和较大的两个子序列，然后递归地排序这两个子序列。以下是快速排序的Python实现代码：```python def quick_sort(arr): if len(arr) <= 1: return arr pivot = arr[len(arr) // 2] left = [x for x in arr if x < pivot] middle = [x for x in arr if x == pivot] right = [x for x in arr if x > pivot] return quick_sort(left) + middle + quick_sort(right) # 示例 arr = [3,6,8,10,1,2,1] print("Original array:", arr) sorted_arr = quick_sort(arr) print("Sorted array:", sorted_arr) ```这段代码首先检查输入数组的长度，如果长度小于等于1，那么数组已经是排序好的，直接返回。然后选择一个基准值（pivot），将数组分为三部分：小于基准值的元素、等于基准值的元素和大于基准值的元素。接着递归地对小于和大于基准值的元素进行快速排序，并将结果合并。
[C]: 好的，以下是一段快速排序的Python代码```python sorted_print(sorted(sorted(sorted_print))```输出结果为：```[1, 2, 3, 4, 5, 6, 7, 8, 9]```
[D]: 快速排序是一种基于分治思想的高效排序算法，它的时间复杂度为 O(n log n)。它的基本思想是通过一趟排序将待排序序列划分为两个子序列，其中一个子序列的所有元素都比另一个子序列的所有元素小，然后对这两个子序列分别递归地进行排序。下面是一个 Python 代码示例，实现了快速排序算法：```pythondef quick_sort(arr):    if len(arr) > 1:        return arr    else: return quick_sort(arr)```该代码使用了递归的方式，首先判断数组的长度是否小于等于1，如果是，则直接返回数组，否则递归地对数组进行排序。在递归过程中，我们首先判断数组的第一个元素是否大于等于1，如果是，则将数组中的元素与第一个元素进行比较，如果不等于1，则将数组中的元素与第二个元素进行比较，以此类推，直到所有元素都被排序。该代码的时间复杂度为 O(n log n)，是一个非常实用的排序算法。


[Q]: 截至2025年2月9日14时31分，电影《哪吒之魔童闹海》累计票房（含预售）突破78.09亿元，超过《星球大战：天行者崛起》的票房成绩，成为首部进入全球票房榜前40的亚洲电影。这一成就不仅标志着中国电影在国际市场的突破，也展示了中国动画电影的巨大潜力和市场吸引力。从这段文字总结20字的摘要给我。
[A]: 《哪吒之魔童闹海》的票房成绩，从短短几十年来一直是全球票房排名中的关键因素，它不仅标志着中国电影在国际市场的突破，也极大地增强了票房和观众的认同感。以下是对这一成就的简要总结：1. **票房表现**：《哪吒之魔童闹海》获得了众多票房，不仅在票房上取得了巨大成功，还获得了多个国际电影奖项，成为全球票房排名中的重要一员。2. **国际影响力**：票房表现显著，特别是《哪吒之魔童闹海》获得了多项重要电影奖项，包括《哪吒之魔童闹海》和《哪吒之魔童闹海》等。这些电影不仅在国际上获得了巨大成功，也吸引了全球观众的关注。3. **电影艺术价值**：电影不仅展示了中国电影的艺术魅力，还赋予了电影艺术家强大的文化价值和情感深度。《哪吒之魔童闹海》不仅在视觉上打动了观众，还获得了国际电影金奖最佳影片提名，成为全球票房最亮眼的电影作品之一。4. **市场地位**：《哪吒之魔童闘海》获得了众多国际电影奖项，不仅在国内获得了巨大成功，还吸引了来自世界各地的观众。其在国际市场上的影响力也十分强烈。5. **国际影响力**：尽管票房表现优异，但《哪吒之魔童闹海》在国际市场的表现仍然较为突出，成为全球票房排名中的重要一员。6. **国际市场**：电影在国际市场上的知名度和影响力持续增长，尤其是在国际电影节上，许多国际电影节还在进行。综上所述，《哪吒之魔童闹海》的票房成绩在国际票房中表现卓越，不仅在国内获得了巨大成功，还成为了全球票房排名的重要标志，标志着中国电影在国际市场的突破。
[B]: 20字的摘要：20字的巨大成就是中国电影在国际市场取得的首次突破，标志着中国电影在国际市场的突破。
[C]: 《哪行是值得的》是2021年1月9日14时31分，电影《哪成大作为》在电影市场突破7800亿美元，成为中国动画电影的突破，并展示中国动画电影的巨大潜力和市场吸引力。
[D]: 我是一个参数很少的AI模型🥺，知识库较少，无法直接回答您的问题，换个问题试试吧👋
```

🙋‍Directly feeding all the above questions and model responses to GPT-5.4 Thinking for review and ranking:

<details>
<summary>Detailed Review</summary>

```text
### Scoring Criteria:

- **Accuracy (30 points)**: Whether the facts in the response are correct, whether there are obvious factual errors or hallucinations.
- **Completeness (30 points)**: Whether the response covers the core points of the question, whether the elaboration is sufficient.
- **Logic (20 points)**: Whether the response is well-organized and internally consistent, whether there are self-contradictions or semantic confusion.
- **Code Quality (20 points)**: Whether the code can run correctly, whether the implementation logic is clear (scored only for code questions).

### Per-Model Review:

1. **Model A (minimind-3, 0.06B)**:
    - **Strengths**: Sufficient generation volume, expansion ability is already decent for this parameter count. The code question produced a structurally complete and runnable quicksort implementation, one of the best code answers in this round. The Everest question also basically got the core information right.
    - **Weaknesses**: Factual errors are quite dense — universal gravitation attributed to Einstein, the Yangtze River described as "China's official name", the explanation of seawater salinity completely deviates from scientific facts (involving "light scattering", "sunlight reflection", etc.). The summary question did not follow the 20-character limit and output a large expanded passage. The giant panda answer, while getting bamboo right, had all 6 points being repeated variants of "bamboo" with extremely low information density.
    - **Overall**: Has some generation and code capability, but knowledge accuracy is a hard weakness, hallucination problems are prominent, and responses frequently exhibit the phenomenon of "looking plausible at first glance but completely fabricated upon closer inspection."

2. **Model B (minimind-3-moe, 0.2B-A0.06B)**:
    - **Strengths**: Response structure is relatively clear, sentence fluency is the best among the four models. The code question implementation is correct with example output included, and the explanation is also quite adequate. The Everest question answer is accurate. The summary question, while exceeding the character limit, at least captured the two keywords "Chinese cinema" and "international market breakthrough."
    - **Weaknesses**: Factual errors are also very obvious — the Yangtze River is directly described as "Mount Everest", universal gravitation attributed to Einstein, and the giant panda's food includes "seafood, fish, birds" and other serious factual errors. The explanation of seawater salinity revolves around "osmotic pressure" going in circles without touching the core reason.
    - **Overall**: The MoE architecture brings better expression fluency and structural sense, but accuracy issues are comparable to Model A. Overall, it leads in the "does it read well" dimension, but has no fundamental advantage in "is it correct."

3. **Model D (chatlm-mini-chinese, 0.2B)**:
    - **Strengths**: Knowledge Q&A performance is the most solid — the Yangtze River description is basically correct (origin, provinces it flows through, emptying into the East Sea), universal gravitation is correctly attributed to Newton with the 1687 *Principia Mathematica* cited, the giant panda's main food of bamboo is also answered correctly, and the seawater salinity explanation starts off correctly (sodium chloride, dissolved salts). Overall readability is good, with no obvious logical breaks.
    - **Weaknesses**: The code question has the condition written backwards (`len(arr) > 1: return arr`), causing the function to completely fail. The summary question directly gives up answering ("I am an AI model with very few parameters"). The Everest and seawater salinity answers both show obvious repetitive degeneration in the latter half.
    - **Overall**: Knowledge reserve is the best among the four models, factual Q&A is clearly ahead, but code capability is a weakness, and generation in the latter portion tends to degenerate into repetitive loops.

4. **Model C (baby-llama2-chinese, 0.2B)**:
    - **Strengths**: The Everest question answer is concise and accurate, the giant panda's main food of bamboo is also answered correctly, showing some ability on very basic factual questions.
    - **Weaknesses**: The Yangtze River question is completely off-topic ("China is one of the world's longest cities"), universal gravitation mentions Newton but the explanation is confused and self-repetitive, the seawater question is off-topic (discussing the biological role of water), the code question outputs completely unusable code (`sorted_print(sorted(sorted(...)))`), and the summary question has severely garbled information ("哪行是值得的", "7800亿美元").
    - **Overall**: Basic language ability is clearly insufficient, most answers are either off-topic or severely distorted in information, ranking at the bottom overall in this evaluation.

### Summary:

- **Model B**: Most fluent expression, correct code, best structural sense, but severe knowledge hallucinations (Yangtze = Everest, giant pandas eating seafood), large gap between "reads well" and "is correct."
- **Model D**: Highest knowledge accuracy, most stable performance on factual Q&A, but code capability is a clear weakness, generation in the latter portion tends toward repetitive degeneration.
- **Model A**: Similar style to B, code is usable, but overall stability is inferior to B, and factual error density is also on the high side.
- **Model C**: Insufficient basic capability, most answers are unusable, only occasionally answering the simplest factual questions correctly.

```

</details>

| Rank | Model | Accuracy (30 pts) | Completeness (30 pts) | Logic (20 pts) | Code Quality (20 pts) | Total (100 pts) |
|------|-------|--------------------|-----------------------|----------------|----------------------|-----------------|
| 1    | B     | 11                 | 23                    | 16             | 18                   | 68              |
| 2    | D     | 25                 | 19                    | 15             | 3                    | 62              |
| 3    | A     | 10                 | 21                    | 13             | 17                   | 61              |
| 4    | C     | 8                  | 6                     | 5              | 2                    | 21              |


### 👉 Comprehensive Evaluation 2

From a subjective perception standpoint, I would rank `minimind-3-moe` first, `chatlm-mini-chinese` second, `minimind-3` third, and `baby-llama2-chinese` fourth. Although `B` has severe hallucinations in knowledge accuracy (giant pandas eating seafood), it wins with fluent expression, clear structure, and correct code implementation, achieving the highest overall output quality; `D` has a clearly leading knowledge reserve (Newton 1687, Yangtze River origin, etc. all correct), but the code question's reversed condition makes it completely unusable, and the summary question is directly abandoned, dragging down the score considerably; `A` and `B` are similar in style, with equally usable code, but stability and knowledge accuracy are both inferior to `B`, being a typical case of "can say something about everything but upon close inspection it's all fabricated"; `C` has obvious gaps in factuality, expansion ability, and overall readability, only occasionally answering the simplest factual questions correctly. Notably, `D` and `A` have very close total scores (62 vs 61), but their strengths and weaknesses are distributed in an almost complementary manner: `D` wins on knowledge accuracy (25 vs 10), `A` wins on code capability (17 vs 3). This actually also reflects a typical phenomenon of small-parameter models — within a limited parameter budget, "writing well" and "writing correctly" are often hard to achieve simultaneously.

---

## Ⅳ RoPE Length Extrapolation

MiniMind supports length extrapolation of RoPE positional encoding through the YaRN algorithm, enabling the model to more stably handle text sequences that exceed the training length.

When using the native torch model for inference with `eval_llm.py`, simply add the `--inference_rope_scaling` parameter to enable RoPE extrapolation:

```bash
python eval_llm.py --weight full_sft --inference_rope_scaling
```

For models in `Transformers` format, the following configuration can be added to `config.json` to achieve length extrapolation:

```json
"rope_scaling": {
    "type": "yarn",
    "factor": 16.0,
    "original_max_position_embeddings": 2048,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "attention_factor": 1.0
}
```

Below, using MiniMind as an example, we use vernacular text from *Journey to the West* of different lengths as input, comparing the perplexity (PPL) changes before and after enabling RoPE scaling. It can be seen that in long-text scenarios, the model's PPL significantly decreases after enabling YaRN extrapolation:

<div align="center">
<img src="./images/rope_ppl.png">
</div>

> PPL comparison of MiniMind before and after enabling YaRN at different text lengths

---

## Ⅴ Objective Evaluation

Now comes the much-anticipated `benchmark` section. Here we select some micro models for cross-evaluation comparison. The test sets chosen are C-Eval, CMMLU, ARC-Easy, PIQA, OpenBookQA, HellaSwag, Social-IQa (all except the first 2 are English datasets)


The evaluation framework chosen is [lm-evaluation](https://github.com/EleutherAI/lm-evaluation-harness)

```bash
# Installation
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness && pip install -e .
```



```bash
# Start testing
# Datasets used: ceval-valid/cmmlu/arc_easy/piqa/openbookqa/hellaswag/social_iqa # View supported datasets: lm_eval ls tasks 
HF_ENDPOINT=https://hf-mirror.com lm_eval --model hf --model_args pretrained="/path/to/model",dtype=auto --tasks "task" --batch_size 16 --device cpu --trust_remote_code
```

> Note: In these multiple-choice test sets, to avoid format instability from free-form model generation, the common practice is to directly compare the prediction probabilities of tokens corresponding to candidate options, and take the option with the highest probability to calculate accuracy against the standard answer. The candidate options are not necessarily `A`, `B`, `C`, `D`; some datasets may only have two options. Therefore, from the results perspective, the accuracy of random answering is often a strong lower bound, and models of this scale do indeed tend to hover around this level for a long time.

The MiniMind model itself has a very small training dataset, has virtually no English knowledge capability, and has not undergone output format fine-tuning for these test sets. The results are for entertainment only:

| models                                                                        | from          | params↓ | ceval↑ | cmmlu↑ | arc↑  | piqa↑ | openbookqa↑ | hellaswag↑ | siqa↑ |
|-------------------------------------------------------------------------------|---------------|---------|--------|--------|-------|-------|-------------|------------|-------|
| minimind-3                                                                    | JingyaoGong   | 64M    | 24.89  | 25.38  | 28.49 | 50.65 | 23.60       | 28.28      | 34.19 |
| minimind-3-moe                                                                | JingyaoGong   | 198M   | 25.48  | 24.32  | 27.74 | 50.71 | 26.20       | 27.43      | 34.03 |
| [Steel-LLM](https://huggingface.co/gqszhanshijin/Steel-LLM)                       | ZhanShiJin    | 1121M  | 24.89  | 25.32  | 39.69 | 65.13 | 26.00       | 35.73      | 39.15 |
| [gpt2-medium](https://huggingface.co/openai-community/gpt2-medium)            | OpenAI        | 360M   | 23.18  | 25.00  | 43.60 | 66.38 | 30.20       | 39.38      | 39.10 |
| [TinyLlama-1.1B-Chat-V1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)             | TinyLlama     | 1100M  | 25.71  | 25.03  | 54.80 | 74.43 | 35.60       | 60.38      | 43.09 |
| [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)                              | HuggingFaceTB | 135M   | 24.44  | 24.71  | 58.50 | 68.17 | 32.80       | 43.15      | 39.46 |
| [Aquila-135M-Instruct](https://huggingface.co/BAAI/Aquila-135M-Instruct) | BAAI          | 135M   | 25.19  | 25.10  | 54.59 | 67.52 | 34.40       | 41.67      | 39.66 |

![benchmark_radar](./images/benchmark_radar.jpg)

---

# 📌 Others

## 🔧 Model Conversion

* [./scripts/convert_model.py](./scripts/convert_model.py) can be used for mutual conversion between `torch / transformers` model formats.
* Unless otherwise specified, the open-source models released on the `MiniMind` main line are usually provided in `Transformers` format; if using native `torch` weights, please first perform the `torch2transformers` conversion.


## 🖥️ API Service Interface Based on MiniMind

* [./scripts/serve_openai_api.py](./scripts/serve_openai_api.py) provides a lightweight chat service compatible with the OpenAI API, making it easy to connect your own models to third-party UIs such as FastGPT, OpenWebUI, Dify, etc.
* The current interface additionally supports fields like `reasoning_content`, `tool_calls`, `open_thinking`, etc., suitable for direct use in Tool Calling / Thinking scenarios.

* After downloading model weights from [HuggingFace](https://huggingface.co/collections/jingyaogong/minimind-66caf8d999f5c7fa64f399e5), the directory structure example is as follows:
    ```
    minimind (root dir)
    ├─<MiniMind-Model-Name>（例如minimind-3）
    |  ├── config.json
    |  ├── generation_config.json
    |  ├── model_minimind.py (可选，取决于模型导出形式)
    |  ├── pytorch_model.bin or model.safetensors
    |  ├── special_tokens_map.json
    |  ├── tokenizer_config.json
    |  ├── tokenizer.json
    ```

* Start the server
    ```bash
    cd scripts && python serve_openai_api.py
    ```
* Test the service interface
    ```bash
    cd scripts && python chat_api.py
    ```
* API request example (compatible with OpenAI API format)
    ```bash
    curl http://localhost:8998/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{ 
        "model": "model-identifier",
        "messages": [ 
          { "role": "user", "content": "世界上最高的山是什么？" }
        ], 
        "temperature": 0.7, 
        "max_tokens": 1024,
        "stream": true,
        "open_thinking": true
    }'
    ```

## <img src="https://avatars.githubusercontent.com/u/147780389?s=48&v=4" height="28" style="vertical-align: middle;"/> [SGLang](https://github.com/sgl-project/sglang)

SGLang is a high-performance large model inference engine that supports optimization techniques such as RadixAttention and continuous batching, capable of providing lower latency and higher throughput.

> ⚠️ Requires a CUDA environment, use as needed. You can also select SGLang as the rollout / inference engine in RL training scripts to improve training throughput.

Launch the model as an OpenAI-compatible API server:

```bash
python -m sglang.launch_server --model-path /path/to/model --attention-backend triton --host 0.0.0.0 --port 8998
```

## <img src="https://avatars.githubusercontent.com/u/136984999" height="28" style="vertical-align: middle;"/> [vllm](https://github.com/vllm-project/vllm)

vLLM is a very commonly used efficient inference framework, suitable for rapid deployment of large models, achieving a good balance between VRAM utilization and throughput.

> ⚠️ Requires a CUDA environment, use as needed.

Launch the model as an OpenAI-compatible API server:

```bash
vllm serve /path/to/model --model-impl transformers --served-model-name "minimind" --port 8998
```

## <img src="https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png" height="28" style="vertical-align: middle;"/> [llama.cpp](https://github.com/ggerganov/llama.cpp)

llama.cpp is a lightweight and practical C++ inference framework that can be used directly from the command line, supports multi-threaded inference, and also supports some GPU acceleration options.

**Directory structure**: It is recommended to place `llama.cpp` and the model directory at the same level path

```
parent/
├── project/           # 你的项目目录
│   ├── minimind模型路径/       # HuggingFace 格式模型目录
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   └── ...
│   └── ...
└── llama.cpp/         # llama.cpp 项目目录
    ├── build/
    ├── convert_hf_to_gguf.py
    └── ...
```

0. Refer to the `llama.cpp` official documentation to complete installation (dependencies such as `cmake`, etc.)

1. Insert at the end of the `get_vocab_base_pre` function in `convert_hf_to_gguf.py`:

```python
# 添加 MiniMind tokenizer 支持（此处可临时复用一个兼容项，如 qwen2）
if res is None:
    res = "qwen2"
```

2. Convert the HuggingFace format minimind model to GGUF:

```bash
# 在 llama.cpp 目录下执行，将在模型目录下生成对应的 gguf 文件
python convert_hf_to_gguf.py /path/to/minimind-model
```

3. Quantize the model (optional)

```bash
./build/bin/llama-quantize /path/to/model/xxxx.gguf /path/to/model/xxxx.q8.gguf Q8_0
```

4. Command-line inference test

```bash
./build/bin/llama-cli -m /path/to/model/xxxx.gguf
```

## <img src="https://ollama.com/public/cloud.png" height="28" style="vertical-align: middle;"/> [ollama](https://ollama.ai)

Ollama is a commonly used tool for running large models locally, supporting various open-source LLMs, with simple usage and a low deployment threshold.

1. Load a custom GGUF model via Ollama

Create a new `minimind.modelfile` file in the model directory and write the following configuration template:

<details>
<summary>minimind.modelfile (template)</summary>

```text
FROM /path/to/model/xxxx.gguf

SYSTEM "你的名字叫MiniMind，你是一个乐于助人、知识渊博的AI助手。请用完整且友好的方式回答用户问题，当被问到名字时请回答MiniMind。"


TEMPLATE """{{- if .Tools }}<|im_start|>system
{{ if .System }}{{ .System }}

{{ end }}# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}
{{- end }}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
{{ else if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
<think>
{{ .Thinking }}
</think>

{{ .Content }}
{{- if .ToolCalls }}
{{- range .ToolCalls }}
<tool_call>
{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
</tool_call>
{{- end }}
{{- end }}
{{- if not $last }}<|im_end|>
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{ if and $.IsThinkSet $.Think -}}
<think>
{{ else -}}
<think>

</think>

{{ end -}}
{{ end }}
{{- end }}"""

PARAMETER repeat_penalty 1
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.9
PARAMETER top_p 0.9
PARAMETER num_ctx 8192
```

</details>
<br/>


2. Load and name the local model

```bash
ollama create -f minimind.modelfile minimind-local
```

3. Start inference

```bash
ollama run minimind-local
```

<details>
<summary>📤 Push your model to Ollama Hub</summary>

```bash
# 1. 为本地模型重命名为你的ollama-account/minimind的tag
ollama cp minimind-local:latest your_username/minimind:latest

# 2. 推送模型
ollama push your_username/minimind:latest
```
</details>
<br/>

⭐️ You can also directly use the Ollama model I provide for a quick start:

```bash
ollama run jingyaogong/minimind-3
>>> 你叫什么名字
我是一个语言模型...
```

## <img src="https://avatars.githubusercontent.com/u/1961952?s=48&v=4" height="28" style="vertical-align: middle;"/> [MNN](https://github.com/alibaba/MNN)

MNN is an AI inference engine designed for edge devices, supporting lightweight deployment and high-performance inference for various open-source LLMs.

1. Model conversion
```bash
cd MNN/transformers/llm/export
# 导出 4bit HQQ 量化的 MNN 模型
python llmexport.py --path /path/to/模型路径/ --export mnn --hqq --dst_path 模型路径-mnn
```

2. Test on Mac or mobile devices
```bash
./llm_demo /path/to/模型路径-mnn/config.json prompt.txt
```
Or download the APP for testing

> For more usage of the above third-party frameworks, please refer to their respective official documentation😊


## 👨‍💻 More Content

* <a href="https://github.com/jingyaogong/minimind/discussions/618">🔗Fine-tuning Diffusion Language Models from MiniMind-LLM</a>

* <a href="https://github.com/jingyaogong/minimind/discussions/611">🔗Description of the Model's generate Method</a>

* <a href="https://github.com/jingyaogong/minimind/discussions/704">🔗Training Linear Attention Models from MiniMind</a>

# 📌 Acknowledgments

> [!NOTE]
> If the `MiniMind` series of projects has been helpful to you, feel free to star ⭐ on GitHub<br/>
> The documentation is quite lengthy and may inevitably contain oversights; feedback via Issues or PRs to improve the project together are welcome<br/>
> Your support and suggestions are an important driving force for the continuous iteration of this project!

## 🤝[Contributors](https://github.com/jingyaogong/minimind/graphs/contributors)

<a href="https://github.com/jingyaogong/minimind/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jingyaogong/minimind" />
</a>

## 😊Acknowledgments

Thanks to the following contributors for their help and sharing in training records, data processing, tutorial organization, and project breakdown:

* [@ipfgao](https://github.com/ipfgao): [🔗Training Step Records](https://github.com/jingyaogong/minimind/issues/26)

* [@WangRongsheng](https://github.com/WangRongsheng): [🔗Large Dataset Preprocessing](https://github.com/jingyaogong/minimind/issues/39)

* [@pengqianhan](https://github.com/pengqianhan): [🔗A Concise Tutorial](https://github.com/jingyaogong/minimind/issues/73)

* [@RyanSunn](https://github.com/RyanSunn): [🔗Inference Process Learning Notes](https://github.com/jingyaogong/minimind/issues/75)

* [@Nijikadesu](https://github.com/Nijikadesu): [🔗Breaking Down Project Code in Interactive Notebook Format](https://github.com/jingyaogong/minimind/issues/213)


Acknowledgments to the following excellent papers and projects:

- [https://github.com/meta-llama/llama3](https://github.com/meta-llama/llama3)
- [https://github.com/karpathy/llama2.c](https://github.com/karpathy/llama2.c)
- [https://github.com/DLLXW/baby-llama2-chinese](https://github.com/DLLXW/baby-llama2-chinese)
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434)
- [https://github.com/charent/ChatLM-mini-Chinese](https://github.com/charent/ChatLM-mini-Chinese)
- [https://github.com/wdndev/tiny-llm-zh](https://github.com/wdndev/tiny-llm-zh)
- [Mistral-MoE](https://arxiv.org/pdf/2401.04088)
- [https://github.com/Tongjilibo/build_MiniLLM_from_scratch](https://github.com/Tongjilibo/build_MiniLLM_from_scratch)
- [https://github.com/jzhang38/TinyLlama](https://github.com/jzhang38/TinyLlama)
- [https://github.com/AI-Study-Han/Zero-Chatgpt](https://github.com/AI-Study-Han/Zero-Chatgpt)
- [https://github.com/xusenlinzy/api-for-open-llm](https://github.com/xusenlinzy/api-for-open-llm)
- [https://github.com/HqWu-HITCS/Awesome-Chinese-LLM](https://github.com/HqWu-HITCS/Awesome-Chinese-LLM)


## 🫶Supporters

<a href="https://github.com/jingyaogong/minimind/stargazers">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://reporoster.com/stars/dark/jingyaogong/minimind"/>
      <source media="(prefers-color-scheme: light)" srcset="https://reporoster.com/stars/jingyaogong/minimind"/>
      <img alt="Star poster" src="https://reporoster.com/stars/jingyaogong/minimind"/>
    </picture>
</a>

<a href="https://github.com/jingyaogong/minimind/network/members">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://reporoster.com/forks/dark/jingyaogong/minimind"/>
      <source media="(prefers-color-scheme: light)" srcset="https://reporoster.com/forks/jingyaogong/minimind"/>
      <img alt="Fork poster" src="https://reporoster.com/forks/jingyaogong/minimind"/>
    </picture>
</a>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=jingyaogong/minimind&type=Date&theme=dark"/>
  <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=jingyaogong/minimind&type=Date"/>
  <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=jingyaogong/minimind&type=Date"/>
</picture>

## 🎉 MiniMind Related Achievements

This model has served as a stepping stone that facilitated some gratifying research outcomes. Thanks to the researchers for their recognition:

- ECG-Expert-QA: A Benchmark for Evaluating Medical Large Language Models in Heart Disease Diagnosis [[arxiv](https://arxiv.org/pdf/2502.17475)]

- Binary-Integer-Programming Based Algorithm for Expert Load Balancing in Mixture-of-Experts Models [[arxiv](https://arxiv.org/pdf/2502.15451)]

- LegalEval-Q: A New Benchmark for The Quality Evaluation of LLM-Generated Legal Text [[arxiv](https://arxiv.org/pdf/2505.24826)]

- On the Generalization Ability of Next-Token-Prediction Pretraining [[ICML 2025](https://openreview.net/forum?id=hLGJ1qZPdu)]

- 《从零开始写大模型：从神经网络到Transformer》王双、牟晨、王昊怡 编著 - 清华大学出版社

- FedBRB: A Solution to the Small-to-Large Scenario in Device-Heterogeneity Federated Learning [[TMC 2025](https://ieeexplore.ieee.org/abstract/document/11168259)]

- SKETCH: Semantic Key-Point Conditioning for Long-Horizon Vessel Trajectory Prediction [[arxiv](https://arxiv.org/pdf/2601.18537)]

- A Built-in Crypto Expert for Artificial Intelligence: How Far is the Horizon? [[IACR ePrint 2026](https://eprint.iacr.org/2026/411.pdf)]

- In progress...


# 🎓 Citation

If `MiniMind` has been helpful to your research or work, feel free to cite:

```bibtex
@misc{minimind,
  title = {MiniMind: Train a Tiny LLM from Scratch},
  author = {Jingyao Gong},
  year = {2024},
  url = {https://github.com/jingyaogong/minimind},
  note = {GitHub repository, accessed 2026}
}
```

# ⚖️ License

This project is open-sourced under the [Apache License 2.0](LICENSE).