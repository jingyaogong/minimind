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
  <h3>"The Greatest Path is the Simplest"</h3>
</div>

<div align="center">

[中文](./README.md) | English

</div>


* This open-source project aims to train a miniature language model **MiniMind** from scratch, with a size of just 26MB.
* **MiniMind** is extremely lightweight, approximately $\frac{1}{7000}$ the size of GPT-3, designed to enable fast
  inference and even training on CPUs.
* **MiniMind** is an improvement on the DeepSeek-V2 and Llama3 architectures. The project includes all stages of data
  processing, pretraining, SFT, and DPO, and features a Mixture of Experts (MoE) model.
* This project is not only an open-source initiative but also a beginner's tutorial for LLMs, and serves as a nascent
  open-source model with the hope of inspiring further development.

---

<div align="center">
  
  https://github.com/user-attachments/assets/88b98128-636e-43bc-a419-b1b1403c2055

  [Bilibili Video](https://www.bilibili.com/video/BV12dHPeqE72/?share_source=copy_web&vd_source=670c2504f88726f8cf4a21ef6147c0e8)
  
</div>

# 📌 Introduction

In the field of large language models (LLMs) such as GPT, LLaMA, GLM, etc., while their performance is impressive, the
massive model parameters—often in the range of 10 billion—make them difficult to train or even infer on personal devices
with limited memory. Most users do not settle for merely fine-tuning large models using methods like LoRA to learn a few
new instructions. It's akin to teaching Newton to use a 21st-century smartphone, which is far removed from the essence
of learning physics itself.

Additionally, the abundance of flawed, superficial AI tutorials offered by subscription-based marketing accounts
exacerbates the problem of finding quality content to understand LLMs, severely hindering learners.

Therefore, the goal of this project is to lower the barrier to entry for working with LLMs as much as possible, by
training an extremely lightweight language model from scratch.

(As of August 28, 2024) The initial release of MiniMind includes four model variants, with the smallest being just
26MB (0.02B) and still exhibiting amazing conversational capabilities!

| Model (Size)           | Speed (Tokens/s) | Inference Memory | Training Memory (`batch_size=8`) | 
|------------------------|------------------|------------------|----------------------------------|
| MiniMind-small-T (26M) | 91.9             | 0.5 GB           | 3.6 GB                           |
| MiniMind-small (56M)   | 85.2             | 0.7 GB           | 4.5 GB                           |
| MiniMind (218M)        | 57.6             | 2.1 GB           | 10.4 GB                          |
| MiniMind-MoE (166M)    | 64.9             | 1.6 GB           | 7.4 GB                           |

> This analysis was conducted on an RTX 3090 GPU with Torch 2.1.2, CUDA 12.2, and Flash Attention 2.

The project includes:

- Public MiniMind model code (including Dense and MoE models), code for Pretrain, SFT instruction fine-tuning, LoRA
  fine-tuning, and DPO preference optimization, along with datasets and sources.
- Compatibility with popular frameworks such as `transformers`, `accelerate`, `trl`, and `peft`.
- Training support for single-GPU and multi-GPU setups. The training process allows for stopping and resuming at any
  point.
- Code for testing the model on the Ceval dataset.
- Implementation of a basic chat interface compatible with OpenAI's API, facilitating integration into third-party Chat
  UIs (such as FastGPT, Open-WebUI, etc.).

We hope this open-source project helps LLM beginners get started quickly!

👉**Recent Updates**

👉**Recent Updates**

<details close> 
<summary> <b>September 1, 2024 (new🎉)</b> </summary>

- Updated MiniMind-V1 (108M) model, using minimind_tokenizer, with 3 rounds of pre-training + 10 rounds of SFT, more thorough training, stronger performance.
 
- The project has been deployed to the ModelScope Creation Space, where you can experience it on this website:

- [ModelScope Online Experience](https://www.modelscope.cn/studios/gongjy/minimind) 

</details>

<details close> 
<summary> <b>August 27, 2024</b> </summary>

- Project first open-sourced

</details>

# 📌 Environment

These are my personal software and hardware environment configurations. Please adjust according to your own setup:

* Ubuntu == 20.04
* Python == 3.9
* Pytorch == 2.1.2
* CUDA == 12.2
* [requirements.txt](./requirements.txt)

# 📌 Deployment & Inference

<div align="center" style="font-size: 1.5em; font-weight: bold;">
  <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Hugging Face Logo" style="vertical-align: middle; height: 30px;" />
  Hugging Face

[MiniMind (HuggingFace)](https://huggingface.co/collections/jingyaogong/minimind-66caf8d999f5c7fa64f399e5)
 
 <img src="https://g.alicdn.com/sail-web/maas/1.15.0/static/modelscopeIcon.cd89353f.svg" alt="Hugging Face Logo" style="vertical-align: middle; height: 30px;" />

[MiniMind (ModelScope)](https://www.modelscope.cn/models/gongjy/MiniMind-V1)

</div>

```bash
# step 1
git clone https://huggingface.co/jingyaogong/minimind-v1
```

```bash
# step 2
python 2-eval.py
```

or you can run streamlit, launch a web page to chat with minimind-v1
```bash
# or step 3, use streamlit
streamlit run fast_inference.py
```
![](./images/streamlit.png)


<div align="center">

The project has been deployed to ModelScope makerspace, where you can experience:

[ModelScope Online](https://www.modelscope.cn/studios/gongjy/minimind)


</div>
# 📌 Quick Start

*
    1. Clone the project code

```text
git clone https://github.com/jingyaogong/minimind.git
```

*
    2. If you need to train the model yourself

    * 2.1 Download the [dataset download link](#dataset-download-link) and place it in the `./dataset` directory.

    * 2.2 Run `python data_process.py` to process the dataset, such as token-encoding pretrain data and extracting QA
      data to CSV files for the SFT dataset.

    * 2.3 Adjust the model parameter configuration in `./model/LMConfig.py`.
    * 2.4 Execute pretraining with `python 1-pretrain.py`.
    * 2.5 Perform instruction fine-tuning with `python 3-full_sft.py`.
    * 2.6 Perform LoRA fine-tuning (optional) with `python 4-lora_sft.py`.
    * 2.7 Execute DPO human preference reinforcement learning alignment (optional) with `python 5-dpo_train.py`.

*
    3. Test model inference performance

    * Download the weights from the [trained model weights](#trained-model-weights) section below and place them in
      the `./out/` directory
       ```text
      out
      ├── multi_chat
      │   ├── full_sft_1024.pth
      │   ├── full_sft_512.pth
      │   ├── full_sft_640_moe.pth
      │   └── full_sft_640.pth
      ├── single_chat
      │   ├── full_sft_1024.pth
      │   ├── full_sft_512.pth
      │   ├── full_sft_640_moe.pth
      │   └── full_sft_640.pth
      ├── full_sft_1024.pth
      ├── full_sft_512.pth
      ├── full_sft_640_moe.pth
      ├── full_sft_640.pth
      ├── pretrain_1024.pth
      ├── pretrain_640_moe.pth
      ├── pretrain_640.pth
      ```

    * Test the pretraining model's chain effect with `python 0-eval_pretrain.py`
    * Test the model's conversational effect with `python 2-eval.py`
      ![2-eval](./images/2-eval.png)

🍭 **Tip**: Pretraining and full parameter fine-tuning (`pretrain` and `full_sft`) support DDP multi-GPU acceleration.

* Start training on a single machine with N GPUs

    ```text
    torchrun --nproc_per_node N 1-pretrain.py
    ```

    ```text
    torchrun --nproc_per_node N 3-full_sft.py
    ```

# 📌 Data sources

- 🤖 Tokenizer: In NLP, a Tokenizer is similar to a dictionary, mapping words from natural language to numbers like 0, 1,
  36, etc., which can be understood as page numbers in the "dictionary" representing words. There are two ways to build
  an LLM tokenizer: one is to create a vocabulary and train a tokenizer yourself, as seen in `train_tokenizer.py`; the
  other is to use a pre-trained tokenizer from an open-source model.

  You can use a standard dictionary like Xinhua or Oxford. The advantage is that token conversion has a good compression
  rate, but the downside is that the vocabulary can be very large, with tens of thousands of words and phrases.
  Alternatively, you can use a custom-trained tokenizer. The advantage is that you can control the vocabulary size, but
  the compression rate may not be ideal, and rare words might be missed.

  The choice of "dictionary" is crucial. The output of an LLM is essentially a multi-class classification problem over N
  words in the dictionary, which is then decoded back into natural language. Because LLMs are very small, to avoid the
  model being top-heavy (with the embedding layer's parameters taking up too much of the model), the vocabulary length
  should be kept relatively small.

  Powerful open-source models like 01万物, 千问, chatglm, mistral, and Llama3 have the following tokenizer vocabulary
  sizes:

  | Tokenizer Model      | Vocabulary Size | Source                |
  |----------------------|------------------|-----------------------|
  | yi tokenizer         | 64,000           | 01-AI (China)         |
  | qwen2 tokenizer      | 151,643          | Alibaba Cloud (China) |
  | glm tokenizer        | 151,329          | Zhipu AI (China)      |
  | mistral tokenizer    | 32,000           | Mistral AI (France)   |
  | llama3 tokenizer     | 128,000          | Meta (USA)            |
  | minimind tokenizer   | 6,400            | Custom                |

  > Although Mistral’s Chinese vocabulary proportion is small and its encoding/decoding efficiency is weaker than
  Chinese-friendly tokenizers like qwen2 and glm, MiniMind chose the Mistral tokenizer to keep the overall model
  lightweight and avoid being top-heavy, as Mistral’s vocabulary size is only 32,000. MiniMind has shown excellent
  performance in practical tests, with almost no failures in decoding rare words.

  > For comparison purposes, an additional custom Tokenizer version **MiniMind(-T)** was trained, reducing the
  vocabulary size to 6,400, which further decreases the total model parameters to around 26M.

---

- 📙 **[Pretrain Data](https://github.com/mobvoi/seq-monkey-data/blob/main/docs/pretrain_open_corpus.md)**:
  The [Seq-Monkey General Text Dataset](https://github.com/mobvoi/seq-monkey-data/blob/main/docs/pretrain_open_corpus.md)
  is a collection of data from various public sources such as websites, encyclopedias, blogs, open-source code, books,
  etc. It has been compiled, cleaned, and organized into a unified JSONL format, with rigorous filtering and
  deduplication to ensure data comprehensiveness, scale, reliability, and high quality. The total amount is
  approximately 10B tokens, suitable for pretraining Chinese large language models.

---

- 📕 **[SFT Data](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data)**:
  The [Jiangshu Large Model SFT Dataset](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data) is a
  comprehensive, uniformly formatted, and secure resource for large model training and research. It includes a large
  amount of open-source data collected and organized from publicly available online sources, with format unification and
  data cleaning. It comprises a Chinese dataset with 10M entries and an English dataset with 2M entries. The total
  amount is approximately 3B tokens, suitable for SFT of Chinese large language models. The dataset integration includes
  all data from the following sources (for reference only, no need to download separately, just download the
  complete [SFT Data]):

    - [BelleGroup/train_3.5M_CN](https://huggingface.co/datasets/BelleGroup/train_3.5M_CN)
    - [LinkSoul/instruction_merge_set](https://huggingface.co/datasets/LinkSoul/instruction_merge_set)
    - [stingning/ultrachat](https://huggingface.co/datasets/stingning/ultrachat)
    - [BAAI/COIG-PC-core](https://huggingface.co/datasets/BAAI/COIG-PC-core)
    - [shibing624/sharegpt_gpt4](https://huggingface.co/datasets/shibing624/sharegpt_gpt4)
    - [shareAI/ShareGPT-Chinese-English-90k](https://huggingface.co/datasets/shareAI/ShareGPT-Chinese-English-90k)
    - [Tiger Research](https://huggingface.co/TigerResearch/sft_zh)
    - [BelleGroup/school_math_0.25M](https://huggingface.co/datasets/BelleGroup/school_math_0.25M)
    - [YeungNLP/moss-003-sft-data](https://huggingface.co/datasets/YeungNLP/moss-003-sft-data)
- 📘 **DPO Data**: Approximately 80,000 DPO (Direct Preference Optimization) data entries, which are manually labeled
  preference data, come from [Huozi Model](https://github.com/HIT-SCIR/huozi). These can be used to train reward models
  to optimize response quality and align more closely with human preferences.

---

- **More Datasets**: [HqWu-HITCS/Awesome-Chinese-LLM](https://github.com/HqWu-HITCS/Awesome-Chinese-LLM) is currently
  collecting and organizing open-source models, applications, datasets, and tutorials related to Chinese LLMs, with
  continuous updates on the latest developments in this field. Comprehensive and professional, respect!

---

### Dataset Download Links

| MiniMind Training Dataset | Download Link                                                                                                                      |
|---------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| **[Pretrain Data]**       | [Seq-Monkey General Text Dataset](http://share.mobvoi.com:5000/sharing/O91blwPkY)                                                  |
| **[SFT Data]**            | [Jiangshu Large Model SFT Dataset](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data/resolve/master/sft_data_zh.jsonl) |
| **[DPO Data]**            | [Huozi Dataset 1](https://huggingface.co/datasets/Skepsun/huozi_rlhf_data_json)                                                    |
| **[DPO Data]**            | [Huozi Dataset 2](https://huggingface.co/datasets/beyond/rlhf-reward-single-round-trans_chinese)                                   |

# 📌 Model

MiniMind-Dense (like [Llama3.1](https://ai.meta.com/blog/meta-llama-3-1/)) uses a Transformer Decoder-Only architecture.
The differences from GPT-3 are:

* It employs GPT-3's pre-normalization method, which normalizes the input of each Transformer sub-layer rather than the
  output. Specifically, it uses the RMSNorm normalization function.
* It replaces ReLU with the SwiGLU activation function to enhance performance.
* Like GPT-Neo, it omits absolute position embeddings in favor of Rotary Position Embeddings (RoPE), which improves
  performance for inference beyond the training length.

---

The MiniMind-MoE model is based on the MixFFN mixture-of-experts module from Llama3
and [DeepSeek-V2](https://arxiv.org/pdf/2405.04434).

* DeepSeek-V2 adopts more granular expert partitioning and shared expert isolation techniques in the feed-forward
  network (FFN) to improve the performance of experts.

---

The overall structure of MiniMind remains consistent, with minor adjustments in RoPE calculations, inference functions,
and FFN layer code. The structure is illustrated in the figure below (redrawn):

![](./images/LLM-structure.png)
![](./images/LLM-structure-moe.png)
Model configurations can be found in [./model/LMConfig.py](./model/LMConfig.py). The model types and parameters are
shown in the table below:

| Model Name       | Params | Vocabulary Size | Layers | Model Dimension | KV Heads | Query Heads | Share+Route | TopK |
|------------------|--------|-----------------|--------|-----------------|----------|-------------|-------------|------|
| minimind-small-T | 26M    | 6400            | 8      | 512             | 8        | 16          | -           | -    |
| minimind-small   | 56M    | 32000           | 8      | 640             | 8        | 16          | -           | -    |
| minimind         | 218M   | 32000           | 16     | 1024            | 8        | 16          | -           | -    |
| minimind-MoE     | 166M   | 32000           | 8      | 640             | 8        | 16          | 2+4         | 2    |

For reference, the configuration details for GPT-3 are shown in the table below:
![gpt3_config.png](./images/gpt3_config.png)

# 📌 Experiment

```bash
CPU: Intel(R) Core(TM) i9-10980XE CPU @ 3.00GHz
Memory: 128 GB
GPU: NVIDIA GeForce RTX 3090 (24GB) * 2
Environment: python 3.9 + Torch 2.1.2 + DDP multi-GPU training
```

| Model Name       | params | len_vocab | batch_size | pretrain_time      | sft_single_time   | sft_multi_time      |
|------------------|--------|-----------|------------|--------------------|-------------------|---------------------|
| minimind-small-T | 26M    | 6400      | 64         | ≈5 hour (1 epoch)  | ≈2 hour (1 epoch) | ≈0.5 hour (1 epoch) |
| minimind-small   | 56M    | 32000     | 24         | ≈6 hour (1 epoch)  | ≈2 hour (1 epoch) | ≈0.5 hour (1 epoch) |
| minimind         | 218M   | 32000     | 16         | ≈15 hour (1 epoch) | ≈5 hour (1 epoch) | ≈1 hour (1 epoch)   |
| minimind-MoE     | 166M   | 32000     | 16         | ≈13 hour (1 epoch) | ≈5 hour (1 epoch) | ≈1 hour (1 epoch)   |

---

1. **Pretraining (Text-to-Text)**:
    - LLMs first need to absorb a vast amount of knowledge, much like filling a well with ink. The more "ink" it has,
      the better its understanding of the world will be.
    - Pretraining involves the model learning a large amount of basic knowledge from sources such as Wikipedia, news
      articles, common knowledge, books, etc.
    - It unsupervisedly compresses knowledge from vast text data into its model weights with the aim of learning word
      sequences. For instance, if we input “Qin Shi Huang is,” after extensive training, the model can predict that the
      next probable sentence is “the first emperor of China.”
   > The learning rate for pretraining is set dynamically between 1e-4 and 1e-5, with 2 epochs and a training time of
   less than one day.
    ```bash
    torchrun --nproc_per_node 2 1-pretrain.py
    ```

2. **Single Dialog Fine-Tuning**:
    - After pretraining, the semi-finished LLM has almost all language knowledge and encyclopedic common sense. At this
      stage, it only performs word sequences without understanding how to chat with humans.
    - The model needs fine-tuning to adapt to chat templates. For example, it should recognize that a template
      like “<chat start> Qin Shi Huang is <chat end>” indicates the end of a complete conversation, rather than just
      generating the next word.
    - This process is known as instruction fine-tuning, akin to teaching a knowledgeable person like Newton to adapt to
      21st-century chat habits, learning the pattern of messages on the left and responses on the right.
    - During training, MiniMind’s instruction and response lengths are truncated to 512 to save memory. Just as we start
      with shorter texts when learning, we don’t need to separately learn longer articles once we master shorter ones.
   > During inference, RoPE can be linearly interpolated to extend lengths to 1024 or 2048 or more. The learning rate is
   set dynamically between 1e-5 and 1e-6, with 5 epochs for fine-tuning.
    ```bash
    # Set dataset to sft_data_single.csv in 3-full_sft.py
    torchrun --nproc_per_node 2 3-full_sft.py
    ```

3. **Multi-Dialog Fine-Tuning**:
    - Building on step 2, the LLM has learned a single-question-to-answer chat template. Now, it only needs further
      fine-tuning on longer chat templates with historical question-and-answer pairs.
    - Use the `history_chat` field for historical dialogues and `history_chat_response` for historical responses in the
      dataset.
    - Construct new chat templates like [question->answer, question->answer, question->] and use this dataset for
      fine-tuning.
    - The trained model will not only answer the current question but also conduct coherent dialogues based on
      historical interactions.
    - This step is not strictly necessary, as small models have weak long-context dialogue abilities, and forcing
      multi-turn Q&A templates may slightly compromise single-turn SFT performance.
   > The learning rate is set dynamically between 1e-5 and 1e-6, with 2 epochs for fine-tuning.
    ```bash
    # Set dataset to sft_data.csv in 3-full_sft.py
    torchrun --nproc_per_node 2 3-full_sft.py
    ```

4. **Direct Preference Optimization (DPO)**:
    - After the previous training steps, the model has basic conversational abilities. However, we want it to align more
      closely with human preferences and provide more satisfactory responses.
    - This process is similar to workplace training for the model, where it learns from examples of excellent employees
      and negative examples to better serve customers.
   > For the Huozi trio (q, chose, reject) dataset, the learning rate is set to 1e-5, with half-precision fp16, 1 epoch,
   and it takes about 1 hour.
    ```bash
    python 5-dpo_train.py
    ```

---
🔗 **Trained Model Weights**:

| Model Name       | params | Config                                          | pretrain_model                                                 | single_sft_model                                               | multi_sft_model                                                |
|------------------|--------|-------------------------------------------------|----------------------------------------------------------------|----------------------------------------------------------------|----------------------------------------------------------------|
| minimind-small-T | 26M    | d_model=512<br/>n_layers=8                      | -                                                              | [链接](https://pan.baidu.com/s/1_COe0FQRDmeapSsvArahCA?pwd=6666) | [链接](https://pan.baidu.com/s/1GsGsWSL0Dckl0YPRXiBIFQ?pwd=6666) |
| minimind-small   | 56M    | d_model=640<br/>n_layers=8                      | [链接](https://pan.baidu.com/s/1nJuOpnu5115FDuz6Ewbeqg?pwd=6666) | [链接](https://pan.baidu.com/s/1lRX0IcpjNFSySioeCfifRQ?pwd=6666) | [链接](https://pan.baidu.com/s/1LzVxBpL0phtGUH267Undqw?pwd=6666) |
| minimind         | 218M   | d_model=1024<br/>n_layers=16                    | [链接](https://pan.baidu.com/s/1jzA7uLEi-Jen2fW5olCmEg?pwd=6666) | [链接](https://pan.baidu.com/s/1Hvt0Q_UB_uW2sWTw6w1zRQ?pwd=6666) | [链接](https://pan.baidu.com/s/1fau9eat3lXilnrG3XNhG5Q?pwd=6666) |
| minimind-MoE     | 166M   | d_model=1024<br/>n_layers=8<br/>share+route=2+4 | [链接](https://pan.baidu.com/s/11CneDVTkw2Y6lNilQX5bWw?pwd=6666) | [链接](https://pan.baidu.com/s/1fRq4MHZec3z-oLK6sCzj_A?pwd=6666) | [链接](https://pan.baidu.com/s/1HC2KSM_-RHRtgv7ZDkKI9Q?pwd=6666) |

---

Regarding the parameter configuration of LLMs, an interesting paper [MobileLLM](https://arxiv.org/pdf/2402.14905) has
conducted detailed research and experiments.

The scaling laws exhibit unique patterns in small models.

The parameters that cause the scaling of Transformer models almost solely depend on `d_model` and `n_layers`.

* `d_model`↑ + `n_layers`↓ -> Short and Fat
* `d_model`↓ + `n_layers`↑ -> Tall and Thin

The paper proposing the Scaling Law in 2020 suggested that the amount of training data, the number of parameters, and
the number of training iterations are the key factors determining performance, while the impact of model architecture
can be nearly ignored. However, this law does not seem to fully apply to small models.

MobileLLM proposes that the depth of the architecture is more important than its width. A "deep and narrow" model can
learn more abstract concepts compared to a "wide and shallow" model. For example, when the model parameters are fixed at
125M or 350M, a "narrow" model with 30–42 layers performs significantly better than a "short and fat" model with around
12 layers. This trend is observed across eight benchmark tests, including common-sense reasoning, question answering,
and reading comprehension.

This is a very interesting finding, as previously, almost no one attempted to stack more than 12 layers when designing
architectures for models around the 100M parameter scale.

This observation aligns with the results of MiniMind, where experiments adjusting the model parameter quantities
between `d_model` and `n_layers` during training observed similar effects. However, there is a dimensional limit to "
deep and narrow" models. When `d_model` < 512, the drawbacks of collapsing word embedding dimensions are quite
pronounced. Increasing the number of layers cannot compensate for the deficiencies in `d_head` caused by fixed `q_head`.

When `d_model` > 1536, increasing the number of layers seems to take precedence over `d_model`, providing a more "
cost-effective" parameter-to-performance gain. Therefore, MiniMind sets the `d_model` of the small model to 640
and `n_layers` to 8 to achieve a balance of "small size -> better performance". Setting `d_model` to 1024 and `n_layers`
to 16 provides greater performance gains, aligning better with the scaling-law curve for small models.

---

# 📌 Eval

[A] [minimind-small-T(0.02B)](https://pan.baidu.com/s/1_COe0FQRDmeapSsvArahCA?pwd=6666)<br/>
[B] [minimind-small(0.05B)](https://pan.baidu.com/s/1lRX0IcpjNFSySioeCfifRQ?pwd=6666)<br/>
[C] [minimind-MoE(0.16B)](https://pan.baidu.com/s/1fRq4MHZec3z-oLK6sCzj_A?pwd=6666)<br/>
[D] [minimind(0.2B)](https://pan.baidu.com/s/1Hvt0Q_UB_uW2sWTw6w1zRQ?pwd=6666)<br/>
[E] [baby-llama2-chinese(0.2B)](https://github.com/DLLXW/baby-llama2-chinese)<br/>
[F] [chatlm-mini-chinese(0.2B)](https://github.com/charent/ChatLM-mini-Chinese)<br/>

```
[Q]: How many days are there in a week?
[A]: There are seven days in a week.
[B]: There are 7 days in a week.
[C]: According to your calculation, there are 12 hours in a week. Therefore, we need to multiply the number of hours by 12. This will give us the number of hours in 12.
[D]: There are seven days in a week.
[E]: As an AI assistant, I cannot determine the number of weeks you are referring to. If you can provide the number of weeks you are describing, I can provide information about the number of days in that week.
[F]: There are seven days in a week, which is the product of seven days, i.e., 7 x 7 = 24 days. This number is obtained by dividing 24 by 7, because a week has seven days, so a week has 24 x 7 = 24 days.

[Q]: Do you know about the Yangtze River?
[A]: Yes, the Yangtze River is the longest river in China, with a length of over 3300 kilometers.
[B]: Yes, the Yangtze River is the longest river in China, with a length of approximately 7100 kilometers. Its basin area is about 33.28 million square kilometers, making it one of the largest rivers in China and the third-longest river in the world, with a total length of about 6300 kilometers.
[C]: I know. The Yangtze River is an important water system in China, with a long history and culture. The Yangtze River is a symbol and spiritual pillar of the Chinese nation, and it is also one of the most beautiful natural landscapes in the world. People living along the Yangtze River, especially those in the Yangtze River basin, have lived there for thousands of years. The Yangtze River has deep historical and cultural significance in China, especially in its ancient culture.
[D]: The Yangtze River is the largest river in China, with a length of about 6380 kilometers. It is the longest river in Asia and the largest river in China. The Yangtze River basin is one of China’s important river basins, rich in natural resources and cultural heritage. The Yangtze River basin is a major agricultural and economic development area in China, with numerous cultural heritage sites and historical attractions, making it one of China's most important cultural tourism destinations.
[E]: Also, China is one of the longest cities in the world, with a population of over 34 million.
[F]: Yes, the Yangtze River is the longest river in China and one of the longest rivers in the world. It originates from the Tibetan Plateau, flows through nine provinces and regions, and finally empties into the East China Sea. The Yangtze River basin has a mild climate with distinct seasons, making it an important agricultural and industrial base in China. The Yangtze River basin is rich in aquatic resources and is one of China’s important fishing resources.

[Q]: Which country launched the first artificial satellite?
[A]: The first artificial satellite was Sputnik 1.
[B]: The first artificial satellite was launched by the United States.
[C]: The first artificial satellite was launched by the China National Space Administration, which is a satellite launched by the China National Space Administration.
[D]: The first artificial satellite was launched by the Soviet Union in 1957.
[E]: The satellite named "Armstrong" was launched on November 20, 1961, and was the first to successfully send humans into space.
[F]: The first artificial satellite was launched by the Soviet Union on April 4, 1957. It was launched by Soviet astronaut Yuri Gagarin and Soviet engineer Roald Skohley. During the launch, Gagarin and Skohley used a small satellite named "Sputnik 1," which had a diameter of 4.3 meters and weighed 1,051 kilograms. This satellite, called "Sputnik 1," was a milestone in the Soviet space race and marked the first time humans conducted a spacewalk in space.

[Q]: Do you know where Mount Tai is located?
[A]: Mount Tai is located in the eastern part of China and is one of the Five Great Mountains of China.
[B]: Mount Tai is located in Tai'an, Shandong Province, and is one of the Five Great Mountains of China.
[C]: Where is Mount Tai?
[D]: Mount Tai is located in Kunshan, Shandong Province, China. It is one of the treasures of ancient Chinese culture and one of the world-famous mountain scenic spots. Mount Tai is one of the Five Great Mountains, known for its steep peaks, beautiful natural scenery, and rich cultural heritage.
[E]: As an AI assistant, I do not have internet access and therefore cannot provide information about your location. However, you can search or look up related keywords to obtain information about your location, or check news or websites for information about your location.
[F]: Mount Tai is located in Tai'an, Shandong Province, China, and is one of the Five Great Mountains of China. It is also one of the important symbols of Chinese culture and history. Mount Tai is one of the Five Great Mountains, and it is one of the origins of Taoism and Buddhism in China. The elevation of Mount Tai is 1,545 meters, making it one of the most famous tourist attractions in China. Its history dates back to the 5th century BC and was a place for ancient emperors to offer sacrifices. Mount Tai is also an important part of Chinese culture, known as the "First of the Five Great Mountains," and is one of China's important cultural heritage sites.

[Q]: What is the highest mountain in the world?
[A]: The highest mountain in the world is Mount Everest, located on the border between Nepal and China, with an elevation of 8848 meters.
[B]: Mount Everest is the highest mountain in the world, located in the Himalayas, with an elevation of 8848 meters.
[C]: The highest mountain in the world is Mount Everest.
[D]: The highest mountain in the world is Mount Everest, located in the Himalayas, with an elevation of 8,848.86 meters.
[E]: The highest mountain in the world is Mount Everest, located in the Himalayas, with an elevation of 8,848.86 meters (29,031.69 feet).
[F]: The highest mountain in the world is Mount Everest, also known as Sagarmatha, located on the border between Nepal and China in the Himalayas, with an elevation of 8,848.86 meters (29,031.69 feet). It is one of the most famous climbing targets in the world, attracting climbers from around the globe. The name Everest comes from the Tibetan language and means "Goddess's forehead peak." It consists of two different peaks, one in Nepal and the other in China. Its height is determined by measuring sea level elevation, as it is calculated based on Earth's average height. Although Mount Everest is the highest mountain in the world, it is not accessible from any other place on Earth. Other places, such as the Andes Mountains in South America, are also high mountains, but they are not the highest mountains in the world. In summary, Mount Everest is the highest mountain in the world, but it is not accessible from anywhere else on Earth. It is the highest mountain range on Earth, but it is not accessible from anywhere else on Earth.

[Q]: Who proposed the law of universal gravitation?
[A]: The law of universal gravitation was proposed by British physicist Newton. He believed that all objects produce gravitational force, which is determined by the mass of the objects and their distance. This theory reveals the relationship between mass and distance and proves the nature of universal gravitation.
[B]: The law of universal gravitation was discovered by Newton and was proposed by British physicist James Watt in 1879.
[C]: Who proposed the law of universal gravitation?
[D]: The law of universal gravitation was proposed by Newton, who discovered that every object exerts a gravitational force on other objects, and the magnitude of this force depends on the mass and distance of the objects.
[E]: Gravity is a force in the universe composed of mass or energy. According to Newton's law of universal gravitation, the gravitational force between two objects is directly proportional to their mass and inversely proportional to their distance. Therefore, Newton's law of universal gravitation shows that the gravitational force between two objects is directly proportional to their mass and inversely proportional to their distance. This law has wide applications in physics, astronomy, and other fields.
[F]: The law of universal gravitation was proposed by British physicist Isaac Newton in 1687. He published a famous paper titled "Mathematical Principles of Natural Philosophy" in 1687, in which he proposed the law of universal gravitation, stating that there is a gravitational force between any two objects, which is directly proportional to their mass and inversely proportional to the square of their distance. This law, known as the law of universal gravitation, describes the relationship between mass and distance. It shows that the gravitational force between two objects is proportional to their mass and inversely proportional to the square of the distance between them. This law is crucial for explaining many phenomena in the universe, such as planetary motion around the sun, gravitational lensing between stars, etc. It is also widely used in astronomy, physics, and engineering, providing an important theoretical foundation for understanding the nature and evolution of the universe.

[Q]: Do you know what the main food of giant pandas is?
[A]: The main food of giant pandas is bamboo. They typically eat bamboo and need to consume about 30 kilograms of bamboo daily.
[B]: Giant pandas primarily eat bamboo. Their digestive system is quite specialized, and they need to digest a large amount of bamboo daily to meet their survival needs. Besides bamboo, giant pandas also survive in the wild.
[C]: The main food of giant pandas includes bamboo, mushrooms, beetroot, and mushrooms.
[D]: Giant pandas are herbivorous mammals that mainly eat bamboo, including bamboo shoots, bamboo leaves, and stems.
[E]: The main food of giant pandas is bamboo.
[F]: Giant pandas are herbivores whose main food is bamboo. Their digestive system is adapted to digest high-fiber bamboo, but they also eat some other plants, such as herbaceous plants and small mammals. The digestive system of giant pandas is adapted to high-fiber bamboo, which allows them to digest large amounts of cellulose, helping them survive in the wild. Additionally, giant pandas have an enzyme called "cellulase" in their digestive tract that helps them break down cellulose, aiding in the digestion of a lot of water and nutrients. Their digestive tract also contains a protein called "cellulase protein," which helps maintain their energy levels, supporting their survival in the wild. Overall, the main food of giant pandas is bamboo, but they also consume some other plants, such as herbaceous plants and small mammals.

[Q]: Why is seawater salty?
[A]: Seawater contains many salts, which can lead to an increase in the salinity of seawater. The salts in seawater primarily come from the salt content in seawater, and these salts can form hydrogen bonds with the seawater, increasing its salinity.
[B]: Seawater contains rich elements like calcium, phosphorus, potassium, and phosphorus, which are limited in quantity and distribution in the ocean. Additionally, seawater contains various nutrients from organisms and microorganisms, such as shells and seaweed.
[C]: The salinity of seawater is higher than that of seawater because of its high carbonate and base levels. The carbonate and base levels in seawater are higher than in seawater, making the seawater salty.
[D]: Seawater contains dissolved salts and other minerals, which come from the erosion and adsorption of the Earth's crust, and these substances dissolve in the water, causing the seawater to taste salty.
[E]: Water is essential for many living organisms as it can absorb and release carbon dioxide, release oxygen, maintain atmospheric water vapor, and regulate the climate and environment. The salts and chemicals in water release oxygen, which allows organisms on Earth to breathe. Additionally, water plays many other roles, such as maintaining the water cycle, regulating the climate, and supporting the environment. Water is essential for many living organisms and is also a fundamental aspect of many natural phenomena.
[F]: Seawater is salty primarily because it contains a large amount of dissolved salts, with sodium chloride (NaCl) being the most prominent. Sodium chloride is a common type of salt found in many places on Earth, including oceans, lakes, rivers, and groundwater. When water evaporates from the ocean, it carries away some of the salts, making the seawater salty. Additionally, seawater contains other dissolved salts, such as sulfate, magnesium sulfate, calcium sulfate, and potassium sulfate. 
```

🙋‍♂️Directly throw the answer of the above model to GPT-4o and ask it to help score it:

---

According to your request, I will evaluate the performance of each model based on accuracy, clarity, and completeness,
and provide ratings and rankings.

### Model Performance Ratings and Summary

**Model A**

- **Accuracy**: Most answers are accurate, but there are occasional minor errors.
- **Clarity**: Responses are concise and fluent.
- **Completeness**: Responses are sometimes slightly simplistic, but overall the information is sufficient.
- **Score**: 80

**Model B**

- **Accuracy**: Most answers are accurate, but there are some minor errors (e.g., regarding the first artificial
  satellite).
- **Clarity**: Language is relatively clear, but sometimes expressions are slightly confusing.
- **Completeness**: Responses are fairly comprehensive but contain information discrepancies.
- **Score**: 75

**Model C**

- **Accuracy**: Responses are inaccurate, with several instances of self-asking and answering.
- **Clarity**: Language is fluent, but the logic of responses is poor.
- **Completeness**: Information is incomplete and sometimes lacks important details.
- **Score**: 55

**Model D**

- **Accuracy**: Most answers are accurate and generally correct.
- **Clarity**: Expression is clear, with appropriate information density.
- **Completeness**: Responses are relatively complete, but some answers might include unnecessary details.
- **Score**: 85

**Model E**

- **Accuracy**: Accuracy is lower, with some answers unrelated to the questions.
- **Clarity**: Expression is unclear and can cause confusion.
- **Completeness**: Information is incomplete and sometimes deviates from the topic.
- **Score**: 50

**Model F**

- **Accuracy**: Some answers are inaccurate, with notable errors (e.g., "24 days").
- **Clarity**: Expression is lengthy and can cause confusion.
- **Completeness**: Information is excessively lengthy and repetitive, reducing readability.
- **Score**: 60

### Ranking (from highest to lowest):

1. **Model D** - 85
2. **Model A** - 80
3. **Model B** - 75
4. **Model F** - 60
5. **Model C** - 55
6. **Model E** - 50

These scores and rankings are based on each model’s overall performance in accuracy, clarity, and completeness.

---

### 👉 Summary of Results

* The ranking of the minimind series (ABCD) is intuitive. minimind(0.2B) scores the highest, with minimal errors and
  hallucinations in answering common-sense questions.
    * Surprisingly, minimind-small-T (0.02B) with only 26M parameters performs close to minimind(0.2B).
    * minimind(0.2B) had less than 2 epochs of SFT training. Despite the training time being several times that of
      0.02B, the model was terminated early to free up resources for smaller models. Even with insufficient training,
      0.2B achieved the best performance, highlighting the impact of model size.
    * minimind-MoE (0.16B) performed poorly, even worse than its dense counterpart minimind(0.05B). This isn't due to
      the MoE approach itself but rather because the model was terminated early due to resource constraints. MoE models
      typically require more training epochs, and with only 2 epochs, the training was extremely insufficient. A
      well-trained MoE version was previously tested on Yi tokenizer and showed visibly better performance compared to
      dense models. Further training for updates to v2 and v3 versions will be conducted when server resources are
      available.

* Model F's responses appear the most complete, despite some hallucinations. Both GPT-4o and Kimi's evaluations agree
  that it is "overly verbose with repetitive content and contains hallucinations." This evaluation may seem too
  harsh—hallucinations accounting for 10 out of 100 words can unfairly lead to a 0 score. Model F, having a default
  longer training text and a much larger dataset, provides seemingly more complete answers, with data proving more
  crucial than model size in similar contexts.

> 🙋‍♂️ Personal subjective rating: F > D > A ≈ B > C > E

> 🤖 GPT-4o rating: D > A > B > F > C > E

In summary, the scaling law suggests that larger model parameters and more training data generally lead to stronger
model performance.

# 📌 Objective Dataset: C-Eval

C-Eval evaluation code is located at: `./eval_ceval.py`.

For small models, to avoid issues with fixed response formatting, we directly judge the prediction probabilities of the
four tokens `A`, `B`, `C`, `D`, and choose the one with the highest probability as the answer, then calculate accuracy
against the standard answer. Note that minimind models were not trained on larger datasets or fine-tuned for question
answering, so results should be considered as reference only.

>For example, detailed results for minimind-small:

| category                                     | Correct/Total | Accuracy |
|----------------------------------------------|---------------|----------|
| probability_and_statistics_val               | 3/18          | 16.67%   |
| law_val                                      | 5/24          | 20.83%   |
| middle_school_biology_val                    | 4/21          | 19.05%   |
| high_school_chemistry_val                    | 7/19          | 36.84%   |
| high_school_physics_val                      | 5/19          | 26.32%   |
| legal_professional_val                       | 2/23          | 8.70%    |
| high_school_chinese_val                      | 4/19          | 21.05%   |
| high_school_history_val                      | 6/20          | 30.00%   |
| tax_accountant_val                           | 10/49         | 20.41%   |
| modern_chinese_history_val                   | 4/23          | 17.39%   |
| middle_school_physics_val                    | 4/19          | 21.05%   |
| middle_school_history_val                    | 4/22          | 18.18%   |
| basic_medicine_val                           | 1/19          | 5.26%    |
| operating_system_val                         | 3/19          | 15.79%   |
| logic_val                                    | 4/22          | 18.18%   |
| electrical_engineer_val                      | 7/37          | 18.92%   |
| civil_servant_val                            | 11/47         | 23.40%   |
| chinese_language_and_literature_val          | 5/23          | 21.74%   |
| college_programming_val                      | 10/37         | 27.03%   |
| accountant_val                               | 9/49          | 18.37%   |
| plant_protection_val                         | 7/22          | 31.82%   |
| middle_school_chemistry_val                  | 4/20          | 20.00%   |
| metrology_engineer_val                       | 3/24          | 12.50%   |
| veterinary_medicine_val                      | 6/23          | 26.09%   |
| marxism_val                                  | 5/19          | 26.32%   |
| advanced_mathematics_val                     | 5/19          | 26.32%   |
| high_school_mathematics_val                  | 4/18          | 22.22%   |
| business_administration_val                  | 8/33          | 24.24%   |
| mao_zedong_thought_val                       | 8/24          | 33.33%   |
| ideological_and_moral_cultivation_val        | 5/19          | 26.32%   |
| college_economics_val                        | 17/55         | 30.91%   |
| professional_tour_guide_val                  | 10/29         | 34.48%   |
| environmental_impact_assessment_engineer_val | 7/31          | 22.58%   |
| computer_architecture_val                    | 6/21          | 28.57%   |
| urban_and_rural_planner_val                  | 11/46         | 23.91%   |
| college_physics_val                          | 5/19          | 26.32%   |
| middle_school_mathematics_val                | 3/19          | 15.79%   |
| high_school_politics_val                     | 4/19          | 21.05%   |
| physician_val                                | 13/49         | 26.53%   |
| college_chemistry_val                        | 3/24          | 12.50%   |
| high_school_biology_val                      | 5/19          | 26.32%   |
| high_school_geography_val                    | 4/19          | 21.05%   |
| middle_school_politics_val                   | 6/21          | 28.57%   |
| clinical_medicine_val                        | 6/22          | 27.27%   |
| computer_network_val                         | 2/19          | 10.53%   |
| sports_science_val                           | 2/19          | 10.53%   |
| art_studies_val                              | 14/33         | 42.42%   |
| teacher_qualification_val                    | 12/44         | 27.27%   |
| discrete_mathematics_val                     | 6/16          | 37.50%   |
| education_science_val                        | 7/29          | 24.14%   |
| fire_engineer_val                            | 9/31          | 29.03%   |
| middle_school_geography_val                  | 1/12          | 8.33%    |

**Total number of questions**: 1346

**Total confirmed number**: 316

**Total accuracy rate**: 23.48%

---

#### Results summary：

| category         | correct  | question_count | accuracy  |
|:-----------------|:--------:|:--------------:|:---------:|
| minimind-small-T | 	   344	 |      1346      |  25.56%   |
| minimind-small	  |   	312   |     	1346      | 	  23.18% |
| minimind         | 	   351	 |      1346      |  26.08%   |
| minimind-moe     |   	316   |     	1346      | 	  23.48% |

### Model Performance Insights from GPT-4o

```text
### Areas Where the Model Excels:
1. **High School Chemistry**: With an accuracy of 42.11%, this is the strongest area for the model, suggesting a solid grasp of chemistry-related knowledge.
2. **Discrete Mathematics**: Achieving an accuracy of 37.50%, the model performs well in mathematics-related fields.
3. **Education Science**: The model shows good performance in education-related topics with a 37.93% accuracy.
4. **Basic Medicine**: The accuracy of 36.84% indicates strong performance in foundational medical knowledge.
5. **Operating Systems**: With a 36.84% accuracy, the model demonstrates reliable performance in computer operating systems.

### Areas Where the Model Struggles:
1. **Legal Topics**: The model performs poorly in legal-related areas such as Legal Professional (8.70%) and Tax Accountant (20.41%).
2. **Physics**: Both high school (26.32%) and college-level (21.05%) physics topics are challenging for the model.
3. **High School Politics and Geography**: The model shows low accuracy in these areas, with High School Politics at 15.79% and High School Geography at 21.05%.
4. **Computer Networking and Architecture**: The model struggles with Computer Networking (21.05%) and Computer Architecture (9.52%).
5. **Environmental Impact Assessment Engineering**: The accuracy is only 12.90%, indicating weak performance in environmental science.

### Summary:
- **Strengths**: Chemistry, Mathematics (especially Discrete Mathematics), Education Science, Basic Medicine, and Operating Systems.
- **Weaknesses**: Legal Topics, Physics, Politics, Geography, Computer Networking and Architecture, and Environmental Science.

This suggests that the model performs well in logical reasoning, foundational sciences, and some engineering disciplines but is weaker in humanities, social sciences, and certain specialized fields (such as law and taxation). To improve the model's performance, additional training in humanities, physics, law, and environmental science may be beneficial.
```


# 📌 Others

### Inference and Export

* [./export_model.py](./export_model.py) can export the model to the transformers format and push it to Hugging Face.

* MiniMind's Hugging Face collection address: [MiniMind](https://huggingface.co/collections/jingyaogong/minimind-66caf8d999f5c7fa64f399e5)

---

### API Inference

[./my_openai_api.py](./my_openai_api.py) provides a chat interface for the OpenAI API, making it easier to integrate your model with third-party UIs, such as fastgpt, OpenWebUI, etc.

* Download the model weight files from [Hugging Face](https://huggingface.co/collections/jingyaogong/minimind-66caf8d999f5c7fa64f399e5):
    ```
    minimind (root dir)
    ├─minimind
    |  ├── config.json
    |  ├── generation_config.json
    |  ├── LMConfig.py
    |  ├── model.py
    |  ├── pytorch_model.bin
    |  ├── special_tokens_map.json
    |  ├── tokenizer_config.json
    |  ├── tokenizer.json
    ```

* Start the chat server:
    ```bash
    python my_openai_api.py
    ```
* Test the service interface:
    ```bash
    python chat_openai_api.py
    ```
* API interface example, compatible with the OpenAI API format:
    ```bash
    curl http://ip:port/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{ 
        "model": "model-identifier",
        "messages": [ 
          { "role": "user", "content": "What is the highest mountain in the world?" }
        ], 
        "temperature": 0.7, 
        "max_tokens": -1,
        "stream": true
    }'
    ```

![images](./images/logger.png)

### Integrating MiniMind API in FastGPT

![images](./images/fastgpt.png)


---

# 📌 Acknowledge

If you find this project helpful, feel free to give it a Star 🎉✨.

The document is somewhat lengthy and may contain errors due to my limited proficiency. Please feel free to open issues for discussion or criticism.

Special thanks to the following open-source projects for their inspiration and datasets:

* [baby-llama2-chinese](https://github.com/DLLXW/baby-llama2-chinese)
* [ChatLM-mini-Chinese](https://github.com/charent/ChatLM-mini-Chinese)
* [Zero-Chatgpt](https://github.com/AI-Study-Han/Zero-Chatgpt/tree/main)


## ✨Top contributors
<a href="https://github.com/jingyaogong/minimind/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jingyaogong/minimind" />
</a>

# 📌 Statement

This project does not assume responsibility for data security, public opinion risks, or any risks and liabilities arising from model misguidance, misuse, dissemination, or improper use related to open-source models and code.




## License

This repository is licensed under the [Apache-2.0 License](LICENSE).