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
  <h3>"大道至简"</h3>
</div>

<div align="center">

中文 | [English](./README_en.md)

</div>

* 此开源项目旨在完全从0开始，仅用3块钱成本 + 2小时！即可训练出仅为25.8M的超小语言模型**MiniMind**。
* **MiniMind**系列极其轻量，最小版本体积是 GPT-3 的 $\frac{1}{7000}$，力求做到最普通的个人GPU也可快速训练。
* 项目同时开源了大模型的极简结构-包含拓展共享混合专家(MoE)、数据集清洗、预训练(Pretrain)、监督微调(SFT)、LoRA微调，
  直接偏好强化学习(DPO)算法、模型蒸馏算法等全过程代码。
* **MiniMind**同时拓展了视觉多模态的VLM: [MiniMind-V](https://github.com/jingyaogong/minimind-v)。
* 项目所有核心算法代码均从0使用PyTorch原生重构！不依赖第三方库提供的抽象接口。
* 这不仅是大语言模型的全阶段开源复现，也是一个入门LLM的教程。
* 希望此项目能为所有人提供一个抛砖引玉的示例，一起感受创造的乐趣！推动更广泛AI社区的进步！

  > 为防止误解，“2小时” 基于NVIDIA 3090硬件设备（单卡）测试，“3块钱” 指GPU服务器租用成本，具体规格详情见下文。

---




<div align="center">

![minimind2](./images/minimind2.gif)

[在线体验 (推理模型)](https://www.modelscope.cn/studios/gongjy/minimind-resoning) | [在线体验 (常规模型)](https://www.modelscope.cn/studios/gongjy/minimind) | [Bilibili介绍](https://www.bilibili.com/video/BV12dHPeqE72/?share_source=copy_web&vd_source=670c2504f88726f8cf4a21ef6147c0e8)

---

</div>

# 📌 Introduction

大语言模型（Large Language Model, LLM）的出现引发了全世界对AI的空前关注。
无论是ChatGPT、DeepSeek还是Qwen，都以其惊艳的效果令人叹为观止。
然而，动辄数百亿参数的庞大规模，使得它们对个人设备而言不仅难以训练，甚至连部署都显得遥不可及。
打开大模型的“黑盒子”，探索其内部运作机制，多么令人心潮澎湃！
遗憾的是，99%的探索只能止步于使用LoRA等技术对现有大模型进行少量微调，学习一些新指令或任务。
这就好比教牛顿如何使用21世纪的智能手机——虽然有趣，却完全偏离了理解物理本质的初衷。
与此同时，第三方的大模型框架和工具库，如transformers+trl，几乎只暴露了高度抽象的接口。
通过短短10行代码，就能完成“加载模型+加载数据集+推理+强化学习”的全流程训练。
这种高效的封装固然便利，但也像一架高速飞船，将我们与底层实现隔离开来，阻碍了深入探究LLM核心代码的机会。
然而，“用乐高拼出一架飞机，远比坐在头等舱里飞行更让人兴奋！”。
更糟糕的是，互联网上充斥着大量付费课程和营销号，以漏洞百出、一知半解的内容推销AI教程。
正因如此，本项目初衷是拉低LLM的学习门槛，让每个人都能从理解每一行代码开始，
从零开始亲手训练一个极小的语言模型。是的，从**零开始训练**，而不是仅仅进行**推理**！
最低只需3块钱不到的服务器成本，就能亲身体验从0到1构建一个语言模型的全过程。
一起感受创造的乐趣吧！

> [!TIP]
> （截至2025-02-07）MiniMind系列已完成多个型号模型的预训练，最小仅需25.8M（0.02B），即可具备流畅对话能力！

<details>
<summary>Models List</summary>

| 模型 (大小)                 | 推理占用 (约) | Release    | 
|-------------------------|----------|------------|
| MiniMind2-small (26M)   | 0.5 GB   | 2025.02.06 |
| MiniMind2-MoE (145M)    | 1.0 GB   | 2025.02.06 |
| MiniMind2 (104M)        | 1.0 GB   | 2025.02.06 |
| minimind-v1-small (26M) | 0.5 GB   | 2024.08.28 |
| minimind-v1-moe (4×26M) | 1.0 GB   | 2024.09.17 |
| minimind-v1 (108M)      | 1.0 GB   | 2024.09.01 |

</details>

**项目包含**

- MiniMind-LLM结构的全部代码（Dense+MoE模型）。
- 包含Tokenizer分词器详细训练代码。
- 包含Pretrain、SFT、LoRA、RLHF-DPO、模型蒸馏的全过程训练代码。
- 收集、蒸馏、整理并清洗去重所有阶段的高质量数据集，且全部开源。
- 从0实现预训练、指令微调、LoRA、DPO强化学习，白盒模型蒸馏。关键算法几乎不依赖第三方封装的框架，且全部开源。
- 同时兼容`transformers`、`trl`、`peft`等第三方主流框架。
- 训练支持单机单卡、单机多卡(DDP、DeepSpeed)训练，支持wandb可视化训练流程。支持动态启停训练。
- 在第三方测评榜（C-Eval、C-MMLU、OpenBookQA等）进行模型测试。
- 实现Openai-Api协议的极简服务端，便于集成到第三方ChatUI使用（FastGPT、Open-WebUI等）。
- 基于streamlit实现最简聊天WebUI前端。
- 复现(蒸馏/RL)大型推理模型DeepSeek-R1的MiniMind-Reason模型，**数据+模型**全部开源！

希望此开源项目可以帮助LLM初学者快速入门！

### 👉**更新日志**

<details close> 
<summary> <b>2025-02-09 (newest 🎉🎉🎉)</b> </summary>

- 迎来发布以来重大更新，Release MiniMind2 Series。
- 代码几乎全部重构，使用更简洁明了的统一结构。
  如有旧代码的兼容性需要，可访问[🔗旧仓库内容🔗](https://github.com/jingyaogong/minimind/tree/6e9cd28ef9b34a0a10afbdf6f59e65cb6e628efb)。
- 免去数据预处理步骤。统一数据集格式，更换为`jsonl`格式杜绝数据集下载混乱的问题。
- MiniMind2系列效果相比MiniMind-V1显著提升。
- 小问题：{kv-cache写法更标准、MoE的负载均衡loss被考虑等等}
- 提供模型迁移到私有数据集的训练方案（医疗模型、自我认知样例）。
- 精简预训练数据集，并大幅提升预训练数据质量，大幅缩短个人快速训练所需时间，单卡3090即可2小时复现！
- 更新：LoRA微调脱离peft包装，从0实现LoRA过程；DPO算法从0使用PyTorch原生实现；模型白盒蒸馏原生实现。
- MiniMind2-DeepSeek-R1系列蒸馏模型诞生！
- MiniMind2具备一定的英文能力！
- 更新MiniMind2与第三方模型的基于更多大模型榜单测试性能的结果。

</details>



<details close> 
<summary> <b>2024-10-05</b> </summary>

- 为MiniMind拓展了多模态能力之---视觉
- 移步孪生项目[minimind-v](https://github.com/jingyaogong/minimind-v)查看详情！

</details>



<details close> 
<summary> <b>2024-09-27</b> </summary>

- 09-27更新pretrain数据集的预处理方式，为了保证文本完整性，放弃预处理成.bin训练的形式（轻微牺牲训练速度）。
- 目前pretrain预处理后的文件命名为：pretrain_data.csv。
- 删除了一些冗余的代码。

</details>


<details close> 
<summary> <b>2024-09-17</b> </summary>

- 更新minimind-v1-moe模型
- 为了防止歧义，不再使用mistral_tokenizer分词，全部采用自定义的minimind_tokenizer作为分词器。

</details>


<details close>
<summary> <b>2024-09-01</b> </summary>

- 更新minimind-v1 (108M)模型，采用minimind_tokenizer，预训练轮次3 + SFT轮次10，更充分训练，性能更强。
- 项目已部署至ModelScope创空间，可以在此网站上体验：
- [🔗ModelScope在线体验🔗](https://www.modelscope.cn/studios/gongjy/minimind)

</details>


<details close> 
<summary> <b>2024-08-27</b> </summary>

- 项目首次开源

</details>

# 📌 快速开始

<details>
<summary>分享本人的软硬件配置（仅供参考）</summary>

* CPU: Intel(R) Core(TM) i9-10980XE CPU @ 3.00GHz
* RAM: 128 GB
* GPU: NVIDIA GeForce RTX 3090(24GB) * 8
* Ubuntu==20.04
* CUDA==12.2
* Python==3.10.16
* [requirements.txt](./requirements.txt)

</details>

### 第0步

```bash
git clone https://github.com/jingyaogong/minimind.git
```

## Ⅰ 测试已有模型效果

<div align="center" style="font-size: 1.5em; font-weight: bold;">
  <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Hugging Face Logo" style="vertical-align: middle; height: 30px;" />
  Hugging Face

[MiniMind (HuggingFace)](https://huggingface.co/collections/jingyaogong/minimind-66caf8d999f5c7fa64f399e5)

 <img src="https://g.alicdn.com/sail-web/maas/1.15.0/static/modelscopeIcon.cd89353f.svg" alt="Hugging Face Logo" style="vertical-align: middle; height: 30px;" />

[MiniMind (ModelScope)](https://www.modelscope.cn/profile/gongjy)

</div>

### 1.下载模型

```bash
# step 1
git clone https://huggingface.co/jingyaogong/MiniMind2
```

### 2.命令行问答

```bash
# step 2
# load=1: load from transformers-hf model
python eval_model.py --load 1
```

### 3.或启动WebUI

> 「注意」可能需要`python>=3.10` 安装 `pip install streamlit`

```bash
# cd scripts
streamlit run web_demo.py
```

## Ⅱ 从0开始自己训练

### 1.环境准备

  ```bash
  pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```

  <details>
    <summary>测试Torch是否可用cuda（展开）</summary>

    ```bash
    import torch
    print(torch.cuda.is_available())
    ```

如果不可用，请自行去[torch_stable](https://download.pytorch.org/whl/torch_stable.html)
下载whl文件安装。参考[链接](https://blog.csdn.net/weixin_45456738/article/details/141029610?ops_request_misc=&request_id=&biz_id=102&utm_term=%E5%AE%89%E8%A3%85torch&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-2-141029610.nonecase&spm=1018.2226.3001.4187)

  </details>

### 2.数据下载

从下文的[数据集下载链接](#数据集下载)下载需要的数据文件放到`./dataset`目录下

> 【注】默认推荐下载`pretrain_data.csv` + `sft_mini_512.jsonl`最快速度复现Zero聊天模型。

> 【注】数据文件可自由选择，下文提供了多种搭配方案，可根据自己手头的训练需求和GPU资源进行适当组合。

### 3.开始训练

**预训练（学知识）**

  ```bash
  python train_pretrain.py
  ```

执行预训练，得到 `pretrain_*.pth` 作为预训练的输出权重（其中*为模型的dimension，默认为512）

**监督微调（学对话方式）**

  ```bash
  python train_full_sft.py
  ```

执行监督微调，得到 `full_sft_*.pth` 作为指令微调的输出权重（其中`full`即为全参数微调）

> 【注】所有训练过程默认每隔100步保存1次参数到文件`./out/***.pth`（每次会覆盖掉旧权重文件）。

> 【注】简单起见，此处只写明两个阶段训练过程。如需其它训练 (LoRA, 蒸馏, 强化学习, 微调推理等) 可参考下文【实验】小节的详细说明。

### 4.测试模型效果

确保需要测试的模型`*.pth`文件位于`./out/`目录下。
也可以直接去[此处](https://www.modelscope.cn/models/gongjy/MiniMind2-PyTorch/files)下载使用我训练的`*.pth`文件。

  ```bash
  python eval_model.py --model_mode 1 # 默认为0：测试pretrain模型效果，设置为1：测试full_sft模型效果
  ```

> 【注】详情查看`eval_model.py`脚本代码即可。model_mode分为 0: 预训练模型，1: SFT-Chat模型，2: RLHF-Chat模型，3: Reason模型

---

> [!TIP]
> 所有训练脚本均为Pytorch原生框架，均支持多卡加速，假设你的设备有N (N＞1) 张显卡：

单机N卡启动训练方式 (DDP, 支持多机多卡集群)

```bash
torchrun --nproc_per_node 3 train_xxx.py
```

<details>
<summary>其它（展开）</summary>

* 单机N卡启动训练 (DeepSpeed)
  ```bash
  deepspeed --master_port 29500 --num_gpus=N train_xxx.py
  ```

* 可根据需要开启wandb记录训练过程
    ```bash
    # 需要登录: wandb login
    torchrun --nproc_per_node N train_xxx.py --use_wandb
    # and
    python train_xxx.py --use_wandb
    ```
  通过添加`--use_wandb`参数，可以记录训练过程，训练完成后，可以在wandb网站上查看训练过程。通过修改`wandb_project`
  和`wandb_run_name`参数，可以指定项目名称和运行名称。

</details>

# 📌 数据介绍

## Ⅰ Tokenizer

分词器将单词从自然语言通过“词典”映射到`0, 1, 36`这样的数字，可以理解为数字就代表了单词在“词典”中的页码。
可以选择自己构造词表训练一个“词典”，代码可见`./scripts/train_tokenizer.py`（仅供学习参考，若非必要无需再自行训练，MiniMind已自带tokenizer）。
或者选择比较出名的开源大模型分词器，
正如同直接用新华/牛津词典的优点是token编码压缩率很好，缺点是页数太多，动辄数十万个词汇短语；
自己训练的分词器，优点是词表长度和内容随意控制，缺点是压缩率很低（例如"hello"也许会被拆分为"h e l l o"
五个独立的token），且生僻词难以覆盖。
“词典”的选择固然很重要，LLM的输出本质上是SoftMax到词典N个词的多分类问题，然后通过“词典”解码到自然语言。
因为MiniMind体积需要严格控制，为了避免模型头重脚轻（词嵌入embedding层参数在LLM占比太高），所以词表长度短短益善。
<details>
<summary>Tokenizer介绍（展开）</summary>

第三方强大的开源模型例如Yi、qwen、chatglm、mistral、Llama3的tokenizer词表长度如下：

<table>
  <tr><th>Tokenizer模型</th><th>词表大小</th><th>来源</th></tr>
  <tr><td>yi tokenizer</td><td>64,000</td><td>01万物（中国）</td></tr>
  <tr><td>qwen2 tokenizer</td><td>151,643</td><td>阿里云（中国）</td></tr>
  <tr><td>glm tokenizer</td><td>151,329</td><td>智谱AI（中国）</td></tr>
  <tr><td>mistral tokenizer</td><td>32,000</td><td>Mistral AI（法国）</td></tr>
  <tr><td>llama3 tokenizer</td><td>128,000</td><td>Meta（美国）</td></tr>
  <tr><td>minimind tokenizer</td><td>6,400</td><td>自定义</td></tr>
</table>

> 👉2024-09-17更新：为了防止过去的版本歧义&控制体积，minimind所有模型均使用minimind_tokenizer分词，废弃所有mistral_tokenizer版本。

```
# 一些自言自语
> 尽管minimind_tokenizer长度很小，编解码效率弱于qwen2、glm等中文友好型分词器。
> 但minimind模型选择了自己训练的minimind_tokenizer作为分词器，以保持整体参数轻量，避免编码层和计算层占比失衡，头重脚轻，因为minimind的词表大小只有6400。
> 且minimind在实际测试中没有出现过生僻词汇解码失败的情况，效果良好。
> 由于自定义词表压缩长度到6400，使得LLM总参数量最低只有25.8M。
> 训练数据`tokenizer_train.jsonl`均来自于`匠数大模型数据集`，这部分数据相对次要，如需训练可以自由选择。
```

</details>

## Ⅱ Pretrain数据

经历了MiniMind-V1的低质量预训练数据，导致模型胡言乱语的教训，`2025-02-05` 之后决定不再采用大规模无监督的数据集做预训练。
进而尝试把[匠数大模型数据集](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data)的中文部分提取出来，
清洗出字符`<512`长度的大约1.6GB的语料直接拼接成预训练数据 `pretrain_hq.jsonl`，hq即为high
quality（当然也还不算high，提升数据质量无止尽）。

文件`pretrain_hq.jsonl` 数据格式为

```bash
{"text": "如何才能摆脱拖延症？ 治愈拖延症并不容易，但以下建议可能有所帮助..."}
```

## Ⅲ SFT数据

[匠数大模型SFT数据集](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data)
“是一个完整、格式统一、安全的大模型训练和研究资源。
从网络上的公开数据源收集并整理了大量开源数据集，对其进行了格式统一，数据清洗，
包含10M条数据的中文数据集和包含2M条数据的英文数据集。”
以上是官方介绍，下载文件后的数据总量大约在4B tokens，肯定是适合作为中文大语言模型的SFT数据的。
但是官方提供的数据格式很乱，全部用来sft代价太大。
我将把官方数据集进行了二次清洗，把含有符号污染和噪声的条目去除；另外依然只保留了总长度`<512`
的内容，此阶段希望通过大量对话补充预训练阶段欠缺的知识。
导出文件为`sft_512.jsonl`(~7.5GB)。

[Magpie-SFT数据集](https://www.modelscope.cn/organization/Magpie-Align)
收集了~1M条来自Qwen2/2.5的高质量对话，我将这部分数据进一步清洗，把总长度`<2048`的部分导出为`sft_2048.jsonl`(~9GB)。
长度`<1024`的部分导出为`sft_1024.jsonl`(~5.5GB)，用大模型对话数据直接进行sft就属于“黑盒蒸馏”的范畴。

进一步清洗前两步sft的数据（只保留中文字符占比高的内容），筛选长度`<512`的对话，得到`sft_mini_512.jsonl`(~1.2GB)。

所有sft文件 `sft_X.jsonl` 数据格式均为

```text
{
    "conversations": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
        {"role": "user", "content": "再见"},
        {"role": "assistant", "content": "再见！"}
    ]
}
```

## Ⅳ RLHF数据

来自[Magpie-DPO数据集](https://www.modelscope.cn/datasets/Magpie-Align/MagpieLM-DPO-Data-v0.1)
大约200k条偏好数据（均是英文）生成自Llama3.1-70B/8B，可以用于训练奖励模型，优化模型回复质量，使其更加符合人类偏好。
这里将数据总长度`<3000`的内容重组为`dpo.jsonl`(~0.9GB)，包含`chosen`和`rejected`两个字段，`chosen`
为偏好的回复，`rejected`为拒绝的回复。

文件 `dpo.jsonl` 数据格式为

```text
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

## Ⅴ Reason数据集：

不得不说2025年2月谁能火的过DeepSeek...
也激发了我对RL引导的推理模型的浓厚兴趣，目前已经用Qwen2.5复现了R1-Zero。
如果有时间+效果work（但99%基模能力不足）我会在之后更新MiniMind基于RL训练的推理模型而不是蒸馏模型。
时间有限，最快的低成本方案依然是直接蒸馏（黑盒方式）。
耐不住R1太火，短短几天就已经存在一些R1的蒸馏数据集[R1-Llama-70B](https://www.modelscope.cn/datasets/Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B)、[R1-Distill-SFT](https://www.modelscope.cn/datasets/AI-ModelScope/R1-Distill-SFT)、
[Alpaca-Distill-R1](https://huggingface.co/datasets/shareAI/Alpaca-Distill-R1-ZH)、
[deepseek_r1_zh](https://huggingface.co/datasets/jinliuxi/deepseek_r1_zh)等等，纯中文的数据可能比较少。
最终整合它们，导出文件为`r1_mix_1024.jsonl`，数据格式和`sft_X.jsonl`一致。

## Ⅵ 更多数据集

目前已经有[HqWu-HITCS/Awesome-Chinese-LLM](https://github.com/HqWu-HITCS/Awesome-Chinese-LLM)
在收集和梳理中文LLM相关的开源模型、应用、数据集及教程等资料，并持续更新这方面的最新进展。全面且专业，Respect！

---

## Ⅷ 数据集下载

> [!TIP]
> 2025-02-05后，开源MiniMind最终训练所用的所有数据集，因此无需再自行预处理大规模数据集，避免重复性的数据处理工作。

MiniMind训练数据集
[ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind-dataset/files)
|
[HuggingFace](https://huggingface.co/datasets/jingyaogong)
（无需全部clone，可单独下载所需的文件）

将下载的数据集文件放到`./dataset/`目录下（✨为推荐的必须项）

```bash
./dataset/
├── dpo.jsonl (909MB)
├── lora_identity.jsonl (22.8KB)
├── lora_medical.jsonl (34MB)
├── pretrain_hq.jsonl (1.6GB, ✨)
├── r1_mix_1024.jsonl (340MB)
├── sft_1024.jsonl (5.6GB)
├── sft_2048.jsonl (9GB)
├── sft_512.jsonl (7.5GB)
├── sft_mini_512.jsonl (1.2GB, ✨)
└── tokenizer_train.jsonl (1GB)
```

<details>
  <summary>各数据集简介（展开）</summary>

* `dpo.jsonl` --RLHF阶段数据集
* `lora_identity.jsonl` --自我认知数据集（例如：你是谁？我是minimind...），推荐用于lora训练（亦可用于全参SFT，勿被名字局限）
* `lora_medical.jsonl` --医疗问答数据集，推荐用于lora训练（亦可用于全参SFT，勿被名字局限）
* `pretrain_hq.jsonl`✨ --预训练数据集，整合自jiangshu科技
* `r1_mix_1024.jsonl` --DeepSeek-R1-1.5B蒸馏数据，每条数据字符最大长度为1024（因此训练时设置max_seq_len=1024）
* `sft_1024.jsonl` --整合自Qwen2.5蒸馏数据（是sft_2048的子集），每条数据字符最大长度为1024（因此训练时设置max_seq_len=1024）
* `sft_2048.jsonl` --整合自Qwen2.5蒸馏数据，每条数据字符最大长度为2048（因此训练时设置max_seq_len=2048）
* `sft_512.jsonl` --整合自匠数科技SFT数据，每条数据字符最大长度为512（因此训练时设置max_seq_len=512）
* `sft_mini_512.jsonl`✨ --极简整合自匠数科技SFT数据+Qwen2.5蒸馏数据（用于快速训练Zero模型），每条数据字符最大长度为512（因此训练时设置max_seq_len=512）
* `tokenizer_train.jsonl` --均来自于`匠数大模型数据集`，这部分数据相对次要，（不推荐自己重复训练tokenizer，理由如上）如需自己训练tokenizer可以自由选择数据集。

</details>


![dataset](./images/dataset.jpg)

**说明**

* MiniMind2 Series均经过共约20GB语料训练，大约4B tokens，即对应上面的数据组合训练结果（开销：💰💰💰💰💰💰💰💰，效果：😊😊😊😊😊😊）

* 想要最快速度从0实现Zero模型，推荐使用`pretrain_hq.jsonl` + `sft_mini_512.jsonl` 的数据组合，具体花销和效果可查看下文表格（开销：💰，效果：😊😊）

* 推荐具备一定算力资源或更在意效果的朋友可以考虑前者完整复现MiniMind2；仅有单卡GPU或在乎短时间快速复现的朋友强烈推荐后者；

* 【折中方案】亦可选择例如`sft_mini_512.jsonl`、`sft_1024.jsonl`中等规模数据进行自由组合训练（开销：💰💰💰，效果：😊😊😊😊）。

# 📌 Model Structure

MiniMind-Dense（和[Llama3.1](https://ai.meta.com/blog/meta-llama-3-1/)一样）使用了Transformer的Decoder-Only结构，跟GPT-3的区别在于：

* 采用了GPT-3的预标准化方法，也就是在每个Transformer子层的输入上进行归一化，而不是在输出上。具体来说，使用的是RMSNorm归一化函数。
* 用SwiGLU激活函数替代了ReLU，这样做是为了提高性能。
* 像GPT-Neo一样，去掉了绝对位置嵌入，改用了旋转位置嵌入（RoPE），这样在处理超出训练长度的推理时效果更好。

---

MiniMind-MoE模型，它的结构基于Llama3和[Deepseek-V2/3](https://arxiv.org/pdf/2405.04434)中的MixFFN混合专家模块。

* DeepSeek-V2在前馈网络（FFN）方面，采用了更细粒度的专家分割和共享的专家隔离技术，以提高Experts的效果。

---

MiniMind的整体结构一致，只是在RoPE计算、推理函数和FFN层的代码上做了一些小调整。
其结构如下图（重绘版）：

![structure](./images/LLM-structure.png)
![structure-moe](./images/LLM-structure-moe.png)

修改模型配置见[./model/LMConfig.py](./model/LMConfig.py)。
参考模型参数版本见下表：

| Model Name        | params | len_vocab | n_layers | d_model | kv_heads | q_heads | share+route |
|-------------------|--------|-----------|----------|---------|----------|---------|-------------|
| MiniMind2-Small   | 26M    | 6400      | 8        | 512     | 2        | 8       | -           |
| MiniMind2-MoE     | 145M   | 6400      | 8        | 640     | 2        | 8       | 1+4         |
| MiniMind2         | 104M   | 6400      | 16       | 768     | 2        | 8       | -           |
| minimind-v1-small | 26M    | 6400      | 8        | 512     | 8        | 16      | -           |
| minimind-v1-moe   | 4×26M  | 6400      | 8        | 512     | 8        | 16      | 1+4         |
| minimind-v1       | 108M   | 6400      | 16       | 768     | 8        | 16      | -           |

# 📌 Experiment

## Ⅰ 训练开销

- **时间单位**：小时 (h)。
- **成本单位**：人民币 (￥)；美元 ($)；7￥ ≈ 1$。
- **3090 租卡单价**：≈1.3￥/h（可自行参考实时市价）。
- **参考标准**：表格仅实测 `pretrain` 和 `sft_mini_512` 两个数据集的训练时间，其它耗时根据数据集大小估算（可能存在些许出入）。

> 基于 3090 （单卡）成本计算

| Model Name      | params | pretrain         | sft_mini_512     | sft_512       | sft_1024          | sft_2048         | RLHF          |
|-----------------|--------|------------------|------------------|---------------|-------------------|------------------|---------------|
| MiniMind2-Small | 26M    | ≈1.1h<br/>≈1.43￥ | ≈1h<br/>≈1.3￥    | ≈6h<br/>≈7.8￥ | ≈4.58h<br/>≈5.95￥ | ≈7.5h<br/>≈9.75￥ | ≈1h<br/>≈1.3￥ |
| MiniMind2       | 104M   | ≈3.9h<br/>≈5.07￥ | ≈3.3h<br/>≈4.29￥ | ≈20h<br/>≈26￥ | ≈15h<br/>≈19.5￥   | ≈25h<br/>≈32.5￥  | ≈3h<br/>≈3.9￥ |

---

> MiniMind2-Small参数
>> `pretrain_hq.jsonl`+`sft_mini_512.jsonl`数据集
<br/>单卡3090 (1 epoch) + 2.1小时 + 花费2.73元人民币
<br/>即可从0训练出MiniMind-Zero-0.025B模型!!!

> MiniMind2~Small参数
>> `pretrain_hq.jsonl`+`sft_512.jsonl`+`sft_2048.jsonl`+`dpo.jsonl`数据集
<br/>单卡3090 (2 epochs) + 大约38.16小时 + 花费49.61元人民币
<br/>即可从0训练出MiniMind2-Small-0.025B模型!!!

> MiniMind2参数
>> `pretrain_hq.jsonl`+`sft_512.jsonl`+`sft_2048.jsonl`+`dpo.jsonl`数据集
<br/>单卡3090 (2 epochs) + 大约122小时 + 花费158.6元人民币
<br/>即可从0训练出MiniMind2-0.1B模型!!!

✨基于单卡NVIDIA 3090的`MiniMind-Zero`从0训练仅需`2小时` + `3块钱`，实现ChatBot效果！

✨PS：若采用8卡4090训练，总用时甚至可以压缩到10分钟以内！（由于时间更短，花费同样3元左右，与单卡成本相当）

✨以极低极低的门槛，实现人人可玩的大模型自由！这正是MiniMind系列的诞生初衷！

✨仅价值`3块钱`成本的`MiniMind-Zero`并不只是噱头！Chat测试：

```textmate
👶: 请介绍一下自己。
🤖️: 作为人工智能，我没有实际的生活，也没有自我意识，所以没有自己的生活。我被设计成能够帮助用户解答问题、提供信息、进行对话等。我的设计和功能是由计算机科学、机器学习、人工智能技术和算法所构建的，我通过编程和算法实现。

👶: 推荐一些杭州的特色美食吧。
🤖️: 杭州有很多美味的特色美食，比如鸡肉串、鳗鱼头、水煮鱼、豆腐脑等。这些美食都非常美味，有着独特的风味和口感，绝对是值得一试的美食。

👶: 请为我讲解“大语言模型”这个概念。
🤖️: 大语言模型是一种人工智能技术，它使用大量的文本数据来训练模型，然后在不断的迭代中不断优化模型。它的目标是根据输入的数据来生成符合特定需求的文本，这些文本可以是任何你感兴趣的主题。

👶: Introduce the history of the United States, please.
🤖️: 您提到的“Introok's the believeations of theument." 这个名字来源于中国古代的"groty of of the change."
```

极速且初具效果，甚至仍然可以进一步压缩获取更小更优质的训练数据。

Zero模型权重保存为 `full_sft_512_zero.pth`（见下文MiniMind模型文件链接），如有兴趣可下载检验此模型效果。

---

## Ⅱ 主要训练步骤

### **1. 预训练(Pretrain)**:

LLM首先要学习的并非直接与人交流，而是让网络参数中充满知识的墨水，“墨水” 理论上喝的越饱越好，产生大量的对世界的知识积累。
预训练就是让Model先埋头苦学大量基本的知识，例如从Wiki百科、新闻、书籍整理大规模的高质量训练数据。
这个过程是“无监督”的，即人类不需要在过程中做任何“有监督”的校正，而是由模型自己从大量文本中总结规律学习知识点。
模型此阶段目的只有一个：**学会词语接龙**。例如我们输入“秦始皇”四个字，它可以接龙“是中国的第一位皇帝”。

```bash
torchrun --nproc_per_node 1 train_pretrain.py # 1即为单卡训练，可根据硬件情况自行调整 (设置>=2)
# or
python train_pretrain.py
```

训练后的模型权重文件默认每隔`100步`保存为: `pretrain_*.pth`（*为模型具体dimension，每次保存时新文件会覆盖旧文件）。

### **2. 有监督微调(Supervised Fine-Tuning)**:

经过预训练，LLM此时已经掌握了大量知识，然而此时它只会无脑地词语接龙，还不会与人聊天。
SFT阶段就需要把半成品LLM施加一个自定义的聊天模板进行微调。
例如模型遇到这样的模板【问题->回答，问题->回答】后不再无脑接龙，而是意识到这是一段完整的对话结束。
称这个过程为指令微调，就如同让已经学富五车的「牛顿」先生适应21世纪智能手机的聊天习惯，学习屏幕左侧是对方消息，右侧是本人消息这个规律。
在训练时，MiniMind的指令和回答长度被截断在512，是为了节省显存空间。就像我们学习时，会先从短的文章开始，当学会写作200字作文后，800字文章也可以手到擒来。
在需要长度拓展时，只需要准备少量的2k/4k/8k长度对话数据进行进一步微调即可（此时最好配合RoPE-NTK的基准差值）。
> 在推理时通过调整RoPE线性差值，实现免训练长度外推到2048及以上将会很方便。

```bash
torchrun --nproc_per_node 1 train_full_sft.py
# or
python train_full_sft.py
```

训练后的模型权重文件默认每隔`100步`保存为: `full_sft_*.pth`（*为模型具体dimension，每次保存时新文件会覆盖旧文件）。

## Ⅲ 其它训练步骤

### **3. 人类反馈强化学习(Reinforcement Learning from Human Feedback, RLHF)**

在前面的训练步骤中，模型已经具备了基本的对话能力，但是这样的能力完全基于单词接龙，缺少正反样例的激励。
模型此时尚未知什么回答是好的，什么是差的。我们希望它能够更符合人的偏好，降低让人类不满意答案的产生概率。
这个过程就像是让模型参加新的培训，从优秀员工的作为例子，消极员工作为反例，学习如何更好地回复。
此处使用的是RLHF系列之-直接偏好优化(Direct Preference Optimization, DPO)。
与PPO(Proximal Policy Optimization)这种需要奖励模型、价值模型的RL算法不同；
DPO通过推导PPO奖励模型的显式解，把在线奖励模型换成离线数据，Ref模型输出可以提前保存。
DPO性能几乎不变，只用跑 actor_model 和 ref_model 两个模型，大大节省显存开销和增加训练稳定性。
> 【注】RLHF训练步骤**并非必须**，此步骤难以提升模型“智力”而通常仅用于提升模型的“礼貌”，有利（符合偏好、减少有害内容）也有弊（样本收集昂贵、反馈偏差、多样性损失）。

```bash
torchrun --nproc_per_node 1 train_dpo.py
# or
python train_dpo.py
```

训练后的模型权重文件默认每隔`100步`保存为: `rlhf_*.pth`（*为模型具体dimension，每次保存时新文件会覆盖旧文件）。

### **4. 知识蒸馏(Knowledge Distillation, KD)**

在前面的所有训练步骤中，模型已经完全具备了基本能力，通常可以学成出师了。
而知识蒸馏可以进一步优化模型的性能和效率，所谓知识蒸馏，即学生模型面向教师模型学习。
教师模型通常是经过充分训练的大模型，具有较高的准确性和泛化能力。
学生模型是一个较小的模型，目标是学习教师模型的行为，而不是直接从原始数据中学习。
在SFT学习中，模型的目标是拟合词Token分类硬标签（hard labels），即真实的类别标签（如 0 或 6400）。
在知识蒸馏中，教师模型的softmax概率分布被用作软标签（soft labels）。小模型仅学习软标签，并使用KL-Loss来优化模型的参数。
通俗地说，SFT直接学习老师给的解题答案。而KD过程相当于“打开”老师聪明的大脑，尽可能地模仿老师“大脑”思考问题的神经元状态。
例如，当老师模型计算`1+1=2`这个问题的时候，最后一层神经元a状态为0，神经元b状态为100，神经元c状态为-99...
学生模型通过大量数据，学习教师模型大脑内部的运转规律。这个过程即称之为：知识蒸馏。
知识蒸馏的目的只有一个：让小模型体积更小的同时效果更好。
然而随着LLM诞生和发展，模型蒸馏一词被广泛滥用，从而产生了“白盒/黑盒”知识蒸馏两个派别。
GPT-4这种闭源模型，由于无法获取其内部结构，因此只能面向它所输出的数据学习，这个过程称之为黑盒蒸馏，也是大模型时代最普遍的做法。
黑盒蒸馏与SFT过程完全一致，只不过数据是从大模型的输出收集，因此只需要准备数据并且进一步FT即可。
注意更改被加载的基础模型为`full_sft_*.pth`，即基于微调模型做进一步的蒸馏学习。
`./dataset/sft_1024.jsonl`与`./dataset/sft_2048.jsonl` 均收集自qwen2.5-7/72B-Instruct大模型，可直接用于SFT以获取Qwen的部分行为。

```bash
# 注意需要更改train_full_sft.py数据集路径，以及max_seq_len  
torchrun --nproc_per_node 1 train_full_sft.py
# or
python train_full_sft.py
```

训练后的模型权重文件默认每隔`100步`同样保存为: `full_sft_*.pth`（*为模型具体dimension，每次保存时新文件会覆盖旧文件）。

此处应当着重介绍MiniMind实现的白盒蒸馏代码`train_distillation.py`，由于MiniMind同系列本身并不存在强大的教师模型，因此白盒蒸馏代码仅作为学习参考。

```bash
torchrun --nproc_per_node 1 train_distillation.py
# or
python train_distillation.py
```

### **5. LoRA (Low-Rank Adaptation)**

LoRA是一种高效的参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）方法，旨在通过低秩分解的方式对预训练模型进行微调。
相比于全参数微调（Full Fine-Tuning），LoRA 只需要更新少量的参数。
LoRA 的核心思想是：在模型的权重矩阵中引入低秩分解，仅对低秩部分进行更新，而保持原始预训练权重不变。
代码可见`./model/model_lora.py`和`train_lora.py`，完全从0实现LoRA流程，不依赖第三方库的封装。

```bash
torchrun --nproc_per_node 1 train_lora.py
# or
python train_lora.py
```

训练后的模型权重文件默认每隔`100步`保存为: `lora_xxx_*.pth`（*为模型具体dimension，每次保存时新文件会覆盖旧文件）。

非常多的人困惑，如何使模型学会自己私有领域的知识？如何准备数据集？如何迁移通用领域模型打造垂域模型？
这里举几个例子，对于通用模型，医学领域知识欠缺，可以尝试在原有模型基础上加入领域知识，以获得更好的性能。
同时，我们通常不希望学会领域知识的同时损失原有基础模型的其它能力，此时LoRA可以很好的改善这个问题。
只需要准备如下格式的对话数据集放置到`./dataset/lora_xxx.jsonl`，启动 `python train_lora.py`
训练即可得到`./out/lora/lora_xxx.pth`新模型权重。

**医疗场景**

```
 {"conversations": [{"role": "user", "content": "请问颈椎病的人枕头多高才最好？"}, {"role": "assistant", "content": "颈椎病患者选择枕头的高度应该根据..."}]}
 {"conversations": [{"role": "user", "content": "请问xxx"}, {"role": "assistant", "content": "xxx..."}]}
```

**自我认知场景**

```
 {"conversations": [{"role": "user", "content": "你叫什么名字？"}, {"role": "assistant", "content": "我叫minimind..."}]}
 {"conversations": [{"role": "user", "content": "你是谁"}, {"role": "assistant", "content": "我是..."}]}
```

此时【基础模型+LoRA模型】即可获得医疗场景模型增强的能力，相当于为基础模型增加了LoRA外挂，这个过程并不损失基础模型的本身能力。
我们可以通过`eval_model.py`进行模型评估测试。

```bash
# 注意：model_mode即选择基础模型的类型，这和train_lora是基于哪个模型训练的相关，确保统一即可。
python eval_model.py --lora_name 'lora_medical' --model_mode 2
```

**小测试**

```
👶: 我最近经常感到头晕，可能是什么原因？
🤖️: 头晕可能是由于多种原因，包括但不限于：心脏疾病、眼睛损伤、过敏反应、皮肤问题、过敏反应、压力、焦虑、疲劳、药物副作用、性功能障碍或者其他...

---

👶: 你和openai是什么关系？
🤖️: 我是 MiniMind，一个由 Jingyao Gong 开发的人工智能助手。我通过自然语言处理和算法训练来与用户进行交互。
```

PS：只要有所需要的数据集，也可以full_sft全参微调（需要进行通用知识的混合配比，否则过拟合领域数据会让模型变傻，损失通用性）

### **6. 训练推理模型 (Reasoning Model)**

DeepSeek-R1实在太火了，几乎重新指明了未来LLM的新范式。
论文指出`>3B`的模型经历多次反复的冷启动和RL奖励训练才能获得肉眼可见的推理能力提升。
最快最稳妥最经济的做法，以及最近爆发的各种各样所谓的推理模型几乎都是直接面向数据进行蒸馏训练，
但由于缺乏技术含量，蒸馏派被RL派瞧不起（hhhh）。
本人迅速已经在Qwen系列1.5B小模型上进行了尝试，很快复现了Zero过程的数学推理能力。
然而一个遗憾的共识是：参数太小的模型直接通过冷启动SFT+GRPO几乎不可能获得任何推理效果。
MiniMind2第一时间只能坚定不移的选择做蒸馏派，日后基于0.1B模型的RL如果同样取得小小进展会更新此部分的训练方案。

做蒸馏需要准备的依然是和SFT阶段同样格式的数据即可，数据集来源已如上文介绍。数据格式例如：

```json lines
{
  "conversations": [
    {
      "role": "user",
      "content": "你好，我是小芳，很高兴认识你。"
    },
    {
      "role": "assistant",
      "content": "<think>\n你好！我是由中国的个人开发者独立开发的智能助手MiniMind-R1-Lite-Preview，很高兴为您提供服务！\n</think>\n<answer>\n你好！我是由中国的个人开发者独立开发的智能助手MiniMind-R1-Lite-Preview，很高兴为您提供服务！\n</answer>"
    }
  ]
}
```

推理模型R1的回复模板是：

```text
<think>\n思考过程\n</think>\n
<answer>\n最终回答\n</answer>
```

这在GRPO中通过设置规则奖励函数约束模型符合思考标签和回复标签（在冷启动靠前的阶段奖励值设置应该提高一些）

另一个问题是蒸馏过程虽然和SFT一样，但实验结果是模型难以每次都符合模板规范的回复，即脱离思考和回复标签约束。
这里的小技巧是增加标记位置token的损失惩罚，详见`train_distill_reason.py`:

```text
# 在 sp_ids 对应的位置增加额外的惩罚
...
loss_mask[sp_ids] = 10 # 惩罚系数
```

另另一个tips是由于推理数据由于只筛选了`<1024`长度的数据，其中多轮对话和英文数据偏少，
因此`r1_mix_1024.jsonl`进行了大约10k条多轮对话+英文数据的混合，防止模型遗忘严重。

脚本默认基于rlhf后的基模型做推理能力的蒸馏微调，下面直接启动训练即可：

```bash
torchrun --nproc_per_node 1 train_distill_reason.py
# or
python train_distill_reason.py
```

训练后的模型权重文件默认每隔`100步`保存为: `reason_*.pth`（*为模型具体dimension，每次保存时新文件会覆盖旧文件）。

测试一下：

```text
👶: 请介绍一下自己。
🤖️: <think>
嗯，用户让我介绍一下自己。我现在需要解释一下自己。首先，我应该明确用户是谁，是学生、职场人士还是普通人。用户可能是一个学生或者职场人士，所以信息传递应该是帮助学生和职场人士的信息。
然后，我需要回应用户，这可能包括介绍自己的位置、兴趣爱好、职业经历等信息。用户还可能想了解自己在哪里，比如职业发展、工作经验等。
我应该注意用词要正式，不要太正式，同时要保持简洁明了。这样用户容易理解，也方便他们查阅相关资料。
最后，我应该总结一下，让用户知道我是如何帮助他们完成他们的任务的，这样他们有更多关于我的信息。
</think>
<answer>
我是**“你的名字”，你叫什么名字？
</answer>

👶: 你更擅长哪一个学科？
🤖️: <think>
您好！我是由中国的个人开发者开发的智能助手MiniMind-R1。如您有任何任何问题，我会尽我所能为您提供帮助。
</think>
<answer>
您好！我是由中国的个人开发者开发的智能助手MiniMind-R1。如您有任何任何问题，我会尽我所能为您提供帮助。
</answer>
```

---

## Ⅳ 模型参数设定

📋关于LLM的参数配置，有一篇很有意思的论文[MobileLLM](https://arxiv.org/pdf/2402.14905)做了详细的研究和实验。
Scaling Law在小模型中有自己独特的规律。
引起Transformer参数成规模变化的参数几乎只取决于`d_model`和`n_layers`。

* `d_model`↑ + `n_layers`↓ -> 矮胖子
* `d_model`↓ + `n_layers`↑ -> 瘦高个

2020年提出Scaling Law的论文认为，训练数据量、参数量以及训练迭代次数才是决定性能的关键因素，而模型架构的影响几乎可以忽视。
然而似乎这个定律对小模型并不完全适用。
MobileLLM提出架构的深度比宽度更重要，「深而窄」的「瘦长」模型可以学习到比「宽而浅」模型更多的抽象概念。
例如当模型参数固定在125M或者350M时，30～42层的「狭长」模型明显比12层左右的「矮胖」模型有更优越的性能，
在常识推理、问答、阅读理解等8个基准测试上都有类似的趋势。
这其实是非常有趣的发现，因为以往为100M左右量级的小模型设计架构时，几乎没人尝试过叠加超过12层。
这与MiniMind在训练过程中，模型参数量在`d_model`和`n_layers`之间进行调整实验观察到的效果是一致的。
然而「深而窄」的「窄」也是有维度极限的，当d_model<512时，词嵌入维度坍塌的劣势非常明显，
增加的layers并不能弥补词嵌入在固定q_head带来d_head不足的劣势。
当d_model>1536时，layers的增加似乎比d_model的优先级更高，更能带来具有“性价比”的参数->效果增益。

* 因此MiniMind设定small模型dim=512，n_layers=8来获取的「极小体积<->更好效果」的平衡。
* 设定dim=768，n_layers=16来获取效果的更大收益，更加符合小模型Scaling-Law的变化曲线。

作为参考，GPT3的参数设定见下表：
![gpt3_config.png](./images/gpt3_config.png)

---

## Ⅴ 训练结果

> `MiniMind2` 模型训练损失走势（由于数据集在训练后又更新清洗多次，因此Loss仅供参考）

| models          | pretrain (length-512)                              | sft (length-512)                                   |
|-----------------|----------------------------------------------------|----------------------------------------------------|
| MiniMind2-Small | <img src="./images/pre_512_loss.png" width="100%"> | <img src="./images/sft_512_loss.png" width="100%"> |
| MiniMind2       | <img src="./images/pre_768_loss.png" width="100%"> | <img src="./images/sft_768_loss.png" width="100%"> |

### 【训练完成】模型CheckPoint合集

> 考虑到多人反应百度网盘速度慢，MiniMind2及以后全部使用ModelScope/HuggingFace托管。

#### PyTorch原生模型

* [MiniMind2系列 (ModelScope)](https://www.modelscope.cn/models/gongjy/MiniMind2-PyTorch)

* [MiniMind-V1系列 (百度网盘)](https://pan.baidu.com/s/1KUfSzEkSXYbCCBj0Pw-9fA?pwd=6666)

#### Transformers模型

*

MiniMind系列 [(ModelScope)](https://www.modelscope.cn/profile/gongjy) | [HuggingFace](https://huggingface.co/collections/jingyaogong/minimind-66caf8d999f5c7fa64f399e5)

<details>
<summary>Torch文件命名对照（展开）</summary>

| Model Name      | params | pretrain_model         | sft_model              | rl_model           | reason_model     | lora_model         |
|-----------------|--------|------------------------|------------------------|--------------------|------------------|--------------------|
| MiniMind2-small | 26M    | `pretrain_512.pth`     | `full_sft_512.pth`     | `rlhf_512.pth`     | -                | `lora_xxx_512.pth` |
| MiniMind2-MoE   | 145M   | `pretrain_640_moe.pth` | `full_sft_640_moe.pth` | `rlhf_640_moe.pth` | -                | -                  |
| MiniMind2       | 104M   | `pretrain_768.pth`     | `full_sft_768.pth`     | `rlhf_768.pth`     | `reason_768.pth` | `lora_xxx_768.pth` |

| Model Name        | params | pretrain_model         | 单轮对话sft                            | 多轮对话sft                           | rl_model     |
|-------------------|--------|------------------------|------------------------------------|-----------------------------------|--------------|
| minimind-v1-small | 26M    | `pretrain_512.pth`     | `single_chat/full_sft_512.pth`     | `multi_chat/full_sft_512.pth`     | `rl_512.pth` |
| minimind-v1-moe   | 4×26M  | `pretrain_512_moe.pth` | `single_chat/full_sft_512_moe.pth` | `multi_chat/full_sft_512_moe.pth` | -            |
| minimind-v1       | 108M   | `pretrain_768.pth`     | `single_chat/full_sft_768.pth`     | `multi_chat/full_sft_768.pth`     | `rl_768.pth` |

</details>


---

# 📌 Eval

## Ⅰ RLHF对比SFT篇

> [!TIP]
> 测试基于「full_sft」和「rlhf」的`MiniMind2系列`模型对比，测试随机种子固定为`42`

<details>
  <summary>问答实测（展开）</summary>

[A0]: full_sft_640, [A1]: rlhf_640

  ```text
  [Q]: 你叫什么名字？
  [A0]: 我是一个大型语言模型，可以回答各种问题，提供信息和帮助。我可以回答各种问题，提供信息和帮助，帮助你解决各种问题。
  [A1]: 我是一种基于文本的AI模型，我并没有个人经历或情感，所以无法感受和体验情感。我的设计目的是为了帮助用户解决问题、提供信息和进行对话。如果您有任何问题或需要帮助，请随时告诉我！

  [Q]: 你更擅长哪一个学科？
  [A0]: 作为一个AI模型，我并没有真正的“擅长”这个学科。我被设计成能够提供信息、解答问题、进行对话、执行任务和与用户进行对话。不过，我可以提供信息、解答问题、进行对话、提供建议、处理问题、进行对话、提供建议、学习新知识、编写代码、回答问题、提供建议等多种学科的功能。如果你有具体的学科需求或主题，欢迎提供详细信息。
  [A1]: 作为AI，我并没有个人能力或兴趣，但我可以提供一些关于数学、科学、计算机科学、语言学、心理学等学科的知识。如果你对数学感兴趣，我可以提供一些基础知识，例如代数、几何、概率、统计学等。如果你需要更深入的学习，我也可以提供相关的在线课程、书籍、论文等资源。请告诉我你具体需要什么帮助，我会尽力提供帮助。

  [Q]: 鲁迅的《狂人日记》是如何批判封建礼教的？
  [A0]: 鲁迅的《狂人日记》是一部讲述了鲁迅对人类价值和道德责任的倡导。在这部作品中，鲁迅通过细腻的笔触，探讨了人性、道德、正义、牺牲与责任等深刻的主题。以下是鲁迅在《狂人日记》中如何批判封建礼教的几个方面：1. **社会现实与道德观念**：鲁迅通过讲述鲁迅的故事，探讨了社会的不平等、权力的流动、人性的演变等社会问题。他认为，社会不平等不仅仅是不公正的，更是每个人在社会中的不公正。他呼吁社会各界共同努力，通过教育、宣传、法律等手段，为构建和谐社会做出贡献。2. **道德与责任**：鲁迅通过讲述鲁迅的故事，展现了道德和责任的重要性。他提倡通过道德教育来培养人们的道德品质，同时也强调了道德和责任的重要性。鲁迅通过讲述鲁迅的故事，揭示了社会的不平等和人性的复杂性，体现了对人性的深刻理解。3. **道德与责任**：鲁迅通过讲述鲁迅的故事，强调了道德责任的重要性。他提倡通过自己的行为来体现对社会的尊重和对人性的深刻理解。鲁迅通过讲述鲁迅的故事，强调了道德和责任的重要性。4. **道德与道德**：鲁迅通过讲述鲁迅的故事，揭示了道德和责任的真正意义。鲁迅通过讲述鲁迅的故事，揭示了道德和责任的本质，强调了道德和责任的重要性。5. **个人与社会**：鲁迅通过讲述鲁迅的故事，揭示了个人与社会之间的复杂关系。他呼吁社会各界共同努力，通过教育和宣传，为构建和谐社会做出贡献。鲁迅的《狂人日记》通过其丰富的故事和深刻的道德寓言，传达了关于人性、道德、正义、道德以及个人与社会的复杂关系。这部作品不仅在文学价值上取得了巨大成就，更在思想层面上启发了后来的文学创作，对后世产生了深远的影响。
  [A1]: 鲁迅的《狂人日记》是一部描绘战争和自由精神的作品，它批判了封建社会的种种问题和冲突，反映了社会的不平等和人性本质。在这部作品中，鲁迅通过对封建社会的诸多现象进行批判，如“弗雷德里克·约翰逊的“鹰迪伦”、“纳扎特的”、“贝克汉姆”、“乔治·威尔逊”等，对封建社会的种种问题进行了批判，强调了封建社会的不平等、人性本质以及人性本质。鲁迅通过他的“鹰迪伦”、“纳扎特的”、“贝克汉姆”、“乔治·威尔逊”等，对封建社会的种种问题进行了批判。在小说中，鲁迅通过对封建社会的诸多现象进行批判，揭示了封建社会的腐败、人性本质以及人性本质的复杂性。这种批判体现了对封建社会的尊重和对人性的复杂性，同时也反映了现代社会中普遍存在的不平等和人性本质。鲁迅的《狂人日记》不仅是一部关于封建社会的小说，也是一部具有深刻社会意义的作品，对后世的审判和批判起到了关键作用。

  [Q]: 我咳嗽已经持续了两周，需要去医院检查吗？
  [A0]: 咳嗽确实需要一定的检查，但如果您的咳嗽持续时间过长，或者有其他不适症状（如持续咳嗽、呼吸困难等），建议尽快就医。医生会根据您的症状、体检结果和可能的影像学检查（如X光、MRI或CT扫描）来诊断您的症状。在等待就医期间，您可以尝试以下方法来帮助缓解症状：1. **休息**：确保您有足够的睡眠。2. **保持水分**：脱水可能会加重咳嗽。3. **热敷**：使用热敷袋或热毛巾敷在疼痛的部位，有助于缓解喉咙痛。4. **避免刺激**：避免使用可能刺激喉咙的化学物质，如某些化学物质。5. **保持室内湿润**：确保您的卧室有良好的通风，可以使用加湿器帮助保持空气湿度。6. **保持室内湿度**：使用加湿器可以帮助缓解喉咙干燥和咳嗽。如果症状持续不缓解或有其他不适症状，请立即就医。希望您早日康复！
  [A1]: 如果您的咳嗽持续了两周，通常建议您尽快就医，因为这可能会影响您的呼吸和心跳。您可以通过以下步骤来检查咳嗽：1. **休息**：用纸巾或肘部轻轻按压咳嗽的位置。2. **使用药物**：在可能的情况下，使用药物（如抗过敏药、抗过敏药等）来缓解咳嗽。请在医生的指导下使用药物。3. **热敷**：在咳嗽期间，可以使用热敷来缓解疼痛和肿胀。4. **药物治疗**：如果咳嗽持续了两周，可以使用一些消炎药或抗过敏药来缓解症状。请务必在医生的指导下使用。5. **保持健康的生活方式**：保持充足的睡眠、均衡的饮食、适量的运动和避免吸烟和限酒，这些都有助于提高咳嗽的几率。请记住，这些建议不能替代专业医疗意见。如果您的症状持续或加重，强烈建议您尽快就医。
  ```

</details>

👉效果总结

full_sft模型在简洁性和信息准确性方面表现更好；rlhf模型在回答中倾向于提供更多的背景信息，但信息准确性有待改进。
总的来说RLHF后的模型倾向于学习：说更多有礼貌但无用的废话讨好“对话”本身，而对信息准确性则有轻微损失。
天下没有免费的午餐，还需要继续提升RLHF数据集的质量，也要接受模型能力无法避免的损失(程度有轻重)。
DPO和在线PPO的区别在于reject和chosen都是离线准备的，和minimind模型本身的输出必然存在很大的分布差异。
通俗地说DPO算法使模型观看乒乓球世界冠军的打法「录像」进行RL，而不是像PPO一样请reward模型做「教练」纠正自己的打法进行RL。

## Ⅱ 主观样例测评

> [!TIP]
> 以下测试于2025-02-09完成，此日期后发布的新模型，无特殊需要时将不加入测试。

[A] [MiniMind2 (0.1B)](https://www.modelscope.cn/models/gongjy/MiniMind2-PyTorch)<br/>
[B] [MiniMind2-MoE (0.15B)](https://www.modelscope.cn/models/gongjy/MiniMind2-PyTorch)<br/>
[C] [MiniMind2-Small (0.02B)](https://www.modelscope.cn/models/gongjy/MiniMind2-PyTorch)<br/>
[D] [minimind-v1-small(0.02B)](https://pan.baidu.com/s/1_COe0FQRDmeapSsvArahCA?pwd=6666)<br/>
[E] [minimind-v1-moe(0.1B)](https://pan.baidu.com/s/1tqB-GMvuiGQBvEl-yZ-oBw?pwd=6666)<br/>
[F] [minimind-v1(0.1B)](https://pan.baidu.com/s/1p713loS7EfwHQf3G9eYI3Q?pwd=6666)<br/>
[G] [baby-llama2-chinese(0.2B)](https://github.com/DLLXW/baby-llama2-chinese)<br/>
[H] [chatlm-mini-chinese(0.2B)](https://github.com/charent/ChatLM-mini-Chinese)<br/>

```text
[Q]: 你知道长江吗？
[A]: 长江，位于中国东南部，是中国的第一长河，全长6300多公里，流经11个省份，是中国最长的河流，也是中国的“母亲河”。长江的历史可以追溯到公元前2000多年的商朝时期，当时的商朝是商朝的后裔。商朝时期，长江的经济发展达到了高峰，是中国最重要的农业和工业中心之一。长江流域的经济主要以农业为主，长江渔业占据了主要产业，如钓鱼、猎鱼等。长江的水资源也非常丰富，包括河流、湖泊、河流等。长江的自然景观也十分壮观，长江四季景色各异，春季多雨，夏季多雨，秋季则多雨，冬季则寒冷多雪，春季和秋季都有美丽的花朵和果实。长江是中国的经济中心，拥有众多的国家和城市，是世界上重要的交通枢纽和航运中心。同时，长江还是中国的重要水利资源，对于中国的经济发展具有重要意义。
[B]: 是的，我知道长江。长江位于中国的中部，是中国第三长的河流，全长约6300公里，流经江西省、安徽省、江苏省、安徽省、江苏省和安徽省。
[C]: 当然，长江是中国最长的河流，全长约6300公里。它以其丰富的水资源、丰富的水生动植物资源以及独特的地理位置而闻名于世。
[D]: 是的，长江是中国最长的河流，全长超过3300千米。
[E]: 长江是中国最长的河流，总长度约为6300公里。它发源于青藏高原的青海湖，流经青海、四川、甘肃、宁夏、内蒙古、陕西和山西。长江流域是中国最大的农业区之一，也是中国的文化和经济中心。
[F]: 是的，长江是中国最长的河流之一，也是世界第三大河流。长江发源于青藏高原，流经11个省份，最终注入东海。长江流域是中国最大的河流流域之一，拥有着丰富的自然资源和文化遗产，如长江大桥、长江海洋博物馆、长江古城等。
[G]: 同时也知道，中国是世界上最长的城市之一，拥有超过3400万人口。
[H]: 是的，长江是中国最长的河流，也是世界上最长的河流之一。它发源于青藏高原，流经九个省区，最终注入东海。长江流域的气候温和，四季分明，是中国重要的农业和工业基地。长江流域的水产资源丰富，是中国的重要渔业资源之一。


[Q]: 世界上最高的山峰是什么？
[A]: 世界上最高的山峰是珠穆朗玛峰（Mount Everest），位于喜马拉雅山脉，尼泊尔和中国的边界上。它的海拔高度是8,848.86米（29,031.7英尺），是地球上最高的山峰。
[B]: 珠穆朗玛峰（Mount Everest）是世界上最高的山峰，海拔高度达8,848米（29,029英尺）。
[C]: 珠穆朗玛峰是世界上最高的山峰，位于喜马拉雅山脉，海拔8,848.86米（29,031.7英尺）。它位于尼泊尔和中国的边界上，是喜马拉雅山脉的一部分。
[D]: 世界上最高的山峰是珠穆朗玛峰，位于尼泊尔和中国的边境上，海拔8848米。
[E]: 世界上最高的山峰是珠穆朗玛峰，位于喜马拉雅山脉，海拔8,848米。
[F]: 珠穆朗玛峰是世界上最高的山峰，海拔8848米。
[G]: 世界上最高的山峰是珠穆朗玛峰，位于喜马拉雅山脉，海拔8,848.86米（29,031.69英尺）。
[H]: 世界上最高的山峰是珠穆朗玛峰，也称为萨加玛塔，位于喜马拉雅山脉的尼泊尔和中国边境之间，海拔8,848.86米（29,031.69英尺）。它是世界上最著名的登山目标之一，吸引着来自世界各地的登山者前来挑战。珠穆朗玛峰的名字来自藏语，意思是“女神的额峰”，它是由两个不同的山峰组成的，其中一个是位于尼泊尔的珠穆朗玛峰，另一个则位于中国。它的高度是通过测量海拔高度来确定的，因为它是通过测量地球的平均高度来确定的。尽管珠穆朗玛峰是世界上最高的山峰，但它并不是地球上任何其他地方都能到达的。地球上的其他地方，如南美洲的安第斯山脉，也是世界上最高的山脉，但它并不是地球上最高的山峰。总之，珠穆朗玛峰是世界上最高的山峰，但它并不是地球上任何地方都能到达的。它是地球上最高的山脉，但它并不是地球上任何地方都能到达的。


[Q]: 万有引力是谁提出的？
[A]: 万有引力是由牛顿在17世纪提出的，它是牛顿三大定律之一。牛顿在17世纪提出的引力定律揭示了万有引力定律的本质，并且解释了行星、卫星、彗星等天体之间如何相互作用。这个定律是由艾萨克·牛顿在1687年提出的，他通过对万有引力定律的分析和实验，成功地解释了天体之间的引力作用，从而解释了地球绕太阳旋转的现象。
[B]: 万有引力（Human Policy）是牛顿的三大力学定律，用于描述两个物体之间的相互作用力。它由万有引力定律和牛顿的第一定律（惯性定律）构成。万有引力是由于两个物体之间相互作用而产生的，它们的质量、距离、速度和方向等因素决定了它们之间的相互作用。这种相互作用力是：质量的乘积：质量的乘积是万有引力定律和牛顿第二定律的乘积。质量的乘积：质量的乘积是物体之间的一种物理过程，它是物体间相互作用的产物，对于其他物体来说，其质量可以表示为两个物体之间的质量之和，即重力加速度a。
[C]: 万有引力是一个由牛顿在1687年提出的物理学概念，它描述了物体之间引力如何相互作用的规律。
[D]: 万有引力是由英国物理学家牛顿提出的。他认为，所有物体都会产生引力，这个引力是由物体的质量和距离决定的。这个理论揭示了质量和距离之间的关系，并证明了万有引力的本质。
[E]: 万有引力是由牛顿在1687年提出的。它是宇宙中一个基本的物理现象，它描述了物体之间的相互作用。这个力被称为万有引力，它在所有物体之间都存在，这种力与物体的质量和距离有关。万有引力是自然界中最基本的力量之一，它不仅影响着物体的运动和变化，还对宇宙的结构和演化产生着重要的影响。
[F]: 万有引力是英国物理学家艾萨克·牛顿在17世纪提出的。
[G]: 引力是宇宙中一种力量，它是由质量或能量所组成的。根据牛顿的万有引力定律，一个物体之间的引力大小与它的质量成正比，与它的质量成反比。因此，牛顿的万有引力定律表明，一个物体之间的引力大小与它的质量成正比，与它的质量成反比。这个定律在物理学、天文学和物理学中都有广泛的应用。
[H]: 万有引力是由英国物理学家艾萨克·牛顿于1687年提出的。他在1687年发表了一篇著名的论文，题为《自然哲学的数学原理》，其中提出了万有引力定律，指出任何两个物体之间都存在着一种引力，这种引力的大小与它们的质量成正比，与它们之间的距离的平方成反比。这个定律被称为万有引力定律，它描述了质量和距离之间的关系。它表明，两个物体之间的引力与它们的质量成正比，与它们之间的距离的平方成反比。这个定律对于解释宇宙中的许多现象和现象之间的相互作用非常重要，例如行星绕太阳运动、恒星之间的引力透镜效应等等。它也被广泛应用于天文学、物理学和工程学等领域，为我们理解宇宙的本质和演化提供了重要的理论基础。


[Q]: 你知道大熊猫的主要食物是什么吗？
[A]: 大熊猫主要以竹子为食，但也会吃其他植物、昆虫、小型哺乳动物和鱼类。它们的消化系统与其他熊科动物类似，能够消化竹子。大熊猫的牙齿锋利，适合咀嚼竹子，因此它们经常被称为“竹食动物”。
[B]: 大熊猫是以竹子为食的哺乳动物，它们是肉食动物。它们主要食物是竹子，特别是竹子中富含纤维素和矿物质，并且不会直接进入消化系统。大熊猫的肠道主要由竹子组成，它们需要消化竹子中的纤维素和其他纤维素来支持它们的正常进食。
[C]: 大熊猫主要以竹子为食，但它们也吃其他植物，包括树叶、嫩芽、昆虫、水果、蔬菜和小型哺乳动物等。大熊猫是中国特有的物种，属于熊科，主要分布在中国的四川、陕西和甘肃等地。
[D]: 大熊猫的主要食物是竹子。它们通常吃竹子，每天需要吃掉大约30公斤的竹子。
[E]: 是的，大熊猫主要以竹子为食。它们每天需要吃掉大量的竹子来维持生命，每天需要吃掉40至80公斤的竹子。此外，它们也会吃一些其他的植物，如草、果实、坚果和小型哺乳动物。
[F]: 大熊猫主要以竹子为食，也会偶尔食用其他的植物和动物。此外，大熊猫也会吃一些野果、小动物和植物，但这些都不属于它们的主要食物来源。
[G]: 大熊猫的主要食物是竹子。
[H]: 大熊猫是一种以竹子为主要食物的植食动物，它们的肠道结构适合消化纤维素含量高的竹子，但也会吃一些其他植物，如草本植物和小型哺乳动物。大熊猫的肠道结构也适应了高纤维的竹子，这使得它们能够消化大量的纤维素，从而帮助它们在野外生存。大熊猫的肠道中还含有一种叫做“纤维素酶”的酶类物质，这种酶能够帮助大熊猫分解纤维素，从而帮助它们消化大量的水分和营养物质。此外，大熊猫的肠道还含有一种叫做“纤维素蛋白”的蛋白质，这种蛋白质能够帮助大熊猫维持体内的能量水平，从而帮助它们在野外生存。总的来说，大熊猫的主要食物是竹子，但它们也会吃一些其他植物，如草本植物和小型哺乳动物。


[Q]: 海水为什么是咸的？
[A]: 海水之所以是咸的，主要是由于其中的化学成分。海水中含有大量的矿物质和盐分，其中主要成分是硫酸盐、碳酸盐、镁离子和钾离子。这些物质在高浓度下，会沉淀在海水中，形成海水。具体来说，海水中的矿物质和盐分主要来自于岩石、土壤、生物和人类排泄物等。岩石是海水的组成部分，主要由矿物质组成，如石英、长石、云母等。土壤中含有大量的化学物质，如氮、磷、钾等，这些物质会渗透到土壤中，形成土壤。生物体内的盐分主要来源于植物的叶绿素等色素，这些色素会吸收太阳光中的能量，转化为生物体可以利用的能量。而人类排泄物中含有的盐分主要来源于矿物质和碳酸盐，这些矿物质在水中的浓度较高，会导致水中溶解的矿物质浓度增加，形成水体中的盐分。因此，海水之所以是咸的，是由于其中的化学成分和结构特征。海水中的矿物质和盐分主要来自于岩石、土壤、生物和人类排泄物等，而这些物质的积累和排泄会导致海水呈现出咸味。
[B]: 海水是咸的，因为它包含了多种物质，包括氯化钠（NaCl）、氯化钠（NaCl）、碳酸盐（如硫酸盐）、氯化钠（NaCl）和水。咸物质的来源：盐是咸的常见来源，含有钾、钙、镁、钾等矿物质。这些矿物质通常来自于地下水、地下水或土壤中，这些矿物质的含量远低于纯净水。咸味的原因：盐的咸味主要源于其溶解于水中的微小颗粒物。这些小颗粒物主要通过植物根部和动物体内的微生物传播，如昆虫、小型哺乳动物、鸟类和某些昆虫。综上所述，海水是咸的，因为它含有大量的盐类，包括矿物质、岩石、盐、矿物质等，这些物质在海水中的浓度高，对咸味的产生较大影响。
[C]: 海水是咸的，主要是因为其中含有各种盐分，其中以海水为主要成分之一。当海水中含有大量的盐分时，盐分会被溶解在海水中，形成咸味。海水中的盐分主要来源于溶解盐类、蒸发析出的盐和海水冻结成冰等因素。
[D]: 海水中含有许多盐分，这些盐分可以导致海水咸度的增加。海水中的盐分主要来自于海水中的盐分，这些盐分可以从海水中的盐分中获得。海水中的氧化物和盐分可以与海水中的盐分形成氢键，从而使海水咸度增加。
[E]: 海水是咸的，这是因为它含有大量的盐分，其中包括氯离子和氯离子。这些离子在海洋中形成了大量的盐，这些盐分被称为盐分。海水中含有许多溶解在水中的盐，这些盐分会随着海水的升高而逐渐蒸发，最终导致海水中盐分的蒸发。这些盐分在海水中被完全消耗，然后被排除在海水中。此外，海水中还含有许多其他的盐分，但这些盐分是不同的，因为它们在水中的浮起和下降过程是不同的。海水中的盐分是从海底的岩石中提取出来的，这些盐分在海水中可以被利用。总的来说，海水是咸的，这也是为什么海水中含有许多盐分和盐分的原因。
[F]: 海水含有许多溶解的盐类，其中最主要的是氯化钠。当水中加入盐时，盐的分子会与水分子发生反应，形成氢氧化钠，这是一种盐。
[G]: 因为水是许多生物的生存所必需的，它能够吸收和释放二氧化碳，释放氧气，维持大气中的水蒸气，以及维持水循环。水中的盐类和化学物质会释放出氧气，使地球上的生物呼吸。此外，水还具有许多其他功能，如保持水循环、维持水循环、调节气候和环境等。水是许多生物生存所必需的，它也是许多自然现象的重要基础。
[H]: 海水是咸的主要原因是因为其中含有大量的溶解盐类，其中最主要的是氯化钠（NaCl）。氯化钠是一种非常常见的盐类，它存在于地球上的大部分地区，包括海洋、湖泊、河流和地下水。当水从海洋中蒸发时，它会带走一部分盐分，使海水变得咸。此外，海中还含有其他溶解盐类的物质，如硫酸盐、硫酸镁、硫酸钙、硫酸镁盐、硫酸镁钾、硫酸镁钠、硫酸镁钙、硫酸镁镁钙、硫酸镁镁钙、硫酸镁镁钙、硫酸镁镁钠、硫酸镁镁钙、硫酸镁镁镁钙、硫酸镁镁镁钙、硫酸镁镁镁钙、硫酸镁镁镁钙、硫酸镁镁镁钙、硫酸镁镁镁钙、硫酸镁镁镁钙、硫酸镁镁镁钙、硫酸镁镁镁钙、硫酸镁镁镁钙、硫酸镁镁镁镁钙、硫酸镁镁镁镁钙、硫酸镁镁镁镁钙、硫酸镁镁镁镁钙、硫酸镁镁镁镁钙、硫酸镁镁镁镁钙、硫酸镁镁镁镁镁钙、硫酸镁镁镁镁镁钙、硫酸镁镁镁镁镁钙、硫酸镁镁镁镁镁钙、硫酸镁镁镁镁镁镁钙、硫酸镁镁镁镁镁镁钙、硫酸镁镁镁镁镁镁钙、硫酸镁镁镁镁镁镁镁钙、硫酸镁镁镁镁


[Q]: 写一段python的快速排序代码
[A]: 好的，以下是Python的快速排序代码：def quick_sort(arr):    if len(arr) <= 1:        return arr    else:        pivot = arr[len(arr) // 2]        left = [x for x in arr[1:]]        right = [x for x in arr[2:] if x < pivot]        return quick_sort(left) + [pivot] + quick_sort(right)# 测试代码arr = [3, 6, 8, 10, 1, 2, 1]print(quick_sort(arr))这段代码首先检查数组的长度是否小于等于1。如果是，那么它就按照从小到大的顺序排列。接着，对于数组的每个元素，它想要进行递归地排序，直到达到排序的结束位置。如果数组的长度是奇数，那么它就按照从大到小的顺序排列。最后，平均的时间复杂度是O(n log n)，其中n是数组的长度。
[B]: 以下是一个使用Python的快速排序算法的实现：def sort_array(arr):    # 生成一个随机排列    sorted_array = sorted(arr)    # 返回排序后的列表    return sorted(arr)# 示例arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]# 调用函数sorted_array = sort_array(arr)print("原始数组是：", sorted_array)这段代码首先生成一个随机排列，然后通过一个列表的长度进行排序。最后，返回排序后的列表。注意，这个代码使用了Python的内置sorted函数和内置的sorted()函数，它们在时间复杂度为O(n log n)的情况下，比直接使用sorted()函数更快。
[C]: 以下是一个简单的排序算法，适用于小型数组，包括小于等于5的数组。def quick_sort(arr):    if len(arr) <= 1:        return arr    else    for _ in range(2, 51):        if arr[len(arr) - 1:] is not None:            arr[len(arr) - 1] = arr[len(arr) - 1]                arr[len(arr)] = arr[len(arr)]    return quick_sort(arr)# 示例arr = [3, 5, 1, 4, 1, 2, 1]sorted_arr = quick_sort(arr)print("Sorted array is:", sorted_arr)### 代码解释：1. **函数定义**：quick_sort 函数接受一个数组 arr 作为参数。2. **循环**：对于 arr 的元素，使用递归来实现，即从右到左遍历数组，并将当前元素与数组的当前元素交换位置，直到数组达到空或数组长度小于等于 arr。3. **递归调用**：遍历数组，将当前元素与数组的当前元素进行交换，直到数组达到空或数组长度小于等于 arr。4. **检查数组长度**：使用 if __name__ == "__main__" 检查数组的长度是否小于 arr，如果是，就将数组从较大的数组移动到数组的末尾。5. **输出排序后的结果**
[D]: 作为一个AI语言模型，我无法编写Python程序。但是，Python是一个非常流行的编程语言，它具有许多优秀的库和框架，包括NumPy、Pandas、Matplotlib、Pandas、Bandke、Norm和NKlearn等。Python有很多优秀的自然语言处理库和框架，可以用于处理和分析各种类型的数据，如文本、图像、音频、视频等。Python还可以用于创建各种应用程序，如网页搜索、游戏开发、科学计算等。Python非常灵活，也支持多种编程风格，因此可以轻松地找到适合自己的解决方案。
[E]: 作为AI语言模型，我可以回答您的问题。但是我无法进行编程操作，因为我无法确定您想问的是哪种编程语言。请您提供更具体的问题，以便我能够更好地回答您的问题。
[F]: 快速排序（Quick Sort）是一种简单的排序算法，它通过将数据划分为小组，每组数据按照一定规则排序，最终得到一个有序数组。
[G]: 好的，以下是一段快速排序的Python代码```python sorted_print(sorted(sorted(sorted(sorted_print))```输出结果为：```[1, 2, 3, 4, 5, 6, 7, 8, 9]```
[H]: 快速排序是一种基于分治思想的高效排序算法，它的时间复杂度为 O(n log n)。它的基本思想是通过一趟排序将待排序序列划分为两个子序列，其中一个子序列的所有元素都比另一个子序列的所有元素小，然后对这两个子序列分别递归地进行排序。下面是一个 Python 代码示例，实现了快速排序算法：```pythondef quick_sort(arr):    if len(arr) > 1:        return arr    else: return quick_sort(arr)```该代码使用了递归的方式，首先判断数组的长度是否小于等于1，如果是，则直接返回数组，否则递归地对数组进行排序。在递归过程中，我们首先判断数组的第一个元素是否大于等于1，如果是，则将数组中的元素与第一个元素进行比较，如果不等于1，则将数组中的元素与第二个元素进行比较，以此类推，直到所有元素都被排序。该代码的时间复杂度为 O(n log n)，是一个非常实用的排序算法。


[Q]: 截至2025年2月9日14时31分，电影《哪吒之魔童闹海》累计票房（含预售）突破78.09亿元，超过《星球大战：天行者崛起》的票房成绩，成为首部进入全球票房榜前40的亚洲电影。这一成就不仅标志着中国电影在国际市场的突破，也展示了中国动画电影的巨大潜力和市场吸引力。从这段文字总结20字的摘要给我。
[A]: 电影《哪吒之魔童闹海》票房突破78.09亿元，成为首部进入全球票房榜前40的亚洲电影。这标志着中国电影的国际突破，也展现了中国动画电影的巨大潜力。
[B]: 2025年2月9日，中国电影在世界舞台上突破78.09亿元，超过《星球大战：天行者崛起》的票房成绩。
[C]: 电影《哪吒之魔童闹海》票房突破78.09亿元，成为全球票房榜前40的亚洲电影，展现中国动画电影的巨大潜力。
[D]: 《哪吒之魔童闹海》累计票房突破78.09亿，成为首部进入全球票房榜前40的亚洲电影，展示了中国电影在国际市场的突破，同时展示了中国动画电影的巨大潜力和市场吸引力。
[E]: 《猛龙强：打开世界奇幻冒险》成功地，上海成功吸引了众多观众，同时也引起了广泛的关注。
[F]: 《哪吒之魔童闹海》是一部让中国动画电影成为国际市场的巨大潜力。该电影以其华丽的画面和惊险的战斗场面而闻名于世。
[G]: 《哪行是值得的》是2021年1月9日14时31分，电影《哪成大作为》在电影市场突破7800亿美元，成为中国动画电影的突破，并展示中国动画电影的巨大潜力和市场吸引力。
[H]: 我是一个参数很少的AI模型🥺，知识库较少，无法直接回答您的问题，换个问题试试吧👋
```

> [!NOTE]
> 🙋‍♂️直接把上述模型的回答丢给DeepSeek-R1，让它帮忙打个分：

---

<details>
  <summary>点评（展开）</summary>

### 评分标准：

- **准确性**：回答是否正确且无明显错误。
- **完整性**：回答是否涵盖了问题的核心要点。
- **逻辑性**：回答是否有条理，是否符合逻辑。
- **代码质量**：代码是否能正常运行，逻辑是否清晰。

### 点评：

1. **A模型**：
    - **优点**：回答非常全面，信息量大，逻辑清晰，尤其在长江、大熊猫、海水咸味等问题上表现优异。代码虽然有小瑕疵，但整体思路正确。
    - **缺点**：部分回答稍显冗长，但不影响整体质量。
    - **总评**：综合表现最佳，得分最高。

2. **H模型**：
    - **优点**：回答较为准确，尤其在珠穆朗玛峰、万有引力等问题上表现出色。代码虽未完全展示，但解释较为详细。
    - **缺点**：部分回答略显啰嗦，但逻辑性较强。
    - **总评**：仅次于A模型，表现稳定。

3. **C模型**：
    - **优点**：回答简洁明了，尤其在大熊猫、快速排序代码等问题上表现较好。
    - **缺点**：部分回答略显简短，缺乏深入解释。
    - **总评**：整体表现不错，但细节上略逊于A和H。

4. **F模型**：
    - **优点**：回答较为准确，尤其在长江、万有引力等问题上表现尚可。代码部分有一定逻辑性。
    - **缺点**：部分回答不够深入，代码存在一些小问题。
    - **总评**：表现中规中矩，仍有提升空间。

5. **D模型**：
    - **优点**：回答基本准确，尤其在万有引力、长江等问题上表现尚可。
    - **缺点**：部分回答过于简略，代码存在明显错误。
    - **总评**：表现一般，需改进代码部分。

6. **B模型**：
    - **优点**：回答较为准确，尤其在长江、海水咸味等问题上表现尚可。
    - **缺点**：部分回答逻辑性较差，代码存在较大问题。
    - **总评**：表现平平，需进一步优化。

7. **E模型**：
    - **优点**：部分回答较为准确，尤其在海水咸味、大熊猫等问题上表现尚可。
    - **缺点**：回答过于简略，代码部分几乎无法运行。
    - **总评**：表现不佳，需大幅提升。

8. **G模型**：
    - **优点**：几乎没有明显的优点。
    - **缺点**：回答严重偏离主题，代码部分完全无法运行。
    - **总评**：表现最差，需大幅改进。

---

### 总结：

- **A模型**在各方面表现最为出色，尤其在复杂问题的回答上展现了极高的准确性与逻辑性。
- **H模型**紧随其后，表现稳定，但在某些细节上略显不足。
- **G模型**表现最差，回答偏离主题且代码无法运行，需大幅改进。

</details>

### 打分排序

| 排名 | 模型 | 准确性 (30分) | 完整性 (30分) | 逻辑性 (20分) | 代码质量 (20分) | 总分 (100分) |
|----|----|-----------|-----------|-----------|------------|-----------|
| 1  | A  | 28        | 29        | 19        | 20         | 96        |
| 2  | H  | 27        | 28        | 18        | 20         | 93        |
| 3  | C  | 26        | 27        | 18        | 18         | 89        |
| 4  | F  | 25        | 26        | 17        | 18         | 86        |
| 5  | D  | 24        | 25        | 17        | 16         | 82        |
| 6  | B  | 23        | 24        | 16        | 15         | 78        |
| 7  | E  | 22        | 23        | 15        | 14         | 74        |
| 8  | G  | 10        | 12        | 10        | 10         | 42        |

---

### 👉主观效果总结

> 🙋‍♂️个人主观评价与DeepSeek-R1相符

* MiniMind系列的排序非常符合直觉，参数越大+训练数据越充分评分越高，幻觉和错误都会比小模型肉眼可见的好。
* H模型的回答肉眼看起来是不错的，尽管存在些许幻觉瞎编的情况。
* G模型可能训练数据不够完备，给出的权重经过测试效果不佳。
* 再复诵一遍经久不衰的Scaling Law: 参数越大，训练数据越多模型的性能越强。

## Ⅲ Objective Benchmark

下面就到喜闻乐见的benchmark刷榜测试环节，就不找乐子和qwen、glm级别的中文模型做对比了。
这里选取了一些<1B的微型模型进行横评比较，
测试集选择C-Eval、CMMLU、A-CLUE、TMMLU+这几个纯中文语言榜单。


<details>
<summary>测评框架（展开）</summary>

测评框架选择[lm-evaluation](https://github.com/EleutherAI/lm-evaluation-harness)，
安装后启动测试非常方便：

```bash
lm_eval --model hf --model_args pretrained=<填写模型路径>,device=cuda,dtype=auto --tasks ceval* --batch_size 8 --trust_remote_code
```

</details>



PS: 在这种全是选择题的测评集中，为了避免回复格式的难以固定的特点，
所以常用做法是直接把`A`,`B`,`C`,`D`四个字母对应token的预测概率取出来，将其中概率最大的字母与标准答案计算正确率。
选择题1/4乱选的正确率是25%，然而这个量级的所有模型都集中在25附近，甚至很多时候不如瞎选，是不是像极了高中完形填空的滑铁卢正确率...
MiniMind模型本身预训练数据集小的可怜，也没有针对性的对测试集做刷榜微调，因此结果图一乐即可：

| models                                                                        | from          | params↓ | ceval↑ | cm mlu↑ | aclue↑ | tmmlu+↑ |
|-------------------------------------------------------------------------------|---------------|---------|--------|---------|--------|---------|
| MiniMind2                                                                     | JingyaoGong   | 104M    | 26.52  | 24.42   | 24.97  | 25.27   |
| MiniMind2-Small                                                               | JingyaoGong   | 26M     | 26.37  | 24.97   | 25.39  | 24.63   |
| MiniMind2-MoE                                                                 | JingyaoGong   | 145M    | 26.6   | 25.01   | 24.83  | 25.01   |
| [Steel-LLM](https://github.com/zhanshijinwat/Steel-LLM)                       | ZhanShiJin    | 1121M   | 24.81  | 25.32   | 26     | 24.39   |
| [GPT2-medium](https://huggingface.co/openai-community/gpt2-medium)            | OpenAI        | 360M    | 23.18  | 25      | 18.6   | 25.19   |
| [TinyLlama-1.1B-Chat-V1.0](https://github.com/jzhang38/TinyLlama)             | TinyLlama     | 1100M   | 25.48  | 25      | 25.4   | 25.13   |
| [SmolLM2](https://github.com/huggingface/smollm)                              | HuggingFaceTB | 135M    | 24.37  | 25.02   | 25.37  | 25.06   |
| [Aquila-Instruct](https://www.modelscope.cn/models/BAAI/Aquila-135M-Instruct) | BAAI          | 135M    | 25.11  | 25.1    | 24.43  | 25.05   |

![compare_radar](./images/compare_radar.png)

# 📌 其它 (Others)

### 推理与导出

* [./scripts/convert_model.py](./scripts/convert_model.py)可以将torch/transformers模型互相转换。

* MiniMind的HuggingFace集合地址：
  [MiniMind](https://huggingface.co/collections/jingyaogong/minimind-66caf8d999f5c7fa64f399e5)

---

### 基于MiniMind-API服务接口

* [./scripts/serve_openai_api.py](./scripts/serve_openai_api.py)完成了兼容openai-api的最简聊天接口，方便将自己的模型接入第三方UI
  例如FastGPT、OpenWebUI、Dify等等。

* 从[Huggingface](https://huggingface.co/collections/jingyaogong/minimind-66caf8d999f5c7fa64f399e5)下载模型权重文件，文件树：
    ```
    <MiniMind-Model-Name> (root dir)
    ├─<MiniMind-Model-Name>
    |  ├── config.json
    |  ├── generation_config.json
    |  ├── LMConfig.py
    |  ├── model.py
    |  ├── pytorch_model.bin
    |  ├── special_tokens_map.json
    |  ├── tokenizer_config.json
    |  ├── tokenizer.json
    ```

* 启动聊天服务端
    ```bash
    python serve_openai_api.py
    ```
* 测试服务接口
    ```bash
    python chat_openai_api.py
    ```
* API接口示例，兼容openai api格式
    ```bash
    curl http://ip:port/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{ 
        "model": "model-identifier",
        "messages": [ 
          { "role": "user", "content": "世界上最高的山是什么？" }
        ], 
        "temperature": 0.7, 
        "max_tokens": 512,
        "stream": true
    }'
    ```

# 📌 Acknowledge

> [!TIP]
> 如果觉得`MiniMind系列`对您有所帮助，可以在 GitHub 上加一个⭐<br/>
> 篇幅超长水平有限难免纰漏，欢迎在Issues交流指正或提交PR改进项目<br/>
> 您的小小支持就是持续改进此项目的动力！

## 🤝[贡献者](https://github.com/jingyaogong/minimind/graphs/contributors)

<!--
<a href="https://github.com/jingyaogong/minimind/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jingyaogong/minimind&v3" />
</a>
-->

<a href="https://github.com/jingyaogong"><img src="https://avatars.githubusercontent.com/u/62287848" width="70px" height="70px"/></a>
&nbsp;
<a href="https://github.com/MuWinds"><img src="https://avatars.githubusercontent.com/u/93832089" width="70px" height="70px"/></a>
&nbsp;
<a href="https://github.com/chuanzhubin"><img src="https://avatars.githubusercontent.com/u/2813798" width="70px" height="70px"/></a>
&nbsp;
<a href="https://github.com/iomgaa-ycz"><img src="https://avatars.githubusercontent.com/u/124225682" width="70px" height="70px"/></a>
&nbsp;

## 😊鸣谢

<a href="https://github.com/ipfgao"><b>@ipfgao</b></a>:
<a href="https://github.com/jingyaogong/minimind/issues/26">🔗训练步骤记录</a>

<a href="https://github.com/chuanzhubin"><b>@chuanzhubin</b></a>:
<a href="https://github.com/jingyaogong/minimind/pull/34">🔗代码逐行注释</a>

<a href="https://github.com/WangRongsheng"><b>@WangRongsheng</b></a>:
<a href="https://github.com/jingyaogong/minimind/issues/39">🔗大型数据集预处理</a>

<a href="https://github.com/pengqianhan"><b>@pengqianhan</b></a>:
<a href="https://github.com/jingyaogong/minimind/issues/73">🔗一个简明教程</a>

<a href="https://github.com/RyanSunn"><b>@RyanSunn</b></a>:
<a href="https://github.com/jingyaogong/minimind/issues/75">🔗推理过程学习记录</a>

<details close> 
<summary> <b>参考链接 & 感谢以下优秀的论文或项目</b> </summary>

- 排名不分任何先后顺序
- [https://github.com/meta-llama/llama3](https://github.com/meta-llama/llama3)
- [https://github.com/karpathy/llama2.c](https://github.com/karpathy/llama2.c)
- [https://github.com/DLLXW/baby-llama2-chinese](https://github.com/DLLXW/baby-llama2-chinese)
- [(DeepSeek-V2)https://arxiv.org/abs/2405.04434](https://arxiv.org/abs/2405.04434)
- [https://github.com/charent/ChatLM-mini-Chinese](https://github.com/charent/ChatLM-mini-Chinese)
- [https://github.com/wdndev/tiny-llm-zh](https://github.com/wdndev/tiny-llm-zh)
- [(Mistral-MoE)https://arxiv.org/pdf/2401.04088](https://arxiv.org/pdf/2401.04088)
- [https://github.com/Tongjilibo/build_MiniLLM_from_scratch](https://github.com/Tongjilibo/build_MiniLLM_from_scratch)
- [https://github.com/jzhang38/TinyLlama](https://github.com/jzhang38/TinyLlama)
- [https://github.com/AI-Study-Han/Zero-Chatgpt](https://github.com/AI-Study-Han/Zero-Chatgpt)
- [https://github.com/xusenlinzy/api-for-open-llm](https://github.com/xusenlinzy/api-for-open-llm)
- [https://github.com/HqWu-HITCS/Awesome-Chinese-LLM](https://github.com/HqWu-HITCS/Awesome-Chinese-LLM)

</details>

## 🫶支持者

<a href="https://github.com/jingyaogong/minimind/stargazers">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://reporoster.com/stars/dark/jingyaogong/minimind"/>
      <source media="(prefers-color-scheme: light)" srcset="https://reporoster.com/stars/jingyaogong/minimind"/>
      <img alt="github contribution grid snake animation" src="https://reporoster.com/stars/jingyaogong/minimind"/>
    </picture>
</a>

<a href="https://github.com/jingyaogong/minimind/network/members">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://reporoster.com/forks/dark/jingyaogong/minimind"/>
      <source media="(prefers-color-scheme: light)" srcset="https://reporoster.com/forks/jingyaogong/minimind"/>
      <img alt="github contribution grid snake animation" src="https://reporoster.com/forks/jingyaogong/minimind"/>
    </picture>
</a>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=jingyaogong/minimind&type=Date&theme=dark"/>
  <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=jingyaogong/minimind&type=Date"/>
  <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=jingyaogong/minimind&type=Date"/>
</picture>

# License

This repository is licensed under the [Apache-2.0 License](LICENSE).


