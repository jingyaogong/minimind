# SearchShortQA：搜索增强短问答强化学习系统

## 业务场景

面向企业知识库、客服问答、运营内搜等场景，用户提出一个短问题后，系统基于检索片段生成简短答案，并必须给出引用来源。核心目标不是开放式聊天，而是降低幻觉、提升引用可追溯性和短答案可用性。

简历标题建议使用：

> SearchShortQA：面向企业知识库的搜索增强短问答对齐系统

模型表述建议使用：

> 基于 LLaMA/Qwen-style Decoder-only Transformer 架构，构建 100M/300M 两档小参数搜索增强短问答模型；2B+ 模型仅作为 Qwen/LLaMA 开源模型 LoRA/QLoRA 对照，不表述为从零训练。

## 数据来源

推荐用公开 QA/RAG 数据做底座，再转换成企业知识库样式数据：

| 数据源 | 用途 | 备注 |
| --- | --- | --- |
| DuReader / DuReader Robust / DuReader Retrieval | 中文主数据 | 真实搜索问题、证据文档、人工答案，最适合中文搜索增强问答 |
| Natural Questions | 英文开域 QA | Google 搜索问题 + Wikipedia 答案 |
| MS MARCO | 英文检索/QA | Bing 搜索问题、passage ranking、QA |
| KILT | provenance 评测 | 强调答案来源追溯，适合引用合法率指标 |

落地时保留 `manifest.json`，记录原始数据路径、构造参数、切分比例和随机种子。简历里只写“基于公开 QA/RAG 数据构造”，不要包装成真实企业私有数据。

## 数据格式

原始 JSONL 推荐字段：

```json
{
  "id": "case_001",
  "question": "退款多久到账？",
  "answer": "原路退回通常需要1-3个工作日",
  "gold_doc_ids": ["D2"],
  "contexts": [
    {"id": "D1", "title": "会员规则", "text": "..."},
    {"id": "D2", "title": "退款规则", "text": "退款审核通过后，原路退回通常需要1-3个工作日。"}
  ]
}
```

兼容字段别名：

- 问题：`question` / `query` / `prompt`
- 答案：`answer` / `answers`
- 检索片段：`contexts` / `documents` / `snippets` / `search_results`
- 引用标签：`gold_doc_ids` / `citations` / `supporting_doc_ids`

## 构造 SearchShortQA 数据

把本地下载好的 DuReader/NQ/MS MARCO/KILT 风格 JSONL 转成统一格式：

```bash
python scripts/build_search_shortqa_dataset.py \
  --inputs dataset/raw/dureader_train.jsonl dataset/raw/msmarco_train.jsonl \
  --output_dir dataset/search_shortqa \
  --train_size 30000 \
  --dev_size 1000 \
  --test_size 1000 \
  --max_contexts 6 \
  --num_negatives 3 \
  --negative_mining bm25
```

如果服务器已安装 `sentence_transformers`，可以切到 embedding hard negative：

```bash
python scripts/build_search_shortqa_dataset.py \
  --inputs dataset/raw/dureader_train.jsonl dataset/raw/msmarco_train.jsonl \
  --output_dir dataset/search_shortqa \
  --train_size 30000 \
  --dev_size 1000 \
  --test_size 1000 \
  --max_contexts 6 \
  --num_negatives 3 \
  --negative_mining embedding \
  --embedding_model BAAI/bge-small-zh-v1.5
```

构造逻辑：

- 正样本：原始证据文档或包含答案的文档作为 `gold_doc_ids`。
- 负样本：默认从全局语料用 BM25-ish 词面相似度采样 hard negatives，也可使用 embedding 相似度采样。
- 截断策略：优先保留 `gold_doc_ids` 对应证据片段，再补充 hard negatives，避免标签引用的文档被 `max_contexts` 截掉。
- 输出文件：
  - `dataset/search_shortqa/search_shortqa_train.jsonl`
  - `dataset/search_shortqa/search_shortqa_dev.jsonl`
  - `dataset/search_shortqa/search_shortqa_test.jsonl`
  - `dataset/search_shortqa/manifest.json`

构建后先校验字段和引用合法性：

```bash
python scripts/validate_search_shortqa_dataset.py \
  --data dataset/search_shortqa/search_shortqa_train.jsonl

python scripts/validate_search_shortqa_dataset.py \
  --data dataset/search_shortqa/search_shortqa_dev.jsonl
```

推荐规模：

| Split | 推荐规模 |
| --- | ---: |
| SFT train | 30k-100k |
| GRPO train | 3k-10k prompts |
| Dev/Test | 1k-3k each |
| Bad cases | 每轮 100-300 |

## SFT 冷启动

先把业务数据转成 SFT 对话格式：

```bash
python scripts/build_search_shortqa_sft.py \
  --input dataset/search_shortqa/search_shortqa_train.jsonl \
  --output dataset/search_shortqa/search_shortqa_sft.jsonl \
  --max_contexts 6
```

然后执行 SFT：

```bash
cd trainer
deepspeed --num_gpus=7 train_full_sft_deepspeed.py \
  --model_profile SearchLM-300M \
  --data_path ../dataset/search_shortqa/search_shortqa_sft.jsonl \
  --save_weight full_sft_search \
  --from_weight full_sft \
  --dtype float16
```

## GRPO 强化训练

使用业务奖励函数训练短答案、引用和格式：

```bash
cd trainer
deepspeed --num_gpus=7 train_search_grpo_deepspeed.py \
  --model_profile SearchLM-300M \
  --data_path ../dataset/search_shortqa/search_shortqa_train.jsonl \
  --from_weight full_sft_search \
  --save_weight search_grpo \
  --num_generations 6 \
  --batch_size 2 \
  --dtype float16
```

奖励函数位于 `trainer/search_shortqa_reward.py`，包含：

- `answer_f1`：预测答案与标准答案 token F1
- `exact_match`：短答案精确匹配
- `citation_score`：引用 precision / recall / valid rate
- `format_score`：是否输出合法 JSON
- `brevity_score`：短答案长度约束
- `repetition_penalty`：重复惩罚

## 离线评测

先生成预测文件：

```bash
python scripts/predict_search_shortqa.py \
  --model_profile SearchLM-300M \
  --data dataset/search_shortqa/search_shortqa_dev.jsonl \
  --output outputs/search_shortqa_dev_pred.jsonl \
  --save_dir out \
  --weight search_grpo \
  --load_from model \
  --dtype float16
```

预测文件每行至少包含 `prediction` / `response` / `output` 之一：

```json
{"id": "case_001", "prediction": "{\"answer\":\"原路退回通常需要1-3个工作日\", \"citations\":[\"D2\"]}"}
```

评测：

```bash
python scripts/eval_search_shortqa.py \
  --data dataset/search_shortqa/search_shortqa_dev.jsonl \
  --pred outputs/search_shortqa_dev_pred.jsonl \
  --by id \
  --experiment searchlm300_grpo_search \
  --json_output reports/searchlm300_grpo_search_eval.json \
  --bad_cases_output reports/searchlm300_grpo_search_bad_cases.jsonl
```

输出指标可直接写入实验报告：

- `exact_match`
- `answer_f1`
- `citation_precision`
- `citation_recall`
- `citation_valid`
- `format_score`
- `reward_total`

## 交互 Demo

启动 SearchShortQA 业务演示页：

```bash
streamlit run scripts/search_shortqa_demo.py -- \
  --data dataset/examples/search_shortqa_raw_demo.jsonl
```

Demo 支持编辑问题、检索片段和模型 JSON 输出，实时展示 reward、F1、引用合法率、格式正确率、奖励分解和引用检查表。它适合面试时展示“模型输出是否基于证据、引用是否合法、bad case 如何定位”。

## 模型规模建议

不要把主线写成“从零训练 2B”。7×RTX3090 更适合：

| 模型 | 用途 |
| --- | --- |
| SearchLM-100M | debug、结构消融、快速验证 |
| SearchLM-300M | 简历主模型，SFT + GRPO |
| SearchLM-500M | 可选大一档实验 |
| Qwen2.5-1.5B/3B LoRA | 大模型对照，不写成自研底座 |

查看推荐配置和估算参数：

```bash
python scripts/estimate_searchlm_scale.py
python scripts/estimate_searchlm_scale.py --json
```

profile 配置位于 `configs/searchlm_profiles.json`。训练入口支持：

```bash
--model_profile SearchLM-100M
--model_profile SearchLM-300M
--model_profile SearchLM-500M
```

DeepSpeed 服务器环境安装：

```bash
pip install deepspeed
```

默认使用 `trainer/ds_config_zero2.json`，采用 ZeRO-2 切分 optimizer state 和 gradient，适合 7×RTX3090 的 100M/300M/500M 训练链路。完整 DeepSpeed checkpoint 保存在 `../checkpoints/`，同时 rank0 会导出一份轻量 `.pth` 权重到 `../out/` 用于评测和推理。

## 断点续训

DeepSpeed 训练脚本支持 `--from_resume 1`。恢复的是 `../checkpoints/` 下的完整训练状态，包括模型、optimizer、scheduler、DeepSpeed engine 和已训练到的 `epoch/step`；`../out/` 下的 `.pth` 只是轻量推理权重，不能单独用于连续训练。

第一次训练建议降低 `save_interval`，减少在线 GPU 中断后的损失：

```bash
cd trainer

deepspeed --num_gpus=7 train_pretrain_deepspeed.py \
  --model_profile SearchLM-300M \
  --data_path ../dataset/pretrain_t2t_mini.jsonl \
  --save_weight pretrain_searchlm \
  --dtype float16 \
  --save_interval 200
```

中断后用同样的核心参数恢复：

```bash
cd trainer

deepspeed --num_gpus=7 train_pretrain_deepspeed.py \
  --model_profile SearchLM-300M \
  --data_path ../dataset/pretrain_t2t_mini.jsonl \
  --save_weight pretrain_searchlm \
  --dtype float16 \
  --save_interval 200 \
  --from_resume 1
```

恢复时需要保持以下参数一致：`--model_profile`、模型结构相关参数、`--save_weight`、`--data_path`、`--max_seq_len`、GPU 数量、`--batch_size` 和 `--accumulation_steps`。`--epochs` 表示总目标 epoch 数，不是剩余 epoch 数。

恢复成功时日志会出现：

```text
DeepSpeed checkpoint loaded: ...
Epoch [x/y]: 跳过前N个step，从step N+1开始
```

为了避免中断时覆盖最后一个可用 checkpoint，DeepSpeed checkpoint 会保存成 step 级目录，例如 `search_grpo_1024_mla_epoch0_step50`，并用 `search_grpo_1024_mla_latest` 指针记录最后完整保存的 checkpoint。

推荐完整训练顺序：

```bash
cd trainer

# 1. 通用预训练：使用通用文本，而不是 SearchShortQA
deepspeed --num_gpus=7 train_pretrain_deepspeed.py \
  --model_profile SearchLM-300M \
  --data_path ../dataset/pretrain_t2t_mini.jsonl \
  --save_weight pretrain_searchlm \
  --dtype float16

# 2. 通用 SFT：保留指令跟随能力
deepspeed --num_gpus=7 train_full_sft_deepspeed.py \
  --model_profile SearchLM-300M \
  --data_path ../dataset/sft_t2t_mini.jsonl \
  --from_weight pretrain_searchlm \
  --save_weight full_sft_searchlm \
  --dtype float16

# 3. SearchShortQA SFT：学习检索片段、短答案和引用格式
deepspeed --num_gpus=7 train_full_sft_deepspeed.py \
  --model_profile SearchLM-300M \
  --data_path ../dataset/search_shortqa/search_shortqa_sft.jsonl \
  --from_weight full_sft_searchlm \
  --save_weight full_sft_search \
  --dtype float16

# 4. SearchShortQA GRPO：用可验证奖励优化效果
deepspeed --num_gpus=7 train_search_grpo_deepspeed.py \
  --model_profile SearchLM-300M \
  --data_path ../dataset/search_shortqa/search_shortqa_train.jsonl \
  --from_weight full_sft_search \
  --save_weight search_grpo \
  --num_generations 6 \
  --dtype float16
```

2B+ 全参数 DDP 不推荐作为主线，因为 24GB 3090 上 actor/ref/optimizer/activation 会让 PPO/GRPO 显存压力过高；如果要体现 2B+，用 Qwen/LLaMA 开源权重做 LoRA/QLoRA 对照即可。

已有 64M 权重的处理建议：

- 如果只是把训练后端从 DDP 换成 DeepSpeed，模型结构、tokenizer、hidden/layer/head 配置都不变，则不需要重新预训练，可以继续做通用 SFT、SearchShortQA SFT 和 GRPO。
- 如果改成 SearchLM-300M/500M，或者切换到 MLA/MoE、改变 `hidden_size`、层数、head 数、词表/tokenizer，则旧 64M 不能作为同构 checkpoint 直接复用，需要重新预训练或只把 64M 当 baseline。
- 简历主线建议保留 64M 作为 baseline，同时至少补一版 SearchLM-100M 或 SearchLM-300M 的 DeepSpeed 预训练日志；这样可以写“64M baseline -> 300M 主模型”的规模消融和工程优化。

## 简历表达

可以包装为：

> 构建 SearchShortQA 搜索增强短问答数据集，基于 DuReader/NQ/MS MARCO 等公开 QA/RAG 数据统一转换为“问题-检索片段-短答案-引用来源”格式，并通过 BM25-ish hard negative 采样构造干扰片段；基于 DeepSpeed ZeRO-2 在 7×RTX3090 上训练 300M 级 LLaMA/Qwen-style Decoder-only 模型，完成 SFT 冷启动与 GRPO 强化对齐，在固定测试集上评估 EM、F1、引用合法率、JSON 格式正确率和推理吞吐。

如果同时使用 MLA：

> 在 SearchShortQA 场景下对比 GQA/MQA/MLA 注意力结构，MLA 将 KV Cache 理论占用由 768 降至 176 floats/token/layer，降低 77.1%，压缩约 4.36x，并评估其对检索长上下文问答的显存、吞吐和答案质量影响。
