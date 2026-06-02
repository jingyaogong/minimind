# SearchShortQA Resume Experiment Plan

目标：把项目从“跑通过训练流程”推进到“有业务场景、有工程优化、有可复现实验、有简历指标”的状态。所有数字以服务器实际日志和评测结果为准，不提前编造。

## 目标简历叙事

项目名称建议：

> SearchShortQA：面向企业知识库的搜索增强短问答对齐系统

核心叙事：

> 基于公开 QA/RAG 数据构造 SearchShortQA 数据集，训练 64M baseline 与 100M/300M 级 Decoder-only 模型；将训练后端从 DDP 升级为 DeepSpeed ZeRO-2，并通过 SFT + GRPO 优化短答案准确率、引用合法率和 JSON 格式稳定性。

## 必跑实验

| ID | 实验名 | 目的 | 产出指标 |
| --- | --- | --- | --- |
| E0 | `searchlm64_sft_baseline` | 保留你已有 64M 作为 baseline | loss、F1、引用合法率、格式正确率、延迟 |
| E1 | `searchlm100_pretrain_ds` | 验证 DeepSpeed 预训练链路 | loss、吞吐、峰值显存、checkpoint 恢复 |
| E2 | `searchlm100_sft_search` | SearchShortQA SFT 冷启动 | dev F1、EM、引用合法率、格式正确率 |
| E3 | `searchlm100_grpo_search` | 对比 SFT 与 GRPO | reward、F1、引用合法率、格式正确率提升 |
| E4 | `searchlm300_pretrain_ds` | 简历主模型预训练 | DeepSpeed 7卡训练日志、峰值显存、有效 batch |
| E5 | `searchlm300_sft_grpo` | 简历主模型业务对齐 | 主结果表，写入简历 |
| E6 | `searchlm300_mla_ablation` | 注意力结构消融 | KV cache 理论压缩、吞吐/显存/效果变化 |
| E7 | `ddp_vs_zero2_100m` | 工程优化对比 | DDP vs ZeRO-2 峰值显存、可用 batch、吞吐 |
| E8 | `bm25_vs_embedding_negative` | 数据构造消融 | hard negative 命中率、SFT/GRPO dev 指标变化 |

如果时间不够，优先完成 E0-E5。E6/E7 是加分项。

## 服务器运行建议

目录约定：

```bash
mkdir -p logs reports outputs/search_shortqa
```

DeepSpeed 主链路：

```bash
cd trainer

deepspeed --num_gpus=7 train_pretrain_deepspeed.py \
  --model_profile SearchLM-300M \
  --data_path ../dataset/pretrain_t2t_mini.jsonl \
  --save_weight pretrain_searchlm \
  --dtype float16 \
  --batch_size 8 \
  --accumulation_steps 8

deepspeed --num_gpus=7 train_full_sft_deepspeed.py \
  --model_profile SearchLM-300M \
  --data_path ../dataset/search_shortqa/search_shortqa_sft.jsonl \
  --from_weight pretrain_searchlm \
  --save_weight full_sft_search \
  --dtype float16 \
  --batch_size 4 \
  --accumulation_steps 4

deepspeed --num_gpus=7 train_search_grpo_deepspeed.py \
  --model_profile SearchLM-300M \
  --data_path ../dataset/search_shortqa/search_shortqa_train.jsonl \
  --from_weight full_sft_search \
  --save_weight search_grpo \
  --num_generations 6 \
  --dtype float16 \
  --batch_size 1 \
  --accumulation_steps 4
```

断点续训：

```bash
cd trainer

deepspeed --num_gpus=7 train_search_grpo_deepspeed.py \
  --model_profile SearchLM-300M \
  --data_path ../dataset/search_shortqa/search_shortqa_train.jsonl \
  --from_weight full_sft_search \
  --save_weight search_grpo \
  --num_generations 6 \
  --dtype float16 \
  --batch_size 1 \
  --accumulation_steps 4 \
  --save_interval 25 \
  --from_resume 1
```

注意：

- 恢复时 `--epochs` 是总 epoch，不是剩余 epoch。
- 严格连续训练时不要改变 GPU 数量、batch、accumulation、数据文件和模型结构。
- `../checkpoints/` 是完整训练状态；`../out/` 是推理权重。在线 GPU 环境要保证这两个目录都在持久磁盘上。

3090 不支持 bf16 的完整高效路径时，优先用 `float16`。如果 GRPO 显存吃紧，先降：

- `--batch_size 1`
- `--num_generations 4`
- `--max_seq_len 768`
- `--max_gen_len 128`

## 日志与评测产物

训练日志解析：

```bash
python scripts/parse_train_log.py \
  --log logs/searchlm300_grpo_search.log \
  --experiment searchlm300_grpo_search \
  --output reports/searchlm300_grpo_search_train.json \
  --csv reports/searchlm300_grpo_search_train.csv \
  --drop_records
```

离线预测：

```bash
python scripts/predict_search_shortqa.py \
  --model_profile SearchLM-300M \
  --data dataset/search_shortqa/search_shortqa_dev.jsonl \
  --output outputs/search_shortqa/searchlm300_grpo_dev_pred.jsonl \
  --save_dir out \
  --weight search_grpo \
  --load_from model \
  --dtype float16
```

评测并落盘：

```bash
python scripts/eval_search_shortqa.py \
  --data dataset/search_shortqa/search_shortqa_dev.jsonl \
  --pred outputs/search_shortqa/searchlm300_grpo_dev_pred.jsonl \
  --by id \
  --experiment searchlm300_grpo_search \
  --json_output reports/searchlm300_grpo_search_eval.json \
  --bad_cases_output reports/searchlm300_grpo_search_bad_cases.jsonl
```

聚合报告：

```bash
python scripts/aggregate_experiment_report.py \
  --inputs reports/searchlm64_sft_baseline_train.json reports/searchlm64_sft_baseline_eval.json reports/searchlm300_grpo_search_train.json reports/searchlm300_grpo_search_eval.json \
  --baseline searchlm64_sft_baseline \
  --output_md reports/search_shortqa_experiment_report.md \
  --output_csv reports/search_shortqa_experiment_report.csv
```

业务 Demo：

```bash
streamlit run scripts/search_shortqa_demo.py -- \
  --data dataset/examples/search_shortqa_raw_demo.jsonl
```

数据构造消融：

```bash
python scripts/build_search_shortqa_dataset.py \
  --inputs dataset/raw/dureader_train.jsonl \
  --output_dir dataset/search_shortqa_bm25 \
  --negative_mining bm25

python scripts/build_search_shortqa_dataset.py \
  --inputs dataset/raw/dureader_train.jsonl \
  --output_dir dataset/search_shortqa_embedding \
  --negative_mining embedding \
  --embedding_model BAAI/bge-small-zh-v1.5
```

## 简历指标填充模板

跑完实验后，把报告里的数字填进去：

> 构建 SearchShortQA 搜索增强短问答数据集，统一问题、检索片段、短答案和引用来源字段；基于 DeepSpeed ZeRO-2 在 7×RTX3090 上训练 300M 级 Decoder-only 模型，有效 batch size 为 `xx`，峰值显存为 `xxGB`。

> 设计 GRPO 奖励函数，综合答案 F1、引用合法率、JSON 格式正确率、简洁性和重复惩罚；相比 SFT baseline，dev 集答案 F1 从 `xx.x%` 提升至 `xx.x%`，引用合法率从 `xx.x%` 提升至 `xx.x%`，格式正确率提升至 `xx.x%`。

> 对比 GQA/MQA/MLA 注意力结构，MLA 将 KV Cache 理论占用由 `xx` 降至 `xx` floats/token/layer，压缩 `xx.x%`，并在 SearchShortQA 长上下文评测中对比吞吐、显存和答案质量。

> 构建可交互 SearchShortQA Demo，支持检索片段编辑、模型 JSON 输出检查、引用合法性高亮和 reward 分解，用于 bad case 分析和业务侧效果展示。

## 64M 模型处理

不需要直接废弃 64M。它最适合做 baseline：

- 如果结构、tokenizer、hidden/layer/head 配置没变，只是把 DDP 改成 DeepSpeed，不需要重新预训练。
- 如果改成 SearchLM-100M/300M、MLA/MoE、新 FFN 或新 tokenizer，需要重新预训练。
- 简历写法是“64M baseline + 300M 主模型 + 结构/训练后端消融”，不要把 64M 写成主成果。

## 面试可追问点

准备回答这些问题：

- 为什么 7×3090 不从零训练 2B？
- DeepSpeed ZeRO-2 比 DDP 省在哪里，ZeRO-1/2/3 区别是什么？
- GRPO 为什么比 PPO 简化，reference model 和 KL 惩罚分别做什么？
- 引用合法率如何定义，如何避免模型凭空编 citation？
- MLA 的 KV cache 为什么能降，和 GQA/MQA 的区别是什么？
- 如果检索片段本身没有答案，奖励函数会怎样处理？
