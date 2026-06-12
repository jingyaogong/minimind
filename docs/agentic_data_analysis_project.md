# Agentic DataAnalysis：面向运营数据分析的 Agentic RL 系统

## 项目定位

面向运营数据分析场景，构建一个 300M 级 Agentic LLM：模型需要先理解指标口径，再调用 `retriever`、`python_executor`、`calculator` 等工具完成多步分析，最终输出可验证的业务结论。

这条线不重新预训练，直接使用已有 `SearchLM-300M` 预训练权重：

```text
out/pretrain_searchlm_1024_mla.pth
```

后续主流程：

```text
Agent 轨迹 SFT -> Agentic GRPO/CISPO -> 离线评测 -> Demo 展示
```

## 数据构造

生成运营数据表、指标文档、任务、标准答案和专家工具轨迹：

```bash
python scripts/build_agentic_data_tasks.py \
  --output_dir dataset/agentic_data \
  --train_size 30000 \
  --dev_size 1000 \
  --test_size 1000
```

输出：

- `dataset/agentic_data/tables/orders.csv`
- `dataset/agentic_data/tables/marketing.csv`
- `dataset/agentic_data/tables/users.csv`
- `dataset/agentic_data/train.jsonl`
- `dataset/agentic_data/dev.jsonl`
- `dataset/agentic_data/test.jsonl`
- `dataset/agentic_data/manifest.json`

训练前校验：

```bash
python scripts/validate_agentic_data.py \
  --data dataset/agentic_data/train.jsonl
```

## SFT 冷启动

把专家轨迹转换成现有 `SFTDataset` 支持的 conversations 格式：

```bash
python scripts/build_agentic_sft.py \
  --input dataset/agentic_data/train.jsonl \
  --output dataset/agentic_data/sft.jsonl
```

使用已有 300M 预训练模型进行 Agent SFT：

```bash
cd trainer

deepspeed --num_gpus=7 train_full_sft_deepspeed.py \
  --model_profile SearchLM-300M \
  --from_weight pretrain_searchlm \
  --save_weight agent_sft \
  --data_path ../dataset/agentic_data/sft.jsonl \
  --dtype float16 \
  --save_interval 200
```

完整 14GB 通用 SFT 数据建议先训练一轮，再进入 Agent SFT。服务器可直接使用一键脚本：

```bash
bash scripts/run_general_sft_300m.sh
```

脚本默认使用 7 张 GPU、每卡 batch 1、梯度累积 8，有效 batch 为 56；如果检测到
`checkpoints/general_sft_14g_1024_mla_latest`，会自动加 `--from_resume 1`。日志写入 `logs/`。

常用覆盖参数：

```bash
USE_WANDB=1 SAVE_INTERVAL=200 bash scripts/run_general_sft_300m.sh
```

通用 SFT 完成后，将下面 Agent SFT 命令中的 `--from_weight` 改为 `general_sft_14g`。

## Agentic RL

使用 trajectory-level reward 优化多步工具调用行为：

```bash
cd trainer

deepspeed --num_gpus=7 train_agent_deepspeed.py \
  --model_profile SearchLM-300M \
  --from_weight agent_sft \
  --save_weight agent_grpo \
  --data_path ../dataset/agentic_data/train.jsonl \
  --num_generations 6 \
  --batch_size 1 \
  --accumulation_steps 4 \
  --dtype float16 \
  --save_interval 25
```

断点续训：

```bash
deepspeed --num_gpus=7 train_agent_deepspeed.py \
  --model_profile SearchLM-300M \
  --from_weight agent_sft \
  --save_weight agent_grpo \
  --data_path ../dataset/agentic_data/train.jsonl \
  --num_generations 6 \
  --batch_size 1 \
  --accumulation_steps 4 \
  --dtype float16 \
  --save_interval 25 \
  --from_resume 1
```

`../checkpoints/` 保存完整 DeepSpeed 训练状态；`../out/agent_grpo_1024_mla.pth` 是轻量推理权重。

## Reward 指标

奖励函数位于 `agentic/data_analysis_env.py`，按整条轨迹评分：

- `task_success`：最终答案是否命中标准答案中的文本和数值校验。
- `answer_score`：文本 contains 与数值 tolerance 的平均得分。
- `tool_selection_f1`：调用工具集合与期望工具集合的 F1。
- `schema_valid_rate`：工具名、JSON 和必填参数是否合法。
- `exec_success_rate`：工具能否成功执行。
- `invalid_action_rate`：非法 action 占比。
- `avg_turns`：平均交互轮数。
- `format_score`：`<tool_call>` 标签和 JSON 是否闭合合法。

## 离线评测

直接跑模型生成：

```bash
python scripts/eval_agentic_data.py \
  --data dataset/agentic_data/test.jsonl \
  --weight agent_grpo \
  --save_dir out \
  --output reports/agentic_grpo_pred.jsonl \
  --json_output reports/agentic_grpo_metrics.json \
  --bad_cases_output reports/agentic_grpo_bad_cases.jsonl
```

也可以评测已有预测文件：

```bash
python scripts/eval_agentic_data.py \
  --data dataset/agentic_data/test.jsonl \
  --pred reports/agentic_grpo_pred.jsonl \
  --json_output reports/agentic_grpo_metrics.json
```

建议固定对比：

```text
pretrain_searchlm -> agent_sft -> agent_grpo
```

核心简历指标使用固定 test split，不使用训练集结果。

## Demo

```bash
streamlit run scripts/agentic_data_demo.py -- \
  --data dataset/agentic_data/dev.jsonl
```

Demo 展示问题、数据表、指标文档、工具轨迹、最终答案和 reward 分解，适合面试时说明 Agent 行为如何被训练和评估。

## 简历表达

可以写成：

> 构建面向运营数据分析的 Agentic RL 系统，基于 300M Decoder-only 预训练模型完成工具轨迹 SFT 与 GRPO/CISPO 后训练；设计 retriever、python executor、calculator 交互环境和 trajectory-level reward，在固定测试集上评估任务成功率、工具选择准确率、无效 action 率、Python 执行成功率和平均交互轮数。
