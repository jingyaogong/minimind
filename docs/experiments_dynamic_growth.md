# 动态神经元生长论文实验指南（MiniMind）

这份文档帮助你把“动态神经元生长（FFN 级别）”做成**可复现实验**，并形成论文所需的对照与记录。

也可直接使用一键流水线：

```bash
python scripts/run_paper_pipeline.py --run
```

配置文件在：`eval/pipeline_config.json`

---

## 1. 实验主线与对照设置

**主方法**
- 活动 + 梯度驱动生长（`grow_method=act_grad`）

**对照基线**
- Baseline：无生长（`neuron_growth=0`）
- Random：随机生长（`grow_method=random`）
- Grad-only：纯梯度生长（`grow_method=act_grad, grow_score_alpha=0, grow_score_beta=1`）
- Act-only：纯活动生长（`grow_method=act_grad, grow_score_alpha=1, grow_score_beta=0`）

---

## 2. 推荐实验命令（预训练）

> 下面命令以 `train_pretrain.py` 为例；SFT 使用 `train_full_sft.py` 同理。

**Baseline（无生长）**
```bash
python trainer/train_pretrain.py \
  --save_weight pretrain_baseline \
  --neuron_growth 0
```

**Random 生长**
```bash
python trainer/train_pretrain.py \
  --save_weight pretrain_random \
  --neuron_growth 1 \
  --init_active_ratio 0.8 \
  --grow_method random \
  --grow_interval 100 \
  --grow_ratio 0.02 \
  --max_active_ratio 0.99
```

**Grad-only 生长（只看梯度）**
```bash
python trainer/train_pretrain.py \
  --save_weight pretrain_grad \
  --neuron_growth 1 \
  --init_active_ratio 0.8 \
  --grow_method act_grad \
  --grow_interval 100 \
  --grow_ratio 0.02 \
  --max_active_ratio 0.99 \
  --grow_score_alpha 0.0 \
  --grow_score_beta 1.0
```

**Act-only 生长（只看活动）**
```bash
python trainer/train_pretrain.py \
  --save_weight pretrain_act \
  --neuron_growth 1 \
  --init_active_ratio 0.8 \
  --grow_method act_grad \
  --grow_interval 100 \
  --grow_ratio 0.02 \
  --max_active_ratio 0.99 \
  --grow_score_alpha 1.0 \
  --grow_score_beta 0.0
```

**Act+Grad（主方法）**
```bash
python trainer/train_pretrain.py \
  --save_weight pretrain_actgrad \
  --neuron_growth 1 \
  --init_active_ratio 0.8 \
  --grow_method act_grad \
  --grow_interval 100 \
  --grow_ratio 0.02 \
  --max_active_ratio 0.99 \
  --grow_score_alpha 1.0 \
  --grow_score_beta 1.0
```

---

## 3. 推荐实验命令（SFT）

将脚本换成 `trainer/train_full_sft.py`，其余参数相同。

```bash
python trainer/train_full_sft.py \
  --save_weight full_sft_actgrad \
  --neuron_growth 1 \
  --init_active_ratio 0.8 \
  --grow_method act_grad \
  --grow_interval 100 \
  --grow_ratio 0.02 \
  --max_active_ratio 0.99 \
  --grow_score_alpha 1.0 \
  --grow_score_beta 1.0
```

---

## 4. 关键参数解释（建议先固定）

- `init_active_ratio`：初始激活比例，推荐 0.8
- `grow_interval`：每隔多少次“优化器更新步”生长一次（非 batch）
- `grow_ratio`：每次激活比例（建议 0.01~0.05）
- `max_active_ratio`：最大激活比例（0.95~1.0）
- `grow_score_alpha/beta`：活动/梯度的权重
- `neuron_ema_beta`：活动 EMA 的系数（0.05~0.2）

---

## 5. 记录模板（写论文时必备）

建议你每次训练记录以下信息（最好表格化）：

| Run Name | Model Size | Data Version | Steps | LR | Batch | Seq Len | Growth Method | init_ratio | grow_interval | grow_ratio | max_ratio | PPL | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

**示例记录**
- Run Name: pretrain_actgrad_v1
- Model Size: 26M (hidden=512, layers=8)
- Data Version: pretrain_hq_v1
- Steps: 20k
- Growth Method: act_grad
- PPL: 18.3
- Notes: stable, active_ratio≈0.97

训练脚本会自动在 `save_dir` 下生成 `*_config.json`，包含：
- 超参配置
- 随机种子
- git commit
- 训练耗时、总 tokens（训练结束后写入）

---

## 6. 轻量评测（建议统一流程）

建议使用固定 prompt 列表进行对比评测：

```bash
python scripts/eval_fixed_prompts.py \
  --weight pretrain_actgrad \
  --prompts_file eval/prompts_minimal.jsonl \
  --out_dir eval_runs \
  --config eval/eval_config.json
```

建议先生成一个**固定验证集**（保证实验可复现）：

```bash
python scripts/make_val_split.py \
  --data_path dataset/pretrain_hq.jsonl \
  --out_path eval/val_pretrain.jsonl \
  --val_size 2000 \
  --seed 42
```

建议再跑一次 PPL/交叉熵评测，作为论文中的定量指标：

```bash
python scripts/eval_ppl.py \
  --weight pretrain_actgrad \
  --data_path eval/val_pretrain.jsonl \
  --max_seq_len 340 \
  --batch_size 8
```

结果可填入模板：`eval/results_template.csv`

可以用汇总脚本生成 CSV：

```bash
python scripts/aggregate_eval.py --inputs eval_runs_ppl --out eval/summary_ppl.csv
```

并用绘图脚本快速画图：

```bash
python scripts/plot_growth.py --csv eval/summary_ppl.csv --x weight --y ppl --out eval/plot_ppl.png
```

如果你想统计多 seed 的均值/方差：

```bash
python scripts/aggregate_eval.py \
  --inputs eval_runs_ppl \
  --out eval/summary_ppl.csv \
  --group_by method \
  --out_grouped eval/summary_ppl_grouped.csv
```

---

## 7. 最小可发表的实验组合

1. Baseline
2. Random
3. Grad-only
4. Act-only
5. Act+Grad（主方法）

只要主方法在相同预算下优于 1~4，即有论文价值。

建议每个设置至少跑 3 个随机种子，可用批量脚本：

```bash
python scripts/run_growth_sweep.py \
  --script trainer/train_pretrain.py \
  --prefix exp \
  --seeds 42,123,2026 \
  --base_args \"--epochs 1 --batch_size 32\" 
```
