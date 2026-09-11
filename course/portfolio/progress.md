# Portfolio Progress

记录每个阶段的完成情况。

| 阶段 | 状态 | 产物 | 备注 |
|---|---|---|---|
| 模型结构 | 进行中 | `core/model_parts.py`, `core/causal_lm.py`, `core/generation.py` | 第 13 课已加入 FFN/MoE 手写任务 |
| Pretrain | 进行中 | `core/train_loop.py`, `train_pretrain_impl.py` | 第 15 课已加入 checkpoint/resume 手写任务 |
| SFT | 进行中 | `core/datasets.py`, `train_sft_impl.py` | 第 17 课已固定 course_pretrain -> course_sft 阶段连接 |
| LoRA | 进行中 | `core/lora.py`, `train_lora_impl.py` | 第 18 课已加入 LoRA 核心模块手写任务 |
| DPO | 未开始 | `train_dpo_impl.py` |  |
| PPO/GRPO | 未开始 | `train_grpo_impl.py` |  |
