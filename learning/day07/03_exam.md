# Day 7 Final Exam

## A. 闭卷概念题

1. 从零训练 MiniMind，按顺序需要哪些阶段？
2. Pretrain 和 SFT 的 labels 有什么区别？
3. 为什么 `ignore_index=-100` 是训练中的关键设计？
4. GRPO 为什么不需要 critic？
5. 蒸馏中的 CE 和 KL 分别学什么？
6. checkpoint 为什么要保存 optimizer？
7. 为什么训练 loss 不能替代生成评测？
8. Tool call 能力在 MiniMind 中主要从哪里来？

## B. 代码定位题

写出以下功能所在文件和函数：

1. MiniMind causal LM loss。
2. SFT assistant-only labels。
3. Pretrain 数据 padding 和 label mask。
4. Checkpoint 保存和恢复。
5. GRPO reward 计算。
6. Rollout engine 抽象。
7. 推理时 chat template 构造。
8. Tool call 自动测试入口。

## C. 实战题

重新从空日志开始，写出你会执行的完整命令：

1. 环境检查。
2. pretrain。
3. resume pretrain。
4. SFT。
5. eval pretrain。
6. eval full_sft。
7. eval toolcall。

每条命令后写一句：这一步验证什么。

## D. Debug 场景题

给出排查路径：

1. SFT loss 正常下降，但模型推理输出乱码。
2. 模型只会复述用户问题。
3. GRPO reward 上升，但人工看回答变差。
4. 训练恢复后 step 对不上。
5. Tool call JSON 经常非法。

## E. 最终报告题

写一份 2 页以内最终报告，必须包含：

- 训练环境
- 数据文件
- 模型配置
- 训练命令
- loss 记录
- pretrain vs full_sft 对比
- 失败样例
- 你对 GRPO 或蒸馏的理解
- 下一步最值得做的 3 个实验

## F. 评分标准

总分 100：

- 20：成功训练并推理 `pretrain` / `full_sft`
- 20：理解 dataset、template、label mask
- 15：理解模型结构和 causal LM loss
- 15：理解训练工程和 checkpoint
- 15：能做结构化评测和失败分析
- 15：理解至少一个后训练算法

通过线：75。

优秀线：90，且最终报告里有清晰失败分析和下一步实验设计。
