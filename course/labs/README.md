# Course Labs

这个目录放课程配套的最小实验文件。

设计原则：

- 数据尽量小，优先验证流程，不追求模型效果。
- 脚本默认不下载数据、不启动长训练。
- 实验输出尽量打印关键 shape、tokens、labels、loss。

当前文件：

- `check_minimind_import.py`：确认核心依赖和 MiniMind 模型实例化。
- `check_project_readiness.py`：检查当前环境、数据、权重和可选依赖是否齐备。
- `compare_pretrain_sft.py`：对比 tiny pretrain 与 tiny SFT 的 input_ids / labels / loss 区域。
- `inspect_sft_dataset.py`：打印 tiny SFT 样本的 prompt、input_ids、labels 和 assistant 训练区域。
- `inspect_model_config_params.py`：实例化不同 MiniMindConfig，打印模型 shape、参数量和模块分解。
- `inspect_tokenizer.py`：观察 tokenizer 的特殊 token、encode/decode、chat template 输出。
- `run_minimind3_once.py`：使用下载好的 `minimind-3` transformers 模型跑一次真实推理。
- `trace_attention_shapes.py`：追踪 Attention 的 Q/K/V projection、head reshape、causal mask 和输出投影。
- `trace_block_shapes.py`：手动追踪 embedding、RMSNorm、残差、block 和 lm_head 的 shape 流。
- `trace_pretrain_step.py`：在 CPU 上跑一个 tiny pretrain step，打印 config、batch、logits、loss、梯度和参数更新。
- `trace_cli_inference.py`：用随机 tiny 模型追踪 CLI 推理中的 prompt、token、logits、generate。
- `trace_loss_shift.py`：手动复现 causal LM 的 logits/labels shift 和 `ignore_index=-100` loss。
- `trace_rope_kv_cache.py`：追踪 RoPE 的 cos/sin shape、KV cache 增长，以及 cache/no-cache 生成一致性。
- `trace_distillation_loss.py`：用 tiny student/teacher 追踪 CE loss、KL 蒸馏 loss 和 alpha 混合总 loss。
- `trace_dpo_dataset_reference.py`：追踪 DPO chosen/rejected 数据、x/y/mask、policy/ref logprob。
- `trace_dpo_loss.py`：从 token logprob 手算 sequence logprob、DPO logratio 和 DPO loss。
- `trace_ppo_batch_flow.py`：用 tiny synthetic rollout 追踪 PPO old/new logprob、GAE、clip policy loss 和 value loss。
- `trace_grpo_cispo_loss.py`：用 synthetic group rewards 对比 GRPO 与 CISPO 的 advantage、KL 和 loss。
- `trace_reward_logprob_kl.py`：用固定张量复盘 reward、old/current/ref logprob、ratio、KL penalty 和 DPO reference logratio。
- `trace_deploy_surfaces.py`：检查 transformers 模型目录、chat template 渲染，以及 API 侧 `reasoning_content` / `tool_calls` 解析。
- `tiny_pretrain.jsonl`：极小预训练样本。
- `tiny_sft.jsonl`：极小 SFT 对话样本。
- `tiny_dpo.jsonl`：极小 DPO 偏好样本。

后续课程会继续加入：

- tokenizer 调试脚本。
- tiny pretrain / SFT 训练命令说明。
- LoRA 参数冻结检查脚本。
