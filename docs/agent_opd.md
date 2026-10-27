# Multi-turn tool-use on-policy distillation

`trainer/train_agent_opd.py` is a single-device reference trainer for distilling a
frozen MiniMind teacher along the student's own tool interactions. It uses the
local tool implementations in `train_agent.py`; weather, exchange rates and other
mock tool results are **not live services**. It does not load a reward model.

The student generates a turn, its tool calls are executed, and the returned
observations condition the next student turn. The teacher sees exactly these
student-generated token prefixes and observations. No teacher answer replaces a
student rollout. Prompt text, inserted role markers, tool observations and padding
are context only. Every generated assistant token, including each turn's first
EOS, is a distillation target. Later assistant turns remain targets after earlier
EOS tokens.

## Objective

Following the on-policy data collection idea in
[GKD (Agarwal et al., ICLR 2024)](https://arxiv.org/abs/2306.13649), the default loss is
the mean full-vocabulary `T² KL(teacher_T || student_T)` over assistant tokens on
the sampled interaction. `--kl_direction reverse_kl` instead uses
`T² KL(student_T || teacher_T)` at the same prefixes. This is a GKD-style
semi-gradient: sampling and environment execution are detached. It is not a
REINFORCE estimator of trajectory-level reverse KL.

Sampling defaults to temperature 1, top-p 1 and top-k 0. `--temperature` controls
student sampling; `--distill_temperature` controls only the loss distributions.
Both checkpoints must be native MiniMind weights using the **same tokenizer and
token-to-ID mapping**. A teacher with an unrelated vocabulary is not supported.

## Run

Run from the repository root, after installing the repository dependencies:

```bash
python trainer/train_agent_opd.py \
  --student_checkpoint out/full_sft_768.pth \
  --teacher_checkpoint out/agent_768.pth \
  --data_path dataset/agent_rl.jsonl \
  --device cuda:0 --dtype bfloat16 \
  --max_turns 3 --max_gen_len 256 --max_total_len 2048 \
  --batch_size 1 --accumulation_steps 4 --max_steps 100
```

Use an independently trained, stronger teacher. The example filenames refer to
weights you already have; the trainer does not download them. Architecture flags
(`--student_hidden_size`, `--teacher_num_layers`, `--teacher_use_moe`, etc.) must
match their checkpoints. Weights are loaded strictly. For a CPU correctness run,
use `--device cpu --dtype float32` and small checkpoints. This initial trainer
supports one process/device; it does not provide DDP or SGLang training.

Data may use `conversations` (as in `agent_rl.jsonl`) or `messages`. Tool schemas
can appear in the system message's `tools` field or the record's top-level
`tools` field. A trailing fixed assistant answer is ignored. `gt` is not required:

```json
{"messages":[{"role":"user","content":"请用工具计算 2+2。"}],"tools":[{"type":"function","function":{"name":"calculate_math","description":"计算数学表达式","parameters":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}}}]}
```

Only tools declared for the sample are available. Unknown tools and invalid
arguments produce an error observation. Context and turn limits stop further
interaction while retaining already sampled targets. Context is never silently
left-truncated into a different teacher prefix. Oversized initial prompts raise
an error; adjust the data or `--max_total_len`.

The output includes an inference weight file, `out/agent_opd/agent_opd_768.pth`,
and a full-precision `agent_opd_768_resume.pth` containing optimizer/scaler state,
the next data cursor and random states. Checkpoints are saved after complete
optimizer updates, including a final partial accumulation window. Resume with
the original arguments plus:

```bash
--resume out/agent_opd/agent_opd_768_resume.pth --max_steps 200
```

`--max_steps` counts total optimizer steps, including resumed steps. Keep the
teacher, data, architecture, optimizer and sampling arguments unchanged. The
logged `loss`, `mean_turns`, `tool_calls`, `assistant_tokens` and
`truncated_fraction` describe training, not task success rates.

## Validation and limits

```bash
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  python -m unittest discover -s tests -p test_agent_opd.py -v
```

Tests use the real repository tokenizer, local tools and tiny CPU MiniMind
models. Scripted generation covers deterministic multi-turn boundaries, including
tool errors, per-turn EOS and multiple calls; other tests exercise actual model
generation, optimization, teacher freezing and exact CPU checkpoint continuation.
A controlled two-turn optimization test verifies that teacher divergence falls.
No pretrained task-quality improvement or CUDA/MPS performance result is claimed.

Before claiming better tool use, evaluate the starting student, teacher and OPD
checkpoint on held-out prompts with identical decoding and tool limits. Compare
against fixed-data distillation as well, and report final-answer accuracy,
successful tool execution, truncation, wall time and peak memory. A falling
distillation loss alone is not evidence of better answers.

## 中文说明

这是面向多轮工具交互的 OPD 参考实现：学生自己生成工具调用，读取工具结果后继续生成，
教师沿着学生实际经历的上下文提供分布监督。工具结果参与上下文，但不作为预测标签；每轮
学生生成的 EOS 都参与损失，后续轮次不会因第一轮 EOS 而被屏蔽。默认使用前向 KL，支持
反向 KL 的 GKD 式半梯度。教师被冻结，数据不需要 `gt` 或标准答案。

首版仅支持单进程、共享分词器的 MiniMind 教师与学生，复用仓库的本地模拟工具。
请根据实际权重指定教师/学生结构。测试验证训练链路和数值正确性；正式模型上的能力提升
仍需独立留出集实验。断点文件保存完整精度和随机状态，恢复时沿用原始参数并加上 `--resume`。
