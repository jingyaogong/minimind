# Code Generation Agent

This extension adds an execution-feedback loop for Python algorithm generation while keeping the original MiniMind model and checkpoint format unchanged.

## Architecture

```mermaid
flowchart LR
    A[Algorithm task] --> B[MiniMind or OpenAI-compatible model]
    B --> C[Python code extraction]
    C --> D[AST policy validation]
    D --> E[Isolated subprocess]
    E --> F[Deterministic tests]
    F --> G[Structured feedback]
    G -->|repair prompt| B
    F --> H[Verifiable reward]
    H --> I[GRPO or CISPO]
    F --> J[pass@k and error metrics]
```

The implementation is split into small, testable modules:

- `code_agent/verifier.py`: code extraction, AST policy checks, subprocess execution, timeout and structured results.
- `code_agent/reward.py`: transparent reward components and grouped GRPO scoring.
- `code_agent/agent.py`: generate-execute-feedback-repair loop with hidden-test-safe feedback.
- `code_agent/evaluation.py`: compile rate, execution rate, test pass rate, error counts and unbiased pass@k.
- `code_agent/dataset.py`: JSONL code tasks adapted to the MiniMind chat template.
- `code_agent/backends.py`: OpenAI-compatible inference adapter for MiniMind, vLLM and SGLang.

## Task format

Each JSONL record contains a prompt, a required function name, and JSON-serializable tests:

```json
{"task_id":"add-two","prompt":"Implement add(a, b).","entry_point":"add","tests":[{"args":[1,2],"expected":3}]}
```

`dataset/code_rl_mini.jsonl` contains four smoke-test tasks. It is deliberately too small for a meaningful training claim; use it to validate the pipeline before preparing a larger train/validation/test split.

For a standard public benchmark, convert the compatible subset of MBPP-sanitized while preserving its official train/validation/test split:

```bash
python scripts/prepare_mbpp.py --output-dir out/mbpp_sanitized
```

The converter accepts direct, JSON-serializable function assertions, validates every converted task against the published reference solution, and reports all skipped records. It also emits `train_sft.jsonl` from the official training references, enabling an SFT → execution-reward GRPO ablation without leaking validation or test solutions. Generated benchmark files remain under the ignored `out/` directory.

## Evaluate saved candidates

Generate multiple candidates from a running OpenAI-compatible model service with deterministic per-request seeds:

```bash
python scripts/generate_codegen.py \
  --tasks dataset/code_rl_mini.jsonl \
  --samples-per-task 10 \
  --seed 42 \
  --output out/code_predictions.jsonl
```

Then execute and aggregate them:

```bash
python scripts/eval_codegen.py \
  --tasks dataset/code_rl_mini.jsonl \
  --predictions out/code_predictions.jsonl \
  --ks 1 5 10
```

The report separates:

- `compile_rate`: candidate passed syntax and policy validation;
- `execution_rate`: candidate completed without a timeout or runtime exception;
- `test_pass_rate`: fraction of deterministic tests passed;
- `pass@k`: probability that at least one of `k` sampled candidates is correct;
- `status_counts`: syntax, policy, timeout, runtime and wrong-answer failure modes.

## Run the repair agent

Start the repository's OpenAI-compatible service, then run:

```bash
python scripts/run_code_agent.py \
  --tasks dataset/code_rl_mini.jsonl \
  --base-url http://localhost:8998/v1 \
  --model minimind \
  --max-attempts 3
```

Hidden-test details are not sent back to the model by default. Add `--reveal-test-details` only for public development tests.

## Train with verifiable rewards

From the repository root:

```bash
cd trainer
python train_grpo.py \
  --reward_type code \
  --data_path ../dataset/code_rl_mini.jsonl \
  --num_generations 6 \
  --loss_type grpo \
  --from_weight full_sft
```

The training log includes `degenerate_group_rate`. A group whose candidates all receive the same reward has zero normalized advantage and contributes no useful policy-gradient signal. This metric makes the failure mode visible before introducing strategies such as dynamic sampling or harder tasks.

## Reward design

The default reward is intentionally transparent and bounded to `[-1, 1]`:

| Outcome | Reward |
|---|---:|
| all tests passed | 1.0 |
| wrong answer | partial credit from test pass rate |
| runtime error | -0.4 |
| timeout | -0.6 |
| syntax error | -0.8 |
| missing code or policy violation | -1.0 |

Keep hidden tests separate from the prompts and report reward distributions by task difficulty. Otherwise the system can overfit public tests or produce groups with all-identical rewards.

## Safety boundary

The local verifier uses AST checks, a temporary working directory, an isolated Python process, a timeout, and POSIX resource limits. These are defense-in-depth measures, not a secure sandbox. Run hostile or externally submitted code inside a dedicated container or micro-VM with network disabled, a read-only filesystem, strict CPU/memory/process limits, and disposable credentials.

## Experimental protocol

Use the same held-out tasks and sampling parameters for three comparisons:

1. single-pass generation baseline;
2. inference-time execution feedback and repair;
3. execution-reward GRPO/CISPO followed by the same inference procedure.

Record pass@1, pass@k, test pass rate, compile rate, timeout rate, average attempts, average generation length, reward mean/std, KL to the reference model, and degenerate-group rate. Publish model results only after this protocol has been run and the raw logs and configuration have been retained for reproduction.

## Attribution

This repository is a derivative of [jingyaogong/minimind](https://github.com/jingyaogong/minimind) under Apache-2.0. The extension preserves upstream model names and checkpoint compatibility; presentation-layer identity prompts are kept neutral.
