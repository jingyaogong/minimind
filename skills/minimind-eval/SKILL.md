---
name: minimind-eval
description: Evaluate and chat with MiniMind checkpoints via eval_llm.py and scripts/eval_toolcall.py. Use for CLI dialogue, comparing weight prefixes, LoRA overlays, open_thinking, or tool-call smoke tests.
---

# MiniMind eval and chat

## CLI chat — `eval_llm.py` (repo root)

```bash
# Transformers-format folder
python eval_llm.py --load_from ./minimind-3

# Native torch weights under ./out (tokenizer/code from ./model)
python eval_llm.py --load_from ./model --weight full_sft
python eval_llm.py --weight pretrain
python eval_llm.py --weight dpo
python eval_llm.py --weight ppo_actor
python eval_llm.py --weight grpo
python eval_llm.py --weight agent
```

Flags from `eval_llm.py`:

| Flag | Default | Notes |
| --- | --- | --- |
| `--load_from` | `model` | `model` = native torch + `out/`; other path = transformers dir |
| `--save_dir` | `out` | Where `*.pth` live for native mode |
| `--weight` | `full_sft` | Filename prefix: `pretrain`, `full_sft`, `dpo`, `ppo_actor`, `grpo`, `agent`, … |
| `--lora_weight` | `None` | Adapter name, e.g. `lora_medical` |
| `--lora_rank` | `16` | Must match LoRA training |
| `--hidden_size` | `768` | Must match trained config |
| `--num_hidden_layers` | `8` | |
| `--use_moe` | `0` | `0` or `1` |
| `--inference_rope_scaling` | off | YaRN-style RoPE extrapolate (store_true) |
| `--max_new_tokens` | `8192` | |
| `--temperature` | `0.85` | |
| `--top_p` | `0.95` | |
| `--open_thinking` | `0` | `1` enables adaptive thinking template |
| `--historys` | `0` | Even number of history turns; `0` = none |
| `--show_speed` | `1` | tokens/s |
| `--device` | `cuda` if available else `cpu` | |

Examples:

```bash
python eval_llm.py --weight full_sft --lora_weight lora_medical
python eval_llm.py --weight full_sft --open_thinking 1 --historys 2
python eval_llm.py --load_from ./minimind-3 --temperature 0.7
```

Weight files are expected as `{save_dir}/{weight}_{hidden_size}[_moe].pth` for native loads.

## Tool-call eval — `scripts/eval_toolcall.py`

Run from `scripts/` so relative defaults resolve:

```bash
cd scripts
python eval_toolcall.py --weight full_sft
python eval_toolcall.py --weight agent
```

| Flag | Default |
| --- | --- |
| `--backend` | `local` (`local` \| `api`) |
| `--load_from` | `../model` |
| `--save_dir` | `../out` |
| `--weight` | `full_sft` |
| `--hidden_size` | `768` |
| `--num_hidden_layers` | `8` |
| `--use_moe` | `0` |
| `--max_new_tokens` | `512` |
| `--temperature` | `0.9` |
| `--top_p` | `0.9` |
| `--show_speed` | `0` |
| `--device` | `cuda`/`cpu` |
| `--api_base_url` | `http://localhost:11434/v1` |
| `--api_key` | `sk-123` |
| `--api_model` | `jingyaogong/minimind-3:latest` |
| `--stream` | `1` |

API backend example:

```bash
python eval_toolcall.py --backend api --api_base_url http://localhost:8998/v1 --api_model model-identifier
```

## WebUI (optional)

Requires transformers-format weights copied under `scripts/`:

```bash
cp -r minimind-3 ./scripts/minimind-3
cd scripts && streamlit run web_demo.py
```

`web_demo.py` auto-scans subdirs that contain weight files.

## Consistency rules

Match `--hidden_size`, `--num_hidden_layers`, and `--use_moe` to the checkpoint. Prefer reading `-h` on the script over guessing flags.
