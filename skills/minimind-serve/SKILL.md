---
name: minimind-serve
description: Serve MiniMind with the OpenAI-compatible API, convert torch↔transformers weights, and use related scripts. Use for serve_openai_api.py, convert_model.py, chat_api.py, web_demo, ollama/vllm/sglang pointers.
---

# MiniMind serving and conversion

## OpenAI-compatible API — `scripts/serve_openai_api.py`

```bash
cd scripts
python serve_openai_api.py
```

Listens on `0.0.0.0:8998` (host/port are fixed in the script, not CLI flags).

Flags:

| Flag | Default | Notes |
| --- | --- | --- |
| `--load_from` | `../model` | native torch loader; other path = transformers dir |
| `--save_dir` | `out` | script loads `../{save_dir}/...pth` (default → repo `out/`) |
| `--weight` | `full_sft` | |
| `--lora_weight` | `None` | |
| `--lora_rank` | `16` | |
| `--hidden_size` | `768` | |
| `--num_hidden_layers` | `8` | |
| `--max_seq_len` | `8192` | |
| `--use_moe` | `0` | |
| `--inference_rope_scaling` | off | store_true |
| `--device` | `cuda`/`cpu` | |

Examples:

```bash
python serve_openai_api.py --load_from ../minimind-3
python serve_openai_api.py --load_from ../model --weight full_sft
python serve_openai_api.py --weight full_sft --lora_weight lora_medical
```

Supports request fields such as `reasoning_content`, `tool_calls`, and `open_thinking` (see README).

Smoke client:

```bash
cd scripts && python chat_api.py
```

curl example (from README):

```bash
curl http://localhost:8998/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model-identifier",
    "messages": [{"role": "user", "content": "世界上最高的山是什么？"}],
    "temperature": 0.7,
    "max_tokens": 1024,
    "stream": true,
    "open_thinking": true
  }'
```

Expected transformers layout when using a downloaded model folder at repo root:

```text
minimind/
└── minimind-3/
    ├── config.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── model.safetensors  # or pytorch_model.bin
```

## Model conversion — `scripts/convert_model.py`

No CLI argparse. Edit paths in the `if __name__ == '__main__':` block, then:

```bash
cd scripts && python convert_model.py
```

Available helpers in that file:

- `convert_torch2transformers(...)` — torch `.pth` → transformers (Qwen-compatible structure used by the script)
- `convert_torch2transformers_minimind(...)` — torch → MiniMind transformers registration path
- `convert_transformers2torch(...)` — reverse
- `convert_merge_base_lora(...)` — merge base + LoRA into one `.pth`

Default `__main__` example converts `../out/full_sft_768.pth` → `../minimind-3` with `MiniMindConfig(hidden_size=768, num_hidden_layers=8, max_seq_len=8192, use_moe=False)`. Adjust `lm_config` and paths to match the checkpoint before running.

## WebUI

```bash
cp -r minimind-3 ./scripts/minimind-3   # transformers weights must live under scripts/
cd scripts && streamlit run web_demo.py
```

## Third-party (documented in README; not MiniMind scripts)

```bash
# ollama
ollama run jingyaogong/minimind-3

# vllm
vllm serve /path/to/model --model-impl transformers --served-model-name "minimind" --port 8998

# sglang
python -m sglang.launch_server --model-path /path/to/model --attention-backend triton --host 0.0.0.0 --port 8998
```

For interactive CLI without a server, use skill `minimind-eval` (`eval_llm.py`).
