---
name: minimind-lora
description: Train and use MiniMind LoRA adapters (train_lora.py). Use for PEFT-style domain adaptation (e.g. medical/identity), lora_weight loading in eval_llm.py, or merging LoRA via convert_model.py.
---

# MiniMind LoRA

LoRA is implemented in-repo (`model/model_lora.py` + `trainer/train_lora.py`), not via peft wrappers. Run training from `trainer/`. Needs a base weight such as `full_sft` under `../out/`.

## Train

```bash
cd trainer
python train_lora.py
# or: torchrun --nproc_per_node N train_lora.py
```

Important defaults from `train_lora.py`:

| Flag | Default |
| --- | --- |
| `--save_dir` | `../out` |
| `--lora_rank` | `16` |
| `--lora_alpha` | `None` (means alpha = rank, scaling = 1) |
| `--lora_name` | `lora_medical` |
| `--epochs` | `10` |
| `--batch_size` | `32` |
| `--learning_rate` | `1e-4` |
| `--max_seq_len` | `340` |
| `--data_path` | `../dataset/lora_medical.jsonl` |
| `--from_weight` | `full_sft` |
| `--from_resume` | `0` |
| `--hidden_size` | `768` |
| `--num_hidden_layers` | `8` |
| `--use_moe` | `0` |

Output weight: `../out/<lora_name>_<hidden_size>.pth` (e.g. `lora_medical_768.pth`).

Examples:

```bash
python train_lora.py --lora_name lora_identity --data_path ../dataset/lora_identity.jsonl
python train_lora.py --from_resume 1
```

Data format matches SFT multi-turn `conversations` jsonl (see README LoRA section).

## Chat with base + LoRA

From repo root — `--weight` is the **base** checkpoint prefix; `--lora_weight` is the adapter name (not `None`):

```bash
python eval_llm.py --weight full_sft --lora_weight lora_medical
python eval_llm.py --weight full_sft --lora_weight lora_identity --lora_rank 16
```

`--lora_rank` must match training (default `16`).

Same flags exist on `scripts/serve_openai_api.py` (`--lora_weight`, `--lora_rank`).

## Merge LoRA into a full torch weight

`scripts/convert_model.py` has **no argparse**. Edit the `__main__` block under `scripts/`, then:

```bash
cd scripts && python convert_model.py
```

Uncomment / adjust the merge section that calls `convert_merge_base_lora(base_torch_path, lora_path, merged_torch_path)`. Paths in the file are relative to `scripts/` (e.g. `../out/full_sft_768.pth`, `../out/lora_identity_768.pth`).

## Notes

- Prefer LoRA for small vertical datasets; large mixed general+domain data can use `train_full_sft.py` instead.
- Keep `--hidden_size`, `--num_hidden_layers`, `--use_moe` aligned with the base `full_sft` (or other) weight.
