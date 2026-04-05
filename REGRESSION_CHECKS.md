# Regression Checks for Recent Fixes

This file records lightweight checks for the recent high-priority fixes.

## 1) Syntax checks

```bash
python -m py_compile dataset/lm_dataset.py trainer/train_pretrain.py trainer/train_full_sft.py trainer/train_lora.py trainer/train_distillation.py trainer/train_dpo.py trainer/train_grpo.py trainer/train_agent.py eval_llm.py
```

## 2) Dataset label/mask boundary check (no eos after truncation)

```bash
python - <<'PY'
from dataset.lm_dataset import SFTDataset, DPODataset

s = SFTDataset.__new__(SFTDataset)
s.bos_id = [11, 12]
s.eos_id = [13]
s.max_length = 10
raw_ids = [99, 11, 12, 21, 22]  # truncated without eos
labels = s.generate_labels(raw_ids)
pad_len = s.max_length - len(raw_ids)
labels = labels + ([-100] * pad_len)
print("sft_pad_labels_ok", labels[5:] == [-100] * pad_len)

d = DPODataset.__new__(DPODataset)
d.bos_id = [11, 12]
d.eos_id = [13]
d.max_length = 10
mask = d.generate_loss_mask(raw_ids)
mask = mask + ([0] * pad_len)
print("dpo_pad_mask_ok", mask[5:] == [0] * pad_len)
PY
```

Expected output:

```text
sft_pad_labels_ok True
dpo_pad_mask_ok True
```

## 3) Eval backend argument check

```bash
python eval_llm.py -h
```

Expected: help output includes `--backend {auto,torch,hf}`.

## 4) Manual inference smoke tests

HF backend:

```bash
python eval_llm.py --backend hf --load_from ./minimind-3 --max_new_tokens 64 --temperature 0.2 --top_p 0.8
```

Torch backend:

```bash
python eval_llm.py --backend torch --load_from model --weight full_sft --max_new_tokens 64 --temperature 0.2 --top_p 0.8
```
