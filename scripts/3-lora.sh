#!/bin/bash

# LoRA fine-tuning script
# Run with distributed training and save logs

nohup torchrun \
  --nproc_per_node=8 \
  4-lora_sft.py \
  --use_wandb \
  --epochs 3 \
  --out_dir /home/jovyan/zlcode/minimind/out_dir \
  2>&1 > /home/jovyan/zlcode/minimind/log/lora_sft.log &
  
