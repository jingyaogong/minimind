# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

MiniMind is an ultra-small language model (~26M-104M params) designed for training from scratch on consumer GPUs. It demonstrates the complete LLM training pipeline at minimal cost. The codebase is a Chinese-language educational project with English code.

## Common Commands

### Training (all run from `trainer/` directory)

```bash
# Stage 1: Pretraining
cd trainer && torchrun --nproc_per_node=1 train_pretrain.py

# Stage 2: Supervised Fine-Tuning
cd trainer && torchrun --nproc_per_node=1 train_full_sft.py

# Optional: LoRA / DPO / PPO / GRPO / Agent RL / Distillation
cd trainer && torchrun --nproc_per_node=1 train_lora.py
cd trainer && torchrun --nproc_per_node=1 train_dpo.py
cd trainer && torchrun --nproc_per_node=1 train_ppo.py
cd trainer && torchrun --nproc_per_node=1 train_grpo.py
cd trainer && torchrun --nproc_per_node=1 train_agent.py
cd trainer && torchrun --nproc_per_node=1 train_distillation.py
```

Multi-GPU: change `--nproc_per_node=N`. Resume training with `--from_resume 1`.

### Inference

```bash
python eval_llm.py                                          # CLI inference
python eval_llm.py --from_transformers                      # Load transformers-format model
python eval_llm.py --lora_path <path>                       # With LoRA adapter
python scripts/serve_openai_api.py                          # OpenAI-compatible API server
```

### Model Conversion

```bash
python scripts/convert_model.py                             # torch <-> transformers format, LoRA merge
```

### Dependencies

```bash
pip install -r requirements.txt
# PyTorch must be installed separately (commented out in requirements.txt)
```

## Architecture

### Training Pipeline (sequential)

`Pretrain` → `Full SFT` → optional (`LoRA` / `DPO` / `PPO` / `GRPO` / `Agent RL` / `Distillation`)

### Model (`model/model_minimind.py`)

- `MiniMindConfig`: HuggingFace PretrainedConfig with hidden_size=768, 8 layers, vocab_size=6400
- `MiniMindForCausalLM`: Main model class (inherits PreTrainedModel + GenerationMixin)
- Components: RMSNorm, RoPE + YaRN extrapolation, GQA (8 query heads, 4 KV heads), SwiGLU FFN, MoE with top-K routing and aux loss
- Weight tying between input embedding and LM head (`tie_word_embeddings`)
- Custom `generate()` with top-k/top-p sampling, repetition penalty, KV-cache
- Model sizes controlled by `hidden_size`: 512 (minimind-3, 4 layers), 640 (minimind-2, 6 layers), 768 (default, 8 layers)

### LoRA (`model/model_lora.py`)

Pure PyTorch LoRA implementation (rank-16 A*B matrices). Functions: `apply_lora()`, `load_lora()`, `save_lora()`, `merge_lora()`.

### Datasets (`dataset/lm_dataset.py`)

- `PretrainDataset`: plain text JSONL
- `SFTDataset`: multi-turn chat conversations
- `DPODataset`: chosen/rejected pairs
- `RLAIFDataset`: prompts for RL training
- `AgentRLDataset`: multi-turn with tool calls

All data is JSONL format in `dataset/`.

### Shared Utilities (`trainer/trainer_utils.py`)

- `init_model()`: model + tokenizer initialization, handles DDP unwrapping and torch.compile
- `lm_checkpoint()`: atomic checkpoint save/load with `os.replace()`, handles multi-GPU world_size changes on resume
- `init_distributed_mode()`: DDP setup
- `setup_seed()`: deterministic seeding
- `LMForRewardModel`: wrapper for reward models (uses InternLM2-1.8B-Reward)

### Logging

SwanLab (default) or WandB (`--use_wandb`). Checkpoints store wandb_id for resume.

### Tokenizer (`model/`)

Custom BPE tokenizer with 6,400 vocab, 36 special tokens. Chat template supports tools, multi-turn, and reasoning_content. `train_tokenizer.py` is reference-only (pre-trained tokenizer is used).

## Key Conventions

- No test framework exists; `scripts/eval_toolcall.py` provides tool call evaluation
- All training scripts use argparse with defaults matching the minimind-768 config
- Checkpoints are saved to `../checkpoints/` relative to `trainer/`
- Code comments are in Chinese
