#!/bin/bash
# 1B pretrain launcher — 4 卡 DDP，SkyPile stage 1（真 raw web pretrain）
#
# Config 基于 plan L84 + dry-run 学到的教训：
#   - hidden=2048, layers=20, heads=16 (kv=4 GQA) → ~1.02B params
#   - seq=1024 (SkyPile 长度足够)
#   - global batch = 16 per GPU × 4 GPU × accum 4 = 256 → ~256*1024=262k tokens/step
#   - lr 3e-4 (1B from-scratch 标准) + warmup 1000 (前 500 步发散风险 >50%)
#   - eval 每 500 step，用 minimind pretrain 做 proxy holdout（1B 未见过）
#   - save 每 2000 step（2GB × 264 次约 500GB 磁盘，可承受；1B 训 ~150k step 总量）
#   - grad_checkpointing 0（1B seq=1024 在 A100 80GB 用不上，省 30% 时间）
#
# Stage 2 (minimind curated Q&A mid-training) 是另开 run，从 pretrain_1b ckpt 起点继续
#
# 用法:
#   scripts/launch_1b_pretrain.sh                # 默认 4 卡
#   scripts/launch_1b_pretrain.sh --nproc 2       # 覆盖为 2 卡
#   scripts/launch_1b_pretrain.sh -- --learning_rate 5e-4  # 传给 python
set -e

# 默认参数
NPROC=4
# 分离 launcher 参数和 python 参数
PY_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nproc) NPROC="$2"; shift 2 ;;
        --) shift; PY_ARGS+=("$@"); break ;;
        *) PY_ARGS+=("$1"); shift ;;
    esac
done

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT/trainer"

# 数据：SkyPile 是 stage 1 主战场
DATA_PATH="../dataset/skypile_test.jsonl"
# eval holdout：用 minimind pretrain 作 proxy（1B 训练时未见过）
EVAL_DATA="../dataset/pretrain_t2t.jsonl"

echo "[launch_1b_pretrain] NPROC=$NPROC data=$DATA_PATH"
echo "[launch_1b_pretrain] extra args: ${PY_ARGS[*]}"

torchrun --nproc_per_node=$NPROC --standalone train_pretrain.py \
    --hidden_size 2048 \
    --num_hidden_layers 20 \
    --max_seq_len 1024 \
    --batch_size 24 \
    --accumulation_steps 4 \
    --learning_rate 3e-4 \
    --warmup_steps 1000 \
    --grad_checkpointing 0 \
    --dtype bfloat16 \
    --num_workers 4 \
    --use_compile 1 \
    --data_path "$DATA_PATH" \
    --eval_interval 500 \
    --eval_data_path "$EVAL_DATA" \
    --eval_max_batches 20 \
    --save_interval 500 \
    --log_interval 20 \
    --save_weight pretrain_1b \
    --save_dir ../out \
    --wandb_project MiniMind-1B-Pretrain \
    "${PY_ARGS[@]}"
