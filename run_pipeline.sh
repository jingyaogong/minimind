#!/bin/bash
# MiniMind 一键训练流水线：pretrain -> full_sft -> eval（自动测试）
# 用于在新环境/新硬件上快速验证完整训练流程，各阶段参数默认值与对应训练脚本一致。
#
# 用法:
#   bash run_pipeline.sh
#   PRETRAIN_EPOCHS=1 SFT_BATCH=32 bash run_pipeline.sh   # 用环境变量覆盖参数
#   RUN_EVAL=0 bash run_pipeline.sh                       # 跳过最后的自动测试
set -uo pipefail

cd "$(dirname "$0")"
ROOT=$(pwd)
LOG_DIR="$ROOT/logs"
TS=$(date +%Y%m%d_%H%M%S)
PLOG="$LOG_DIR/pipeline_$TS.log"
mkdir -p "$LOG_DIR"

PY=${PYTHON:-python3}

# ---- 可调参数（默认值与各训练脚本一致）----
PRETRAIN_EPOCHS=${PRETRAIN_EPOCHS:-2}
PRETRAIN_BATCH=${PRETRAIN_BATCH:-32}
PRETRAIN_ACCUM=${PRETRAIN_ACCUM:-8}
SFT_EPOCHS=${SFT_EPOCHS:-2}
SFT_BATCH=${SFT_BATCH:-16}
SFT_ACCUM=${SFT_ACCUM:-1}
NUM_WORKERS=${NUM_WORKERS:-8}
LOG_INTERVAL=${LOG_INTERVAL:-100}
RESUME=${RESUME:-0}   # 1=检测到上次中断的检查点时自动续训
RUN_EVAL=${RUN_EVAL:-1}
# ------------------------------------------

# stdout 经管道/tee 时是块缓冲，必须关闭缓冲才能实时看到训练日志
export PYTHONUNBUFFERED=1

# 自动检测训练设备：cuda > mps > cpu（也可用 DEVICE=xxx 覆盖）
DEVICE=${DEVICE:-$("$PY" -c "import torch; print('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))" 2>/dev/null || echo cpu)}
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "[pipeline] device=$DEVICE start $(date '+%F %T')" | tee "$PLOG"

run_stage() {
    local name="$1"; shift
    echo "[pipeline] === stage: $name START $(date '+%F %T') ===" | tee -a "$PLOG"
    local t0
    t0=$(date +%s)
    "$@" 2>&1 | tee "$LOG_DIR/${name}_${TS}.log"
    local rc=${PIPESTATUS[0]}
    local t1
    t1=$(date +%s)
    echo "[pipeline] === stage: $name END rc=$rc elapsed=$(( (t1 - t0) / 60 ))min $(date '+%F %T') ===" | tee -a "$PLOG"
    return $rc
}

cd "$ROOT/trainer"

run_stage pretrain "$PY" train_pretrain.py \
    --device "$DEVICE" --epochs "$PRETRAIN_EPOCHS" \
    --batch_size "$PRETRAIN_BATCH" --accumulation_steps "$PRETRAIN_ACCUM" \
    --num_workers "$NUM_WORKERS" --log_interval "$LOG_INTERVAL" \
    --from_resume "$RESUME" || exit 1

run_stage full_sft "$PY" train_full_sft.py \
    --device "$DEVICE" --epochs "$SFT_EPOCHS" \
    --batch_size "$SFT_BATCH" --accumulation_steps "$SFT_ACCUM" \
    --num_workers "$NUM_WORKERS" --log_interval "$LOG_INTERVAL" \
    --from_resume "$RESUME" || exit 1

if [ "$RUN_EVAL" = "1" ]; then
    cd "$ROOT"
    echo "[pipeline] === stage: eval START $(date '+%F %T') ===" | tee -a "$PLOG"
    echo 0 | "$PY" eval_llm.py --weight full_sft --device "$DEVICE" \
        2>&1 | tee "$LOG_DIR/eval_${TS}.log"
    echo "[pipeline] === stage: eval END $(date '+%F %T') ===" | tee -a "$PLOG"
fi

echo "[pipeline] === ALL DONE $(date '+%F %T') ===" | tee -a "$PLOG"
