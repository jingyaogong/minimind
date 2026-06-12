#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/dataset/sft_t2t.jsonl}"
WEIGHT_PATH="${WEIGHT_PATH:-${REPO_ROOT}/out/pretrain_searchlm_1024_mla.pth}"
SAVE_WEIGHT="${SAVE_WEIGHT:-general_sft_14g}"
NUM_GPUS="${NUM_GPUS:-7}"
EPOCHS="${EPOCHS:-1}"
MICRO_BATCH="${MICRO_BATCH:-1}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
NUM_WORKERS="${NUM_WORKERS:-8}"
USE_WANDB="${USE_WANDB:-0}"
FORCE_RESUME="${FORCE_RESUME:-auto}"

if [[ ! -s "${DATA_PATH}" ]]; then
  echo "SFT dataset not found or empty: ${DATA_PATH}" >&2
  exit 1
fi

if [[ ! -s "${WEIGHT_PATH}" ]]; then
  echo "Pretrained checkpoint not found: ${WEIGHT_PATH}" >&2
  echo "Expected from_weight=pretrain_searchlm -> out/pretrain_searchlm_1024_mla.pth" >&2
  exit 1
fi

if ! command -v deepspeed >/dev/null 2>&1; then
  echo "deepspeed command not found. Install it in the server training environment first." >&2
  exit 1
fi

mkdir -p "${REPO_ROOT}/logs" "${REPO_ROOT}/out" "${REPO_ROOT}/checkpoints"

LATEST_FILE="${REPO_ROOT}/checkpoints/${SAVE_WEIGHT}_1024_mla_latest"
RESUME=0
if [[ "${FORCE_RESUME}" == "1" ]]; then
  RESUME=1
elif [[ "${FORCE_RESUME}" == "auto" && -s "${LATEST_FILE}" ]]; then
  RESUME=1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${REPO_ROOT}/logs/${SAVE_WEIGHT}_${STAMP}.log"

echo "repo_root=${REPO_ROOT}"
echo "data_path=${DATA_PATH}"
echo "weight_path=${WEIGHT_PATH}"
echo "save_weight=${SAVE_WEIGHT}"
echo "num_gpus=${NUM_GPUS}"
echo "effective_batch=$((NUM_GPUS * MICRO_BATCH * ACCUMULATION_STEPS))"
echo "resume=${RESUME}"
echo "log_file=${LOG_FILE}"

WANDB_ARGS=()
if [[ "${USE_WANDB}" == "1" ]]; then
  WANDB_ARGS+=(--use_wandb)
fi

cd "${REPO_ROOT}/trainer"

deepspeed --num_gpus="${NUM_GPUS}" train_full_sft_deepspeed.py \
  --model_profile SearchLM-300M \
  --from_weight pretrain_searchlm \
  --save_weight "${SAVE_WEIGHT}" \
  --data_path "${DATA_PATH}" \
  --epochs "${EPOCHS}" \
  --batch_size "${MICRO_BATCH}" \
  --accumulation_steps "${ACCUMULATION_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --dtype float16 \
  --num_workers "${NUM_WORKERS}" \
  --log_interval "${LOG_INTERVAL}" \
  --save_interval "${SAVE_INTERVAL}" \
  --from_resume "${RESUME}" \
  "${WANDB_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
