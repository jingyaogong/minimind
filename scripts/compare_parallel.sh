#!/bin/bash
# 分布式并行策略对比 smoke tests
#
# 目的：跑 5-6 种并行策略各 100 步，收集 MFU / tokens/s / 每卡显存 → Design Doc #2 素材表
#
# ⚠️ 前置：**必须先 kill 当前 1B 训练**（4 卡全被占）：
#   pkill -9 -f 'train_pretrain|torchrun|multiproc'
#   然后确认 nvidia-smi 显存 <100 MB/卡
#
# 用法：
#   scripts/compare_parallel.sh              # 跑所有对比
#   scripts/compare_parallel.sh ddp           # 只跑 DDP
#   scripts/compare_parallel.sh fsdp_full     # 只跑 FSDP full shard
#   scripts/compare_parallel.sh tp            # 只跑 TP=2×DP=2
#
# 输出：/tmp/parallel_compare/*.log；最后自动 summarize 到 stdout
set -e

REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT_DIR=/tmp/parallel_compare
mkdir -p "$OUT_DIR"

# 公共配置（保持对比公平：同 bs / seq / accum / model）
COMMON_ARGS=(
    --hidden_size 2048
    --num_hidden_layers 20
    --max_seq_len 1024
    --batch_size 16
    --accumulation_steps 1
    --learning_rate 3e-4
    --warmup_steps 50
    --dtype bfloat16
    --num_workers 4
    --use_compile 1
    --data_path "$REPO/dataset/skypile_test.jsonl"
    --eval_interval 0
    --save_interval 100000
    --log_interval 10
    --save_weight parallel_smoke
    --save_dir "$REPO/out"
)

run_config() {
    local name="$1"
    shift
    local strategy_args=("$@")
    local log="$OUT_DIR/$name.log"
    echo ""
    echo "=========================================="
    echo "  Running: $name"
    echo "  extra: ${strategy_args[*]}"
    echo "=========================================="
    cd "$REPO/trainer"
    # timeout 3 min 强制停：100 步跑不完就当异常，不用等
    timeout 180 torchrun --nproc_per_node=4 --standalone train_pretrain.py \
        "${COMMON_ARGS[@]}" \
        "${strategy_args[@]}" > "$log" 2>&1 || true
    echo "  最后 5 条 MFU log："
    grep -E 'MFU' "$log" 2>/dev/null | grep -v 'params_flops' | tail -5 || echo "  (no MFU logs — crashed?)"
    # 收显存 (kill 前抓)
    nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1 > "$OUT_DIR/${name}_gpu.csv" || true
    ps aux | grep -E 'train_pretrain|torchrun|multiproc' | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
    sleep 5
}

CONFIG="${1:-all}"

case "$CONFIG" in
    all)
        run_config "ddp"          --parallel_strategy ddp
        run_config "fsdp_no"      --parallel_strategy fsdp_no
        run_config "fsdp_grad_op" --parallel_strategy fsdp_grad_op
        run_config "fsdp_full"    --parallel_strategy fsdp_full
        run_config "tp2_dp2"      --parallel_strategy tp --tp_size 2
        run_config "tp4_dp1"      --parallel_strategy tp --tp_size 4
        ;;
    ddp|fsdp_no|fsdp_grad_op|fsdp_full)
        run_config "$CONFIG" --parallel_strategy "$CONFIG"
        ;;
    tp|tp2)
        run_config "tp2_dp2" --parallel_strategy tp --tp_size 2
        ;;
    tp4)
        run_config "tp4_dp1" --parallel_strategy tp --tp_size 4
        ;;
    *)
        echo "Unknown config: $CONFIG"
        echo "Valid: all | ddp | fsdp_no | fsdp_grad_op | fsdp_full | tp | tp4"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "  Summary (稳态 = 各 log 最后一条 MFU)"
echo "=========================================="
printf "%-16s %-10s %-10s %-12s\n" "strategy" "it/s" "MFU%" "mem/card"
printf "%-16s %-10s %-10s %-12s\n" "--------" "----" "----" "--------"

for log in "$OUT_DIR"/*.log; do
    [ -f "$log" ] || continue
    name=$(basename "$log" .log)
    line=$(grep -E 'MFU' "$log" 2>/dev/null | grep -v 'params_flops' | tail -1)
    mem_file="$OUT_DIR/${name}_gpu.csv"
    mem=$(cat "$mem_file" 2>/dev/null | tr -d ' ' || echo "?")
    if [ -z "$line" ]; then
        printf "%-16s %-10s %-10s %-12s\n" "$name" "-" "-" "crashed"
    else
        its=$(echo "$line" | grep -oE 'it/s: [0-9.]+' | awk '{print $2}')
        mfu=$(echo "$line" | grep -oE 'MFU: [0-9.]+' | awk '{print $2}')
        printf "%-16s %-10s %-10s %-12s\n" "$name" "$its" "$mfu%" "$mem"
    fi
done

echo ""
echo "详细 log: $OUT_DIR/*.log"
echo "profile 单个: scripts/compare_parallel.sh $CONFIG 时加 --profile_steps 20 到 COMMON_ARGS"
