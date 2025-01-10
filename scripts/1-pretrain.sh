#!/bin/bash

# 环境变量配置
export MASTER_PORT=29500      # 设置通信端口
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 设置使用的 GPU

# 训练配置参数
NUM_GPUS=8
EPOCHS=3
OUT_DIR="/home/jovyan/zlcode/minimind/out_dir"
LOG_DIR="/home/jovyan/zlcode/minimind/log"
LOG_FILE="${LOG_DIR}/pretrain.log"
Batch_size=128
# 创建输出目录和日志目录（如果不存在）
mkdir -p "$OUT_DIR"
mkdir -p "$LOG_DIR

# 设置训练参数的选项
TRAINING_OPTIONS="--use_wandb \
                  --ddp \
                  --epochs=$EPOCHS \
                  --batch_size=${Batch_size} \
                  --out_dir=$OUT_DIR"

# 设置torchrun分布式训练参数
DISTRIBUTED_OPTIONS="--nproc_per_node=$NUM_GPUS \
                     --nnodes=1 \
                     --rdzv_id=123 \
                     --rdzv_backend=c10d"

# 使用 nohup 启动训练任务，并且重定向输出
nohup torchrun $DISTRIBUTED_OPTIONS 1-pretrain.py $TRAINING_OPTIONS \
    2>&1 | tee "$LOG_FILE" &

