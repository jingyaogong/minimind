#!/bin/bash

# 获取脚本所在目录
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_DIR"

# 检查是否已经有实例在运行
if [ -f "train_web_ui.pid" ]; then
    pid=$(cat "train_web_ui.pid")
    if ps -p "$pid" > /dev/null 2>&1; then
        echo "Web UI 服务已经在运行 (PID: $pid)"
        exit 1
    else
        echo "删除旧的PID文件"
        rm "train_web_ui.pid"
    fi
fi

# 创建日志目录
LOG_DIR="../logfile"
mkdir -p "$LOG_DIR"

# 生成时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/web_ui_$TIMESTAMP.log"

echo "启动 MiniMind Web UI 服务..."
echo "日志文件: $LOG_FILE"

# 使用nohup启动服务
nohup python -u train_web_ui.py > "$LOG_FILE" 2>&1 &

# 保存PID
echo $! > "train_web_ui.pid"

sleep 2

echo "服务已启动! PID: $(cat "train_web_ui.pid")"
echo "访问地址: http://localhost:5000"
echo "停止命令: kill $(cat "train_web_ui.pid") or ./trainer_web/stop_web_ui.sh"