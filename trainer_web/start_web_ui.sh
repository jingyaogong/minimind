#!/bin/bash

# 获取脚本所在目录（兼容 macOS）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
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

# 轮询日志以获取实际端口号（最多等待10秒）
PORT=""
for i in {1..20}; do
    # 提取形如 http://0.0.0.0:12345 的地址，再截取端口
    PORT=$(grep -Eo 'http://0\.0\.0\.0:[0-9]+' "$LOG_FILE" | tail -n1 | awk -F: '{print $NF}')
    if [ -n "$PORT" ]; then
        break
    fi
    sleep 0.5
done

# 如果仍未获取到端口，回退为默认提示端口（与后端起始端口一致）
if [ -z "$PORT" ]; then
    PORT=12581
fi

echo "服务已启动! PID: $(cat "train_web_ui.pid")"
echo "访问地址: http://localhost:$PORT"
echo "停止命令: kill $(cat "train_web_ui.pid") or bash trainer_web/stop_web_ui.sh"
