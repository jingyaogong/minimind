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
pid=$!
echo $pid > "train_web_ui.pid"

# 检查进程是否成功启动
if ! ps -p "$pid" > /dev/null 2>&1; then
    echo "错误: Web UI 服务启动失败，进程已不存在"
    # 显示日志文件的最后几行以提供错误信息
    if [ -f "$LOG_FILE" ]; then
        echo "错误日志信息:"
        tail -n 20 "$LOG_FILE"
    fi
    rm -f "train_web_ui.pid"
    exit 1
fi

# 轮询日志以获取实际端口号并检查是否有错误（最多等待10秒）
PORT=""
START_TIME=$(date +%s)
MAX_WAIT_TIME=10
SUCCESS=false

while [ $(( $(date +%s) - $START_TIME )) -lt $MAX_WAIT_TIME ]; do
    # 检查是否有错误信息
    if grep -i "error\|exception\|failed\|traceback" "$LOG_FILE" > /dev/null 2>&1; then
        echo "错误: Web UI 服务启动过程中出现错误"
        echo "错误日志信息:"
        grep -i "error\|exception\|failed\|traceback" "$LOG_FILE"
        tail -n 20 "$LOG_FILE"
        echo "请查看完整日志文件了解详细错误信息: $LOG_FILE"
        rm -f "train_web_ui.pid"
        kill $pid > /dev/null 2>&1
        exit 1
    fi
    
    # 提取形如 http://0.0.0.0:12345 的地址，再截取端口
    PORT=$(grep -Eo 'http://0\.0\.0\.0:[0-9]+' "$LOG_FILE" | tail -n1 | awk -F: '{print $NF}')
    if [ -n "$PORT" ]; then
        SUCCESS=true
        break
    fi
    
    # 再次检查进程是否还在运行
    if ! ps -p "$pid" > /dev/null 2>&1; then
        echo "错误: Web UI 服务启动失败，进程已退出"
        echo "错误日志信息:"
        tail -n 20 "$LOG_FILE"
        rm -f "train_web_ui.pid"
        exit 1
    fi
    
    sleep 0.5
done

# 如果超时且没有找到端口
if [ "$SUCCESS" = false ]; then
    # 再次检查进程是否还在运行
    if ps -p "$pid" > /dev/null 2>&1; then
        echo "警告: 无法在指定时间内确认服务是否完全启动"
        # 如果仍未获取到端口，回退为默认提示端口（与后端起始端口一致）
        PORT=12581
        echo "服务可能已启动，但未能确认。请检查日志文件: $LOG_FILE"
    else
        echo "错误: Web UI 服务启动超时且进程已退出"
        echo "错误日志信息:"
        tail -n 20 "$LOG_FILE"
        rm -f "train_web_ui.pid"
        exit 1
    fi
fi

echo "服务已成功启动! PID: $pid"
echo "访问地址: http://localhost:$PORT"
echo "停止命令: kill $pid or bash trainer_web/stop_web_ui.sh"
