#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_DIR"

if [ -f "train_web_ui.pid" ]; then
    pid=$(cat "train_web_ui.pid")
    if ps -p "$pid" > /dev/null 2>&1; then
        echo "正在停止 Web UI 服务 (PID: $pid)"
        kill "$pid"
        sleep 2
        # 检查是否成功停止
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "强制停止服务..."
            kill -9 "$pid"
        fi
        echo "服务已停止"
    else
        echo "服务未运行，但存在PID文件，已删除"
        rm "train_web_ui.pid"
    fi
else
    echo "服务未运行（未找到PID文件）"
fi