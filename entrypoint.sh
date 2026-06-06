#!/bin/bash
cd /app/backend

# 如果模型目录存在且包含 short.onnx，则启动伏羲推理服务
if [ -f "/app/FuXi_EC/short.onnx" ]; then
    echo "Starting FuXi inference service..."
    python fuxi_server_win.py &
else
    echo "FuXi model not found, skipping FuXi service."
fi

# 启动 FastAPI 主服务
echo "Starting Yunshu backend..."
uvicorn main:app --host 0.0.0.0 --port 8000