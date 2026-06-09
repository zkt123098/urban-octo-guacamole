#!/bin/bash
cd /app/backend

# ===== 自动将前端 API 地址改为容器宿主机的 IP =====
sed -i "s|http://127.0.0.1:8000|http://192.168.18.4:8000|g" /app/frontend/index.html

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