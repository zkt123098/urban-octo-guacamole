# 使用轻量级 Python 3.10 镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 复制后端代码
COPY backend/ /app/backend/
# 复制前端页面
COPY frontend/ /app/frontend/

# 安装依赖
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]