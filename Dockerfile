FROM python:3.10-slim

WORKDIR /app

# 复制后端代码
COPY backend/ /app/backend/
# 复制前端页面
COPY frontend/ /app/frontend/

# 安装依赖
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# 暴露端口
EXPOSE 8000

# 使用 shell 脚本同时启动 FastAPI 和伏羲服务（如果模型存在）
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]