## 🐳 Docker 部署 (Linux 服务器)

### 前提条件
- 已安装 Docker 和 Docker Compose。
- 可选：如果需要伏羲大模型预测功能，请下载 `FuXi_EC` 模型文件夹并放置在项目根目录（确保 `FuXi_EC/short.onnx` 存在）。

### 部署步骤

1. **克隆仓库**
   ```bash
   git clone https://github.com/zkt123098/urban-octo-guacamole.git
   cd urban-octo-guacamole
   ```

2. **配置 DeepSeek API Key**
   编辑 `docker-compose.yml`，将 `DEEPSEEK_API_KEY: your_api_key_here` 替换为你的真实 API Key。也可以创建 `.env` 文件并在 `docker-compose.yml` 中通过 `env_file` 引入，但最简单是直接修改。

3. **（可选）放置模型文件**
   如果下载了伏羲模型，将解压后的 `FuXi_EC` 文件夹放到当前目录（即与 `docker-compose.yml` 同级），确保 `./FuXi_EC/short.onnx` 存在。

4. **启动服务**
   ```bash
   docker compose up -d
   ```

   首次启动会自动拉取镜像、构建并运行，同时启动 MySQL 数据库。MySQL 数据会保存在 `./data/mysql` 目录下。

5. **导入初始数据（重要）**
   系统需要气象文档和台风轨迹数据才能正常工作。你可以通过以下两种方式导入：
   - **方式 A（推荐）**：将 CMA 最佳路径数据集 `.txt` 文件放入 `backend/raw_data/`，然后进入容器手动运行导入脚本：
     ```bash
     docker compose exec yunshu bash
     cd /app/backend
     python import_typhoon_data.py
     python import_data.py
     python train_similarity_model.py
     exit
     ```
   - **方式 B**：准备一个 `init.sql` 文件（包含所有 INSERT 语句），放在项目根目录，重启 MySQL 服务时自动导入（需要在 `docker-compose.yml` 中挂载并设置初始化脚本）。

6. **测试访问**
   - 后端健康检查：`http://<服务器IP>:8000/` 应返回 `{"message":"云舒已经准备好啦，来问我天气吧！"}`
   - 前端页面：直接访问 `http://<服务器IP>:8000/frontend/index.html`（或通过 Nginx 等代理）

### 常用命令
- 查看日志：`docker compose logs -f`
- 停止服务：`docker compose down`
- 重启服务：`docker compose restart`
- 进入容器：`docker compose exec yunshu bash`

### 环境变量说明
所有可配置的环境变量均可在 `docker-compose.yml` 中修改，主要包括：
- `DEEPSEEK_API_KEY`：DeepSeek API 密钥
- `MYSQL_HOST`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DB`：数据库连接信息
- `FUXI_MODEL_DIR`：伏羲模型文件所在目录（默认 `/app/FuXi_EC`）
- `ADMIN_SECRET`：管理接口密钥

### Docker 访问地址
服务启动后，你可以通过以下 URL 访问：
- **API 根路径**：`http://<服务器IP>:8000/`
- **前端页面**：`http://<服务器IP>:8000/frontend/index.html`
（若前端需要跨域，已将 CORS 设置为允许所有来源，无需额外配置）