# ☁️ 云舒 · 气象台风智能问答系统
（期末大作业）
基于 **RAG (检索增强生成)** 的台风与气象智能问答助手，集成 **伏羲气象大模型** 与 **历史相似路径预测**，提供自然语言交互的天气与台风查询、频次预测、强度曲线及路径地图生成。

---

## ✨ 功能预览

| 功能 | 简介 |
|------|------|
| 💬 **智能问答** | 基于气象知识库与历史台风数据库，回答“上海2022年天气”、“台风山竹的路径”等问题。 |
| 📈 **台风预测** | 基于 Prophet 时间序列模型的年度台风频次趋势预测。 |
| 🗺️ **路径预测** | 上传内置坐标的 NetCDF 文件或手动输入轨迹点，生成相似台风路径地图。 | 调用本地部署的伏羲大模型，根据气象初始场文件实时生成台风强度预测曲线。 |

---

## 🛠️ 技术栈

| 层次 | 技术 |
|------|------|
| **前端** | Vue 3 + Element Plus + Axios |
| **后端** | FastAPI + Uvicorn |
| **大语言模型** | DeepSeek-Chat (通过 LangChain 调用) |
| **RAG 框架** | LangChain + Chroma + HuggingFaceEmbeddings (BAAI/bge-small-zh) |
| **关系数据库** | MySQL (存储气象文档、台风轨迹数据) |
| **时间序列预测** | Prophet |
| **相似路径算法** | 基于轨迹形状相似度的历史匹配 (Analogue Method) |
| **AI 气象模型** | 伏羲大模型 (ONNX Runtime 本地推理) |

---

## 🐳 Docker 部署 (Linux 服务器)

### 前置要求
- 安装 Docker 和 Docker Compose
- 获取 DeepSeek API Key（免费注册 https://platform.deepseek.com/）

### 部署步骤

1. **克隆仓库**
   ```bash
   git clone https://github.com/zkt123098/urban-octo-guacamole.git
   cd urban-octo-guacamole
 2. 设置 API Key
编辑 docker-compose.yml，将 DEEPSEEK_API_KEY: your_api_key_here 中的 your_api_key_here 替换为你的真实密钥。

3.准备伏羲模型
如需使用伏羲强度预测，下载 [FuXi_EC.zip](https://pan.baidu.com/s/1w1ov00YhNiucjw9jbS3GNQ 提取码: futy)，解压后将 FuXi_EC 文件夹放在项目根目录（与 docker-compose.yml 同级）。

4.启动所有服务

```bash
docker compose up -d
首次启动会自动构建镜像并拉取 MySQL。MySQL 数据会保存在 ./data/mysql 目录，Chroma 向量库保存在 ./data/chroma。
```
5.导入初始数据
系统需要气象文档和台风轨迹数据才能正常问答。进入容器运行导入脚本：
```bash
docker compose exec yunshu bash
cd /app/backend
python import_data.py               # 导入气象文档
python import_typhoon_data.py       # 导入台风轨迹数据
python train_similarity_model.py    # 生成相似路径特征库
exit
```
6.测试访问
后端健康检查：curl http://<服务器IP>:8000/
前端页面：浏览器打开 http://<服务器IP>:8000/frontend/index.html

7.环境变量说明
所有可配置项均可在 docker-compose.yml 中修改：
DEEPSEEK_API_KEY：你的 DeepSeek 密钥
MYSQL_HOST、MYSQL_USER、MYSQL_PASSWORD、MYSQL_DB：数据库连接信息
FUXI_MODEL_DIR：伏羲模型路径（默认 /app/FuXi_EC）
