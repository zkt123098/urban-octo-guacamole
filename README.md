# ☁️ 云舒 · 气象台风智能问答系统

> **期末大作业**  
> 基于 RAG (检索增强生成) 的台风与气象智能问答助手，集成 **伏羲气象大模型** 与 **历史相似路径预测**，提供自然语言交互的天气与台风查询、频次预测、强度曲线及路径地图生成。

---

## ✨ 功能预览

| 功能 | 简介 |
|------|------|
| 💬 **智能问答** | 基于气象知识库与历史台风数据库，回答“上海2022年天气”、“台风山竹的路径”等问题。 |
| 📈 **台风预测** | 基于 Prophet 时间序列模型的年度台风频次趋势预测。 |
| 🗺️ **路径预测** | 上传内置坐标的 NetCDF 文件或手动输入轨迹点，生成相似台风路径地图。 |
| 🌪️ **伏羲强度预测** | 调用本地部署的伏羲大模型，根据气象初始场文件实时生成台风强度预测曲线。 |

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

🐳 Docker 部署教程（Linux 服务器）
一、准备工作
Linux 服务器（如 Ubuntu 20.04/22.04），确保已安装 Docker 和 Docker Compose。

bash
# 安装 Docker（如果未安装）
curl -fsSL https://get.docker.com | bash
sudo usermod -aG docker $USER
newgrp docker
DeepSeek API Key：前往 platform.deepseek.com 注册并获取 sk- 开头的密钥。

（可选）伏羲模型文件：如果需要强度预测功能，下载 FuXi_EC 文件夹并放置在服务器上（本教程默认跳过，不影响核心问答功能）。

（可选）CMA 台风数据：如果需要历史台风查询功能，需下载原始数据文件（.txt）并导入，后面会详细说明。

二、获取项目代码
bash
git clone https://github.com/zkt123098/urban-octo-guacamole.git
cd urban-octo-guacamole
三、配置环境变量
复制模板文件：

bash
cp backend/.env.example backend/.env
编辑 .env，至少填写你的 DeepSeek API Key：

bash
nano backend/.env
将 DEEPSEEK_API_KEY=your_api_key_here 修改为你的真实密钥，例如：

text
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
其他数据库配置保持默认即可。

四、设置后端服务 IP 地址
项目默认使用 192.168.18.4 作为后端 IP，你需要修改为你的 Linux 服务器实际 IP。

查看服务器 IP：

bash
ip addr show | grep 'inet ' | grep -v '127.0.0.1'
假设输出 192.168.1.100，则你的服务器 IP 就是它。

编辑 docker-compose.yml，找到 BACKEND_HOST 那一行，将 IP 修改为你的真实 IP：

yaml
BACKEND_HOST: "192.168.1.100"
五、启动服务
bash
docker compose up -d
首次启动会自动拉取基础镜像并构建应用镜像，需要耐心等待几分钟。完成后会自动启动 MySQL 数据库和云舒后端服务。

六、验证基础运行
bash
curl http://localhost:8000/
如果返回 {"message":"云舒已经准备好啦，来问我天气吧！"}，说明后端启动成功。

浏览器访问 http://你的服务器IP:8000 应该能看到前端聊天界面。

七、导入历史台风数据（可选，但推荐）
如果需要查询真实台风信息，必须导入 CMA 最佳路径数据集。

下载数据文件压缩包（链接: https://pan.baidu.com/s/1kg9LeV_92keS99xlGdBe3g 提取码: rfpm），解压后得到 77 个 .txt 文件和一个 typhoon_names.csv。

将这些文件上传到服务器的 /home/你的用户名/raw_data/ 目录下（可使用 scp、SFTP 等工具）。

进入容器并运行导入脚本：

bash
docker compose exec yunshu bash
cd /app/backend
python import_data.py               # 导入气象文档（100条）
python import_typhoon_data.py       # 导入台风轨迹（1949-2025年）
python train_similarity_model.py    # 生成相似路径特征库
exit
整个过程可能需要 3-5 分钟，请耐心等待。

导入完成后，重建向量索引：

bash
docker compose exec yunshu rm -rf /app/backend/chroma_weather_index
docker compose restart yunshu
八、测试完整功能
在浏览器中强制刷新页面（Ctrl+Shift+R），依次测试：

智能问答：输入“你好”、“上海2022年天气怎么样？”

台风查询：输入“2023年所有台风有哪些？”、“详细介绍一下山竹台风”

台风预测：选择“台风预测”模式，输入“明年台风多吗？”

路径预测：选择“路径预测”模式，上传一个 .nc 文件（或输入6行坐标），按 Enter 生成路径地图和强度曲线。

九、停止与重启
停止服务：docker compose down

重启服务：docker compose restart

更新代码后重建：修改代码后，执行 docker compose build（利用缓存，很快），然后 docker compose up -d --force-recreate

