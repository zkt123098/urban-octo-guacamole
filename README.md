# ☁️ 云舒 · 气象台风智能问答系统

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

## 📦 快速开始 (Docker 部署)

### 1. 克隆仓库
```bash
git clone https://github.com/zkt123098/urban-octo-guacamole.git
cd urban-octo-guacamole
