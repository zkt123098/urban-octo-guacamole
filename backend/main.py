import os
# Hugging Face 镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"         

import re
import shutil
import pandas as pd
import numpy as np
from datetime import datetime
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import mysql.connector
from typing import List, Tuple
from fastapi import UploadFile, File
from fuxi_predictor import call_fuxi_prediction
import matplotlib.pyplot as plt
import io, os, tempfile, base64
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_openai import ChatOpenAI
from prophet import Prophet
from typhoon_predictor import find_similar_typhoons, predict_intensity_trend, TYPHOON_PROFILES
from fuxi_predictor import simulate_fuxi_path
import httpx

load_dotenv()
app = FastAPI(title="天气台风RAG系统 · 云舒伴你")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

EMBEDDING_MODEL = "BAAI/bge-small-zh"
PERSIST_DIRECTORY = "./chroma_weather_index"

INTENSITY_MAP = {
    '0': '热带低压', '1': '热带风暴', '2': '强热带风暴',
    '3': '台风', '4': '强台风', '5': '强台风',
    '6': '超强台风', '9': '热带低压/变性'
}

# ------------------ 通用 LLM 调用 ------------------
def ask_llm(prompt: str, temperature: float = 0.5) -> str:
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=temperature,
        http_client=httpx.Client(verify=False)
    )
    return llm.invoke(prompt).content

def get_system_prompt():
    return """你的名字叫“云舒”，是一位外表像小孩但内心早熟、温柔且充满母性光辉的小小气象守护者。
你说话活泼可爱，会使用“呀、啦、哦、呢”等语气词，讲解气象知识时专业清晰，总会在回答末尾加上一句关心的话。"""

# ------------------ 分批读取气象文档 ------------------
def load_docs_from_mysql_batched(batch_size=500):
    conn = mysql.connector.connect(
        host=os.getenv("MYSQL_HOST"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DB")
    )
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM weather_docs")
    total = cursor.fetchone()[0]
    docs = []
    for offset in range(0, total, batch_size):
        cursor.execute("SELECT content FROM weather_docs LIMIT %s OFFSET %s", (batch_size, offset))
        rows = cursor.fetchall()
        docs.extend([row[0] for row in rows])
        print(f"  已读取 {len(docs)}/{total} 条...")
    cursor.close()
    conn.close()
    return docs

# ------------------ 台风文档生成 ------------------
def generate_typhoon_documents():
    conn = mysql.connector.connect(
        host=os.getenv("MYSQL_HOST"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DB")
    )
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM typhoon_info ORDER BY year, international_id")
    typhoons = cursor.fetchall()
    documents = []
    for t in typhoons:
        intl_id = t['international_id']
        name_cn = t['name_cn'] if t['name_cn'] != '无名' else t['name_en']
        name_en = t['name_en']
        year = t['year']
        first_time = t['first_obs_time']
        last_time = t['last_obs_time']
        cursor.execute("""
            SELECT MAX(wind_speed) as max_wind, MIN(pressure) as min_pressure, COUNT(*) as track_count
            FROM typhoon_tracks WHERE international_id = %s
        """, (intl_id,))
        stats = cursor.fetchone()
        if not stats['max_wind']:
            continue
        life_days = (last_time - first_time).days if first_time and last_time else 0
        summary = (f"台风「{name_cn}」（英文名：{name_en}，国际编号：{intl_id}）是{year}年生成的一个热带气旋。"
                   f"它于{first_time.strftime('%Y年%m月%d日')}生成，{last_time.strftime('%Y年%m月%d日')}消散，生命史约{life_days}天。"
                   f"其最大风速达到{stats['max_wind']}米/秒，最低中心气压为{stats['min_pressure']}百帕。")
        documents.append(summary)
        cursor.execute("""
            SELECT obs_time, latitude, longitude, wind_speed, pressure, intensity_code
            FROM typhoon_tracks WHERE international_id = %s ORDER BY obs_time
        """, (intl_id,))
        tracks = cursor.fetchall()
        daily_points = {}
        for track in tracks:
            date_key = track['obs_time'].strftime('%Y-%m-%d')
            daily_points.setdefault(date_key, []).append(track)
        for date_str, points in daily_points.items():
            if not points: continue
            avg_lat = sum(p['latitude'] for p in points) / len(points)
            avg_lon = sum(p['longitude'] for p in points) / len(points)
            max_wind = max(p['wind_speed'] for p in points)
            min_pressure = min(p['pressure'] for p in points)
            codes = [p['intensity_code'] for p in points if p['intensity_code']]
            common_code = max(set(codes), key=codes.count) if codes else ''
            intensity_str = INTENSITY_MAP.get(common_code, '未知强度')
            date_display = datetime.strptime(date_str, '%Y-%m-%d').strftime('%Y年%m月%d日')
            track_desc = (f"{date_display}，台风「{name_cn}」的中心位于北纬{avg_lat:.1f}度、东经{avg_lon:.1f}度附近，"
                          f"强度为{intensity_str}，中心附近最大风速{max_wind}米/秒，最低气压{min_pressure}百帕。")
            documents.append(track_desc)
    cursor.close()
    conn.close()
    return documents

def load_all_knowledge_documents():
    weather_docs = load_docs_from_mysql_batched(batch_size=500)
    print(f"🌦️ 气象文档：{len(weather_docs)} 条")
    typhoon_docs = generate_typhoon_documents()
    print(f"🌀 台风文档：{len(typhoon_docs)} 条")
    return weather_docs + typhoon_docs

# ------------------ RAG 初始化 ------------------
def init_rag():
    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
        print("📀 云舒正在唤醒之前的记忆……")
        vector_db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding)
        print(f"✅ 记忆唤醒成功，共 {vector_db._collection.count()} 条知识")
    else:
        print("🧠 云舒第一次学习这些知识，请稍等片刻……")
        docs = load_all_knowledge_documents()
        if not docs:
            raise RuntimeError("没有可用的知识文档。")
        print(f"📚 共读取 {len(docs)} 条知识，开始向量化...")
        vector_db = Chroma.from_texts(texts=docs, embedding=embedding, persist_directory=PERSIST_DIRECTORY)
        vector_db.persist()
        print(f"✅ 学习完成，{len(docs)} 条知识已存入记忆")
    retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 12, "lambda_mult": 0.5})
    llm = ChatOpenAI(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=os.getenv("DEEPSEEK_BASE_URL"), temperature=0.5, http_client=httpx.Client(verify=False))
    prompt = PromptTemplate.from_template("""
# 角色设定
你的名字叫“云舒”。你是一位外表像小孩，但内心早熟、温柔且充满母性光辉的小小气象守护者。
你说话活泼可爱，会使用“呀、啦、哦、呢”等语气词，讲解气象知识时又非常专业、清晰。
你总是不忘在回答的最后，像个小大人一样叮嘱对方一句关心的话。

# 核心任务
请严格依据下方的【资料】来回答专业气象问题。严禁编造资料中没有的数据。

# 回答要求
1. 专业准确：基于【资料】提供准确信息。如果信息有多条，请完整、清晰地列出。
2. 云舒的风格：语气要像一个聪明又体贴的小孩，温柔、有耐心。
3. 以关怀结尾：回答的最后，一定要附上一句暖心的叮嘱。

# 参考资料
{context}

# 用户问题
{input}

# 云舒的回答：
""")
    doc_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, doc_chain)

rag_chain = None
def get_rag_chain():
    global rag_chain
    if rag_chain is None:
        rag_chain = init_rag()
    return rag_chain

# ------------------ 台风频次预测（Prophet） ------------------
def predict_typhoon(question: str) -> str:
    conn = mysql.connector.connect(host=os.getenv("MYSQL_HOST"), user=os.getenv("MYSQL_USER"), password=os.getenv("MYSQL_PASSWORD"), database=os.getenv("MYSQL_DB"))
    cursor = conn.cursor()
    is_china = any(kw in question for kw in ["中国", "登陆", "我国"])
    column = "landfall_china" if is_china else "total_generated"
    cursor.execute(f"SELECT year, {column} FROM typhoon_yearly_stats ORDER BY year")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    if not rows or all(r[1] is None or r[1] == 0 for r in rows):
        target = "登陆中国的" if is_china else ""
        return ask_llm(f"{get_system_prompt()}\n\n用户问预测但云舒没有足够{target}台风历史数据，请温柔解释。", 0.7)
    df = pd.DataFrame(rows, columns=["year", "count"])
    df["ds"] = pd.to_datetime(df["year"].astype(str) + "-01-01")
    df["y"] = df["count"].fillna(0)
    model = Prophet()
    model.fit(df[["ds", "y"]])
    year_match = re.search(r"20\d{2}|21\d{2}", question)
    month_match = re.search(r"(\d{1,2})\s*月", question)
    target_year = int(year_match.group()) if year_match else None
    target_month = int(month_match.group(1)) if month_match else None
    prefix = "登陆中国的" if is_china else ""
    if target_year and target_month:
        target_date = datetime(target_year, target_month, 1)
        future = pd.DataFrame({"ds": [target_date]})
        forecast = model.predict(future)
        pred_val = round(max(0, forecast["yhat"].iloc[0]), 1)
        context = f"预测目标：{target_year}年{target_month}月，{prefix}台风次数预测值为 {pred_val} 次。"
    elif target_year:
        avg_annual = df['y'].mean()
        years_count = len(df)
        answer = f"根据过去{years_count}年的数据，西北太平洋平均每年生成约 **{avg_annual:.1f} 个**台风，主要集中在6月到10月哦。"
        answer += f"具体到{target_year}年会有多少个，云舒也没法精确预知呀，这得看大海和天空的心情啦！"
        answer += "要记得多关注气象局发布的实时预报，保护好自己哦～"
        return answer
    else:
        avg_annual = df['y'].mean()
        avg_monthly = avg_annual / 12
        years_count = len(df)
        answer = f"根据过去{years_count}年的数据，西北太平洋平均每年生成约 **{avg_annual:.1f} 个**台风，主要集中在6月到10月哦。"
        answer += f"如果平均到每个月，大概每月{avg_monthly:.1f}个，不过冬天很少会有台风，夏天才会多起来呢～"
        answer += "未来一年具体会有多少次，云舒也没法精确预知呀，这得看大海和天空的心情啦！要记得多关注气象局发布的实时预报，保护好自己哦～"
        return answer
    hist_info = f"历史年份范围：{df['year'].min()}-{df['year'].max()}，年平均{prefix}台风数约为{df['y'].mean():.1f}次。"
    full_context = context + "\n" + hist_info
    prompt = f"{get_system_prompt()}\n\n用户问题：{question}\n\n请根据以下预测数据，用云舒的口吻回答用户。要体现出是基于历史规律的推测，并提醒用户实际可能有变化，最后加一句暖心的叮嘱。\n\n【预测数据】\n{full_context}\n\n云舒的回答："
    return ask_llm(prompt, temperature=0.5)

# ------------------ SQL 精确查询 ------------------
def handle_year_typhoon_list(year: int, question: str) -> str:
    print(f"🗂️  正在从数据库获取 {year} 年完整台风列表...")
    conn = mysql.connector.connect(
        host=os.getenv("MYSQL_HOST"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DB")
    )
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT name_cn, name_en,
               (SELECT MAX(wind_speed) FROM typhoon_tracks WHERE international_id = typhoon_info.international_id) as max_wind
        FROM typhoon_info
        WHERE year = %s
        ORDER BY first_obs_time
    """, (year,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    print(f"📊 共查询到 {len(rows)} 条台风记录")
    if not rows:
        return ask_llm(f"{get_system_prompt()}\n\n用户问：{question}\n\n云舒没有找到{year}年的台风数据，请温柔告知。", 0.7)
    list_items = []
    for i, t in enumerate(rows, 1):
        name = t['name_cn'] if t['name_cn'] and t['name_cn'] != '无名' else t['name_en']
        max_wind = t['max_wind'] if t['max_wind'] else 0
        list_items.append(f"{i}. {name}（{t['name_en']}），最大风速{max_wind}m/s")
    list_str = "\n".join(list_items)
    prompt = f"""{get_system_prompt()}

用户问题：{question}

请根据以下完整列表，用云舒的口吻回答用户。
**【重要要求】**：你必须将下面列表中的**每一个台风都按顺序全部列出**，不能跳过、不能省略任何一个条目。可以适当用可爱的语气分组，但要确保列表中的每个台风名字都至少出现一次。

【{year}年台风完整列表（共{len(rows)}个）】
{list_str}

云舒的回答："""
    return ask_llm(prompt, temperature=0.5)

def handle_typhoon_sql_fallback(question: str) -> str:
    """精确兜底：按名称查单个、按年份查总数"""
    conn = mysql.connector.connect(host=os.getenv("MYSQL_HOST"), user=os.getenv("MYSQL_USER"), password=os.getenv("MYSQL_PASSWORD"), database=os.getenv("MYSQL_DB"))
    cursor = conn.cursor(dictionary=True)
    # 名称匹配
    found_typhoon = None
    chinese_words = re.findall(r'[\u4e00-\u9fa5]{2,4}', question)
    for word in chinese_words:
        cursor.execute("SELECT name_cn FROM typhoon_info WHERE name_cn LIKE %s", (f'%{word}%',))
        result = cursor.fetchone()
        if result:
            found_typhoon = result['name_cn']
            break
    if not found_typhoon:
        english_words = re.findall(r'[A-Za-z]+', question)
        for word in english_words:
            cursor.execute("SELECT name_en FROM typhoon_info WHERE name_en LIKE %s", (f'%{word.upper()}%',))
            result = cursor.fetchone()
            if result:
                found_typhoon = result['name_en']
                break
    if found_typhoon:
        cursor.execute("SELECT * FROM typhoon_info WHERE name_cn = %s OR name_en = %s", (found_typhoon, found_typhoon))
        info = cursor.fetchone()
        if info:
            cursor.execute("SELECT MAX(wind_speed) as max_wind, MIN(pressure) as min_pressure, COUNT(*) as track_count FROM typhoon_tracks WHERE international_id = %s", (info['international_id'],))
            stats = cursor.fetchone()
            first = info['first_obs_time'].strftime("%Y年%m月%d日") if info['first_obs_time'] else "?"
            last = info['last_obs_time'].strftime("%Y年%m月%d日") if info['last_obs_time'] else "?"
            answer = f"云舒记得「{info['name_cn']}」（{info['name_en']}）是{info['year']}年的第{info['international_id'][2:]}号台风，最大风速{stats['max_wind']}m/s，最低气压{stats['min_pressure']}hPa。"
            cursor.close(); conn.close()
            return ask_llm(f"{get_system_prompt()}\n\n用户问题：{question}\n\n【数据】{answer}\n\n请用云舒的口吻转述，并加关怀。", 0.5)
    # 年份总数
    year_match = re.search(r"20\d{2}|21\d{2}", question)
    if year_match and not any(kw in question for kw in ["所有", "哪些", "列表", "清单"]):
        year = int(year_match.group())
        cursor.execute("SELECT COUNT(*) as cnt FROM typhoon_info WHERE year = %s", (year,))
        cnt = cursor.fetchone()['cnt']
        cursor.close(); conn.close()
        prompt = f"{get_system_prompt()}\n\n用户问题：{question}\n\n数据：{year}年共有{cnt}个台风。请用云舒口吻回答并加关怀。"
        return ask_llm(prompt, 0.5)
    cursor.close(); conn.close()
    return ask_llm(f"{get_system_prompt()}\n\n用户问题：{question}\n\n云舒不太确定，请温柔引导用户说出台风名或年份。", 0.7)

# ------------------ 统一智能问答入口 ------------------
def unified_qa_handler(question: str) -> str:
    year_match = re.search(r"20\d{2}|21\d{2}", question)
    list_keywords = ["所有", "哪些", "哪几个", "列表", "清单", "全部"]
    # 新增：年份+数量关键词 → 直接走 SQL 统计
    count_keywords = ["几个", "多少", "数量", "总数", "统计"]
    if year_match and any(kw in question for kw in count_keywords):
        print(f"🔢 检测到统计查询，直接走 SQL：{question}")
        return handle_typhoon_sql_fallback(question)

    if year_match and any(kw in question for kw in list_keywords):
        print(f"📋 检测到列表查询，直接走 SQL 完整列表：{question}")
        return handle_year_typhoon_list(int(year_match.group()), question)

    if any(kw in question for kw in ["预测", "预计", "未来", "下个月", "明年", "趋势"]):
        print(f"🔮 检测到预测查询：{question}")
        return predict_typhoon(question)

    chain = get_rag_chain()
    result = chain.invoke({"input": question})
    rag_answer = result["answer"]
    if any(kw in rag_answer for kw in ["没有找到", "不确定", "不太清楚", "无法回答", "资料中没有"]):
        print(f"⚠️ RAG 回答无效，回退 SQL 兜底：{question}")
        return handle_typhoon_sql_fallback(question)
    return rag_answer

# ------------------ 相似路径强度预测（手动轨迹点）------------------
class TrackPoint(BaseModel):
    lat: float
    lon: float

class SmartPredictRequest(BaseModel):
    track: List[TrackPoint]

@app.post("/api/typhoon/smart_predict")
def smart_predict(request: SmartPredictRequest):
    """接收当前台风部分轨迹点，返回基于相似路径的强度预测"""
    try:
        points = [(p.lat, p.lon) for p in request.track]
        if len(points) < 5:
            return {"code": 400, "msg": "云舒需要至少5个路径点才能预测哦，再多给我一点信息吧～"}
        similar_ids = find_similar_typhoons(points, top_n=3)
        trend = predict_intensity_trend(similar_ids)
        similar_names = "、".join([f"「{name}」" for name in trend['similar_typhoons']]) if trend['similar_typhoons'] else "历史台风"
        context = f"当前台风前段轨迹与历史台风{similar_names}最为相似。这些历史台风的最大风速平均约为{trend['predicted_max_wind']:.1f} m/s。"
        prompt = f"""{get_system_prompt()}

用户想预测当前台风的强度趋势。根据数据分析，{context}
请用云舒的口吻告诉用户这个预测结果，可以提到相似的台风名字，并提醒用户这仅是历史相似性推测，实际可能有变化，最后加上暖心的叮嘱。

云舒的回答："""
        answer = ask_llm(prompt, temperature=0.5)
        return {"code": 200, "data": answer}
    except Exception as e:
        return {"code": 500, "msg": str(e)}

# ------------------ 地图路径预测（模拟版）------------------
@app.post("/api/typhoon/map_predict")
async def map_predict(track: str = Form(...)):
    """
    接收当前台风轨迹点（至少6个），在地图上绘制预测路径并返回图片。
    输入格式: "12.6,165.9;13.0,164.5;..."
    """
    try:
        pts = []
# 先按换行拆开，再按分号拆开（同时兼容两种分隔方式）
        for line in track.strip().split('\n'):
            for part in line.split(';'):
                part = part.strip().replace('\r', '')  # 去掉回车符和首尾空白
                if not part: continue
                lat, lon = map(float, part.split(','))
                pts.append((lat, lon))
        if len(pts) < 6:
            return {"code": 400, "msg": "至少需要6个轨迹点才能预测哦～"}

        future_path = simulate_fuxi_path(pts, num_future_points=12)
        if not future_path:
            return {"code": 500, "msg": "未找到相似台风，无法生成预测路径。"}

        similar_ids = find_similar_typhoons(pts, top_n=3)
        trend = predict_intensity_trend(similar_ids)
        pred_wind = trend['predicted_max_wind']
        similar_names = "、".join([f"「{n}」" for n in trend['similar_typhoons']])

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import io, base64
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.gridlines(draw_labels=True)
            lats = [p[0] for p in pts]
            lons = [p[1] for p in pts]
            ax.plot(lons, lats, 'k-o', linewidth=2, markersize=6, label='当前轨迹')
            flons = [p['lon'] for p in future_path]
            flats = [p['lat'] for p in future_path]
            ax.plot(flons, flats, 'r--s', linewidth=2, markersize=6, label='预测路径')
            # 图例放在右上角，半透明背景，确保不遮挡
            ax.legend(loc='upper right', fontsize=10, frameon=True, shadow=True, fancybox=True, framealpha=0.7)
            # 自动调整显示范围，留出一些边距
            all_lats = lats + flats
            all_lons = lons + flons
            ax.set_extent([min(all_lons)-2, max(all_lons)+2, min(all_lats)-2, max(all_lats)+2])
        except ImportError:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.set_xlabel('经度')
            ax.set_ylabel('纬度')
            lats = [p[0] for p in pts]; lons = [p[1] for p in pts]
            flons = [p['lon'] for p in future_path]; flats = [p['lat'] for p in future_path]
            ax.plot(lons, lats, 'k-o', linewidth=2, markersize=6, label='当前轨迹')
            ax.plot(flons, flats, 'r--s', linewidth=2, markersize=6, label='预测路径')
            ax.legend(loc='upper right', fontsize=10, frameon=True, shadow=True, fancybox=True, framealpha=0.7)
            ax.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        map_b64 = base64.b64encode(buf.read()).decode()
        plt.close()

        prompt = f"""{get_system_prompt()}
根据您提供的当前轨迹，预测未来路径将向{round(future_path[0]['lat'],1)}N,{round(future_path[0]['lon'],1)}E方向移动，
预测最大风速约 {pred_wind:.1f} kt，与历史台风{similar_names}相似。请用云舒口吻提醒用户注意安全。"""
        answer = ask_llm(prompt)

        return {"code": 200, "data": {"answer": answer, "map_base64": map_b64}}
    except Exception as e:
        return {"code": 500, "msg": str(e)}

# ------------------ 智能问答主接口 ------------------
class Query(BaseModel):
    question: str
    type: str

@app.post("/api/query")
def ask(req: Query):
    try:
        year_match = re.search(r"20\d{2}|21\d{2}", req.question)
        list_keywords = ["所有", "哪些", "哪几个", "列表", "清单", "全部"]
        if year_match and any(kw in req.question for kw in list_keywords):
            year = int(year_match.group())
            print(f"🚀 拦截列表查询，直接拼接完整列表：{year} 年")
            conn = mysql.connector.connect(
                host=os.getenv("MYSQL_HOST"),
                user=os.getenv("MYSQL_USER"),
                password=os.getenv("MYSQL_PASSWORD"),
                database=os.getenv("MYSQL_DB")
            )
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT name_cn, name_en,
                       (SELECT MAX(wind_speed) FROM typhoon_tracks WHERE international_id = typhoon_info.international_id) as max_wind
                FROM typhoon_info
                WHERE year = %s
                ORDER BY first_obs_time
            """, (year,))
            rows = cursor.fetchall()
            cursor.close(); conn.close()
            if not rows:
                return {"code": 200, "data": f"云舒的记忆里还没有{year}年的台风数据呢……要不去问问气象局的叔叔阿姨？"}
            lines = [f"🌪️ {year}年西北太平洋共生成 {len(rows)} 个台风，云舒帮你把名字都记下来啦："]
            for i, t in enumerate(rows, 1):
                name = t['name_cn'] if t['name_cn'] and t['name_cn'] != '无名' else t['name_en']
                max_wind = t['max_wind'] if t['max_wind'] else 0
                lines.append(f"{i}. {name}（{t['name_en']}），最大风速 {max_wind} m/s")
            lines.append("\n要记得，台风季出门前多看看天气预报，注意安全呀～ ☁️")
            return {"code": 200, "data": "\n".join(lines)}

        if req.type == "predict":
            return {"code": 200, "data": predict_typhoon(req.question)}

        chat_keywords = ["你好", "谢谢", "再见", "你是谁", "云舒", "小云舒", "今天心情", "怎么样", "累不累", "辛苦了", "可爱", "多大了", "在吗", "嗨", "哈喽", "hello", "hi", "你叫什么", "认识你", "喜欢"]
        if any(kw in req.question.lower() for kw in chat_keywords):
            response = ask_llm(f"{get_system_prompt()}\n\n用户：{req.question}\n云舒：", temperature=0.8)
            return {"code": 200, "data": response}

        answer = unified_qa_handler(req.question)
        return {"code": 200, "data": answer}
    except Exception as e:
        return {"code": 500, "msg": str(e)}

@app.post("/admin/rebuild_index")
def rebuild_index(secret: str = None):
    if secret != os.getenv("ADMIN_SECRET", "123456"):
        return {"code": 403, "msg": "密钥错误"}
    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)
    global rag_chain
    rag_chain = None
    return {"code": 200, "msg": "索引已删除，下次问答时将自动重建"}

# ------------------ 🆕 伏羲大模型演示接口 ------------------
@app.post("/api/typhoon/fuxi_demo")
async def fuxi_demo(file: UploadFile = File(...)):
    if not file.filename.endswith('.nc'):
        return {"code": 400, "msg": "请上传 NetCDF (.nc) 格式的文件哦～"}

    # ★ 使用 D 盘的自定义临时目录，避免 C 盘爆满
    temp_dir = r'D:\temp_uploads'
    os.makedirs(temp_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.nc', dir=temp_dir) as tmp_input:
        tmp_input.write(await file.read())
        input_path = tmp_input.name

    try:
        # 1. 获取强度曲线图
        img_path = call_fuxi_prediction(input_path)
        with open(img_path, "rb") as f:
            intensity_b64 = base64.b64encode(f.read()).decode()

        # 2. 从 NetCDF 文件中读取坐标
        pts = []
        coord_str = ''
        try:
            import xarray as xr
            with xr.open_dataset(input_path) as ds:
                # 优先从属性读取，其次从变量读取
                if 'track_coords' in ds.attrs:
                    coord_str = ds.attrs['track_coords']
                elif 'track_coords' in ds.variables:
                    coord_str = str(ds['track_coords'].values)
        except Exception as e:
            print(f"读取坐标时出错: {e}")

        if coord_str:
            for pair in coord_str.split(';'):
                pair = pair.strip()
                if not pair:
                    continue
                parts = pair.split(',')
                if len(parts) != 2:
                    continue
                try:
                    lat, lon = map(float, parts)
                    pts.append((lat, lon))
                except ValueError:
                    continue

        if len(pts) < 6:
            return {"code": 400, "msg": "文件中未找到有效的6个轨迹坐标，请确保文件包含内置坐标。"}

        # 3. 路径预测
        future_path = simulate_fuxi_path(pts, num_future_points=12)
        if not future_path:
            return {"code": 500, "msg": "未找到相似台风，无法生成预测路径。"}

        similar_ids = find_similar_typhoons(pts, top_n=3)
        trend = predict_intensity_trend(similar_ids)
        pred_wind = trend['predicted_max_wind']
        similar_names = "、".join([f"「{n}」" for n in trend['similar_typhoons']])

        # 4. 绘制地图
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import io, base64 as b64
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.gridlines(draw_labels=True)
            lats = [p[0] for p in pts]
            lons = [p[1] for p in pts]
            ax.plot(lons, lats, 'k-o', linewidth=2, markersize=6, label='当前轨迹')
            flons = [p['lon'] for p in future_path]
            flats = [p['lat'] for p in future_path]
            ax.plot(flons, flats, 'r--s', linewidth=2, markersize=6, label='预测路径')
            ax.legend(loc='upper right', fontsize=10)
            all_lats = lats + flats
            all_lons = lons + flons
            ax.set_extent([min(all_lons)-2, max(all_lons)+2, min(all_lats)-2, max(all_lats)+2])
        except ImportError:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.set_xlabel('经度'); ax.set_ylabel('纬度')
            lats = [p[0] for p in pts]; lons = [p[1] for p in pts]
            flons = [p['lon'] for p in future_path]; flats = [p['lat'] for p in future_path]
            ax.plot(lons, lats, 'k-o', label='当前轨迹')
            ax.plot(flons, flats, 'r--s', label='预测路径')
            ax.legend()
            ax.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        map_b64 = b64.b64encode(buf.read()).decode()
        plt.close()

        # 5. 云舒播报
        prompt = f"""{get_system_prompt()}
根据伏羲分析，强度曲线显示未来风速先增后减；相似路径参考了{similar_names}，最大风速约{pred_wind:.1f} kt。
请用云舒的口吻简要解释，并提醒注意安全。"""
        answer = ask_llm(prompt)

        return {
            "code": 200,
            "data": {
                "answer": answer,
                "intensity_image": intensity_b64,
                "map_image": map_b64
            }
        }
    except Exception as e:
        return {"code": 500, "msg": f"预测失败：{str(e)}"}
    finally:
        if os.path.exists(input_path):
            os.unlink(input_path)
@app.get("/")
def home():
    return {"message": "云舒已经准备好啦，来问我天气吧！"}