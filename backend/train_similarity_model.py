# 训练相似路径特征库的脚本（生成 model_cache/typhoon_profiles.pkl）
import pickle
import mysql.connector
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

def load_all_tracks():
    conn = mysql.connector.connect(
        host=os.getenv("MYSQL_HOST"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DB")
    )
    query = """
    SELECT international_id, obs_time, latitude, longitude, wind_speed, pressure
    FROM typhoon_tracks
    ORDER BY international_id, obs_time
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def build_typhoon_profiles(df):
    profiles = {}
    for tid, group in df.groupby('international_id'):
        if len(group) < 5:
            continue
        # 存储完整轨迹点序列
        track_points = group[['latitude', 'longitude']].values.tolist()
        profiles[tid] = {
            'name_cn': '',  # 稍后从 typhoon_info 补全
            'name_en': '',
            'year': 0,
            'max_wind': float(group['wind_speed'].max()),
            'avg_wind': float(group['wind_speed'].mean()),
            'track_points': track_points,
            'length': len(track_points)
        }
    return profiles

def enrich_names(profiles):
    """从 typhoon_info 表补充中英文名"""
    conn = mysql.connector.connect(
        host=os.getenv("MYSQL_HOST"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DB")
    )
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT international_id, name_cn, name_en, year FROM typhoon_info")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    for row in rows:
        tid = row['international_id']
        if tid in profiles:
            profiles[tid]['name_cn'] = row['name_cn'] if row['name_cn'] and row['name_cn'] != '无名' else row['name_en']
            profiles[tid]['name_en'] = row['name_en']
            profiles[tid]['year'] = row['year']
    return profiles

if __name__ == "__main__":
    print("正在加载所有台风轨迹...")
    df = load_all_tracks()
    print(f"共 {df['international_id'].nunique()} 个台风，{len(df)} 条记录")

    print("正在构建台风特征库...")
    profiles = build_typhoon_profiles(df)
    profiles = enrich_names(profiles)

    os.makedirs("model_cache", exist_ok=True)
    with open("model_cache/typhoon_profiles.pkl", "wb") as f:
        pickle.dump(profiles, f)
    print(f"特征库已保存到 model_cache/typhoon_profiles.pkl，共 {len(profiles)} 个台风。")