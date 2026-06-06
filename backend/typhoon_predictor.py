# 相似路径预测算法模块
import pickle
import numpy as np

# 加载预计算的特征库
with open("model_cache/typhoon_profiles.pkl", "rb") as f:
    TYPHOON_PROFILES = pickle.load(f)

def compute_track_distance(track1, track2):
    """计算两条轨迹的近似距离（均匀采样后平均欧氏距离）"""
    min_len = min(len(track1), len(track2))
    if min_len < 3:
        return float('inf')
    idx1 = np.linspace(0, len(track1)-1, min_len, dtype=int)
    idx2 = np.linspace(0, len(track2)-1, min_len, dtype=int)
    pts1 = np.array([track1[i] for i in idx1])
    pts2 = np.array([track2[i] for i in idx2])
    dist = np.mean(np.sqrt(np.sum((pts1 - pts2)**2, axis=1)))
    return dist

def find_similar_typhoons(current_track_points, top_n=3):
    """
    current_track_points: list of (lat, lon) 当前台风已走过的路径点（至少5个）
    返回最相似的历史台风 ID 列表
    """
    similarities = []
    for tid, profile in TYPHOON_PROFILES.items():
        hist_track = profile['track_points'][:len(current_track_points)]
        if len(hist_track) < 5:
            continue
        dist = compute_track_distance(current_track_points, hist_track)
        similarities.append((tid, dist))
    similarities.sort(key=lambda x: x[1])
    return [sim[0] for sim in similarities[:top_n]]

def predict_intensity_trend(similar_ids):
    """根据相似台风的后续强度变化，预测趋势"""
    trends = []
    for tid in similar_ids:
        if tid not in TYPHOON_PROFILES:
            continue
        profile = TYPHOON_PROFILES[tid]
        trends.append({
            'id': tid,
            'name_cn': profile['name_cn'],
            'max_wind': profile['max_wind']
        })
    if not trends:
        return {'similar_typhoons': [], 'predicted_max_wind': 0, 'advice': '数据不足'}
    avg_max = np.mean([t['max_wind'] for t in trends])
    return {
        'similar_typhoons': [t['name_cn'] for t in trends],
        'predicted_max_wind': round(avg_max, 1),
        'advice': '可能达到强台风级别' if avg_max > 50 else '可能为台风或以下级别'
    }