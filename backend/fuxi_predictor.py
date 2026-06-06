import numpy as np
import os
import shutil
from gradio_client import Client, handle_file

FUXI_SERVER_IP = "127.0.0.1"  

def call_fuxi_prediction(local_nc_path, output_dir="model_cache"):
    """调用伏羲模拟服务，将生成的图片复制到 output_dir，返回图片路径。"""
    client = Client(f"http://{FUXI_SERVER_IP}:7860/")
    result = client.predict(
        input_file=handle_file(local_nc_path),
        api_name="/forecast"
    )
    if isinstance(result, str) and os.path.exists(result):
        dst = os.path.join(output_dir, "fuxi_intensity.png")
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy2(result, dst)
        return dst
    else:
        raise Exception(f"FuXi 服务未返回有效图片，结果：{result}")

def simulate_fuxi_path(current_track, num_future_points=12):
    """模拟伏羲预测路径，复用原有逻辑。"""
    from typhoon_predictor import find_similar_typhoons, TYPHOON_PROFILES
    if len(current_track) < 6:
        return []
    points = [(p[0], p[1]) for p in current_track]
    similar_ids = find_similar_typhoons(points, top_n=1)
    if not similar_ids:
        return []
    sim_id = similar_ids[0]
    sim_track = TYPHOON_PROFILES[sim_id]['track_points']
    sim_max_wind = TYPHOON_PROFILES[sim_id]['max_wind']
    start_lat, start_lon = points[-1]
    best_idx = 0
    best_dist = float('inf')
    for i, (lat, lon) in enumerate(sim_track):
        dist = np.sqrt((lat - start_lat) ** 2 + (lon - start_lon) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    future_sim = sim_track[best_idx:best_idx + num_future_points]
    if len(future_sim) < num_future_points:
        future_sim = sim_track[best_idx:]
    pred_wind = sim_max_wind
    result = []
    for lat, lon in future_sim:
        result.append({'lat': lat, 'lon': lon, 'wind_speed': pred_wind})
    return result