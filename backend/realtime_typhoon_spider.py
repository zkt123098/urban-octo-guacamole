# 实时台风爬虫（暂时失效）（后续通过https://dev.qweather.com/获取实时数据）
import requests
import re
import json

def get_current_typhoon_info():
    """
    从中国天气网获取当前活跃台风的基础信息。
    返回一个列表，若无活跃台风或出错则返回空列表。
    """
    url = "https://typhoon.weather.com.cn/gis/typhoon/json/active"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://typhoon.weather.com.cn/"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.encoding = 'utf-8'
        text = resp.text

        # 提取 JSON 部分（去除回调函数包裹）
        match = re.search(r'typhoonCallback\((.*)\)', text, re.S)
        if not match:
            print("无法解析 JSONP 响应")
            return []
        data = json.loads(match.group(1))
        typhoons = data.get('typhoonList', [])
        active = []
        for t in typhoons:
            # state=1 表示活跃
            if t.get('state') == 1:
                active.append({
                    'id': str(t.get('typhoonId', '')),
                    'name_cn': t.get('name', ''),
                    'name_en': t.get('enName', ''),
                    'typhoon_num': str(t.get('num', ''))
                })
        return active
    except Exception as e:
        print(f"爬取台风列表时出错: {e}")
        return []

def get_typhoon_track_points(typhoon_id):
    """
    获取指定台风ID的详细轨迹点（中国天气网接口）。
    """
    url = f"https://typhoon.weather.com.cn/gis/typhoon/json/{typhoon_id}"
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://typhoon.weather.com.cn/"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        text = resp.text
        match = re.search(r'typhoonCallback\((.*)\)', text, re.S)
        if not match:
            return []
        data = json.loads(match.group(1))
        points = data.get('typhoon', {}).get('points', [])
        track = []
        for p in points:
            track.append({
                'time': p.get('time'),
                'lat': float(p.get('lat', 0)),
                'lng': float(p.get('lng', 0)),
                'pressure': int(p.get('pressure', 0)),
                'wind': int(p.get('wind', 0))
            })
        return track
    except Exception as e:
        print(f"获取台风轨迹失败: {e}")
        return []

if __name__ == '__main__':
    current = get_current_typhoon_info()
    if current:
        print("当前活跃台风：")
        for t in current:
            print(f"  - {t['name_cn']} ({t['name_en']})，编号 {t['typhoon_num']}")
            track = get_typhoon_track_points(t['id'])
            print(f"    轨迹点数：{len(track)}")
    else:
        print("当前没有活跃台风。")