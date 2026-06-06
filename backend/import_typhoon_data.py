# 导入台风数据的脚本
import os
import re
import csv
import mysql.connector
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ------------------ 加载中英文映射表 ------------------
def load_name_mapping(csv_path='raw_data/typhoon_names.csv'):   # 修改路径
    mapping = {}
    if not os.path.exists(csv_path):
        print(f"警告：未找到 {csv_path}，将使用英文名作为中文名。")
        return mapping
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row['name_en'].upper()] = row['name_cn']
    return mapping

NAME_MAP = load_name_mapping()

def get_chinese_name(en_name):
    if en_name.upper() == '(NAMELESS)' or en_name.upper() == 'NAMELESS':
        return '无名'
    return NAME_MAP.get(en_name.upper(), en_name)

# ------------------ 解析 CMA 最佳路径文件 ------------------
def parse_cma_bst_file(filepath):
    typhoons_dict = {}
    tracks = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('#'):
            i += 1
            continue

        # 头记录匹配
        match = re.match(r'^66666\s+\d{4}\s+\d+\s+\d{4}\s+(\d{4})\s+\d\s+\d\s+([A-Za-z\(\)]+)', line)
        if match:
            intl_id = match.group(1).strip()
            name_raw = match.group(2).strip()
            name_en = name_raw if name_raw != '(nameless)' else 'Nameless'
            name_cn = get_chinese_name(name_en)
            year = 2000 + int(intl_id[:2]) if int(intl_id[:2]) < 50 else 1900 + int(intl_id[:2])

            first_time = None
            last_time = None
            points = []

            i += 1
            while i < len(lines):
                data_line = lines[i].strip()
                if not data_line:
                    i += 1
                    continue
                if data_line.startswith('66666'):
                    break

                parts = data_line.split()
                if len(parts) >= 6:
                    time_str = parts[0]
                    obs_time = datetime.strptime(time_str, '%Y%m%d%H')
                    intensity_code = parts[1]
                    lat = int(parts[2]) / 10.0
                    lng = int(parts[3]) / 10.0
                    pressure = int(parts[4])
                    wind = int(parts[5])

                    if first_time is None:
                        first_time = obs_time
                    last_time = obs_time

                    points.append({
                        'obs_time': obs_time,
                        'intensity_code': intensity_code,
                        'latitude': lat,
                        'longitude': lng,
                        'pressure': pressure,
                        'wind_speed': wind
                    })
                i += 1

            # 只保留有效编号且至少有一个轨迹点
            if points and intl_id != '0000':
                if intl_id in typhoons_dict:
                    typhoons_dict[intl_id]['last_obs_time'] = max(
                        typhoons_dict[intl_id]['last_obs_time'], last_time
                    )
                    typhoons_dict[intl_id]['first_obs_time'] = min(
                        typhoons_dict[intl_id]['first_obs_time'], first_time
                    )
                else:
                    typhoons_dict[intl_id] = {
                        'international_id': intl_id,
                        'name_en': name_en,
                        'name_cn': name_cn,
                        'year': year,
                        'first_obs_time': first_time,
                        'last_obs_time': last_time
                    }
                for p in points:
                    p['international_id'] = intl_id
                    tracks.append(p)
            continue
        else:
            i += 1

    return list(typhoons_dict.values()), tracks

# ------------------ 导入所有数据到 MySQL ------------------
def import_all_data():
    conn = mysql.connector.connect(
        host=os.getenv("MYSQL_HOST"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DB")
    )
    cursor = conn.cursor()

    print("清空旧数据...")
    cursor.execute("SET FOREIGN_KEY_CHECKS=0")
    cursor.execute("TRUNCATE TABLE typhoon_tracks")
    cursor.execute("TRUNCATE TABLE typhoon_info")
    cursor.execute("TRUNCATE TABLE typhoon_yearly_stats")
    cursor.execute("SET FOREIGN_KEY_CHECKS=1")
    conn.commit()

 
    data_dir = 'raw_data'
    if not os.path.exists(data_dir):
        print(f"错误：数据目录 {data_dir} 不存在！")
        return
    txt_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith('CH') and f.endswith('.txt')]
    total_typhoons = 0
    total_tracks = 0

    for filepath in txt_files:
        filename = os.path.basename(filepath)
        print(f"正在处理 {filename}...")
        typhoons, tracks = parse_cma_bst_file(filepath)

        for t in typhoons:
            cursor.execute("""
                INSERT IGNORE INTO typhoon_info (international_id, name_en, name_cn, year, first_obs_time, last_obs_time)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (t['international_id'], t['name_en'], t['name_cn'], t['year'], t['first_obs_time'], t['last_obs_time']))

        track_values = [(t['international_id'], t['obs_time'], t['intensity_code'],
                         t['latitude'], t['longitude'], t['pressure'], t['wind_speed']) for t in tracks]
        cursor.executemany("""
            INSERT INTO typhoon_tracks (international_id, obs_time, intensity_code, latitude, longitude, pressure, wind_speed)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, track_values)

        conn.commit()
        total_typhoons += len(typhoons)
        total_tracks += len(tracks)
        print(f"  -> 导入 {len(typhoons)} 个台风，{len(tracks)} 条路径")

    # 生成年度统计
    print("生成年度统计数据...")
    cursor.execute("""
        INSERT INTO typhoon_yearly_stats (year, total_generated)
        SELECT year, COUNT(*) FROM typhoon_info GROUP BY year
        ON DUPLICATE KEY UPDATE total_generated = VALUES(total_generated)
    """)
    conn.commit()

    cursor.close()
    conn.close()
    print(f"\n✅ 全部完成！共导入 {total_typhoons} 个台风，{total_tracks} 条路径记录。")

if __name__ == "__main__":
    import_all_data()