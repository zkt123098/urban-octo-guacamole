#查看特征库的脚本（可看到台风国际编号；中文名；英文名；生成年份； 生命史最大风速；生命史平均风速；完整轨迹点序列；轨迹点总数）
import pickle

with open("model_cache/typhoon_profiles.pkl", "rb") as f:
    profiles = pickle.load(f)

# 查看 ‘数字’的完整内容
print(profiles['2001'])