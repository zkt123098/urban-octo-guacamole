# create_typhoon_1.py —— 模拟台风“鲇鱼”(2022) 早期轨迹
import numpy as np
import xarray as xr
import os

# ===================== 坐标参数（2022年鲇鱼早期6个点） =====================
COORDS = [
    (6.5, 148.0),
    (7.0, 146.8),
    (7.6, 145.5),
    (8.2, 144.0),
    (8.8, 142.5),
    (9.4, 141.0)
]
TYPHOON_NAME = "鲇鱼(2022)"

# ===================== 生成 NetCDF 数据 =====================
lat = np.linspace(90, -90, 721, dtype=np.float32)
lon = np.linspace(0, 359.75, 1440, dtype=np.float32)
time_steps = [0, 6]

variable_names = [
    'Z50','Z100','Z150','Z200','Z250','Z300','Z400','Z500','Z600',
    'Z700','Z850','Z925','Z1000',
    'T50','T100','T150','T200','T250','T300','T400','T500','T600',
    'T700','T850','T925','T1000',
    'U50','U100','U150','U200','U250','U300','U400','U500','U600',
    'U700','U850','U925','U1000',
    'V50','V100','V150','V200','V250','V300','V400','V500','V600',
    'V700','V850','V925','V1000',
    'R50','R100','R150','R200','R250','R300','R400','R500','R600',
    'R700','R850','R925','R1000',
    'T2M','U10','V10','MSL','TP'
]

np.random.seed(42)  # 固定种子，保证可复现
data_array = np.zeros((2, len(variable_names), 721, 1440), dtype=np.float32)

for i, var in enumerate(variable_names):
    if var.startswith('Z'):    base, scale = 50000, 5000
    elif var.startswith('T'):  base, scale = 250, 20
    elif var.startswith('U') or var.startswith('V'): base, scale = 0, 10
    elif var.startswith('R'):  base, scale = 50, 20
    elif var == 'T2M':         base, scale = 290, 10
    elif var in ('U10','V10'): base, scale = 0, 8
    elif var == 'MSL':         base, scale = 101325, 2000
    elif var == 'TP':          base, scale = 0, 0.01
    else:                      base, scale = 0, 1
    
    field = np.random.randn(721, 1440) * scale + base
    field = np.clip(field, base-4*scale, base+4*scale)
    data_array[0,i] = field.astype(np.float32)
    data_array[1,i] = (field + np.random.randn(721,1440)*scale*0.02).astype(np.float32)

if 'TP' in variable_names:
    tp_idx = variable_names.index('TP')
    data_array[:,tp_idx] = np.abs(data_array[:,tp_idx])

ds = xr.Dataset(
    {'input': (['time','variable','lat','lon'], data_array)},
    coords={'time': time_steps, 'variable': variable_names, 'lat': lat, 'lon': lon}
)

nc_file = 'typhoon_1_meigui.nc'
ds.to_netcdf(nc_file)
print(f"✅ 已生成 {nc_file}")

# 同时生成坐标文件
coord_file = 'typhoon_1_coords.txt'
with open(coord_file, 'w') as f:
    for lat, lon in COORDS:
        f.write(f"{lat},{lon}\n")
print(f"✅ 已生成 {coord_file}（{TYPHOON_NAME}坐标）")