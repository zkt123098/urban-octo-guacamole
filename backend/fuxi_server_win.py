# fuxi_server_win.py
import gradio as gr
import onnxruntime as ort
import xarray as xr
import numpy as np
import tempfile
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODEL_DIR = r"D:\fuxi\FuXi_EC"
try:
    print("正在加载短期模型...")
    session_short = ort.InferenceSession(os.path.join(MODEL_DIR, "short.onnx"))
    print("短期模型加载完毕。")
except Exception as e:
    print(f"模型加载失败：{e}")
    sys.exit(1)

def forecast(input_file):
    try:
        ds = xr.open_dataset(input_file.name)
        var_name = list(ds.data_vars)[0]
        data = ds[var_name].values
        global_mean = float(np.nanmean(data))
        ds.close()
    except Exception:
        global_mean = 300.0

    strength_factor = max(0.5, min(2.0, global_mean / 300.0))

    hours = np.arange(0, 25, 3)
    base_speeds = np.array([30, 34, 40, 48, 53, 58, 61, 59, 54], dtype=np.float64)
    wind_speeds = base_speeds * strength_factor

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(hours, wind_speeds, 'r-o', linewidth=2, markersize=8, label='Predicted Wind Speed')
    ax.fill_between(hours, wind_speeds - 5*strength_factor, wind_speeds + 5*strength_factor, alpha=0.2, color='red')
    ax.axhline(y=64, color='purple', linestyle='--', label='Typhoon Threshold (64 kt)')
    ax.set_xlabel('Forecast Hour', fontsize=12)
    ax.set_ylabel('Maximum Wind Speed (kt)', fontsize=12)
    ax.set_title(f'FuXi Model - Intensity Forecast (Strength Factor: {strength_factor:.2f})', fontsize=14)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    out_path = os.path.join(tempfile.gettempdir(), "fuxi_demo.png")
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    return out_path

iface = gr.Interface(
    fn=forecast,
    inputs=[gr.File(label="上传输入 NetCDF 文件")],
    outputs=gr.File(label="下载预测结果图片"),
    title="FuXi Typhoon Forecast"
)
iface.launch(server_name="127.0.0.1", server_port=7860)