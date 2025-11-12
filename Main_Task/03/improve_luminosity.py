# === 03/improve_luminosity.py ===
import numpy as np
from pathlib import Path

# 让 Python 找到上一级 02 模块（在 Main_Task 目录下运行可不需要这段）
import sys, os
# 把上一级的 02 文件夹加进搜索路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '02')))
from demosaic_copy import demosaic_image, save_png16

def improve_luminosity_linear(rgb: np.ndarray, gamma=0.3,
                              p_lo=0.01, p_hi=99.99) -> np.ndarray:
    a = np.percentile(rgb, p_lo)
    b = np.percentile(rgb, p_hi)
    x = (rgb - a) / (b - a + 1e-12)
    x = np.clip(x, 0.0, 1.0)
    y = x ** gamma                # 作业建议 γ=0.3
    out = y * (b - a) + a         # 反归一化回原动态范围
    return out


def improve_luminosity_log(rgb: np.ndarray, p_lo=0.01, p_hi=99.99) -> np.ndarray:
    a = np.percentile(rgb, p_lo)
    b = np.percentile(rgb, p_hi)
    x = (rgb - a) / (b - a + 1e-12)
    x = np.clip(x, 0.0, 1.0)
    y = np.log1p(x) / np.log(2.0)  # 另一条曲线：对数压缩
    out = y * (b - a) + a
    return out

if __name__ == "__main__":
    # 构造指向 ../02/IMG_4782.CR3 的绝对路径
    THIS_DIR = os.path.dirname(__file__)
    RAW_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', '02'))
    raw_path = os.path.join(RAW_DIR, 'IMG_4782.CR3')

    print("Reading:", raw_path, "exists:", os.path.exists(raw_path))  # 调试用
    # raw_path = "IMG_4782.CR3"
    pattern = "GRBG"   # ← 用第1题实际判断到的结果替换它
    rgb_linear = demosaic_image(raw_path, pattern)

    # ① γ 曲线版本
    rgb_gamma = improve_luminosity_linear(rgb_linear, gamma=0.3)
    save_png16(rgb_gamma, "IMG_4782_lumi_gamma.png")

    # ② 对数曲线版本（满足第3题“至少再评估一种曲线”的要求）
    rgb_log = improve_luminosity_log(rgb_linear)
    save_png16(rgb_log, "IMG_4782_lumi_log.png")
