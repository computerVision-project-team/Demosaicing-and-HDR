# === 05/process_raw.py ===
import os
import sys
import numpy as np
import cv2
import rawpy

THIS_DIR = os.path.dirname(__file__)
DIR_02 = os.path.abspath(os.path.join(THIS_DIR, '..', '02'))
DIR_03 = os.path.abspath(os.path.join(THIS_DIR, '..', '03'))
DIR_04 = os.path.abspath(os.path.join(THIS_DIR, '..', '04'))
sys.path.append(DIR_02)
sys.path.append(DIR_03)
sys.path.append(DIR_04)


from demosaic import demosaic_image, bayer_pattern_from_raw
from improve_luminosity import improve_luminosity_linear
from white_balance import gray_world


def process_raw(raw_path: str,
                jpg_path: str,
                jpeg_quality: int = 99) -> None:
    """
    pipeline:
      CR3 → demosaic_image (02) → improve_luminosity_linear (03) →
      gray_world (04) → 归一化 → 高质量 JPG
    """

    # --- 读取 RAW ---
    with rawpy.imread(raw_path) as raw:
        pattern = bayer_pattern_from_raw(raw)
    print("Detected Bayer pattern:", pattern)

    # --- 02: 去马赛克得到线性 RGB (float64) ---
    rgb = demosaic_image(raw_path, pattern)    # H x W x 3, float64

    # --- 03: 亮度增强（γ 曲线， gamma=0.3） ---
    rgb = improve_luminosity_linear(rgb, gamma=0.3)

    # --- 04: 灰世界白平衡 ---
    rgb = gray_world(rgb)

    # --- 色温矩阵 ---
    warm = rgb.copy()
    warm[..., 0] *= 1.03  # R 增强，偏暖关键
    warm[..., 2] *= 0.97  # B 减弱，去冷色

    rgb = warm
    # --- 颜色 & 对比度增强 ---
    rgb_01 = (rgb / np.percentile(rgb, 99.5)).astype(np.float32)
    rgb_01 = np.clip(rgb_01, 0.0, 1.0)

    hsv = cv2.cvtColor(rgb_01, cv2.COLOR_RGB2HSV)
    sat_gain = 1.4
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_gain, 0.0, 1.0)
    rgb_sat = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # --- Local contrast (Unsharp Mask) ---
    blur = cv2.GaussianBlur(rgb_sat, (0, 0), sigmaX=3, sigmaY=3)
    detail = rgb_sat - blur

    amount = 0.8  # 局部对比度强度，可调 0.5~2.0
    rgb_local_contrast = np.clip(rgb_sat + amount * detail, 0.0, 1.0)

    # --- 归一化 & 保存 ---
    intensity = rgb_local_contrast.mean(axis=2)
    a = np.percentile(intensity, 0.1)
    b = np.percentile(intensity, 99.99)
    vis = np.clip((rgb_local_contrast - a) / (b - a + 1e-12), 0.0, 1.0)

    out8 = (vis * 255.0 + 0.5).astype(np.uint8)
    out8_bgr = cv2.cvtColor(out8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(jpg_path, out8_bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])



if __name__ == "__main__":
    example_raw = os.path.join(DIR_02, "IMG_4782.CR3")
    process_raw(example_raw, "IMG_4782_process_raw.jpg")
