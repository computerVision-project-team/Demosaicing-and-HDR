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

    # --- 可视化归一化到 [0,1] ---
    intensity = rgb.mean(axis=2)
    a = np.percentile(intensity, 0.01)
    b = np.percentile(intensity, 99.99)
    vis = np.clip((rgb - a) / (b - a + 1e-12), 0.0, 1.0)

    # --- 转成 8-bit 并保存为高质量 JPG ---
    out8 = (vis * 255.0 + 0.5).astype(np.uint8)      # RGB [0,255]
    out8_bgr = cv2.cvtColor(out8, cv2.COLOR_RGB2BGR) # OpenCV 用 BGR

    cv2.imwrite(
        jpg_path,
        out8_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    )
    print(f"Saved JPG (Q={jpeg_quality}) → {jpg_path}")


if __name__ == "__main__":
    example_raw = os.path.join(DIR_02, "IMG_4782.CR3")
    process_raw(example_raw, "IMG_4782_process_raw.jpg")
