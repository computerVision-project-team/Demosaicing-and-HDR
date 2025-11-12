import sys, os
import numpy as np
import cv2

# === 1. 加入路径 ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '02')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '03')))

# === 2. 导入模块 ===
from demosaic_copy import demosaic_image
from improve_luminosity import improve_luminosity_linear

# === 3. 读 RAW 并做去马赛克 ===
raw_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '02', 'IMG_4782.CR3'))
rgb_linear = demosaic_image(raw_path, pattern="GRBG")

# === 4. 调用第3题亮度增强 ===
rgb_gamma = improve_luminosity_linear(rgb_linear, gamma=0.3)

# === 5. 灰度世界白平衡 ===
def gray_world(rgb):
    means = rgb.mean(axis=(0,1))
    mean_gray = means.mean()
    gains = mean_gray / (means + 1e-9)
    out = rgb * gains
    return np.clip(out, 0, 255)

rgb_balanced = gray_world(rgb_gamma)

# === 6. 导出图像 ===
out = np.clip(rgb_balanced / np.percentile(rgb_balanced, 99.99), 0, 1)
out16 = (out * 65535).astype(np.uint16)
cv2.imwrite("IMG_4782_white_balance.png", cv2.cvtColor(out16, cv2.COLOR_RGB2BGR))

print("己完成")
# 我这块还没做好呢等我改改。。。。
