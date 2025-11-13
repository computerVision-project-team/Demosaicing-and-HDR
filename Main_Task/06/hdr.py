
import os, sys, numpy as np, cv2, rawpy

from scipy.signal import convolve2d
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..', '02')))
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..', '04')))
from demosaic import demosaic_image, bayer_pattern_from_raw, make_masks, demosaic_image
from white_balance import gray_world


def demosaic(raw_arr, pattern):
    h, w = raw_arr.shape
    masks = make_masks(h, w, pattern)
    kernel = np.ones((3, 3), np.float64)
    eps = 1e-12
    rgb = []
    for col in "RGB":
        M = masks[col].astype(np.float64)
        num = convolve2d(M * raw_arr, kernel, mode="same", boundary="symm")
        den = convolve2d(M,         kernel, mode="same", boundary="symm")
        rgb.append(num / (den + eps))
    return np.stack(rgb, axis=-1)



# === 读取 raw 图像路径 ==
folder = os.path.dirname(__file__)
files = sorted(
    os.path.join(folder, f)
    for f in os.listdir(folder)
    if f.lower().endswith(".cr3")
)

# === 最亮的图 ===
raw0 = rawpy.imread(files[0])
h = raw0.raw_image_visible.astype(np.float64)
pattern = bayer_pattern_from_raw(raw0)
raw0.close()

# === 剩下的图 ===
for idx, path in enumerate(files[1:], start=1):
    raw = rawpy.imread(path)
    i = raw.raw_image_visible.astype(np.float64)
    raw.close()

    scale = 2 ** idx
    i_scaled = i * scale

    t = 0.8 * h.max()
    mask = (h > t)
    h[mask] = i_scaled[mask]

# === demosaic和白平衡 ===
rgb = demosaic(h, pattern)
rgb = gray_world(rgb)
np.save("HDR_linear.npy", rgb)

# === log tone mapping ===
hdr_log = np.log1p(rgb)

# === 归一化到 [0,255] ===
a = np.percentile(hdr_log, 0.01)
b = np.percentile(hdr_log, 99.99)
ldr = np.clip((hdr_log - a) / (b - a + 1e-12), 0, 1)

out = (ldr * 255).astype(np.uint8)

cv2.imwrite("HDR_result.png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
print("HDR saved → HDR_result.png")