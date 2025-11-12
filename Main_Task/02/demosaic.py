import numpy as np
import rawpy
from scipy.signal import convolve2d
import cv2

def bayer_pattern_from_raw(raw):
    patt = raw.raw_pattern.astype(int)
    desc = raw.color_desc.decode("ascii")
    idx2c = {i: desc[i] for i in range(len(desc))}
    tile = np.vectorize(idx2c.get)(patt)
    return "".join(tile.flatten())  # e.g. RGGB, GRBG, etc.

def make_masks(h, w, pattern):
    masks = {c: np.zeros((h, w), dtype=bool) for c in "RGB"}
    p = np.array([[pattern[0], pattern[1]], [pattern[2], pattern[3]]])
    yy, xx = np.indices((h, w))
    c = p[yy % 2, xx % 2]
    for color in "RGB":
        masks[color][c == color] = True
    return masks

def demosaic_image(path, pattern):
    raw = rawpy.imread(path)
    raw_arr = np.array(raw.raw_image_visible)
    raw_arr = raw_arr.astype(np.float64)
    h, w = raw_arr.shape
    masks = make_masks(h, w, pattern)
    kernel = np.ones((3,3))  # 简单平均核
    eps = 1e-12

    rgb = []
    for color in "RGB":
        M = masks[color].astype(np.float64)
        num = convolve2d(M * raw_arr, kernel, mode="same", boundary="symm")
        den = convolve2d(M, kernel, mode="same", boundary="symm")
        rgb.append(num / (den + eps))
    return np.stack(rgb, axis=-1)

def save_16bit(rgb, path):
    m = float(np.max(rgb)) if np.max(rgb) > 0 else 1.0
    arr = np.clip(rgb / m, 0.0, 1.0)
    arr16 = (arr * 65535.0 + 0.5).astype(np.uint16)
    bgr16 = cv2.cvtColor(arr16, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(path, bgr16)  # 16-bit PNG



if __name__ == '__main__':
    path = "IMG_4782.CR3"
    raw = rawpy.imread(path)
    pattern = bayer_pattern_from_raw(raw)
    print("Detected Bayer pattern:", pattern)

    rgb = demosaic_image(path, pattern)
    save_16bit(rgb, "IMG_4782_demosaic.png")
    print("Saved -> IMG_4782_demosaic.png")
