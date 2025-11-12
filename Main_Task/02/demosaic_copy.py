# === 02/demosaic.py ===
import numpy as np
import rawpy


def conv3x3_box(img: np.ndarray) -> np.ndarray:
    # 零填充 1 像素（上下左右各 1）
    p = np.pad(img.astype(np.float64), ((1, 1), (1, 1)), mode="constant")
    # 从 padded 里取 9 个相邻窗口并相加；输出形状与原图一致
    out = (
            p[0:-2, 0:-2] + p[0:-2, 1:-1] + p[0:-2, 2:] +
            p[1:-1, 0:-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
            p[2:, 0:-2] + p[2:, 1:-1] + p[2:, 2:]
    )
    return out

def reconstruct(mask_bool: np.ndarray, X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    M = mask_bool.astype(np.float64)
    num = conv3x3_box(M * X)
    den = conv3x3_box(M) + eps
    return num / den

def bayer_masks(h: int, w: int, pattern: str):
    R = np.zeros((h,w), bool)
    G = np.zeros((h,w), bool)
    B = np.zeros((h,w), bool)
    if pattern == "RGGB":
        R[0::2,0::2]=1; G[0::2,1::2]=1; G[1::2,0::2]=1; B[1::2,1::2]=1
    elif pattern == "BGGR":
        B[0::2,0::2]=1; G[0::2,1::2]=1; G[1::2,0::2]=1; R[1::2,1::2]=1
    elif pattern == "GRBG":
        G[0::2,0::2]=1; R[0::2,1::2]=1; B[1::2,0::2]=1; G[1::2,1::2]=1
    elif pattern == "GBRG":
        G[0::2,0::2]=1; B[0::2,1::2]=1; R[1::2,0::2]=1; G[1::2,1::2]=1
    else:
        raise ValueError(f"Unknown Bayer pattern: {pattern}")
    return R, G, B


def demosaic_image(raw_path: str, pattern: str) -> np.ndarray:
    """
    读取 RAW、按 pattern 重建三通道，返回【线性域】RGB（float64）。
    不做白平衡、不做 gamma、不做归一化。
    """
    raw = rawpy.imread(raw_path)
    X = np.array(raw.raw_image_visible).astype(np.float64)
    h, w = X.shape
    Rm, Gm, Bm = bayer_masks(h, w, pattern)
    R_rec = reconstruct(Rm, X)
    G_rec = reconstruct(Gm, X)
    B_rec = reconstruct(Bm, X)
    rgb = np.stack([R_rec, G_rec, B_rec], axis=-1)
    return rgb


# -------- 工具：保存（只在导出时量化）--------
def save_png16(rgb: np.ndarray, path: str):
    import cv2
    out = np.clip(rgb, 0, None)  # 先不强行上限裁剪，避免信息丢失
    out = np.clip(out / np.percentile(out, 99.99), 0, 1)  # 仅用于可视化
    out16 = (out * 65535 + 0.5).astype(np.uint16)
    cv2.imwrite(path, cv2.cvtColor(out16, cv2.COLOR_RGB2BGR))
