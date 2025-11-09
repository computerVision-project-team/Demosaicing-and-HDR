import numpy as np
import rawpy
import cv2

# === 1. 导入 RAW 文件 ===
raw = rawpy.imread('IMG_4782.CR3')
array = np.array(raw.raw_image_visible)
print('Loaded RAW shape:', array.shape, 'dtype:', array.dtype)

# === 2. 定义 Bayer 模式 ===
pattern = "GRBG"

# === 3. 根据 Bayer 模式生成掩码 ===
h, w = array.shape
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

# === 4. 定义 3×3 box 卷积函数 ===
def conv3x3_box(img: np.ndarray) -> np.ndarray:
    # 零填充 1 像素（上下左右各 1）
    p = np.pad(img.astype(np.float64), ((1,1),(1,1)), mode="constant")
    # 从 padded 里取 9 个相邻窗口并相加；输出形状与原图一致
    out = (
        p[0:-2, 0:-2] + p[0:-2, 1:-1] + p[0:-2, 2:  ] +
        p[1:-1, 0:-2] + p[1:-1, 1:-1] + p[1:-1, 2:  ] +
        p[2:  , 0:-2] + p[2:  , 1:-1] + p[2:  , 2:  ]
    )
    return out


# === 5. 按讲义公式做权平均重建 ===
X = array.astype(np.float64)  # 原始马赛克
eps = 1e-12

def reconstruct(mask_bool: np.ndarray) -> np.ndarray:
    M = mask_bool.astype(np.float64)
    num = conv3x3_box(M * X)            # (Mc * X) ⊗ K
    den = conv3x3_box(M) + eps          # (Mc) ⊗ K
    return num / den


R_rec = reconstruct(R)
G_rec = reconstruct(G)
B_rec = reconstruct(B)
rgb = np.stack([R_rec, G_rec, B_rec], axis=-1)

# === 6. 简单白平衡 & Gamma ===
def gray_world(rgb):
    means = rgb.mean(axis=(0,1))
    g = means[1]
    gains = g / (means + 1e-9)
    return rgb * gains
rgb = gray_world(rgb)
rgb = np.clip(rgb / np.percentile(rgb,99.5), 0, 1)**(1/2.2)

# === 7. 导出 16-bit 图像（只在此步降精度） ===
out16 = (rgb * 65535 + 0.5).astype(np.uint16)
cv2.imwrite('IMG_4782_demosaic.png', cv2.cvtColor(out16, cv2.COLOR_RGB2BGR))
