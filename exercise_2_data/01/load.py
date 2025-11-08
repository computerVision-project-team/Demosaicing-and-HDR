import numpy as np

array = np.load('IMG_9939.npy')
print('Loaded array of size', array.shape)
print('The pens, from top to bottom, are red, green and blue')

# 统计 2×2 四个相位的平均亮度
means = {}
for i in (0,1):
    for j in (0,1):
        means[(i, j)] = array[i::2, j::2].mean()
for k, v in means.items():
    print(f"{k}: {float(v):.2f}")