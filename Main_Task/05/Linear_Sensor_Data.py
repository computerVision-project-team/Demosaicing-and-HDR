import rawpy, numpy as np, matplotlib.pyplot as plt

# 曝光时间对应列表（秒）
times = [1/10, 1/20, 1/40, 1/80, 1/160, 1/320]
files = [f"IMG_{i}.CR3" for i in range(3044, 3050)]
means = []
for f in files:
    raw = rawpy.imread(f)
    arr = np.array(raw.raw_image_visible, dtype=np.float64)
    black = np.mean(raw.black_level_per_channel)
    white = getattr(raw, "white_level", np.max(arr))
    valid = np.clip(arr - black, 0, white - black)
    means.append(np.mean(valid))
    raw.close()

plt.plot(times, means, 'o-', label='average RAW')
plt.xlabel('Exposure time (s)')
plt.ylabel('Mean raw value (minus black level)')
plt.title('Linearity check of RAW sensor response')
plt.legend()
plt.grid()
plt.show()
