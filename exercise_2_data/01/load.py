import numpy as np

array = np.load('IMG_9939.npy')
print('Loaded array of size', array.shape)
print('The pens, from top to bottom, are red, green and blue')

# Average brightness of the four phases
means = {}
for i in (0,1):
    for j in (0,1):
        means[(i, j)] = array[i::2, j::2].mean()
for k, v in means.items():
    print(f"{k}: {float(v):.2f}")

# Find green
greens = sorted(means.items(), key=lambda x: x[1], reverse=True)[:2]
print("Green:", [g[0] for g in greens])

