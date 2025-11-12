import numpy as np
import matplotlib.pyplot as plt

array = np.load('IMG_9939.npy')
print('Loaded array of size', array.shape)
print('The pens, from top to bottom, are red, green and blue')

# Store the array slices, not just the mean values
tiles = {}
for i in (0, 1):
    for j in (0, 1):
        tiles[(i, j)] = array[i::2, j::2]  # Store the actual array slice

# Calculate mean values for each phase
tile_means = {}
print("\nBrightness of Image:")
for pos, tile_array in tiles.items():
    tile_means[pos] = tile_array.mean()
    print(f"{pos}: {tile_means[pos]:.2f}")

# Find green channels (highest values)
green_positions = [g[0] for g in sorted(tile_means.items(), key=lambda x: x[1], reverse=True)[:2]]
print("Green positions:", green_positions,"\n")

# Get the ROI by hand
plt.imshow(array, cmap='gray')
plt.axis('on')
plt.show()

roi_red = (slice(1400, 2000), slice(2000, 3000))
roi_blue = (slice(2500, 3000), slice(1800, 2800))


# Calculate mean values in ROIs for each phase
def roi_mean(tile_array, roi):
    rr, cc = roi
    start_row = rr.start // 2
    stop_row = rr.stop // 2
    start_col = cc.start // 2
    stop_col = cc.stop // 2

    return tile_array[start_row:stop_row, start_col:stop_col].mean()


means_red = {k: float(roi_mean(v, roi_red)) for k, v in tiles.items()}
means_blue = {k: float(roi_mean(v, roi_blue)) for k, v in tiles.items()}

print("Brightness of ROI Red:")
for k, v in means_red.items():
    print(f"{k}: {v:.2f}")

print("\nBrightness of ROI Blue:")
for k, v in means_blue.items():
    print(f"{k}: {v:.2f}")

print("\nDetermining Bayer pattern:")

# In red pen area, red channel should be brightest among non-green channels
non_green = [p for p in tiles.keys() if p not in green_positions]
red_position = max(((p, means_red[p]) for p in non_green),  key=lambda x: x[1])[0]
blue_position = max(((p, means_blue[p]) for p in non_green), key=lambda x: x[1])[0]
print(f"Red position: {red_position}")
print(f"Blue position: {blue_position}")

# Final Bayer pattern determination
bayer_pattern = {}
for pos in tiles.keys():
    if pos in green_positions:
        bayer_pattern[pos] = 'G'
    elif pos == red_position:
        bayer_pattern[pos] = 'R'
    else:
        bayer_pattern[pos] = 'B'

print("\nFinal Bayer pattern:")
print(f"Position (0,0): {bayer_pattern[(0, 0)]}")
print(f"Position (0,1): {bayer_pattern[(0, 1)]}")
print(f"Position (1,0): {bayer_pattern[(1, 0)]}")
print(f"Position (1,1): {bayer_pattern[(1, 1)]}")