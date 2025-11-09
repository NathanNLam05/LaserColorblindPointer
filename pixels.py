import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, lab2rgb, deltaE_ciede2000
import matplotlib.pyplot as plt
import webcolors
from collections import deque

# === CONFIG ===
IMG_NAME = "IMG_2746.png"
# IMG_NAME = "AV-ProwCave.jpg"
N_COLORS = 36                # initial K-Means colors
DELTA_E_THRESHOLD = 15       # merge threshold for perceptual similarity
IMG_PATH = fr"./images/{IMG_NAME}"
OUTPUT_PATH = f"./images2/{IMG_NAME}_{N_COLORS}-{DELTA_E_THRESHOLD}-copy.png"


# # === 1️⃣ LIGHTING NORMALIZATION (optional) ===
# def normalize_lighting(img_bgr):
#     """Equalize L-channel in LAB space to normalize brightness."""
#     img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(img_lab)
#     l_eq = cv2.equalizeHist(l)
#     img_eq = cv2.merge((l_eq, a, b))
#     return cv2.cvtColor(img_eq, cv2.COLOR_LAB2BGR)


img_bgr = cv2.imread(IMG_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h, w, _ = img_rgb.shape

# === 2️⃣ COLOR EXTRACTION (K-Means in Lab / ΔE space) ===
pixels_lab = rgb2lab(img_rgb.reshape(-1, 1, 3) / 255.0)[:, 0, :]
print(f"Running KMeans on {len(pixels_lab):,} pixels in Lab space...")

kmeans = KMeans(n_clusters=N_COLORS, random_state=42, n_init="auto")
kmeans.fit(pixels_lab)

palette_lab = kmeans.cluster_centers_
labels = kmeans.labels_

# Convert cluster centers back to RGB for visualization
palette_rgb = (lab2rgb(palette_lab[np.newaxis, :, :])[0] * 255).astype(np.uint8)
quantized_lab = palette_lab[labels].reshape(h, w, 3)
quantized_rgb = (lab2rgb(quantized_lab) * 255).astype(np.uint8)

# === 3️⃣ COLOR MERGING (ΔE in Lab space) ===
merged_palette_lab = []
used = np.zeros(len(palette_lab), dtype=bool)

for i, c1 in enumerate(palette_lab):
    if used[i]:
        continue
    group = [i]
    for j, c2 in enumerate(palette_lab):
        if i != j and not used[j]:
            if deltaE_ciede2000(c1, c2) < DELTA_E_THRESHOLD:
                group.append(j)
                used[j] = True
    used[i] = True
    avg_lab = np.mean(palette_lab[group], axis=0)
    merged_palette_lab.append(avg_lab)

merged_palette_lab = np.array(merged_palette_lab)
merged_palette_rgb = (lab2rgb(merged_palette_lab[np.newaxis, :, :])[0] * 255).astype(np.uint8)

print(f"Merged {len(palette_lab)} → {len(merged_palette_lab)} colors.")


# === 4️⃣ LABEL ASSIGNMENT (Manual mapping) ===
LABELS = [
    "Red", "Yellow", "Green", "Blue",
    "Purple", "Orange", "Pink", "Brown",
    "Gray", "Black", "White", "Other"
]
label_map = {i: LABELS[i % len(LABELS)] for i in range(len(merged_palette_lab))}


# === 5️⃣ SEGMENTATION (Cluster-based masking in Lab space) ===
dists = np.linalg.norm(pixels_lab[:, None, :] - merged_palette_lab[None, :, :], axis=2)
closest_idx = np.argmin(dists, axis=1)
segmented_lab = merged_palette_lab[closest_idx].reshape(h, w, 3)
segmented_rgb = (lab2rgb(segmented_lab) * 255).astype(np.uint8)


##### Get colors
def closest_colour(requested_colour):
    distances = {}
    for name in webcolors.names():
        r_c, g_c, b_c = webcolors.name_to_rgb(name)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        distances[name] = rd + gd + bd
    return min(distances, key=distances.get)

def get_colour_name(requested_colour):
    try:
        closest_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
    return closest_name

# Segmented_rgb is the rgb val for each pixel
color_dict = dict()
analyzed_colors = dict()
for r in range(len(segmented_rgb)):
    if r%100 == 0: print(r)
    for c in range(len(segmented_rgb[r])):
        color = tuple([int(col) for col in segmented_rgb[r][c]])
        if color not in color_dict:
            color_name = get_colour_name(color)
            color_dict[color] = color_name

        # color_name = color_dict[color]
        # color = webcolors.name_to_rgb(color_name)
        # segmented_rgb[r][c] = color


print(color_dict)
color_counts = dict()

def find_clusters_midpoints(arr, connectivity=4):
    n, m = arr.shape[:2]
    visited = np.zeros((n, m), dtype=bool)
    midpoints = []

    # directions for 4- or 8-connectivity
    if connectivity == 4:
        directions = [(1,0), (-1,0), (0,1), (0,-1)]
    else:
        directions = [(dr, dc) for dr in (-1,0,1) for dc in (-1,0,1) if not (dr==0 and dc==0)]

    for r in range(n):
        for c in range(m):
            if visited[r, c]:
                continue
            color = tuple(arr[r, c])
            # BFS to find all pixels of the same cluster
            q = deque([(r, c)])
            cluster_pixels = []
            visited[r, c] = True
            while q:
                x, y = q.popleft()
                cluster_pixels.append((x, y))
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < n and 0 <= ny < m and not visited[nx, ny]:
                        if tuple(arr[nx, ny]) == color:
                            visited[nx, ny] = True
                            q.append((nx, ny))
            # compute midpoint
            xs, ys = zip(*cluster_pixels)
            mid_r, mid_c = np.mean(xs), np.mean(ys)
            if len(cluster_pixels) > 7:
                color_name = get_colour_name(color)
                midpoints.append({
                    "color": [int(c) for c in color],
                    "color_name": get_colour_name(color),
                    "midpoint": (float(mid_r), float(mid_c)),
                    "size": len(cluster_pixels)
                })
                if color_name in color_counts:
                    color_counts[color_name] += 1
                else:
                    color_counts[color_name] = 1
    return midpoints
midpoints = find_clusters_midpoints(segmented_rgb)
for c in midpoints:
    print(c)

# === SAVE OUTPUT IMAGE ===
# Image.fromarray(segmented_rgb).save(OUTPUT_PATH)
# print(f"✅ Saved segmented image: {OUTPUT_PATH}")


# === VISUALIZE RESULTS ===
fig, ax = plt.subplots(1, 2, figsize=(18, 6))

ax[0].imshow(img_rgb)
ax[0].set_title("Original (Lighting-Normalized) Image")
ax[0].axis("off")

# ax[1].imshow(quantized_rgb)
# ax[1].set_title(f"K-Means in Lab Space ({N_COLORS} Colors)")
# ax[1].axis("off")

ax[1].imshow(segmented_rgb)
ax[1].set_title("Merged + Labeled Segmentation")
ax[1].axis("off")

bad_colors = [()]
for mp in midpoints:
    y, x = mp["midpoint"]     # note: imshow swaps axes
    color_name = mp["color_name"]
    if color_counts[color_name] < 100:
        # ax[0].plot(x, y, 'wo', color=tuple([c/255.0 for c in mp["color"]]), markersize=6, markeredgecolor='red', markeredgewidth=1.5)
        ax[0].plot(x, y, 'wo', color=tuple([c/255.0 for c in mp["color"]]), markersize=6)
    # Optional: label with color or cluster id

plt.tight_layout()
plt.show()

print(color_counts)