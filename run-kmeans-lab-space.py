import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, lab2rgb, deltaE_ciede2000
import matplotlib.pyplot as plt

# === CONFIG ===
IMG_NAME = "IMG_2746"
N_COLORS = 60                # initial K-Means colors
DELTA_E_THRESHOLD = 10       # merge threshold for perceptual similarity
IMG_PATH = f"./images/{IMG_NAME}.png"
OUTPUT_PATH = f"./images/{IMG_NAME}_holds_quantized_{N_COLORS}_LabSpace_del_e_{DELTA_E_THRESHOLD}.png"


# === 1️⃣ LIGHTING NORMALIZATION (optional) ===
def normalize_lighting(img_bgr):
    """Equalize L-channel in LAB space to normalize brightness."""
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    l_eq = cv2.equalizeHist(l)
    img_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(img_eq, cv2.COLOR_LAB2BGR)


img_bgr = cv2.imread(IMG_PATH)
# img_bgr = normalize_lighting(img_bgr)
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


# === SAVE OUTPUT IMAGE ===
Image.fromarray(segmented_rgb).save(OUTPUT_PATH)
print(f"✅ Saved segmented image: {OUTPUT_PATH}")


# === VISUALIZE RESULTS ===
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

ax[0].imshow(img_rgb)
ax[0].set_title("Original (Lighting-Normalized) Image")
ax[0].axis("off")

ax[1].imshow(quantized_rgb)
ax[1].set_title(f"K-Means in Lab Space ({N_COLORS} Colors)")
ax[1].axis("off")

ax[2].imshow(segmented_rgb)
ax[2].set_title("Merged + Labeled Segmentation")
ax[2].axis("off")

plt.tight_layout()
plt.show()

# === SHOW FINAL COLOR PALETTE ===
fig, ax = plt.subplots(figsize=(10, 2))
ax.imshow([merged_palette_rgb])
ax.set_title("Final Merged Color Palette (Lab-based K-Means)")
ax.set_xticks(range(len(merged_palette_rgb)))
ax.set_xticklabels([label_map[i] for i in range(len(merged_palette_rgb))], rotation=45)
ax.set_yticks([])
plt.show()
