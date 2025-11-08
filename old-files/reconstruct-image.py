from PIL import Image
import numpy as np

# --- Load image ---
fp = "./images/IMG_2746.png"
test = Image.open(fp).convert("RGB")  # ensure RGB
print("Processing an image...")
print(f"Mode: {test.mode}")
print(f"Size: {test.size}")

# --- Define colors ---
def make_color(rgb, long_name, short_name):
    return {"rgb": np.array(rgb, dtype=float), "long_name": long_name, "short_name": short_name}

# colors = [
#     make_color((0, 0, 0), "Black", "0"),
#     make_color((255, 255, 255), "White", "1"),
#     make_color((97, 96, 92), "Wall", "2"),
#     make_color((141, 115, 64), "Yellow", "Y"),
#     make_color((126, 82, 95), "Pink", "P"),
#     make_color((75, 88, 45), "Green", "G"),
# ]

colors = [
    make_color((0, 0, 0),       "Black",        "0"),
    make_color((255, 255, 255), "White",        "1"),
    make_color((97, 96, 92),    "Wall",         "2"),
    make_color((142, 123, 93),  "Tan",          "3"),
    make_color((141, 115, 64),  "Yellow",       "Y"),
    make_color((126, 82, 95),   "Pink",         "p"),
    make_color((75, 88, 45),    "Lime Green",   "g"),
    make_color((143, 139, 127), "Light Wall",   "3"),
    make_color((43, 57, 32),    "Dark Green",   "G"),
    make_color((47, 60, 68),    "Blue",         "B"),
    make_color((49, 42, 50),    "Purple",       "P"),
    make_color((123, 69, 67),   "Orange",       "O"),
    make_color((91, 56, 60),    "Red",          "R"),
]


# --- Convert to numpy array ---
pixels = np.array(test)  # shape = (height, width, 3)
h, w, _ = pixels.shape

# --- Prepare color reference arrays ---
palette = np.stack([c["rgb"] for c in colors])  # (n_colors, 3)

# --- Vectorized nearest-color computation ---
flat_pixels = pixels.reshape(-1, 3).astype(float)  # (n_pixels, 3)
dists = np.sum((flat_pixels[:, None, :] - palette[None, :, :]) ** 2, axis=2)
closest_idx = np.argmin(dists, axis=1)

# --- Reconstruct quantized image ---
quantized_pixels = palette[closest_idx].astype(np.uint8)
quantized_img = quantized_pixels.reshape(h, w, 3)

# --- Save new image ---
output_image = Image.fromarray(quantized_img, mode="RGB")
output_path = "./images/krishlookspants.png"
output_image.save(output_path)

print(f"✅ New color-quantized image saved to: {output_path}")





# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # Enables 3D plotting
# import numpy as np
# from PIL import Image

# # --- Load image ---
# fp = "./images/IMG_2748.png"
# img = Image.open(fp).convert("RGB")

# # --- Optionally resize for faster plotting ---
# img = img.resize((150, 150))  # smaller = fewer points, adjust as needed
# pixels = np.array(img).reshape(-1, 3).astype(float)

# # --- Your hard-coded palette ---
# colors = [
#     {"rgb": (0, 0, 0), "long_name": "Black", "short_name": "0"},
#     {"rgb": (255, 255, 255), "long_name": "White", "short_name": "1"},
#     {"rgb": (97, 96, 92), "long_name": "Wall", "short_name": "2"},
#     {"rgb": (141, 115, 64), "long_name": "Yellow", "short_name": "Y"},
#     {"rgb": (126, 82, 95), "long_name": "Pink", "short_name": "P"},
#     {"rgb": (75, 88, 45), "long_name": "Green", "short_name": "G"},
# ]

# palette = np.array([c["rgb"] for c in colors], dtype=float)
# labels = [c["long_name"] for c in colors]

# # --- Create 3D plot ---
# fig = plt.figure(figsize=(9, 7))
# ax = fig.add_subplot(111, projection='3d')
# ax.set_title("Image Pixels + Defined Palette in RGB Space", fontsize=13, pad=20)

# # --- Plot all image pixels (faint background) ---
# ax.scatter(
#     pixels[:, 0],
#     pixels[:, 1],
#     pixels[:, 2],
#     c=pixels / 255.0,       # actual pixel colors
#     s=3,                    # small dot size
#     alpha=0.07,             # transparency
#     linewidths=0,
# )

# # --- Plot your palette colors (bold points) ---
# ax.scatter(
#     palette[:, 0],
#     palette[:, 1],
#     palette[:, 2],
#     c= palette / 255.0,
#     s=200,                  # larger
#     edgecolor="k",
#     linewidth=0.8
# )

# # --- Label palette colors ---
# for (x, y, z), label in zip(palette, labels):
#     ax.text(x, y, z, label, fontsize=9, ha='center', va='bottom', weight='bold')

# # --- Axis setup ---
# ax.set_xlabel("Red", labelpad=10)
# ax.set_ylabel("Green", labelpad=10)
# ax.set_zlabel("Blue", labelpad=10)
# ax.set_xlim(0, 255)
# ax.set_ylim(0, 255)
# ax.set_zlim(0, 255)

# # Clean visual tweaks
# ax.xaxis.pane.fill = False
# ax.yaxis.pane.fill = False
# ax.zaxis.pane.fill = False
# ax.grid(False)

# plt.tight_layout()
# plt.show()


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PIL import Image

# --- Load image ---
fp = "./images/IMG_2748.png"
img = Image.open(fp).convert("RGB")

# Resize for performance
img = img.resize((150, 150))
pixels = np.array(img).reshape(-1, 3).astype(float)

# --- Brightness boost function ---
def brighten(arr):
    return np.clip(arr * 3, 0, 255)

# --- Your hard-coded palette ---
colors = [
    {"rgb": (0, 0, 0), "long_name": "Black", "short_name": "0"},
    {"rgb": (255, 255, 255), "long_name": "White", "short_name": "1"},
    {"rgb": (97, 96, 92), "long_name": "Wall", "short_name": "2"},
    {"rgb": (141, 115, 64), "long_name": "Yellow", "short_name": "Y"},
    {"rgb": (126, 82, 95), "long_name": "Pink", "short_name": "P"},
    {"rgb": (75, 88, 45), "long_name": "Green", "short_name": "G"},
]

palette = np.array([c["rgb"] for c in colors], dtype=float)
labels = [c["long_name"] for c in colors]

# --- Apply brightness boost ---
pixels_bright = brighten(pixels)
palette_bright = brighten(palette)

# --- Create 3D plot ---
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Brightened Image Pixels + Palette in RGB Space", fontsize=13, pad=20)

# --- Plot all image pixels (faint background) ---
ax.scatter(
    pixels_bright[:, 0],
    pixels_bright[:, 1],
    pixels_bright[:, 2],
    c=pixels_bright / 255.0,
    s=3,
    alpha=0.7,
    linewidths=0,
)

# --- Plot palette colors (bold points) ---
ax.scatter(
    palette_bright[:, 0],
    palette_bright[:, 1],
    palette_bright[:, 2],
    c=palette_bright / 255.0,
    s=220,
    edgecolor="k",
    linewidth=0.8,
)

# --- Label palette points ---
for (x, y, z), label in zip(palette_bright, labels):
    ax.text(x, y, z, label, fontsize=9, ha='center', va='bottom', weight='bold')

# --- Axis setup ---
ax.set_xlabel("Red", labelpad=10)
ax.set_ylabel("Green", labelpad=10)
ax.set_zlabel("Blue", labelpad=10)
ax.set_xlim(0, 255)
ax.set_ylim(0, 255)
ax.set_zlim(0, 255)

# Clean look
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(False)

plt.tight_layout()
plt.show()
