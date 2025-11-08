from PIL import Image
import numpy as np
import math

# --- Load image ---
fp = "./images/IMG_2746.png"
test = Image.open(fp).convert("RGB")  # ensure RGB (no alpha)

print("Processing an image...")
print(f"Mode: {test.mode}")
print(f"Size: {test.size}")

# --- Define colors ---
def make_color(rgb, long_name, short_name):
    return {"rgb": np.array(rgb, dtype=float), "long_name": long_name, "short_name": short_name}

colors = [
    make_color((0, 0, 0),       "Black",    "0"),
    make_color((255, 255, 255), "White",    "1"),
    make_color((97, 96, 92),    "Wall", "2"),
    make_color((142, 123, 93),  "Tan", "3"),
    make_color((141, 115, 64),  "Yellow", "Y"),
    make_color((126, 82, 95),   "Pink", "p"),
    make_color((75, 88, 45),    "Lime Green", "g"),
    make_color((143, 139, 127), "Light Wall", "3"),
    make_color((43, 57, 32),    "Dark Green", "G"),
    make_color((47, 60, 68),    "Blue", "B"),
    make_color((49, 42, 50),    "Purple", "P"),
    make_color((123, 69, 67),   "Orange", "O"),
    make_color((91, 56, 60),    "Red", "R"),
]

# --- Convert to numpy array ---
pixels = np.array(test)  # shape = (height, width, 3)
h, w, _ = pixels.shape

# --- Prepare color reference arrays ---
palette = np.stack([c["rgb"] for c in colors])  # shape (n_colors, 3)
short_names = np.array([c["short_name"] for c in colors])

# --- Vectorized distance computation ---
# Flatten image to (n_pixels, 3)
flat_pixels = pixels.reshape(-1, 3).astype(float)

# Compute squared distances between each pixel and each color
# Efficient: (n_pixels, n_colors)
dists = np.sum((flat_pixels[:, None, :] - palette[None, :, :]) ** 2, axis=2)

# Find index of closest color for each pixel
closest_idx = np.argmin(dists, axis=1)

# Map to short names
mapped_chars = short_names[closest_idx]

# Reshape to original image shape
output_chars = mapped_chars.reshape(h, w)

# --- Combine into output string ---
output_str = "\n".join("".join(row) for row in output_chars)

print(output_str)
