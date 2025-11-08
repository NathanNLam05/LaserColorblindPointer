from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

color_type = "HSV"
n_colors = 50
img_name = "IMG_2747"
img = Image.open(f"./images/{img_name}.png").convert(color_type)
output_image_name = f"{img_name}_KMeans{n_colors}colors{color_type}"
pixels = np.array(img).reshape(-1, 3)

# Cluster colors
kmeans = KMeans(n_clusters=n_colors, random_state=42)
kmeans.fit(pixels)

palette = kmeans.cluster_centers_.astype(int)
print(palette)

def make_color(rgb, long_name, short_name):
    return {"rgb": np.array(rgb, dtype=float), "long_name": long_name, "short_name": short_name}
# colors = [make_color(palette_color) for ]



# --- Convert to numpy array ---
pixels = np.array(img)  # shape = (height, width, 3)
h, w, _ = pixels.shape

# --- Prepare color reference arrays ---
# palette = np.stack([c["rgb"] for c in colors])  # (n_colors, 3)

# --- Vectorized nearest-color computation ---
flat_pixels = pixels.reshape(-1, 3).astype(float)  # (n_pixels, 3)
dists = np.sum((flat_pixels[:, None, :] - palette[None, :, :]) ** 2, axis=2)
closest_idx = np.argmin(dists, axis=1)

# --- Reconstruct quantized image ---
quantized_pixels = palette[closest_idx].astype(np.uint8)
quantized_img = quantized_pixels.reshape(h, w, 3)

# --- Save new image ---
output_image = Image.fromarray(quantized_img, mode="RGB")
output_path = f"./images/{output_image_name}.png"
output_image.save(output_path)

print(f"✅ New color-quantized image saved to: {output_path}")
