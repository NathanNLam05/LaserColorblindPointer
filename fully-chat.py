from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from matplotlib.colors import hsv_to_rgb

# --- Config ---
color_type = "HSV"   # can be "RGB" or "HSV"
n_colors = 50
img_name = "IMG_2748"

# --- Load image ---
img = Image.open(f"./images/{img_name}.png").convert(color_type)
pixels = np.array(img)
h, w, _ = pixels.shape

# --- Prepare data for KMeans ---
flat_pixels = pixels.reshape(-1, 3).astype(float)

# --- Cluster colors ---
print(f"Running KMeans with {n_colors} colors on {color_type} pixels...")
kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init="auto")
kmeans.fit(flat_pixels)

palette = kmeans.cluster_centers_.astype(np.uint8)
print("✅ Palette (cluster centers):")
print(palette)

# --- Assign nearest palette color to each pixel ---
labels = kmeans.predict(flat_pixels)
quantized_pixels = palette[labels]
quantized_img = quantized_pixels.reshape(h, w, 3)

# --- Handle color-space conversion for output ---
if color_type.upper() == "HSV":
    # Convert HSV → RGB before saving (normalize to [0,1] for hsv_to_rgb)
    quantized_img = hsv_to_rgb(quantized_img / 255.0)
    quantized_img = np.clip(quantized_img * 255, 0, 255).astype(np.uint8)

# --- Save image ---
output_image_name = f"{img_name}_KMeans{n_colors}colors{color_type}"
output_path = f"./images/{output_image_name}.png"
output_image = Image.fromarray(quantized_img)
output_image.save(output_path)

print(f"✅ New color-quantized image saved to: {output_path}")
