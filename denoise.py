import cv2
import numpy as np

# Load the image
img = cv2.imread("./images/IMG_2746.png")

# 1. Convert to LAB (helps preserve edges while denoising color)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

# 2. Denoise the lightness channel strongly
l_denoised = cv2.fastNlMeansDenoising(l, None, h=30, templateWindowSize=7, searchWindowSize=21)

# 3. Denoise the color channels more heavily
a_denoised = cv2.fastNlMeansDenoising(a, None, h=50, templateWindowSize=7, searchWindowSize=21)
b_denoised = cv2.fastNlMeansDenoising(b, None, h=50, templateWindowSize=7, searchWindowSize=21)

# 4. Merge back and convert to BGR
lab_denoised = cv2.merge((l_denoised, a_denoised, b_denoised))
img_denoised = cv2.cvtColor(lab_denoised, cv2.COLOR_LAB2BGR)

# 5. Optional: Apply bilateral filter for additional smoothing while preserving edges
img_denoised = cv2.bilateralFilter(img_denoised, d=15, sigmaColor=75, sigmaSpace=75)

# Save the denoised image
cv2.imwrite("climbing_wall_super_denoised.jpg", img_denoised)
print("✅ Saved heavily denoised image as climbing_wall_denoised.jpg")
