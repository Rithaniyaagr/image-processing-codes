import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize
from PIL import Image, ImageDraw, ImageFont
import os

# Input image path
input_path = r"C:\Users\Admin\Downloads\zebra.png"

# Output path
output_path = r"C:\Users\Admin\Downloads\zebra_sampling_comparison_labeled.png"

# Load image and convert to greyscale
img = io.imread(input_path)
gray = rgb2gray(img)
gray = gray.astype(np.float32)

# Sampling factors
factors = [2, 4, 8, 16]

# Frequency Sampling function
def frequency_sampling(image, factor):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    rows, cols = image.shape
    crow, ccol = rows//2, cols//2
    r = min(crow, ccol)//factor
    mask = np.zeros_like(image)
    y, x = np.ogrid[:rows, :cols]
    mask_area = (x - ccol)**2 + (y - crow)**2 <= r**2
    mask[mask_area] = 1
    fshift_filtered = fshift * mask
    img_back = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
    img_back = np.abs(img_back)
    img_back = np.clip(img_back, 0, 1)
    return (img_back * 255).astype(np.uint8)

# Spatial Sampling function
def spatial_sampling(image, factor):
    rows, cols = image.shape
    sampled = image[::factor, ::factor]
    sampled_resized = resize(sampled, (rows, cols), anti_aliasing=True)
    return (sampled_resized * 255).astype(np.uint8)

# Prepare combined image
rows, cols = gray.shape
combined = np.zeros((rows*2, cols*len(factors)), dtype=np.uint8)

# Fill combined image with sampled images
for i, f in enumerate(factors):
    freq_img = frequency_sampling(gray, f)
    spatial_img = spatial_sampling(gray, f)
    combined[0:rows, i*cols:(i+1)*cols] = freq_img
    combined[rows:2*rows, i*cols:(i+1)*cols] = spatial_img

# Convert to PIL Image for labeling
combined_img = Image.fromarray(combined).convert("RGB")
draw = ImageDraw.Draw(combined_img)

# Font setup (default PIL font)
try:
    font = ImageFont.truetype("arial.ttf", 20)
except:
    font = ImageFont.load_default()

# Add labels
for i, f in enumerate(factors):
    # Top row: Frequency
    draw.text((i*cols + 10, 10), f"Freq 1/{f}", fill=(255,0,0), font=font)
    # Bottom row: Spatial
    draw.text((i*cols + 10, rows + 10), f"Spatial 1/{f}", fill=(0,255,0), font=font)

# Save labeled image
combined_img.save(output_path)
print("Labeled combined image saved at:", output_path)
