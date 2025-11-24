from PIL import Image, ImageDraw, ImageFont
import numpy as np
from sklearn.cluster import KMeans
import warnings

# ==========================
# User settings
# ==========================
input_path = r"C:\Users\Admin\Downloads\Screenshot 2025-11-21 201252.png"
output_path = r"C:\Users\Admin\Downloads\quantized_side_by_side.png"
bit_depths = [1, 3, 5]
spacing = 20              # Space between images
font_size = 30
heading_text = "K-Means Quantization Comparison"
heading_font_size = 50
max_width = 800           # Resize large images for faster processing
heading_space = 100       # Space at top for heading
# ==========================

# Suppress warnings
warnings.filterwarnings("ignore")

# Load image and optionally resize for large images
image = Image.open(input_path).convert('RGB')
if image.width > max_width:
    ratio = max_width / image.width
    image = image.resize((max_width, int(image.height * ratio)))

pixels = np.array(image)
h, w, c = pixels.shape

# Single-threaded KMeans quantization function
def kmeans_quantize(image_pixels, bits):
    num_colors = 2 ** bits
    pixels_2d = image_pixels.reshape(-1, 3)

    # Sample pixels for speed if too many
    if len(pixels_2d) > 500000:
        idx = np.random.choice(len(pixels_2d), 500000, replace=False)
        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
        kmeans.fit(pixels_2d[idx])
        labels = kmeans.predict(pixels_2d)
    else:
        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
        kmeans.fit(pixels_2d)
        labels = kmeans.labels_

    cluster_centers = kmeans.cluster_centers_.astype('uint8')
    quantized_pixels = cluster_centers[labels].reshape(h, w, 3)
    return quantized_pixels

# Generate quantized images
output_images = [pixels] + [kmeans_quantize(pixels, b) for b in bit_depths]

# Create side-by-side canvas with extra space for heading
total_width = w * len(output_images) + spacing * (len(output_images)-1)
total_height = h + heading_space
canvas = np.zeros((total_height, total_width, 3), dtype=np.uint8)  # black background

# Paste images below heading
current_x = 0
for img in output_images:
    canvas[heading_space:heading_space+h, current_x:current_x+w, :] = img
    current_x += w + spacing

output_pil = Image.fromarray(canvas)
draw = ImageDraw.Draw(output_pil)

# Load fonts
try:
    font = ImageFont.truetype("arial.ttf", font_size)
    heading_font = ImageFont.truetype("arial.ttf", heading_font_size)
except:
    font = ImageFont.load_default()
    heading_font = ImageFont.load_default()

# Add heading at top
heading_bbox = draw.textbbox((0,0), heading_text, font=heading_font)
heading_w = heading_bbox[2] - heading_bbox[0]
draw.text(((total_width - heading_w)//2, 10), heading_text, fill=(255,255,255), font=heading_font)

# Add labels above each image, below heading
labels = ["Original"] + [f"{b}-bit" for b in bit_depths]
current_x = 0
for label in labels:
    bbox = draw.textbbox((0,0), label, font=font)
    text_w = bbox[2] - bbox[0]
    draw.text((current_x + (w - text_w)//2, heading_space - font_size - 10), label, fill=(255,255,255), font=font)
    current_x += w + spacing

# Save final image
output_pil.save(output_path)
print(f"Side-by-side quantized image with heading saved to: {output_path}")









