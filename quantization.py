from PIL import Image
import numpy as np
import os

# Input & Output paths
input_path = r"C:\Users\Admin\Downloads\image.png"
output_dir = r"C:\Users\Admin\Downloads"

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load image
img = Image.open(input_path).convert("RGB")
np_img = np.array(img)

# --------------------------------------------------------------------
# 1. Greyscale via Desaturation
# --------------------------------------------------------------------
gray_img = np.mean(np_img, axis=2).astype(np.uint8)
gray_img_pil = Image.fromarray(gray_img)
gray_path = os.path.join(output_dir, "output_greyscale_desaturation.png")
gray_img_pil.save(gray_path)

# --------------------------------------------------------------------
# 2. Median Cut Quantization
# --------------------------------------------------------------------
median_cut = img.convert("P", palette=Image.ADAPTIVE, colors=16).convert("RGB")
median_cut_path = os.path.join(output_dir, "output_median_cut.png")
median_cut.save(median_cut_path)

# --------------------------------------------------------------------
# 3. Octree Quantization – Level 1
# --------------------------------------------------------------------
octree_1 = img.quantize(colors=8, method=0).convert("RGB")
octree1_path = os.path.join(output_dir, "output_octree_level1.png")
octree_1.save(octree1_path)

# --------------------------------------------------------------------
# 4. Octree Quantization – Level 2
# --------------------------------------------------------------------
octree_2 = img.quantize(colors=16, method=0).convert("RGB")
octree2_path = os.path.join(output_dir, "output_octree_level2.png")
octree_2.save(octree2_path)

# --------------------------------------------------------------------
# Create Combined 2×2 Grid PNG
# --------------------------------------------------------------------
imgs = [
    Image.open(gray_path).convert("RGB"),
    Image.open(median_cut_path).convert("RGB"),
    Image.open(octree1_path).convert("RGB"),
    Image.open(octree2_path).convert("RGB")
]

# Make all images same size
w, h = imgs[0].size
combined = Image.new("RGB", (w * 2, h * 2))

# Place images in grid
combined.paste(imgs[0], (0, 0))
combined.paste(imgs[1], (w, 0))
combined.paste(imgs[2], (0, h))
combined.paste(imgs[3], (w, h))

# Save output PNG
combined_path = os.path.join(output_dir, "quantization_output.png")
combined.save(combined_path)

print("🎉 Combined PNG created at:", combined_path)
