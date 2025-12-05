from PIL import Image, ImageOps
import numpy as np

# Load the image
img_path = r"C:\Users\Admin\Downloads\bit_plane_splicing.png"
img = Image.open(img_path).convert('L')  # convert to grayscale
img_array = np.array(img)

# Function to extract bit planes
def bit_planes(image):
    planes = []
    for i in range(8):
        plane = (image >> i) & 1
        planes.append(plane)
    return planes

# Extract bit planes
planes = bit_planes(img_array)

# Union of 3 lowest bit planes
union_lowest = (planes[0] | planes[1] | planes[2]) * 255
difference = np.abs(img_array - union_lowest)

# Convert arrays back to images
union_img = Image.fromarray(union_lowest.astype(np.uint8))
diff_img = Image.fromarray(difference.astype(np.uint8))

# Make all images same size if needed
width, height = img.size
union_img = union_img.resize((width, height))
diff_img = diff_img.resize((width, height))

# Create a new blank image to place all three images horizontally
combined = Image.new('L', (width*3, height))
combined.paste(img, (0, 0))
combined.paste(union_img, (width, 0))
combined.paste(diff_img, (width*2, 0))

# Optional: Add simple labels
from PIL import ImageDraw, ImageFont

draw = ImageDraw.Draw(combined)
font = None  # Default font
draw.text((width//4, 10), "Original", fill=255, font=font)
draw.text((width + width//4, 10), "Union 3 LSB", fill=255, font=font)
draw.text((2*width + width//4, 10), "Difference", fill=255, font=font)

# Save combined image
combined.save(r"C:\Users\Admin\Downloads\bit_plane_splicing_combined.png")
print("All images combined saved as 'bit_plane_splicing_combined.png'.")
