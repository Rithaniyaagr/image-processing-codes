import numpy as np
from matplotlib import pyplot as plt
from matplotlib.image import imread, imsave
from scipy.ndimage import convolve1d, gaussian_filter

# ----------------------------
# Load the color image
# ----------------------------
image_path = r"C:\Users\Admin\Downloads\spatial_filtering_image.jpg"
img = imread(image_path)

# Ensure the image is in float format (0-1)
if img.dtype != np.float32 and img.dtype != np.float64:
    img = img.astype(np.float32) / 255.0

# ----------------------------
# Box filter function
# ----------------------------
def box_filter(image, ksize, normalize=True):
    kernel = np.ones((ksize, ksize), dtype=np.float32)
    if normalize:
        kernel /= (ksize * ksize)
    
    # Apply convolution to each channel
    filtered = np.zeros_like(image)
    for c in range(3):
        filtered[..., c] = convolve1d(convolve1d(image[..., c], kernel[:,0], axis=0, mode='reflect'),
                                      kernel[0,:], axis=1, mode='reflect')
    return filtered

# ----------------------------
# Apply box filters
# ----------------------------
box5 = box_filter(img, 5, normalize=False)
box5_norm = box_filter(img, 5, normalize=True)
box20_norm = box_filter(img, 20, normalize=True)

# ----------------------------
# Separable Gaussian filter
# ----------------------------
def separable_gaussian(image, sigma):
    filtered = np.zeros_like(image)
    for c in range(3):
        filtered[..., c] = gaussian_filter(image[..., c], sigma=sigma, mode='reflect')
    return filtered

# Compute sigma and filter sizes (rule of thumb)
ksize_small = 5
ksize_large = 21
sigma_small = 0.3*((ksize_small-1)*0.5 - 1) + 0.8
sigma_large = 0.3*((ksize_large-1)*0.5 - 1) + 0.8

gauss_small = separable_gaussian(img, sigma_small)
gauss_large = separable_gaussian(img, sigma_large)

# Normalized Gaussian (already normalized in scipy)
gauss_small_norm = gauss_small
gauss_large_norm = gauss_large

# ----------------------------
# Display all results on one page
# ----------------------------
plt.figure(figsize=(18,12))

titles = ['Original', 'Box 5x5', 'Box 5x5 Norm', 'Box 20x20 Norm',
          f'Gaussian {ksize_small}x{ksize_small}', f'Normalized Gaussian {ksize_small}x{ksize_small}',
          f'Gaussian {ksize_large}x{ksize_large}', f'Normalized Gaussian {ksize_large}x{ksize_large}']

images = [img, box5, box5_norm, box20_norm, gauss_small, gauss_small_norm, gauss_large, gauss_large_norm]

for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(images[i])
    plt.title(titles[i], fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.show()

# ----------------------------
# Save outputs individually
# ----------------------------
imsave(r"C:\Users\Admin\Downloads\box5.png", np.clip(box5,0,1))
imsave(r"C:\Users\Admin\Downloads\box5_norm.png", np.clip(box5_norm,0,1))
imsave(r"C:\Users\Admin\Downloads\box20_norm.png", np.clip(box20_norm,0,1))
imsave(r"C:\Users\Admin\Downloads\gauss5.png", np.clip(gauss_small,0,1))
imsave(r"C:\Users\Admin\Downloads\gauss21.png", np.clip(gauss_large,0,1))
