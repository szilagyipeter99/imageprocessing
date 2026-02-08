from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import median
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle, denoise_wavelet, denoise_nl_means

# Open the image using PIL
image = Image.open("path-to-resources/noisy-items-?").convert("L")

# Convert image to a NumPy array
data = np.array(image, dtype = np.uint8)

# Footprint for the median filter
custom_footprint = np.array([[1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1]], dtype = np.uint8)

# Perform denoising using various algorithms
denoised_image_med = median(data, footprint = custom_footprint)
denoised_image_bl = denoise_bilateral(data, sigma_color = 0.05, sigma_spatial = 5)
denoised_image_tv = denoise_tv_chambolle(data, weight = 0.1)
denoised_image_wl = denoise_wavelet(data)
denoised_image_nlm = denoise_nl_means(data, h = 0.1, fast_mode = "True", patch_size = 5, patch_distance = 6)

# Display images
filters = [
    ("Original Image", data),
    ("Median filter", denoised_image_med),
    ("Bilateral filter", denoised_image_bl),
    ("Total variation", denoised_image_tv),
    ("Wavelet", denoised_image_wl),
    ("Non-local means", denoised_image_nlm)
]
fig, axes = plt.subplots(2, 3, figsize = (12, 8))
for ax, (title, img) in zip(axes.ravel(), filters):
    ax.imshow(img, cmap = "gray")
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()
