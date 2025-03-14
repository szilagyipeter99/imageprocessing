from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import median
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle, denoise_wavelet, denoise_nl_means

# Open the image using PIL
image = Image.open("path-to-resources/noisy-items.png").convert("L")

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
fig, ax = plt.subplots(2, 3, figsize = (12, 8))
ax[0][0].imshow(data, cmap = "gray")
ax[0][0].set_title("Original Image")
ax[0][0].axis("off")
ax[0][1].imshow(denoised_image_med, cmap = "gray")
ax[0][1].set_title("Median filter")
ax[0][1].axis("off")
ax[0][2].imshow(denoised_image_bl, cmap = "gray")
ax[0][2].set_title("Bilateral filter")
ax[0][2].axis("off")
ax[1][0].imshow(denoised_image_tv, cmap = "gray")
ax[1][0].set_title("Total variation")
ax[1][0].axis("off")
ax[1][1].imshow(denoised_image_wl, cmap = "gray")
ax[1][1].set_title("Wavelet")
ax[1][1].axis("off")
ax[1][2].imshow(denoised_image_nlm, cmap = "gray")
ax[1][2].set_title("Non-local means")
ax[1][2].axis("off")
plt.tight_layout()
plt.show()
