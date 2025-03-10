from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2

# Open the image using PIL
# Source: Seven Digit License Plate Combo's Beginning with "A" Are Overtaking CT Roads — Connecticut by the Numbers
# https://ctbythenumbers.news/ctnews/seven-digit-license-plate-combos-beginning-with-a-are-overtaking-ct-roads
image = Image.open("path-to-resources/license-plates.jpg").convert("L")

# Convert image to a NumPy array
data = np.array(image, dtype=np.uint8)

# Function to build a motion blur kernel
def build_blur_kernel(size, angle, img_shape):
    # Create empty matrix and calculate its center
    kernel = np.zeros((size, size))
    center = size // 2
    # Calculate elements of the kernel matrix
    for i in range(size):
        x = int(center + (i - center) * np.cos(np.deg2rad(angle)))
        y = int(center + (i - center) * np.sin(np.deg2rad(angle)))
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1
    # Normalize the kernel
    normalized_kernel = kernel / kernel.sum()
    return normalized_kernel

# Function to apply convolution in frequency domain
def apply_blur_fft(image, kernel):
    # Convert the image and the kernel to frequency domain
    fft_image = fft2(image)
    fft_kernel = fft2(kernel, s = image.shape)
    # Perform convolution in frequency domain (multiplication)
    blurred_image = fft_image * fft_kernel
    # Convert blurred image back to spatial domain
    blurred_image = np.abs(ifft2(blurred_image))
    blurred_image = np.clip(blurred_image, 0, 255).astype(np.uint8)
    return blurred_image

# Function to apply deconvolution
def apply_deconvolution(image, kernel, K=0.01):
    # Convert the image and the kernel to frequency domain
    fft_image = fft2(image)
    fft_kernel = fft2(kernel, s=image.shape)
    # Replace 0 values with a small value before division
    fft_kernel = np.where(fft_kernel == 0, 1e-8, fft_kernel)
    # Perform deconvolution in frequency domain (division)
    restored_image = fft_image / fft_kernel * (np.abs(fft_kernel) ** 2 / (np.abs(fft_kernel) ** 2 + K))
    # Convert restored image back to spatial domain
    restored_image = np.abs(ifft2(restored_image))
    restored_image = np.clip(restored_image, 0, 255).astype(np.uint8)
    return restored_image

# Create motion blur kernel
kernel = build_blur_kernel(size = 15, angle = 30, img_shape = data.shape)
# Apply motion blur using FFT
blurred_image = apply_blur_fft(image = data, kernel = kernel)
# Apply deconvolution
restored = apply_deconvolution(image = blurred_image, kernel = kernel)

# Display images
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(data, cmap="gray")
ax[0].set_title("Original Image")
ax[0].axis("off")
ax[1].imshow(blurred_image, cmap="gray")
ax[1].set_title("Blurred Image")
ax[1].axis("off")
ax[2].imshow(restored, cmap="gray")
ax[2].set_title("Restored Image")
ax[2].axis("off")
plt.show()
