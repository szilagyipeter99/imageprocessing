# DO NOT USE THIS CODE, IT IS NOT FINAL

import numpy as np
from scipy.fft import fft2, ifft2, fftshift
from PIL import Image
import matplotlib.pyplot as plt

def motion_blur_kernel(size, angle, img_shape):
    """Generates a motion blur kernel and pads it to the image size."""
    kernel = np.zeros((size, size))
    center = size // 2
    angle = np.deg2rad(angle)

    for i in range(size):
        x = int(center + (i - center) * np.cos(angle))
        y = int(center + (i - center) * np.sin(angle))
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1

    kernel /= kernel.sum()  # Normalize

    # Pad kernel to match image size
    pad_kernel = np.zeros(img_shape)
    pad_kernel[:size, :size] = kernel

    # Shift kernel so center is at (0,0) in frequency domain
    pad_kernel = fftshift(pad_kernel)

    return pad_kernel

def apply_motion_blur_fft(image, kernel):
    """Applies motion blur using FFT-based multiplication."""
    image_fft = fft2(image)
    kernel_fft = fft2(kernel, s=image.shape)
    blurred_fft = image_fft * kernel_fft
    blurred = np.abs(ifft2(blurred_fft))
    
    return np.clip(blurred, 0, 255).astype(np.uint8)

def wiener_deconvolution(blurred, kernel, K=0.01):
    """Performs Wiener deconvolution using FFT."""
    blurred_fft = fft2(blurred)
    kernel_fft = fft2(kernel, s=blurred.shape)
    
    kernel_fft = np.where(kernel_fft == 0, 1e-8, kernel_fft)  # Avoid division by zero
    restored_fft = blurred_fft / kernel_fft * (np.abs(kernel_fft)**2 / (np.abs(kernel_fft)**2 + K))
    restored = np.abs(ifft2(restored_fft))
    
    return np.clip(restored, 0, 255).astype(np.uint8)

# Load image
image = Image.open("license_plate.jpg").convert("L")  # Convert to grayscale
image = np.array(image)

# Create motion blur kernel
kernel_size = 15
angle = 30
kernel = motion_blur_kernel(kernel_size, angle, image.shape)

# Apply motion blur using FFT
blurred = apply_motion_blur_fft(image, kernel)

# Apply Wiener deconvolution
restored = wiener_deconvolution(blurred, kernel)

# Save results
Image.fromarray(blurred).save("blurred_fft.jpg")
Image.fromarray(restored).save("restored_fft.jpg")

# Show results
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(image, cmap="gray")
ax[0].set_title("Original")
ax[1].imshow(blurred, cmap="gray")
ax[1].set_title("Blurred (FFT)")
ax[2].imshow(restored, cmap="gray")
ax[2].set_title("Restored (FFT)")
for a in ax:
    a.axis("off")
plt.show()
