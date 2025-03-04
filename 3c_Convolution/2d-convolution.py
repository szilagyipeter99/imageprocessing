import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# Open the image using PIL
image = Image.open("path-to-resources/boglarka_low.jpg")

# Convert image to a NumPy array
data = np.array(image, dtype=np.uint8)

# Define a 5x5 kernel 
kernel = np.array([[1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1]])

# Normalize the kernel
kernel = kernel / kernel.sum()

# Perform the 2D convolution on each color channel
convolved_image_channels = [convolve2d(data[:, :, i], kernel) for i in range(3)]
# Combine the three channels back into one array
convolved_image = np.stack(convolved_image_channels, axis=-1).astype(np.uint8)

# Display images
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(image)
ax[0].set_title("Original Image")
ax[0].axis("off")
ax[1].imshow(convolved_image)
ax[1].set_title("Convolved Image")
ax[1].axis("off")
plt.show()
