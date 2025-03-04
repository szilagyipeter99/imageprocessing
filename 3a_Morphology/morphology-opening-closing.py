from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import  opening, closing

# Open the image using PIL
image = Image.open("path-to-resources/noisy-gears.jpeg").convert("L")

# Convert image to a NumPy array
data = np.array(image, dtype=np.uint8)

# Apply thresholding
threshold = 128
data = np.where(data > threshold, 0, 1)

custom_footprint = np.array([[1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1]], dtype=np.uint8)

# Perform an erosion followed by a dilation 
opened_image = opening(data, custom_footprint)
# Perform a dilation followed by an erosion
closed_image = closing(data, custom_footprint)

# Display images
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(opened_image, cmap="gray")
ax[0].set_title("Opened Image")
ax[0].axis("off")
ax[1].imshow(closed_image, cmap="gray")
ax[1].set_title("Closed Image")
ax[1].axis("off")
plt.show()
