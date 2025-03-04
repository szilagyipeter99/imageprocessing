from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import erosion, dilation

# Open the image using PIL
image = Image.open("path-to-resources/gears.jpeg").convert("L")

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

# Set pixels to the minimum in their neighborhood
eroded_image = erosion(data, custom_footprint)
# Set pixels to the maximum in their neighborhood
dilated_image = dilation(data, custom_footprint)

# Display images
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(eroded_image, cmap="gray")
ax[0].set_title("Eroded Image")
ax[0].axis("off")
ax[1].imshow(dilated_image, cmap="gray")
ax[1].set_title("Dilated Image")
ax[1].axis("off")
plt.show()
