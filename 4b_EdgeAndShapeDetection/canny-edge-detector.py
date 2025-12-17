from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature

# Open the image using PIL
image = Image.open("path-to-resources/cone.jpg").convert("L")

# Convert image to a NumPy array
data = np.array(image, dtype = np.uint8)

# Compute the edges using the Canny filter
edges = feature.canny(data, sigma = 3)

# Display images
fig, ax = plt.subplots(1, 2, figsize = (12, 6))
ax[0].imshow(data, cmap = "gray")
ax[0].set_title("Original Image")
ax[0].axis("off")
ax[1].imshow(edges, cmap = "gray")
ax[1].set_title("Edges")
ax[1].axis("off")
plt.show()
