from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks

# Open the image using PIL
image = Image.open("path-to-resources/cone.jpg").convert("L")

# Convert image to a NumPy array
data = np.array(image, dtype = np.uint8)

# Compute the edges using the Canny filter
edges = canny(data, sigma = 0.5)

# Create test angles and compute their Hough transform
test_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint = False)
hspace, theta, distances = hough_line(edges, theta = test_angles)

# Display images
fig, ax = plt.subplots(1, 2, figsize = (12, 6))
ax[0].imshow(data, cmap = "gray")
ax[0].set_title('Original image')
ax[0].axis("off")
ax[1].imshow(data, cmap = "gray")
ax[1].set_ylim((data.shape[0], 0))
ax[1].set_xlim((0, data.shape[1]))
ax[1].set_title('Detected Lines')
ax[1].axis("off")
# Draw lines
for _, angle, dist in zip(*hough_line_peaks(hspace, theta, distances, threshold = 0.2 * np.max(hspace))):
    (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
    ax[1].axline((x0, y0), slope = np.tan(angle + np.pi / 2))
plt.tight_layout()
plt.show()
