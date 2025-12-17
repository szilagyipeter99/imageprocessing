from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter

# Open the image using PIL
# L. Antonelli, F. Polverino, A. Albu, A. Hada, I.A. Asteriti, F. Degrassi, G. Guarguaglini, L. Maddalena, M.R. Guarracino, 
# ALFI: Cell cycle phenotype annotations of label-free time-lapse imaging data from cultured human cells, Scientific Data 10(677), 2023
# (Image: I_TP03_0014)
image = Image.open("path-to-resources/cells.jpg").convert("L")

# Convert image to a NumPy array
data = np.array(image, dtype = np.uint8)

# Compute the edges using the Canny filter
edges = canny(data, sigma = 0.5)

# Create test circles and compute their Hough transform
test_rs = np.arange(40, 60, 2)
hough_result = hough_circle(edges, test_rs)
_, cx, cy, rs = hough_circle_peaks(hough_result, test_rs, total_num_peaks = 10)

# Empty image to draw circles into
final_image = np.zeros_like(data)

# Draw the circles
for center_y, center_x, radius in zip(cy, cx, rs):
    circy, circx = circle_perimeter(center_y, center_x, radius, shape = data.shape)
    final_image[circy, circx] = 255

# Display images
fig, ax = plt.subplots(1, 2, figsize = (12, 6))
ax[0].imshow(data, cmap = "gray")
ax[0].set_title("Original Image")
ax[0].axis("off")
ax[1].imshow(final_image, cmap = "gray")
ax[1].set_title("Detected Circles")
ax[1].axis("off")
plt.tight_layout()
plt.show()
