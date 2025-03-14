from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Open the image and convert to grayscale
image = Image.open("path-to-resources/wrench.png").convert("L")

# Convert image to a NumPy array
data = np.array(image, dtype = np.uint8)

# Apply thresholding
# Convert image from [0, 255] to {0, 1}
threshold = 135
data = np.where(data > threshold, 1, 0)

# Helper arrays to calculate the center
x_range = np.arange(0, data.shape[1])
y_range = np.arange(0, data.shape[0])

# Calculate area and center
area = data.sum()
x_cntr = np.matmul(data, x_range).sum() / area
y_cntr = np.matmul(data.T, y_range).sum() / area

# --- PCA (Principal Component Analysis) --- 
# Extract foreground pixel coordinates
y, x = np.nonzero(data)
coords = np.column_stack((x, y))
# Centered array
cntr_coords = coords - (x_cntr, y_cntr)
# Covariance matrix
cov_matrix = np.cov(cntr_coords, rowvar = False)
# Eigen value decomposition (EVD) to find the principal components
eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
# Eigenvector corresponding to the largest eigenvalue
pr_eig_vec = eig_vecs[:, np.argmax(eig_vals)]
# Orientation angle in radians
orientation = np.arctan2(pr_eig_vec[1], pr_eig_vec[0])

# Start and End point for the orientation line
half_len = 500
x_line = [x_cntr - half_len * np.cos(orientation), x_cntr + half_len * np.cos(orientation)]
y_line = [y_cntr - half_len * np.sin(orientation), y_cntr + half_len  * np.sin(orientation)]

# Display the image
plt.imshow(data, cmap="gray")
plt.plot(x_line, y_line, color = "red", linewidth = 3)  # Draw a red line on the image
plt.plot(x_cntr, y_cntr, "og", markersize = 5)  # Mark center with a green dot
plt.title("Center and orientation")
plt.axis('off')  # Hide the axis
plt.show()
