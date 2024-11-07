from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Open the image and convert to grayscale
image = Image.open("resources/wrench.png").convert("L")

# Convert image to a NumPy array
data = np.array(image, dtype=np.uint8)

# Apply thresholding
threshold = 135
data = np.where(data > threshold, 255, 0)

# Convert image from {0, 255} to {0, 1} (Normalize)
data = data / 255

# Helper arrays to calculate the center
x_range = np.arange(0, data.shape[1])
y_range = np.arange(0, data.shape[0])

# Calculate area and center
area = data.sum()
x_cntr = np.matmul(data, x_range).sum() / data.sum()
y_cntr = np.matmul(data.T, y_range).sum() / data.sum()

# Display the image
plt.imshow(data, cmap="gray")
plt.plot(x_cntr, y_cntr, "og", markersize=5)  # Mark center with green circle
plt.title("Center point")
plt.axis('off')  # Hide the axis
plt.show()