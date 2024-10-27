from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Open the image and convert to grayscale
image = Image.open("path-to-resources/wrench.png").convert("L")

# Convert image to a NumPy array
data = np.array(image, dtype=np.uint8)

# Apply thresholding
threshold = 135
data = np.where(data > threshold, 255, 0)

# Convert image from {0, 255} to {0, 1} for center calculation
data = data / 255

# Helper arrays to calculate the center
x_range = np.arange(0, data.shape[1])
y_range = np.arange(0, data.shape[0])

# Calculate area and center
area = data.sum()
x_cntr = np.matmul(data, x_range).sum() / area
y_cntr = np.matmul(data.T, y_range).sum() / area

# Calculate a, b, c
a = np.matmul(data, np.square(x_range)).sum() - np.square(x_cntr) * area
b = 2 * (np.matmul(np.matmul(data, x_range), y_range).sum() - x_cntr * y_cntr * area)
c = np.matmul(data.T, np.square(y_range)).sum() - np.square(y_cntr) * area

# Orientation in radians
orientation = np.arctan2(b, a - c) / 2
print(np.rad2deg(orientation))

# Start and end point for the orientation line
half_len = 500
x_line = [x_cntr - half_len * np.cos(orientation), x_cntr + half_len * np.cos(orientation)]
y_line = [y_cntr - half_len * np.sin(orientation), y_cntr + half_len  * np.sin(orientation)]

# Display the image
plt.imshow(data, cmap="gray")
plt.plot(x_line, y_line, color="red", linewidth=3)  # Draw a red line on the image
plt.plot(x_cntr, y_cntr, "og", markersize=5)  # Mark center with green circle
plt.title("Center and orientation")
plt.axis('off')  # Hide the axis
plt.show()
