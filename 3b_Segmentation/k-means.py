from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Open the image using PIL
image = Image.open("path-to-resources/cone.jpg")

# Convert image to a NumPy array
data = np.array(image, dtype=np.uint8)

# Get the dimensions of the image
image_h, image_w, _ = data.shape

# Reshape image data for k-means
pixels = data.reshape(-1, 3)
# Run the k-means algorithm
kmeans = KMeans(n_clusters = 4)
kmeans.fit(pixels)

# Get labels
labels = np.reshape(kmeans.labels_, (image_h, image_w))

# Display images
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(data)
ax[0].set_title("Original Image")
ax[0].axis("off")
ax[1].imshow(labels, cmap = "jet")
ax[1].set_title("Segmented Image")
ax[1].axis("off")
plt.show()
