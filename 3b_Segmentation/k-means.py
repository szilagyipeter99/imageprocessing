from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Open the image using PIL
image = Image.open("resources/boglarka-low.jpg")

# Convert image to a NumPy array
data = np.array(image, dtype=np.uint8)

# Get the dimensions of the image
image_h, image_w, _ = data.shape

# Reshape image data for k-means
pixels = data.reshape(-1, 3)
# Run the k-means algorithm
kmeans = KMeans(n_clusters = 10)
kmeans.fit(pixels)

# Get centers and labels
centers = np.asarray(kmeans.cluster_centers_, dtype=np.uint8)
labels = np.asarray(kmeans.labels_, dtype = np.uint8)
labels = np.reshape(labels, (image_h, image_w))

# Reconstruct the image from the labels
newImage = np.zeros((image_h, image_w, 3), dtype=np.uint8)
for i in range(image_h):
    for j in range(image_w):
            # Assing every pixel the RGB color of their label's center
            newImage[i, j, :] = centers[labels[i, j], :]

# Display images
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(data)
ax[0].set_title("Original Image")
ax[0].axis("off")
ax[1].imshow(newImage)
ax[1].set_title("Segmented Image")
ax[1].axis("off")
plt.show()
