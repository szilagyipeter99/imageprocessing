from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.color import label2rgb

# Open the image using PIL
image = Image.open("path-to-resources/ovals.jpeg").convert("L")

# Convert image to a NumPy array
data = np.array(image, dtype=np.uint8)

# Apply thresholding
threshold = 128
data = np.where(data > threshold, 0, 1)

# Label connected regions (CCA)
labels_from_image = label(data)
# Produce an image where labels are color-coded
image_label_overlay = label2rgb(labels_from_image, image=data, alpha=1.0)

# Display the image
plt.imshow(image_label_overlay)
plt.title("Center point")
plt.axis('off')
plt.show()
