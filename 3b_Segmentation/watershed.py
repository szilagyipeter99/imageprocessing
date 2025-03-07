# DO NOT USE THIS CODE, IT IS NOT FINAL

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import  opening, closing
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

# Open the image using PIL
image = Image.open("resources/cards.png").convert("L")

# Convert image to a NumPy array
data = np.array(image, dtype=np.uint8)

# Apply thresholding
threshold = 175
data = np.where(data > threshold, 1, 0)

# Noise removal
data = opening(data, np.ones((3, 3)))
data = opening(data, np.ones((3, 3)))


distance = ndi.distance_transform_edt(data)
coords = peak_local_max(distance, footprint=np.ones((5, 5)), labels=data)
mask = np.zeros(distance.shape, dtype=bool)
mask[tuple(coords.T)] = True
markers, _ = ndi.label(mask)
labels = watershed(-distance, markers, mask = data)

fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title('Overlapping objects')
ax[1].imshow(-distance, cmap=plt.cm.gray)
ax[1].set_title('Distances')
ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
ax[2].set_title('Separated objects')

for a in ax:
    a.set_axis_off()

fig.tight_layout()
plt.show()
