from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

image = Image.open("path-to-resources/boglarka.jpg").convert("L")

# Convert image to a NumPy array
data = np.array(image, dtype=np.uint8)
print(data.shape)  # (height, width, channels)

# Set brightness and contrast
brightness = 25  # [-127;127] where 0 means unchanged
contrast = 75    # [-127;127] where 0 means unchanged

# Build a grayscale LUT
myLUT=np.zeros(256,dtype=np.uint8)
for li in range(255):
    myLUT[li]= np.clip(li * (contrast / 127 + 1) - contrast + brightness, 0, 255)

# Measure execution time of applying the LUT
start1 = timer()
# Apply the LUT
final_image1 = myLUT[data]
# End timer
end1 = timer()

# Measure execution time of basic processing
start2 = timer()
# Change brigtness and contrast with the usual method
final_image2 = np.clip(data * (contrast / 127 + 1) - contrast + brightness, 0, 255)
final_image2 = np.uint8(final_image2)
# End timer
end2 = timer()

print(f"Basic method: {round((end2 - start2) * 1000, 3)}ms")
print(f"Using a LUT: {round((end1 - start1) * 1000, 3)}ms")

# Display images
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(final_image1, cmap="gray")
ax[0].set_title("LUT-applied Image")
ax[0].axis("off")
ax[1].imshow(final_image2, cmap="gray")
ax[1].set_title("Basic Processing Image")
ax[1].axis("off")
plt.show()