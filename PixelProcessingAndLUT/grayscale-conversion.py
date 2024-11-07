from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Open the image using PIL and ensure the correct (RGB) color mode
image = Image.open("path-to-resources/boglarka.jpg").convert("RGB")

# Convert image to a NumPy array
data = np.array(image, dtype=np.uint8)
print(data.shape)  # (height, width, channels)

# RGB array for the conversion (Rec.709 standard)
conv_arr = [0.2126, 0.7152, 0.0722] # R, G, B

# Perform the conversion
final_image = np.dot(data, conv_arr).astype(np.uint8)

# Display the image
plt.imshow(final_image, cmap="gray")
plt.axis('off')
plt.show()
