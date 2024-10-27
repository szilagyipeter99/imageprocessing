from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Open the image using PIL
image = Image.open("path-to-resources/boglarka.jpg")

# Convert image to a NumPy array
data = np.array(image, dtype=np.uint8)
print(data.shape)  # (height, width, channels)

# Set brightness and contrast
brightness = 25  # [-127;127] where 0 means unchanged
contrast = 75    # [-127;127] where 0 means unchanged

# Extend data type to potential negative values (will be converted back later)
data = np.int16(data) # 16 bit integer
# Perform the modification
data = data * (contrast / 127 + 1) - contrast + brightness
# Clip values between 0 and 255 (<= 0 to 0 and >= 255 to 255)
data = np.clip(data, 0, 255)
# Convert back to 8 bit unsigned integer
final_image = np.uint8(data)

# Display the image
plt.imshow(final_image)
plt.axis('off')
plt.show()
