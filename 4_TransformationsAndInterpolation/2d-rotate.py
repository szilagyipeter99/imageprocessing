from PIL import Image
import matplotlib.pyplot as plt

# Open the image using PIL
image = Image.open("resources/boglarka.jpg")

# Resampling filter options are: 'NEAREST', 'BILINEAR', 'BICUBIC' and more
# Press 'Go to Definition' for more info about the function
rotated_image = image.rotate(angle = 20, resample = Image.Resampling.NEAREST)

# Display the image
plt.imshow(rotated_image)
plt.axis('off')
plt.show()
