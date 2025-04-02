import os
from keras import models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the saved model
model = models.load_model('mnist_full_classifier.keras')

test_folder = 'path-to-resources/mnist-full/test'

# Define the list of classes
class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Function to load and preprocess an image
def preprocess_image(img_path):
    img = Image.open(img_path).convert("L")
    img = img.resize((28, 28))
    img_data = np.array(img, dtype=np.uint8)
    img_data = img_data / 255.0  # Normalize pixel values
    img_data = np.expand_dims(img_data, axis=0)  # Add a new axis at the 0th position (batch)
    return img, img_data

# Get a list of all image files in the folder
test_image_paths = [os.path.join(test_folder, fname) for fname in os.listdir(test_folder) if fname.endswith(('.jpg'))]

# Loop through each image, preprocess, predict, and display the result (probabality distribution)
for i in range(len(test_image_paths)):
    test_image, test_image_prep = preprocess_image(test_image_paths[i])
    fix, ax = plt.subplots(1, 2, figsize = (12, 6))
    ax[0].imshow(test_image, cmap="gray")
    ax[0].axis('off')
    ax[0].set_title("Image")
    ax[1].bar(np.arange(0, 10), model.predict(test_image_prep)[0])
    ax[1].set_xticks(np.arange(0, 10))
    ax[1].set_xticklabels(class_names)
    plt.tight_layout()
    plt.show()
