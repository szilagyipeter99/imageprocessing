import os
from keras import models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the saved model
model = models.load_model('cifar10_cnn.keras')

test_folder = 'path-to-resources/cifar10-cnn-test/'

# Define the list of classes
class_names = [
    "plane", 
    "car",
    "bird", 
    "cat", 
    "deer", 
    "dog", 
    "frog", 
    "horse", 
    "ship", 
    "truck"
]

# Function to load and preprocess an image
def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((32, 32))
    img_data = np.array(img, dtype=np.uint8)
    img_data = img_data / 255.0  # Normalize pixel values
    img_data = np.expand_dims(img_data, axis=0)  # Add a new axis at the 0th position (batch)
    return img, img_data

# Get a list of all image files in the folder
test_image_paths = [os.path.join(test_folder, fname) for fname in os.listdir(test_folder) if fname.endswith(('.jpg', '.jpeg', '.png', '.webp'))]

# Loop through each image, preprocess, predict, and display the result (probabality distribution)
for i in range(len(test_image_paths)):
    test_image, test_image_prep = preprocess_image(test_image_paths[i])
    fix, ax = plt.subplots(1, 2, figsize = (12, 6))
    ax[0].imshow(test_image)
    ax[0].axis('off')
    ax[0].set_title("Image")
    ax[1].bar(np.arange(0, 10), model.predict(test_image_prep)[0])
    ax[1].set_xticks(np.arange(0, 10))
    ax[1].set_xticklabels(class_names)
    plt.tight_layout()
    plt.show()
