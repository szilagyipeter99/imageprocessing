import os
from keras import models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the saved model
model = models.load_model('mnist_binary_classifier.keras')

test_folder = 'path-to-resources/mnist-binary/test'

# Define classes
class_1 = 7 # Would also work as "Seven"
class_2 = 8 # Would also work as "Eight"

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

# Loop through each image, preprocess, predict, and display the result
plt.figure(figsize=(10, 5))
for i in range(10): 
    test_image, test_image_prep = preprocess_image(test_image_paths[i])
    prediction = model.predict(test_image_prep)[0][0]
    predicted_class = class_2 if prediction > 0.5 else class_1
    confidence = prediction if prediction > 0.5 else 1 - prediction
    plt.subplot(2, 5, i + 1)
    plt.imshow(test_image, cmap="gray")
    plt.axis('off')
    plt.title(f"Predicted: {predicted_class}\n{confidence:.8f}")
plt.show()
