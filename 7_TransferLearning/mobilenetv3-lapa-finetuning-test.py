import os
from keras import models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the saved model
model = models.load_model('mobilenetv3_lapa.keras')

test_folder = 'resources/LaPa/test/images/'

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((256, 256))
    img_data = np.array(img, dtype=np.uint8)
    img_data = img_data / 255.0
    return img, img_data

# Get a list of all image files in the folder
test_image_paths = [os.path.join(test_folder, fname) for fname in os.listdir(test_folder) if fname.endswith('.jpg')]

# Loop through each image, preprocess, predict, and display the result (probabality distribution)
for i in range(len(test_image_paths)):
    test_image, test_image_prep = preprocess_image(test_image_paths[i])
    input_data = np.expand_dims(test_image_prep, axis=0)
    predicted_landmarks = model.predict(input_data)[0]
    points = predicted_landmarks.reshape(-1, 2)
    points[:, 0] *= 256
    points[:, 1] *= 256
    plt.figure(figsize=(5, 5))
    plt.imshow(test_image)
    plt.scatter(points[:, 0], points[:, 1], c='red', s=10)
    plt.title(f"Prediction {i+1}")
    plt.axis('off')
    plt.show()
