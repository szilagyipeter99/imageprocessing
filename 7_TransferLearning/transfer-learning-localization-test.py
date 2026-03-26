import os
from keras import models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

model = models.load_model("dog_localization.keras")

test_folder = "resources/dog-sequence"

# Attribution:
# https://www.vecteezy.com/free-videos/adorable - Adorable Stock Videos by Vecteezy

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    img_data = np.array(img, dtype=np.uint8)
    img_data = img_data / 255.0
    img_data = np.expand_dims(img_data, axis=0)
    return img, img_data

test_image_paths = [os.path.join(test_folder, fname) for fname in os.listdir(test_folder) if fname.endswith((".jpg", ".jpeg", ".png", ".webp"))]

# Loop for sequence:

fig, ax = plt.subplots()

for file in sorted(test_image_paths):
    test_img, test_img_prep = preprocess_image(file)
    bounding_box_pred = model.predict(test_img_prep)[0]
    xmin = bounding_box_pred[0] * 224
    ymin = bounding_box_pred[1] * 224
    xmax = bounding_box_pred[2] * 224
    ymax = bounding_box_pred[3] * 224
    box_width = xmax - xmin
    box_height = ymax - ymin
    ax.clear()
    ax.imshow(test_img)
    rect = patches.Rectangle((xmin, ymin), box_width, box_height, linewidth=2, edgecolor = "red", facecolor = "none")
    ax.add_patch(rect)
    ax.axis("off")    
    plt.pause(.033)


# Loop for individual images

# for file in sorted(test_image_paths):
#     test_img, test_img_prep = preprocess_image(file)
#     bounding_box_pred = model.predict(test_img_prep)[0]
#     xmin = bounding_box_pred[0] * 224
#     ymin = bounding_box_pred[1] * 224
#     xmax = bounding_box_pred[2] * 224
#     ymax = bounding_box_pred[3] * 224
#     box_width = xmax - xmin
#     box_height = ymax - ymin
#     fig, ax = plt.subplots()
#     ax.imshow(test_img)
#     rect = patches.Rectangle((xmin, ymin), box_width, box_height, linewidth=2, edgecolor = "red", facecolor = "none")
#     ax.add_patch(rect)
#     ax.axis("off")    
#     plt.show()
