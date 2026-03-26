import os
import xml.etree.ElementTree as ET
import numpy as np
from keras import layers, applications, optimizers, Sequential, losses, utils
from PIL import Image
from sklearn.model_selection import train_test_split

# Dataset source: https://cg.cs.tsinghua.edu.cn/ThuDogs/

DATASET_PATH = "resources/thu-dogs"
IMAGE_DIR = os.path.join(DATASET_PATH, "images")
ANNOTATION_DIR = os.path.join(DATASET_PATH, "annotations")

# Memory-friendly dataset loader
class DogGen(utils.Sequence):
    def __init__(self, annotations, batch_size = 16, image_size = (224, 224), shuffle = True):
        self.annotations = annotations
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.on_epoch_end()
    def __len__(self):
        return int(np.ceil(len(self.annotations) / self.batch_size))
    def __getitem__(self, idx):
        batch = self.annotations[idx * self.batch_size : (idx + 1) * self.batch_size]
        x, y = [], []
        for item in batch:
            img = Image.open(item['path']).convert("RGB").resize(self.image_size)
            x.append(np.array(img, dtype = np.float32) / 255.0)
            y.append(item['bbox'])
        return np.array(x), np.array(y)
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.annotations)

# Find images and head bounding boxes
annotations = []
for root_dir, _, files in os.walk(ANNOTATION_DIR):
    for file in files:
        if not file.endswith(".xml"):
            continue
        xml_path = os.path.join(root_dir, file)
        root = ET.parse(xml_path).getroot()
        bbox = root.find("object").find("headbndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        filename = root.find("filename").text
        folder = root.find("folder").text
        img_path = os.path.join(IMAGE_DIR, folder, filename)
        img = Image.open(img_path)
        width, height = img.size
        annotations.append({
            "path": img_path,
            "bbox": [xmin / width, ymin / height, xmax / width, ymax / height]
        })

# Load the dataset
train_ann, val_ann = train_test_split(annotations, test_size = 0.2)
train_gen = DogGen(train_ann)
val_gen = DogGen(val_ann)

# Load a ResNet50 model pretrained on ImageNet (BACKBONE)
base_model = applications.ResNet50(
    weights = "imagenet",
    include_top = False,
    input_shape = (224, 224, 3)
)

# Disable training for the whole model
for layer in base_model.layers:
    layer.trainable = False

# Add extra layers for localization (HEAD)
model = Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation = "relu"),
    layers.Dropout(0.3),
    layers.Dense(64, activation = "relu"),
    layers.Dropout(0.2),
    layers.Dense(4, activation = "sigmoid")
])

# Train the head
model.compile(optimizer = "adam", loss = losses.Huber())
model.summary()
print("\n--- Training the head: ---")
model.fit(train_gen, validation_data = val_gen, epochs = 1, batch_size = 16)

# Enable training for the last 30 layers of the backbone
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Finetune the model for 3 epochs
model.compile(optimizer = optimizers.Adam(1e-5), loss = losses.Huber())
print("\n--- Finetuning the model: ---")
model.fit(train_gen, validation_data = val_gen, epochs = 3, batch_size = 16)

model.save("dog_localization.keras")
