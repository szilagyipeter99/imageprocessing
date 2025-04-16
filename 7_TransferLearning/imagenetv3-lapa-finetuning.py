import os
import tensorflow as tf
from keras import layers, applications, Model
from PIL import Image
import numpy as np 

dataset_folder = 'path-to-resources/LaPa'

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    img_data = np.array(img, dtype=np.uint8)
    img_data = img_data / 255.0
    return img_data

def load_landmarks(txt_path):
    with open(txt_path, "r") as f:
        lines = f.readlines()[1:]
        coords = [list(map(float, line.strip().split())) for line in lines]
        coords = np.array(coords)
        coords[:, 0] = coords[:, 0] / 224
        coords[:, 1] = coords[:, 1] / 224
        return coords.flatten().astype(np.float32)


def create_dataset(image_dir, txt_dir):
    image_file_names = [fname for fname in os.listdir(image_dir) if fname.endswith('.jpg')]
    def gen():
        for fname in image_file_names:
            img_path = os.path.join(image_dir, fname)
            txt_name = os.path.splitext(fname)[0] + ".txt"
            txt_path = os.path.join(txt_dir, txt_name)
            img_data = preprocess_image(img_path)
            landmarks = load_landmarks(txt_path)
            yield img_data.astype(np.float32), landmarks
    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(212,), dtype=tf.float32)
        )
    )
    return ds.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)

inputs = layers.Input(shape=(224, 224, 3))
backbone = applications.MobileNetV3Small(include_top=False, weights="imagenet", input_tensor=inputs)
backbone.trainable = True
x = backbone.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(212, activation="sigmoid")(x)
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='mse', metrics=['mae']) # Mean Absolute Error is more human-readable during training than MSE
model.summary()

train_ds = create_dataset(f"{dataset_folder}/train/images", f"{dataset_folder}/train/landmarks").repeat()
val_ds = create_dataset(f"{dataset_folder}/val/images", f"{dataset_folder}/val/landmarks").repeat()

model.fit(train_ds, validation_data=val_ds, steps_per_epoch=568, validation_steps=63, epochs=25)

model.save('mobilenetv2_lapa.keras')
