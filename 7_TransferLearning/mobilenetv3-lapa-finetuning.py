import os
import tensorflow as tf
from keras import layers, applications, Model
from PIL import Image
import numpy as np 

dataset_folder = 'resources/LaPa'

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img_h, img_w = img.size[1], img.size[0] 
    img = img.resize((256, 256))
    img_data = np.array(img, dtype=np.uint8)
    img_data = img_data / 255.0
    return img_data, img_h, img_w

def load_landmarks(txt_path, img_h, img_w):
    with open(txt_path, "r") as f:
        lines = f.readlines()[1:]
        coords = [list(map(float, line.strip().split())) for line in lines]
        coords = np.array(coords)
        coords[:, 0] = coords[:, 0] / img_w
        coords[:, 1] = coords[:, 1] / img_h
        return coords.flatten().astype(np.float32)


def create_dataset(image_dir, txt_dir):
    image_file_names = [fname for fname in os.listdir(image_dir) if fname.endswith('.jpg')]
    def gen():
        for fname in image_file_names:
            img_path = os.path.join(image_dir, fname)
            txt_name = os.path.splitext(fname)[0] + ".txt"
            txt_path = os.path.join(txt_dir, txt_name)
            img_data, img_h, img_w = preprocess_image(img_path)
            landmarks = load_landmarks(txt_path, img_h, img_w)
            yield img_data.astype(np.float32), landmarks
    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(212,), dtype=tf.float32)
        )
    )
    return ds.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)

inputs = layers.Input(shape=(256, 256, 3))
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

model.fit(train_ds, validation_data=val_ds, steps_per_epoch=568, validation_steps=63, epochs=5)

model.save('mobilenetv3_lapa.keras')
