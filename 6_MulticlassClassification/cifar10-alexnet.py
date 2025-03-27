from keras import datasets, utils, layers, Sequential
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
(x_train, y_train), (x_val, y_val) = datasets.cifar10.load_data()

# Normalize images
x_train, x_val = x_train / 255.0, x_val / 255.0

# One-hot encoding
y_train = utils.to_categorical(y_train, 10)
y_val = utils.to_categorical(y_val, 10)

# Data Augmentation (AlexNet used heavy augmentation)
data_augmentation = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# AlexNet-inspired model
model = Sequential([
    layers.Input(shape=(32, 32, 3)), # CIFAR-10 is 32x32, AlexNet was 227x227
    data_augmentation,
    layers.Conv2D(96, kernel_size = (3, 3), strides = 1, activation = 'relu', padding = 'same'), # Original AlexNet: (11x11, stride 4)
    layers.BatchNormalization(), # Alexnet used Local Response Normalization (LRN), Batch Normalization (BN) was introduced in 2015 by Sergey Ioffe and Christian Szegedy
    layers.MaxPooling2D(pool_size = (2, 2), strides = 2), # Original AlexNet: (3x3, stride 2)
    layers.Conv2D(256, kernel_size = (3, 3), padding='same', activation='relu'), # Original AlexNet: (5x5)
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size = (2, 2), strides = 2),
    layers.Conv2D(384, kernel_size = (3, 3), padding='same', activation='relu'),
    layers.Conv2D(384, kernel_size = (3, 3), padding='same', activation='relu'),
    layers.Conv2D(256, kernel_size = (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size = (2, 2), strides = 2),
    layers.Flatten(),
    layers.Dense(1024, activation='relu'), # Original AlexNet: 4096
    layers.Dropout(0.5),
    layers.Dense(1024, activation='relu'), # Original AlexNet: 4096
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax') # CIFAR-10 has 10 classes, AlexNet had 1000
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(x_train, y_train, epochs = 10, batch_size = 128, validation_data=(x_val, y_val))

# Evaluate the trained model
score = model.evaluate(x_val, y_val, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Save the model
model.save('cifar10_alexnet.keras')

# Plot accuracy and loss for both training and validation
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(history.history['accuracy'], label='Train Accuracy')
ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Accuracy')
ax[0].legend()
ax[0].set_title('Training & Validation Accuracy')
ax[1].plot(history.history['loss'], label='Train Loss')
ax[1].plot(history.history['val_loss'], label='Validation Loss')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Loss')
ax[1].legend()
ax[1].set_title('Training & Validation Loss')
plt.tight_layout()
plt.show()
