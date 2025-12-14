from keras import datasets, utils, layers, Sequential, optimizers
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
(x_train, y_train), (x_val, y_val) = datasets.cifar10.load_data()

# Normalize images
x_train, x_val = x_train / 255.0, x_val / 255.0

# One-hot encoding
y_train = utils.to_categorical(y_train, 10)
y_val = utils.to_categorical(y_val, 10)

# Data Augmentation
data_augmentation = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1)
])

model = Sequential([
    layers.Input(shape=(32, 32, 3)), # CIFAR-10 images are 32x32
    data_augmentation,
    layers.Conv2D(96, kernel_size = (3, 3), padding='same', activation = None),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size = (2, 2), strides = 2),
    layers.Conv2D(256, kernel_size = (3, 3), padding='same', activation = None),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size = (2, 2), strides = 2),
    layers.Conv2D(384, kernel_size = (3, 3), padding = 'same', activation = 'relu'),
    layers.Conv2D(384, kernel_size = (3, 3), padding = 'same', activation = 'relu'),
    layers.Conv2D(256, kernel_size = (3, 3), padding = 'same', activation = 'relu'),
    layers.MaxPooling2D(pool_size = (2, 2), strides = 2),
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax') # CIFAR-10 has 10 classes
])
# Stochastic Gradient Descent optimizer
optimizer = optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(x_train, y_train, epochs = 30, batch_size = 128, validation_data = (x_val, y_val))

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
