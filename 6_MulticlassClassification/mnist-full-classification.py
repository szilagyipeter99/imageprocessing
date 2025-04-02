from keras import utils, layers, Sequential
import matplotlib.pyplot as plt

# Load and preprocess images
data_dir = "path-to-resources/mnist-full/train"
train_data, val_data = utils.image_dataset_from_directory(
    data_dir,
    labels = "inferred",
    label_mode = "int", # Classes are integers
    batch_size = 128, # Default would be 32
    image_size = (28, 28),
    color_mode = "grayscale",
    subset = "both",
    validation_split = 0.2,  # 20% of data used for validation
    seed = 123  # Optional random seed for shuffling and transformations
)

# Normalize the images
train_data = train_data.map(lambda image, label: (image / 255.0, label))
val_data = val_data.map(lambda image, label: (image / 255.0, label))

# Build the CNN model
model = Sequential([
    layers.Input(shape = (28, 28, 1)),
    layers.Conv2D(32, kernel_size = (3, 3), activation = "relu"),
    layers.MaxPooling2D(pool_size = (2, 2)),
    layers.Conv2D(64, kernel_size = (3, 3), activation = "relu"),
    layers.MaxPooling2D(pool_size = (2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax"), # 10 output neurons with softmax (probability distribution)
])
model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"]) # Not the 'binary_crossentropy' loss function anymore
model.summary()

# Train the model for 10 epochs and save the history
history = model.fit(train_data, epochs = 10, validation_data = val_data)

# Evaluate the trained model
score = model.evaluate(val_data, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Save the model in a '.keras' file
model.save('mnist_full_classifier.keras')

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
