#!/usr/bin/env python
# coding: utf-8

# In[8]:


# ========================== CNN_SCRATCH ============================

# =========================== DATA LOADING ===========================

import os
import numpy as np
import rasterio
import tensorflow as tf
import pathlib
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers

# Paths 
train_dir = r"E:/Sentinel_Dataset/cnn/model_datasets/core_dataset/train"
val_dir   = r"E:/Sentinel_Dataset/cnn/model_datasets/core_dataset/validation"
test_dir  = r"E:/Sentinel_Dataset/cnn/model_datasets/core_dataset/test"

# Helper to load .tif
def load_tif(path):
    with rasterio.open(path) as src:
        img = src.read()  # (bands, H, W)
        img = np.moveaxis(img, 0, -1)  # -> (H, W, bands)
    return img.astype("float32")

# Function to process a directory
def process_dir(data_dir):
    images, labels = [], []
    class_names = sorted(os.listdir(data_dir))
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for file in os.listdir(class_dir):
            if file.endswith(".tif"):
                img = load_tif(os.path.join(class_dir, file))
                images.append(img)
                labels.append(label)
    return np.stack(images), np.array(labels)

# Process and save
train_images, train_labels = process_dir(train_dir)
val_images, val_labels     = process_dir(val_dir)
test_images, test_labels   = process_dir(test_dir)

# Compute per-band mean/std from training set
mean = train_images.mean(axis=(0,1,2))
std  = train_images.std(axis=(0,1,2))

# Normalize
train_images = (train_images - mean) / std
val_images   = (val_images - mean) / std
test_images  = (test_images - mean) / std

# Build tensorflow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(len(train_images)).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset   = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset  = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(32).prefetch(tf.data.AUTOTUNE)

# Optional: Print one batch to check shapes
for data_batch, labels_batch in train_dataset:
    print("data batch shape:", data_batch.shape)
    print("labels batch shape:", labels_batch.shape)
    break

# ==================== CNN ARCHITECTURE DEFINITION ======================

# Input layer accepting images of shape 64x64 with 9 channels
inputs = keras.Input(shape=(64, 64, 9))

# First conv block
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
x = layers.MaxPooling2D((2, 2))(x)   # 64 -> 32

# Second conv block
x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2))(x)   # 32 -> 16

x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2))(x)

# Flatten + Dense layers
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
x = layers.Dropout(0.7)(x)

# Output layer (3 classes: none, weak, strong)
outputs = layers.Dense(3, activation="softmax")(x)

# Create the model
model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()

# ========================== MODEL COMPILATION ==========================

# optimizer = optimizers.xyz(learning_rate=1e-3)
model.compile(optimizer=optimizers.SGD(learning_rate=1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# ========================== MODEL TRAINING ============================

# Save & name the best model as "convnet_core.keras" automatically using a callback
callbacks = [
    keras.callbacks.ModelCheckpoint(filepath="convnet_core_scratch.keras", save_best_only=True, monitor="val_loss")
]

# Train the model
history = model.fit(train_dataset, epochs=20, validation_data=val_dataset, callbacks=callbacks, verbose=1)

# ========================== PLOT THE RESULTS ==========================

history_dict = history.history
training_loss = history_dict["loss"]
val_loss = history_dict["val_loss"]
training_acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]

# Create figure
plt.figure(figsize=(12, 5))

# First subplot
plt.subplot(1, 2, 1)  # 1 row, 2 columns, position 1
epochs = range(1, len(training_loss) + 1)
plt.plot(epochs, training_loss, "bo-", label="Training Loss")
plt.plot(epochs, val_loss, "b-", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
epochs = range(1, len(training_acc) + 1)
plt.plot(epochs, training_acc, "bo-", label="Training Accuracy")
plt.plot(epochs, val_acc, "b-", label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
plt.close()

# ========================== EVALUATE ON TEST DATA ==========================

# Load the best saved model and evaluate on test dataset
test_model = keras.models.load_model("convnet_core_scratch.keras")
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}")


# In[6]:


# ====================== CNN_SCRATCH_AUGMENTATION =======================

# =========================== DATA LOADING ===========================

import os
import numpy as np
import rasterio
import tensorflow as tf
import pathlib
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers

# Paths 
train_dir = r"E:/Sentinel_Dataset/cnn/model_datasets/core_dataset/train"
val_dir   = r"E:/Sentinel_Dataset/cnn/model_datasets/core_dataset/validation"
test_dir  = r"E:/Sentinel_Dataset/cnn/model_datasets/core_dataset/test"

# Helper to load .tif
def load_tif(path):
    with rasterio.open(path) as src:
        img = src.read()  # (bands, H, W)
        img = np.moveaxis(img, 0, -1)  # -> (H, W, bands)
    return img.astype("float32")

# Function to process a directory
def process_dir(data_dir):
    images, labels = [], []
    class_names = sorted(os.listdir(data_dir))
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for file in os.listdir(class_dir):
            if file.endswith(".tif"):
                img = load_tif(os.path.join(class_dir, file))
                images.append(img)
                labels.append(label)
    return np.stack(images), np.array(labels)

# Process and save
train_images, train_labels = process_dir(train_dir)
val_images, val_labels     = process_dir(val_dir)
test_images, test_labels   = process_dir(test_dir)

# Compute per-band mean/std from training set
mean = train_images.mean(axis=(0,1,2))
std  = train_images.std(axis=(0,1,2))

# Normalize
train_images = (train_images - mean) / std
val_images   = (val_images - mean) / std
test_images  = (test_images - mean) / std

# Build tensorflow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(len(train_images)).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset   = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset  = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(32).prefetch(tf.data.AUTOTUNE)

# Optional: Print one batch to check shapes
for data_batch, labels_batch in train_dataset:
    print("data batch shape:", data_batch.shape)
    print("labels batch shape:", labels_batch.shape)
    break

# ====================== DATA AUGMENTATION LAYER ========================

# An object of keras.Sequential class: flip, rotate & zoom
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2)
])

# Optional: Lets have a plot of (10, 10)
plt.figure(figsize=(10,10))

# Fetch only one batch & loop over images only, ignoring labels 
for images, _ in train_dataset.take(1):
    # Each iteration applies random augmentation to the entire batch
    for i in range(6):
        augmented_images = data_augmentation(images)
        # pick the first image from the batch
        img = augmented_images[0].numpy()
        # ---- choose 3 bands for composite (example: 4-3-2 natural color) ----
        r, g, b = img[:, :, 4], img[:, :, 3], img[:, :, 2]
        rgb = np.stack([r, g, b], axis=-1)

        # normalize to 0–1 for display
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

        # plot
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(rgb)
        plt.axis("off")
        plt.title(f"Aug {i+1}")


plt.suptitle("Data Augmentation Examples")
plt.tight_layout()
plt.show()
plt.close()

# ==================== CNN ARCHITECTURE DEFINITION ======================

# Input layer accepting images of shape 64x64 with 9 channels
inputs = keras.Input(shape=(64, 64, 9))

x = data_augmentation(inputs)

# First conv block
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2))(x)   # 64 -> 32

# Second conv block
x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2))(x)   # 32 -> 16

# Third conv block
x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2))(x)   # 16 -> 8

# Flatten + Dense layers
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
x = layers.Dropout(0.7)(x)

# Output layer (3 classes: none, weak, strong)
outputs = layers.Dense(3, activation="softmax")(x)

# Create the model
model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()

# ========================== MODEL COMPILATION ==========================

# optimizer = optimizers.xyz(learning_rate=1e-3)
model.compile(optimizer=optimizers.SGD(learning_rate=1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# ========================== MODEL TRAINING ============================

# Save & name the best model as "convnet_core.keras" automatically using a callback
callbacks = [
    keras.callbacks.ModelCheckpoint(filepath="convnet_core_aug.keras", save_best_only=True, monitor="val_loss")
]

# Train the model
history = model.fit(train_dataset, epochs=50, validation_data=val_dataset, callbacks=callbacks, verbose=1)

# ========================== PLOT THE RESULTS ==========================

history_dict = history.history
training_loss = history_dict["loss"]
val_loss = history_dict["val_loss"]
training_acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]

# Create figure
plt.figure(figsize=(12, 5))

# First subplot
plt.subplot(1, 2, 1)  # 1 row, 2 columns, position 1
epochs = range(1, len(training_loss) + 1)
plt.plot(epochs, training_loss, "bo-", label="Training Loss")
plt.plot(epochs, val_loss, "b-", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
epochs = range(1, len(training_acc) + 1)
plt.plot(epochs, training_acc, "bo-", label="Training Accuracy")
plt.plot(epochs, val_acc, "b-", label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
plt.close()

# ========================== EVALUATE ON TEST DATA ==========================

# Load the best saved model and evaluate on test dataset
test_model = keras.models.load_model("convnet_core_aug.keras")
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}")


# In[ ]:





# In[2]:


# ========== CNN_PRETRAINED(ResNet50)_FINETUNING ============
# ================== END TO END TRAINING ======================

import os
import numpy as np
import rasterio
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
from tensorflow.keras.applications import ResNet50
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="keras.src.applications.resnet")

# =========================== DATA LOADING ===========================

# Paths 
train_dir = r"E:/Sentinel_Dataset/cnn/model_datasets/core_dataset/train"
val_dir   = r"E:/Sentinel_Dataset/cnn/model_datasets/core_dataset/validation"
test_dir  = r"E:/Sentinel_Dataset/cnn/model_datasets/core_dataset/test"

# Helper to load .tif
def load_tif(path):
    with rasterio.open(path) as src:
        img = src.read()  # (bands, H, W)
        img = np.moveaxis(img, 0, -1)  # -> (H, W, bands)
    return img.astype("float32")

# Function to process a directory
def process_dir(data_dir):
    images, labels = [], []
    class_names = sorted(os.listdir(data_dir))
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for file in os.listdir(class_dir):
            if file.endswith(".tif"):
                img = load_tif(os.path.join(class_dir, file))
                images.append(img)
                labels.append(label)
    return np.stack(images), np.array(labels)

# Process and save
train_images, train_labels = process_dir(train_dir)
val_images, val_labels     = process_dir(val_dir)
test_images, test_labels   = process_dir(test_dir)

# Compute per-band mean/std from training set
mean = train_images.mean(axis=(0,1,2))
std  = train_images.std(axis=(0,1,2))

# Normalize
train_images = (train_images - mean) / std
val_images   = (val_images - mean) / std
test_images  = (test_images - mean) / std

# Build tensorflow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(len(train_images)).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset   = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset  = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(32).prefetch(tf.data.AUTOTUNE)

# Optional: Print one batch to check shapes
for data_batch, labels_batch in train_dataset:
    print("data batch shape:", data_batch.shape)
    print("labels batch shape:", labels_batch.shape)
    break            

# ================== LOAD PRETRAINED ResNet50 CONV BASE ==================

# Load pretrained weights trained on the BigEarthNet dataset
# Exclude the fully connected layers at the top of the network

# ResNet50 base (frozen)
conv_base = ResNet50(include_top=False,
                     weights=None,
                     input_shape=(120, 120, 10),
                     pooling=None)

# conv_base.load_weights from checkpoint
ckpt = tf.train.Checkpoint(model=conv_base)
ckpt.restore(r"E:/Sentinel_Dataset/checkpoint_ResNet50/checkpoints-3").expect_partial()

# Freezing ResNet50 convolutional base layers to retain their weight
conv_base.trainable = False

# ===================== DATA AUGMENTATION LAYER ========================

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2)
])

# ================== CNN ARCHITECTURE DEFINITION ====================

# Input (our 9-band images)
inputs = keras.Input(shape=(64, 64, 9))

# Call the data_augmentation object on input images
x = data_augmentation (inputs)

# 1×1 spectral adapter (trainable)
x = layers.Resizing(120, 120)(x)              # upscale to ResNet50 input
x = layers.Conv2D(10, (1, 1), padding="same", activation="linear")(x)  # map 9→10 bands

# Pass through frozen ResNet50
x = conv_base(x)

# Dense classifier
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(3, activation="softmax")(x)  # 3 classes

# Full model
model = keras.Model(inputs, outputs)

# ========================== MODEL COMPILATION ==========================

# optimizer = optimizers.xyz(learning_rate=1e-3)
model.compile(optimizer=optimizers.AdamW(learning_rate=1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# ========================== MODEL TRAINING ============================

# Save & name the best model automatically using a callback
callbacks = [
    keras.callbacks.ModelCheckpoint(filepath="convnet_core_finetuned_1.keras", save_best_only=True, monitor="val_loss")
]

# Train the model
history = model.fit(train_dataset, epochs=50, validation_data = val_dataset, callbacks = callbacks)

# ========================== PLOT THE RESULTS ==========================

history_dict = history.history
training_loss = history_dict["loss"]
val_loss = history_dict["val_loss"]
training_acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]

# Create figure
plt.figure(figsize=(12, 5))

# First subplot
plt.subplot(1, 2, 1)  # 1 row, 2 columns, position 1
epochs = range(1, len(training_loss) + 1)
plt.plot(epochs, training_loss, "bo-", label="Training Loss")
plt.plot(epochs, val_loss, "b-", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
epochs = range(1, len(training_acc) + 1)
plt.plot(epochs, training_acc, "bo-", label="Training Accuracy")
plt.plot(epochs, val_acc, "b-", label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
plt.close()

# ========================== EVALUATE ON TEST DATA ==========================

# Load the best saved model and evaluate on test dataset
test_model = keras.models.load_model("convnet_core_finetuned_1.keras")
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}")


# In[4]:


# As we have successfully trained our custom dense classifier by keeping VGG16 conv_base frozen
# Now we can safely unfreeze & train last 3 layers of conv_base too by fine-tuning
# Therefore LOADING "already trained custom dense classifier" fresh model

model = keras.models.load_model("convnet_core_finetuned_1.keras")

# Confirm CNN Architecture layer that represents frozen conv_base
print("=== Model Layers ===")
for idx, layer in enumerate(model.layers):
    print(f"Layer {idx}: {layer.name}")

# Get the conv_base from the loaded model
conv_base = model.get_layer("resnet50")

# Unfreeze last two ResNet50 stages (conv4_x and conv5_x)
conv_base.trainable = True  

for layer in conv_base.layers:
    if layer.name.startswith("conv4") or layer.name.startswith("conv5"):
        layer.trainable = True   # fine-tune these layers
    else:
        layer.trainable = False  # keep earlier layers frozen

# OPTIONAL: freeze BatchNorm layers to avoid training instability
for layer in conv_base.layers:
    if "bn" in layer.name:  # batch normalization layers contain "bn"
        layer.trainable = False

# Print summary of trainable layers
print("Trainable layers in conv_base:")
for layer in conv_base.layers:
    print(layer.name, "trainable=", layer.trainable)

# ====================== MODEL RE-COMPILATION ======================

# Recompile with a lower learning rate for fine-tuning
# optimizer = optimizers.xyz(learning_rate=1e-3)
model.compile(optimizer=optimizers.AdamW(learning_rate=1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# ================= MODEL RE-TRAINING (End to End) ===================

# It will extract fresh features using 70% frozen & 30% unfrozen parts of conv_base during forward pass
# It will re-train the 30% unfrozen part of conv_base alongwith our custom dense classifier during back pass

fine_tune_callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="convnet_core_finetuned_2.keras",
        save_best_only=True,
        monitor="val_loss")
]

# Re-train the 30% unfrozen part of conv_base alongwith our custom dense classifier on newly extracted features
history = model.fit(train_dataset, epochs=30, validation_data=val_dataset, callbacks=fine_tune_callbacks)

# ========================== PLOT THE RESULTS ==========================

history_dict = history.history
training_loss = history_dict["loss"]
val_loss = history_dict["val_loss"]
training_acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]

# Create figure
plt.figure(figsize=(12, 5))

# First subplot
plt.subplot(1, 2, 1)  # 1 row, 2 columns, position 1
epochs = range(1, len(training_loss) + 1)
plt.plot(epochs, training_loss, "bo-", label="Training Loss")
plt.plot(epochs, val_loss, "b-", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
epochs = range(1, len(training_acc) + 1)
plt.plot(epochs, training_acc, "bo-", label="Training Accuracy")
plt.plot(epochs, val_acc, "b-", label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
plt.close()

# ========================== EVALUATE ON TEST DATA ==========================

# Load the best saved model and evaluate on test dataset
test_model = keras.models.load_model("convnet_core_finetuned_2.keras")
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}")











