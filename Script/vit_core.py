#!/usr/bin/env python
# coding: utf-8

# In[2]:


# ========================== VIT_SCRATCH ============================

import os
import numpy as np
import rasterio
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------- DATA LOADING ----------------------

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
batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(len(train_images)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset   = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_dataset  = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Optional: Print one batch to check shapes
for data_batch, labels_batch in train_dataset.take(1):
    print("data batch shape:", data_batch.shape)  # expected: (batch_size, 64, 64, 9)
    print("labels batch shape:", labels_batch.shape)
    break

# ==================== VISION TRANSFORMER (ViT) DEFINITION ======================

# Input images: (batch_size, 64, 64, 9)
# - Image size = 64 × 64
# - Patch size = 8 × 8
# → Patches per side = 64 / 8 = 8
# → Total patches = 8 × 8 = 64
# Each patch contains 8 × 8 × 9 = 576 values

# Transformers need sequence-like input.
# In NLP, a sentence → split into words → embed words → feed into Transformer.
# In ViT, an image → split into patches → embed patches → feed into Transformer.
# So patches act as the “tokens” of the image.

# Normally, a CNN looks at an image through local filters (kernels) sliding over the image.
# A Transformer, on the other hand, was originally designed for sequences (like words in a sentence).
# To make images look like a “sequence,” we cut the image into smaller square blocks, flatten each block into a vector, and treat each block like a "word token."
# These smaller square blocks are what we call image patches.

# ------- Create Image Patches --------

# Define a custom Keras layer that extracts image patches
class PatchExtractor(layers.Layer):

    # PatchExtractor constructor
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size # patch_size is the side length of each square patch

    # PatchExtractor forward pass
    def call(self, images):

        # images: (batch, Height, Width, Channel)
        batch_size = tf.shape(images)[0] # Extract batch size dynamically
        patch_size = self.patch_size
        
        # Call TensorFlow’s built-in function extract_patches, which cuts out smaller patches from images
        patches = tf.image.extract_patches(
            images=images, # input images
            
            # Tell TF the size of the patch to extract. The list has 4 values because TF expects [batch, height, width, channels].
            # We don’t want to change batch size or channels, so we set those to 1.
            # The middle two values (patch_size, patch_size) define the patch size along height and width. e.g. If patch_size=8, then each extracted patch is 8 × 8 pixels

            sizes=[1, patch_size, patch_size, 1], # [1, 8, 8, 1]
            
            # Tell TF how far to “move the window” when extracting patches. [1, stride_height, stride_width, 1].
            # If we set stride equal to the patch size, it means patches don’t overlap.
            # Here, stride = patch_size, so the patches exactly tile the image without overlap.

            strides=[1, patch_size, patch_size, 1], # non-overlapping
            
            # Mention patch rate. [1, rate_height, rate_width, 1]
            # A rate > 1 would skip pixels inside the patch
            # But here, [1, 1, 1, 1] means normal patches (no skipping)

            rates=[1, 1, 1, 1],
            
            # Define how to handle image edges
            # "VALID" means only extract full patches that fit inside the image (no padding added)

            padding="VALID")
        
        # patches came out of tf.image.extract_patches with shape: (batch, new_h, new_w, patch_size*patch_size*C)
        # batch: number of images in the batch
        # new_h, new_w: how many patches fit along height/width
        # patch_size*patch_size*C: each patch flattened into a vector
        # patches.shape[-1] → the last dimension (the flattened size of each patch)

        patch_dims = patches.shape[-1] # = 576
        
        # Now patches look like: (batch, new_h, new_w, patch_dims)
        # But a Transformer expects a sequence of tokens, not a 2D grid. So we flatten the (new_h, new_w) grid into a single sequence dimension.
        # -1 tells TensorFlow: “compute this dimension automatically.”
        # Reshape method converts the patches 2D grid into a sequence so the image looks like a “sentence of tokens” to the Transformer

        patches = tf.reshape(patches, [batch_size, -1, patch_dims])  # (batch, num_patches, patch_dims) (32, 64, 576)
        return patches

    # Method to save/load the layer
    def get_config(self):
        return {"patch_size": self.patch_size}

# ------- Apply Patch Embedding --------

# Patch embedding is needed because raw patch vectors (576-dim) are too large and arbitrary.
# A Dense layer projects them into a fixed, learnable embedding space (e.g., 64-dim), making them suitable for the Transformer’s attention and aligning image tokens with how words are embedded in NLP.

# Define a custom Keras layer that has num_patches = 64 (from patch extractor) & embed_dim = 64 (desired embedding size).
class PatchEmbedding(layers.Layer):
    def __init__(self, num_patches, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        # Dense layer projects each patch (576-dim) → embedding vector (64-dim)
        self.proj = layers.Dense(embed_dim)
        # apply positional embedding for patches
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(1, num_patches, embed_dim),
            initializer="random_normal",
            trainable=True,
        )
        
    # PatchEmbedding forward pass
    def call(self, patches):
        # Each flattened patch is passed through the Dense layer to get embedded vector
        x = self.proj(patches)  # (batch, num_patches, embed_dim) (32, 64, 64)
        
        # Adds the positional embedding to each patch embedding (element-wise addition)
        # Now, each token (patch) not only carries its content info but also knows its position in the original image
        # Return the final tokens ready to go into the Transformer encoder
        
        x = x + self.pos_embedding
        return x

    # Method to save/load the layer
    def get_config(self):
        return {"num_patches": self.num_patches, "embed_dim": self.embed_dim}

# ------- Transformer Encoder Block ---------

# We are defining our own custom layer, named TransformerEncoder.
# Hey Keras, I want to build my own custom layer, and I will use your built-in Layer class of layer module as the foundation to do it.
# Treat my class TransformerEncoder like a layer.
# I’ll define how this layer behaves (what it does when data flows through it).

class TransformerEncoderBlock(layers.Layer):
    
    # __init__ is a special method in Python. It's called automatically when you create an object from a class.
    # It’s like the constructor — it builds the object and sets it up.
    # self: refers to the current object.
    # embed_dim: size of embedded token vector.
    # dense_dim: size of the hidden layer in the dense network.
    # num_heads: how many attention heads to use.
    # **kwargs: keyword argument, a way to accept extra optional settings, like the layer's name or whether it's trainable
    # This function accepts three known arguments (embed_dim, dense_dim, num_heads)
    # The function also accepts any number of extra arguments, collected into kwargs internally by keras

    def __init__(self, embed_dim, num_heads, mlp_dim, dropout_rate=0.1, **kwargs):

        # Below is the constructor for the parent class, layers.Layer
        # Now pass all those extra arguments to the parent class's constructor.
        # This is where we're passing those extra arguments forward to built-in keras layer (parent class)
        # Keras knows how to handle these extra arguments in model processing

        super().__init__(**kwargs)

        # We are storing these values in the current object (self)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim # dense_dim

        # This is a built-in Keras layer for multi-head attention (used in Transformers).
        # key_dim=embed_dim means that each attention head has vectors of size embed_dim.

        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)

        # We add Normalization layer after every residual connections (add-and-norm), which is standard in Transformer blocks
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        
        # Keras dense layer sequential model definition
        # 1st dense layer uses dense_dim to expand the feature space (learning power).
        # 2nd dense layer uses embed_dim to compress back to the original size for compatibility with residual connections
        
        self.mlp = keras.Sequential(
            [
                layers.Dense(mlp_dim, activation="relu"),
                layers.Dropout(dropout_rate),
                layers.Dense(embed_dim),
                layers.Dropout(dropout_rate),
            ]
        )

    # This is the "forward pass" of our custom layer
    def call(self, inputs, training=False, mask=None):

        # Originally self-attention accepts parameters as:
        # self.attention(query=inputs, value=inputs, key=inputs, attention_mask=mask)
        # But Keras lets you skip key if it's the same as value in self-attention
        # Apply self-attention on inputs

        attn_out = self.attn(query=inputs, value=inputs, key=inputs, attention_mask=mask)

        # Add residual connection and apply layer normalization
        x = self.norm1(inputs + attn_out)

        # Apply feed-forward dense projection
        # training=training Ensures sublayers that dropout is active only during training

        mlp_out = self.mlp(x, training=training)

        # Add second residual connection and apply final normalization
        x = self.norm2(x + mlp_out)
        return x

    # Method to save/load the layer
    def get_config(self):
        return {"embed_dim": self.embed_dim, "num_heads": self.num_heads, "mlp_dim": self.mlp_dim}

# ------------------------ BUILDING ViT MODEL -----------------------

image_size = 64
channels = 9
patch_size = 8
num_patches = (image_size // patch_size) ** 2
embed_dim = 64
num_heads = 4
mlp_dim = 128
num_transformer_blocks = 3
dropout_rate = 0.1
num_classes = 3

# Input
inputs = keras.Input(shape=(image_size, image_size, channels))

# Make patches
patches = PatchExtractor(patch_size=patch_size)(inputs)  # (batch, num_patches, patch_dim)

# Perform patch embedding + positional embedding
x = PatchEmbedding(num_patches=num_patches, embed_dim=embed_dim)(patches)  # (batch, num_patches, embed_dim)

# Stack multiple Transformer Encoder blocks
for _ in range(num_transformer_blocks):
    x = TransformerEncoderBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        dropout_rate=dropout_rate
    )(x)

# Combine all patch embeddings into one feature representing the whole image
x = layers.LayerNormalization(epsilon=1e-6)(x)
image_repr = layers.GlobalAveragePooling1D()(x)   # (batch, embed_dim) (32, 64)

# Classification head
x = layers.Dense(16, activation="relu")(image_repr)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

# Create the model
model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()

# ------------------------- MODEL COMPILATION ------------------------

model.compile(optimizer=optimizers.SGD(learning_rate=1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# ------------------------- MODEL TRAINING -------------------------

callbacks = [
    keras.callbacks.ModelCheckpoint(filepath="vit_core_scratch.keras", save_best_only=True, monitor="val_loss")
]

history = model.fit(train_dataset, epochs=20, validation_data=val_dataset, callbacks=callbacks, verbose=1)

# ---------------------- PLOT THE RESULTS ------------------------

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

# ----------------------- EVALUATE ON TEST DATA -------------------------

# Load the best saved model and evaluate on test dataset
test_model = keras.models.load_model("vit_core_scratch.keras", custom_objects={
    "PatchExtractor": PatchExtractor,
    "PatchEmbedding": PatchEmbedding,
    "TransformerEncoderBlock": TransformerEncoderBlock
})
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}")


# In[4]:


# ----------------------- CLASSIFICATION REPORT -------------------------

from sklearn.metrics import classification_report
import numpy as np

# Get predictions on test set
y_true = []
y_pred = []

for images, labels in test_dataset:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

print(classification_report(y_true, y_pred, digits=3))


# In[6]:


# ----------------------- SAMPLE PREDICTION -------------------------

images, labels = next(iter(val_dataset))
preds = model.predict(images)
print("Sample predictions:", np.argmax(preds, axis=1))
print("True labels:", labels.numpy())


# In[8]:


# ========================== VIT_SCRATCH WITH AUGMENTATION ============================

import os
import numpy as np
import rasterio
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------- DATA LOADING ----------------------

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
batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(len(train_images)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset   = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_dataset  = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Optional: Print one batch to check shapes
for data_batch, labels_batch in train_dataset.take(1):
    print("data batch shape:", data_batch.shape)  # expected: (batch_size, 64, 64, 9)
    print("labels batch shape:", labels_batch.shape)
    break

# ==================== DATA AUGMENTATION ======================
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),            # left-right
    layers.RandomRotation(0.12),                # ±12%
    layers.RandomZoom(0.12),                    # ±12%
    layers.RandomTranslation(0.06, 0.06)       # 6% shift
    # Optional color jitter (if useful for multispectral, skip if it breaks bands):
    # layers.RandomContrast(0.1),
])

# ==================== VISION TRANSFORMER (ViT) DEFINITION ======================

# Input images: (batch_size, 64, 64, 9)
# - Image size = 64 × 64
# - Patch size = 8 × 8
# → Patches per side = 64 / 8 = 8
# → Total patches = 8 × 8 = 64
# Each patch contains 8 × 8 × 9 = 576 values

# Transformers need sequence-like input.
# In NLP, a sentence → split into words → embed words → feed into Transformer.
# In ViT, an image → split into patches → embed patches → feed into Transformer.
# So patches act as the “tokens” of the image.

# Normally, a CNN looks at an image through local filters (kernels) sliding over the image.
# A Transformer, on the other hand, was originally designed for sequences (like words in a sentence).
# To make images look like a “sequence,” we cut the image into smaller square blocks, flatten each block into a vector, and treat each block like a "word token."
# These smaller square blocks are what we call image patches.

# ------- Create Image Patches --------

# Define a custom Keras layer that extracts image patches
class PatchExtractor(layers.Layer):

    # PatchExtractor constructor
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    # PatchExtractor forward pass
    def call(self, images):

        # images: (batch, H, W, C)
        batch_size = tf.shape(images)[0] # Extract batch size dynamically
        patch_size = self.patch_size
        
        # Call TensorFlow’s built-in function extract_patches, which cuts out smaller patches from images
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, patch_size, patch_size, 1], # [1, 8, 8, 1]
            strides=[1, patch_size, patch_size, 1], # non-overlapping
            rates=[1, 1, 1, 1],
            padding="VALID")
        
        patch_dims = patches.shape[-1] # = 576
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])  # (batch, num_patches, patch_dims) (32, 64, 576)
        return patches

    # Method to save/load the layer
    def get_config(self):
        return {"patch_size": self.patch_size}

# ------- Apply Patch Embedding --------

# Define a custom Keras layer that has num_patches = 64 (from patch extractor) & embed_dim = 64 (desired embedding size).
class PatchEmbedding(layers.Layer):
    def __init__(self, num_patches, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        # Dense layer projects each patch (576-dim) → embedding (64-dim)
        self.proj = layers.Dense(embed_dim)
        # positional embedding
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(1, num_patches, embed_dim),
            initializer="random_normal",
            trainable=True,
        )

    # PatchEmbedding forward pass
    def call(self, patches):
        x = self.proj(patches)  # (batch, num_patches, embed_dim)
        x = x + self.pos_embedding
        return x

    # Method to save/load the layer    
    def get_config(self):
        return {"num_patches": self.num_patches, "embed_dim": self.embed_dim}

# ------- Transformer Encoder Block ---------

class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = keras.Sequential(
            [
                layers.Dense(mlp_dim, activation="relu"),
                layers.Dropout(dropout_rate),
                layers.Dense(embed_dim),
                layers.Dropout(dropout_rate),
            ]
        )

    # Transformer encoder forward pass
    def call(self, inputs, training=False, mask=None):
        # Self-attention
        attn_out = self.attn(query=inputs, value=inputs, key=inputs, attention_mask=mask)
        x = self.norm1(inputs + attn_out)
        mlp_out = self.mlp(x, training=training)
        x = self.norm2(x + mlp_out)
        return x
        
    # Method to save/load the layer
    def get_config(self):
        return {"embed_dim": self.embed_dim, "num_heads": self.num_heads, "mlp_dim": self.mlp_dim}

# ------------------------ BUILDING ViT MODEL -----------------------

# Settings
image_size = 64
channels = 9
patch_size = 8
num_patches = (image_size // patch_size) ** 2
embed_dim = 64
num_heads = 4
mlp_dim = 128
num_transformer_blocks = 3
dropout_rate = 0.1
num_classes = 3

# Input
inputs = keras.Input(shape=(image_size, image_size, channels))

# Apply augmentation
augmented = data_augmentation(inputs)

# Make patches
patches = PatchExtractor(patch_size=patch_size)(augmented)  # (batch, num_patches, patch_dim)

# Project patches + positional embedding (no CLS token)
x = PatchEmbedding(num_patches=num_patches, embed_dim=embed_dim)(patches)  # (batch, num_patches, embed_dim)

# Transformer stacks
for _ in range(num_transformer_blocks):
    x = TransformerEncoderBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        dropout_rate=dropout_rate
    )(x)

# Instead of taking CLS token, apply mean pooling across all patches
x = layers.LayerNormalization(epsilon=1e-6)(x)
image_repr = layers.GlobalAveragePooling1D()(x)   # (batch, embed_dim)

# Classification head
x = layers.Dense(16, activation="relu")(image_repr)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

# Create the model
model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()

# ------------------------- MODEL COMPILATION ------------------------

model.compile(optimizer=optimizers.SGD(learning_rate=1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# ------------------------- MODEL TRAINING -------------------------

callbacks = [
    keras.callbacks.ModelCheckpoint(filepath="vit_core_aug.keras", save_best_only=True, monitor="val_loss")
]

history = model.fit(train_dataset, epochs=50, validation_data=val_dataset, callbacks=callbacks, verbose=1)

# ---------------------- PLOT THE RESULTS -----------------------

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

# ----------------------- EVALUATE ON TEST DATA -------------------------

# Load the best saved model and evaluate on test dataset
test_model = keras.models.load_model("vit_core_aug.keras", custom_objects={
    "PatchExtractor": PatchExtractor,
    "PatchEmbedding": PatchEmbedding,
    "TransformerEncoderBlock": TransformerEncoderBlock
})
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}")


# In[10]:


# ----------------------- CLASSIFICATION REPORT -------------------------

from sklearn.metrics import classification_report
import numpy as np

# Get predictions on test set
y_true = []
y_pred = []

for images, labels in test_dataset:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

print(classification_report(y_true, y_pred, digits=3))


# In[12]:


# ----------------------- SAMPLE PREDICTION -------------------------

images, labels = next(iter(val_dataset))
preds = model.predict(images)
print("Sample predictions:", np.argmax(preds, axis=1))
print("True labels:", labels.numpy())


# In[1]:


# ========== ViT_PRETRAINED(BigEarthNet)_FINETUNING ============
import os
import numpy as np
import rasterio
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from vit_keras import vit
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# =========================== DATA LOADING ===========================
train_dir = r"E:/Sentinel_Dataset/cnn/model_datasets/core_dataset/train"
val_dir   = r"E:/Sentinel_Dataset/cnn/model_datasets/core_dataset/validation"
test_dir  = r"E:/Sentinel_Dataset/cnn/model_datasets/core_dataset/test"

def load_tif(path):
    with rasterio.open(path) as src:
        img = src.read()
        img = np.moveaxis(img, 0, -1)   # (C,H,W) → (H,W,C)
    return img.astype("float32")

def process_dir(data_dir):
    images, labels = [], []
    class_names = sorted(os.listdir(data_dir))
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for file in os.listdir(class_dir):
            if file.endswith(".tif"):
                images.append(load_tif(os.path.join(class_dir, file)))
                labels.append(label)
    return np.stack(images), np.array(labels)

train_images, train_labels = process_dir(train_dir)
val_images, val_labels     = process_dir(val_dir)
test_images, test_labels   = process_dir(test_dir)

# Normalize (per-band)
mean = train_images.mean(axis=(0,1,2))
std  = train_images.std(axis=(0,1,2))
train_images = (train_images - mean) / std
val_images   = (val_images - mean) / std
test_images  = (test_images - mean) / std

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(len(train_images)).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset   = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset  = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(32).prefetch(tf.data.AUTOTUNE)

# ==================== DATA AUGMENTATION ======================
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),            # left-right
    layers.RandomRotation(0.12),                # ±12%
    layers.RandomZoom(0.12),                    # ±12%
    layers.RandomTranslation(0.06, 0.06)       # 6% shift
    # Optional color jitter (if useful for multispectral, skip if it breaks bands):
    # layers.RandomContrast(0.1),
])

# ================== ViT MODEL DEFINITION ===========================
inputs = keras.Input(shape=(64, 64, 9))

# Apply augmentation inside the model
x = data_augmentation(inputs)

# Resize + spectral adapter
x = layers.Resizing(112, 112, interpolation="bilinear")(x)  
x = layers.Conv2D(10, (1,1), padding="same", activation="linear")(x)  # 9 → 10 bands

# Using backbone of Vision Transformer ViT_BASE model with 16 patch_size
vit_base = vit.vit_b16(
    image_size=112,
    activation='tanh',     # JSON: _fusion_activation
    pretrained=False,      # we'll load custom weights
    include_top=False,
    pretrained_top=False
)

# Restore pretrained weights from checkpoint
# Config file of checkpoint shows that Pretrained weights were trained on image_size=112 & patch_size=16

checkpoint_path = r"E:/vit_base/vit_base_patch8_224_s2_tf/ckpt-1"
ckpt = tf.train.Checkpoint(model=vit_base)
ckpt.restore(checkpoint_path).expect_partial()
vit_base.trainable = False   # freeze backbone first

# Add custom classification head
x = vit_base(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(3, activation="softmax")(x)

model = keras.Model(inputs, outputs)

# ========================== MODEL COMPILATION ==========================
model.compile(
    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ========================== TRAINING ============================
callbacks = [
    keras.callbacks.ModelCheckpoint("core_vit16_finetuned_1.keras", save_best_only=True, monitor="val_loss")
]

history = model.fit(train_dataset, epochs=50, validation_data=val_dataset, callbacks=callbacks)

# ========================== PLOTS ============================
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history["loss"], "bo-", label="Training Loss")
plt.plot(history.history["val_loss"], "b-", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)

plt.subplot(1,2,2)
plt.plot(history.history["accuracy"], "bo-", label="Training Acc")
plt.plot(history.history["val_accuracy"], "b-", label="Validation Acc")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs"); plt.ylabel("Accuracy"); plt.legend(); plt.grid(True)

plt.tight_layout(); plt.show()

# ========================== TEST EVALUATION ==========================
test_model = keras.models.load_model("core_vit16_finetuned_1.keras", safe_mode=False)
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}")


# In[3]:


# ========================== SAVE FINAL MODEL & WEIGHTS ==========================

save_dir = r"E:/vit_models"
os.makedirs(save_dir, exist_ok=True)

# Save full model (architecture + weights + optimizer state)
final_model_path = os.path.join(save_dir, "core_vit16_finetuned_1.keras")
model.save(final_model_path)
print(f"✅ Full model saved to: {final_model_path}")

# Save just weights (lighter, good for resuming training)
final_weights_path = os.path.join(save_dir, "core_vit16_finetuned_1.weights.h5")
model.save_weights(final_weights_path)
print(f"✅ Weights saved to: {final_weights_path}")


# In[5]:


# ----------------------- CLASSIFICATION REPORT -------------------------

from sklearn.metrics import classification_report
import numpy as np

# Get predictions on test set
y_true = []
y_pred = []

for images, labels in test_dataset:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

print(classification_report(y_true, y_pred, digits=3))


# In[7]:


# ----------------------- SAMPLE PREDICTION -------------------------

images, labels = next(iter(val_dataset))
preds = model.predict(images)
print("Sample predictions:", np.argmax(preds, axis=1))
print("True labels:", labels.numpy())


# In[ ]:


import tensorflow as tf
from tensorflow import keras

# Get the ViT backbone by name
vit_backbone = model.get_layer("vit_b16")

print("=== Internal Layers of vit_b16 ===")
for idx, layer in enumerate(vit_backbone.layers):
    print(f"{idx:03d} | Name: {layer.name:30s} | Trainable: {layer.trainable}")



# In[2]:


# ========== ViT_PRETRAINED(BigEarthNet)_FINETUNING ============
import os
import numpy as np
import rasterio
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from vit_keras import vit
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# =========================== DATA LOADING ===========================
train_dir = r"E:/Sentinel_Dataset/cnn/model_datasets/core_dataset/train"
val_dir   = r"E:/Sentinel_Dataset/cnn/model_datasets/core_dataset/validation"
test_dir  = r"E:/Sentinel_Dataset/cnn/model_datasets/core_dataset/test"

def load_tif(path):
    with rasterio.open(path) as src:
        img = src.read()
        img = np.moveaxis(img, 0, -1)   # (C,H,W) → (H,W,C)
    return img.astype("float32")

def process_dir(data_dir):
    images, labels = [], []
    class_names = sorted(os.listdir(data_dir))
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for file in os.listdir(class_dir):
            if file.endswith(".tif"):
                images.append(load_tif(os.path.join(class_dir, file)))
                labels.append(label)
    return np.stack(images), np.array(labels)

train_images, train_labels = process_dir(train_dir)
val_images, val_labels     = process_dir(val_dir)
test_images, test_labels   = process_dir(test_dir)

# Normalize (per-band)
mean = train_images.mean(axis=(0,1,2))
std  = train_images.std(axis=(0,1,2))
train_images = (train_images - mean) / std
val_images   = (val_images - mean) / std
test_images  = (test_images - mean) / std

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(len(train_images)).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset   = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset  = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(32).prefetch(tf.data.AUTOTUNE)

# ====================== LOAD ALREADY TRAINED MODEL ======================

# Load Stage 1 model with vit_b16 custom object
model = keras.models.load_model(
    r"E:/vit_models/core_vit16_finetuned_1.keras",
    custom_objects={"vit_b16": vit.vit_b16},
    safe_mode=False
)

# ====================== INSPECT MODEL LAYERS ===========================
print("=== Model Layers ===")
for idx, layer in enumerate(model.layers):
    print(f"Layer {idx}: {layer.name}")

# ====================== GET VIT BACKBONE =========================
vit_backbone = model.get_layer("vit_b16")

# ====================== UNFREEZE LAST 3 BLOCKS =========================
# Freeze everything first
for layer in vit_backbone.layers:
    layer.trainable = False

# Unfreeze last 6 encoder blocks + final norm
for layer in vit_backbone.layers:
    if layer.name in [
        "Transformer_encoderblock_9",
        "Transformer_encoderblock_10",
        "Transformer_encoderblock_11",
        "Transformer_encoder_norm"
    ]:
        layer.trainable = True

# ====================== PRINT STATUS ===========================
print("=== Trainable Status After Update ===")
for idx, layer in enumerate(vit_backbone.layers):
    print(f"{idx:03d} | Name: {layer.name:30s} | Trainable: {layer.trainable}")

# ====================== RE-COMPILATION ======================

model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-6, weight_decay=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ====================== RE-TRAINING ======================

fine_tune_callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="core_vit16_finetuned_2.keras",  # full model save
        save_best_only=True,
        monitor="val_loss"
    )
]

history = model.fit(
    train_dataset,
    epochs=30,
    validation_data=val_dataset,
    callbacks=fine_tune_callbacks
)

# ====================== PLOT TRAINING CURVES ======================
history_dict = history.history
training_loss = history_dict["loss"]
val_loss = history_dict["val_loss"]
training_acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]

plt.figure(figsize=(12, 5))

# Loss
plt.subplot(1, 2, 1)
epochs_range = range(1, len(training_loss) + 1)
plt.plot(epochs_range, training_loss, "bo-", label="Training Loss")
plt.plot(epochs_range, val_loss, "b-", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs_range, training_acc, "bo-", label="Training Accuracy")
plt.plot(epochs_range, val_acc, "b-", label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
plt.close()

# ====================== EVALUATE ON TEST DATA ======================
test_model = keras.models.load_model("core_vit16_finetuned_2.keras", safe_mode=False)
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}")


# In[4]:


# ========================== SAVE FINAL MODEL & WEIGHTS ==========================

save_dir = r"E:/vit_models"
os.makedirs(save_dir, exist_ok=True)

# Save full model (architecture + weights + optimizer state)
final_model_path = os.path.join(save_dir, "core_vit16_finetuned_2.keras")
model.save(final_model_path)
print(f"✅ Full model saved to: {final_model_path}")

# Save just weights (lighter, good for resuming training)
final_weights_path = os.path.join(save_dir, "core_vit16_finetuned_2.weights.h5")
model.save_weights(final_weights_path)
print(f"✅ Weights saved to: {final_weights_path}")


# In[6]:


# ----------------------- CLASSIFICATION REPORT -------------------------

from sklearn.metrics import classification_report
import numpy as np

# Get predictions on test set
y_true = []
y_pred = []

for images, labels in test_dataset:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

print(classification_report(y_true, y_pred, digits=3))


# In[8]:


# ----------------------- SAMPLE PREDICTION -------------------------

images, labels = next(iter(val_dataset))
preds = model.predict(images)
print("Sample predictions:", np.argmax(preds, axis=1))
print("True labels:", labels.numpy())


# In[ ]:




