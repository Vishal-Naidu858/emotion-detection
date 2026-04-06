# scripts/train.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dropout,
                                     Flatten, Dense, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

# ── Config ─────────────────────────────────────────────
DATA_DIR    = "datasets/processed"
MODEL_PATH  = "models/emotion_model.h5"
IMG_SIZE    = 48
NUM_CLASSES = 7
BATCH_SIZE  = 64
EPOCHS      = 60

os.makedirs("models", exist_ok=True)

# ── Load preprocessed data ──────────────────────────────
print("Loading data...")
X_train = np.load(f"{DATA_DIR}/X_train.npy")
X_val   = np.load(f"{DATA_DIR}/X_val.npy")
y_train = np.load(f"{DATA_DIR}/y_train.npy")
y_val   = np.load(f"{DATA_DIR}/y_val.npy")

# One-hot encode labels
# WHY: The model outputs 7 probabilities. We need labels in the same format.
#      e.g., class 3 (Happy) becomes [0, 0, 0, 1, 0, 0, 0]
y_train = to_categorical(y_train, NUM_CLASSES)
y_val   = to_categorical(y_val,   NUM_CLASSES)

print(f"Train: {X_train.shape}, Val: {X_val.shape}")

# ── Data Augmentation ────────────────────────────────────
# WHY: Artificially creates more training examples by flipping, shifting,
#      and zooming images. Prevents overfitting (memorising training data).
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    horizontal_flip=True,       # Mirror face left ↔ right
    rotation_range=10,          # Rotate up to 10 degrees
    zoom_range=0.1,             # Zoom in/out by 10%
    width_shift_range=0.1,      # Shift image left/right
    height_shift_range=0.1      # Shift image up/down
)
datagen.fit(X_train)

# ── Build the CNN ────────────────────────────────────────
def build_model():
    model = Sequential([

        # ── Block 1 ──────────────────────────────────────
        # Conv2D: 32 filters, each 3×3. Detects simple edges and textures.
        Conv2D(32, (3,3), activation='relu', padding='same',
               input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        BatchNormalization(),   # Speeds up training, stabilizes gradients
        Conv2D(32, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),     # Reduce 48×48 → 24×24, keep strongest features
        Dropout(0.25),          # Randomly turn off 25% of neurons → prevent overfitting

        # ── Block 2 ──────────────────────────────────────
        # 64 filters: now detects more complex shapes (eye corners, mouth curves)
        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),     # 24×24 → 12×12
        Dropout(0.25),

        # ── Block 3 ──────────────────────────────────────
        # 128 filters: detects high-level features (raised eyebrows, smile shape)
        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),     # 12×12 → 6×6
        Dropout(0.25),

        # ── Fully Connected Head ──────────────────────────
        Flatten(),              # Convert 6×6×128 = 4608 values into a 1D vector
        Dense(256, activation='relu'),   # Learn combinations of features
        BatchNormalization(),
        Dropout(0.5),           # 50% dropout — strongest regularization here

        Dense(128, activation='relu'),
        Dropout(0.3),

        # Output layer: 7 neurons, one per emotion
        # Softmax converts raw scores into probabilities that sum to 1
        Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

model = build_model()
model.summary()  # Prints layer shapes and parameter counts

# ── Compile the Model ────────────────────────────────────
# loss: categorical_crossentropy — standard for multi-class classification
# optimizer: Adam — adaptive learning rate, works great out-of-the-box
# metrics: accuracy — human-readable performance measure
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# ── Callbacks ────────────────────────────────────────────
callbacks = [
    # Save the best model (lowest validation loss) automatically
    ModelCheckpoint(MODEL_PATH, monitor='val_loss',
                    save_best_only=True, verbose=1),

    # Stop training if validation loss doesn't improve for 15 epochs
    EarlyStopping(monitor='val_loss', patience=15,
                  restore_best_weights=True, verbose=1),

    # Reduce learning rate when stuck (helps escape plateaus)
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=7, min_lr=1e-6, verbose=1)
]

# ── Train ────────────────────────────────────────────────
print("\n🚀 Starting training...")
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=callbacks
)

print(f"\n✅ Best model saved to '{MODEL_PATH}'")

# ── Plot Training Curves ─────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['accuracy'],     label='Train Accuracy')
ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
ax1.set_title('Accuracy over Epochs')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
ax1.legend(); ax1.grid(True)

ax2.plot(history.history['loss'],     label='Train Loss')
ax2.plot(history.history['val_loss'], label='Val Loss')
ax2.set_title('Loss over Epochs')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
ax2.legend(); ax2.grid(True)

plt.tight_layout()
plt.savefig("models/training_curves.png")
plt.show()
print("Training curves saved to models/training_curves.png")