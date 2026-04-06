# scripts/preprocess.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# ── Config ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH    = os.path.join(BASE_DIR, "datasets", "raw", "fer2013.csv")
SAVE_DIR    = os.path.join(BASE_DIR, "datasets", "processed")
IMG_SIZE    = 48          # Each image is 48x48 pixels
NUM_CLASSES = 7           # 7 emotions

os.makedirs(SAVE_DIR, exist_ok=True)


# ── Step 1: Load the CSV ────────────────────────────────
# WHY: The entire dataset lives in one CSV file.
print("Loading dataset...")
df = pd.read_csv(CSV_PATH)
print(f"Total samples: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"Emotion distribution:\n{df['emotion'].value_counts()}")

# ── Step 2: Convert pixel strings → numpy arrays ────────
# WHY: Each row has pixels as a long string "70 80 82 ..."
#      We need actual numbers in a 2D array to feed into the CNN.
def pixels_to_array(pixel_string):
    pixels = np.array(pixel_string.split(), dtype='float32')
    return pixels.reshape(IMG_SIZE, IMG_SIZE)   # shape: (48, 48)

print("\nConverting pixel strings to arrays...")
X = np.array([pixels_to_array(row) for row in df['pixels']])
# X.shape is now (35887, 48, 48)

# ── Step 3: Reshape for CNN input ───────────────────────
# WHY: Keras CNN expects (samples, height, width, channels)
#      Grayscale = 1 channel. Color would be 3.
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
# X.shape is now (35887, 48, 48, 1)

# ── Step 4: Normalize pixel values ──────────────────────
# WHY: Pixels range 0–255. Dividing by 255 scales them to 0.0–1.0.
#      Neural networks train much faster and better with small values.
X = X / 255.0
print(f"Pixel value range after normalization: {X.min():.2f} – {X.max():.2f}")

# ── Step 5: Extract labels ───────────────────────────────
y = df['emotion'].values    # shape: (35887,)

# ── Step 6: Train / Validation / Test Split ─────────────
# WHY: We need separate data to train on, tune on, and finally test on.
#      Using test data during training would be "cheating" — the model
#      would look good but fail in the real world.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"\nSplit sizes:")
print(f"  Train : {X_train.shape[0]} samples")
print(f"  Val   : {X_val.shape[0]} samples")
print(f"  Test  : {X_test.shape[0]} samples")

# ── Step 7: Save as .npy files ───────────────────────────
# WHY: Saving preprocessed arrays means we never redo this work again.
#      train.py just loads these files — much faster.
np.save(f"{SAVE_DIR}/X_train.npy", X_train)
np.save(f"{SAVE_DIR}/X_val.npy",   X_val)
np.save(f"{SAVE_DIR}/X_test.npy",  X_test)
np.save(f"{SAVE_DIR}/y_train.npy", y_train)
np.save(f"{SAVE_DIR}/y_val.npy",   y_val)
np.save(f"{SAVE_DIR}/y_test.npy",  y_test)

print(f"\n✅ Preprocessed data saved to '{SAVE_DIR}/'")