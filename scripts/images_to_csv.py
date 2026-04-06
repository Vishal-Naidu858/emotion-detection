# scripts/images_to_csv.py
#
# Converts a folder of emotion images into fer2013.csv format:
#   emotion  |  pixels               | Usage
#   3        |  "70 80 82 75 ..."    | Training
#   0        |  "120 130 ..."        | PublicTest

import os
import cv2
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# ✏️  EDIT THESE TWO PATHS to match where you extracted the zip
# ─────────────────────────────────────────────────────────────
TRAIN_DIR   = "datasets/raw/train"   # folder containing training images
TEST_DIR    = "datasets/raw/test"    # folder containing test images
OUTPUT_CSV  = "datasets/raw/fer2013.csv"

IMG_SIZE    = 48   # resize every image to 48×48

# Emotion label → integer mapping
# Keys must exactly match your folder names (case-insensitive handled below)
EMOTION_MAP = {
    "angry":    0,
    "disgust":  1,
    "fear":     2,
    "happy":    3,
    "sad":      4,
    "surprise": 5,
    "neutral":  6,
}

# Also support numeric folder names (0,1,2 ... 6)
NUMERIC_MAP = {str(v): v for v in EMOTION_MAP.values()}

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# ─────────────────────────────────────────────────────────────
def folder_name_to_label(folder_name: str):
    """
    Accepts emotion name ('angry', 'Happy', etc.) 
    OR numeric name ('0', '3', etc.)
    Returns the integer label, or None if unrecognised.
    """
    name_lower = folder_name.strip().lower()
    if name_lower in EMOTION_MAP:
        return EMOTION_MAP[name_lower]
    if name_lower in NUMERIC_MAP:
        return NUMERIC_MAP[name_lower]
    return None   # unrecognised folder — will be skipped with a warning

# ─────────────────────────────────────────────────────────────
def load_images_from_split(split_dir: str, usage_label: str):
    """
    Walks through split_dir/emotion_folder/*.jpg,
    resizes each image to 48×48 grayscale,
    flattens pixels to a space-separated string,
    and returns a list of row-dicts ready for a DataFrame.

    usage_label : 'Training' or 'PublicTest'  (matches FER-2013 convention)
    """
    rows = []
    
    if not os.path.exists(split_dir):
        print(f"  ⚠️  Directory not found, skipping: {split_dir}")
        return rows

    emotion_folders = sorted(os.listdir(split_dir))
    
    if not emotion_folders:
        print(f"  ⚠️  No sub-folders found inside: {split_dir}")
        return rows

    for folder in emotion_folders:
        folder_path = os.path.join(split_dir, folder)
        
        # Skip files (we only want sub-directories)
        if not os.path.isdir(folder_path):
            continue

        label = folder_name_to_label(folder)
        if label is None:
            print(f"  ⚠️  Skipping unrecognised folder: '{folder}'")
            continue

        image_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        print(f"  📂 {usage_label}/{folder:10s} → label {label} "
              f"| {len(image_files)} images")

        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)

            # ── Load image ───────────────────────────────
            img = cv2.imread(img_path)

            if img is None:
                # File exists but OpenCV can't read it (corrupt/unsupported)
                print(f"    ⛔ Could not read: {img_path} — skipping")
                continue

            # ── Convert to grayscale ─────────────────────
            # WHY: FER-2013 is grayscale. Colour channels add noise for
            #      simple emotion models and triple the data size.
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img   # already grayscale

            # ── Resize to 48×48 ──────────────────────────
            # WHY: The CNN input layer is fixed at 48×48.
            #      All images must be the same size.
            resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE),
                                 interpolation=cv2.INTER_AREA)

            # ── Flatten to pixel string ───────────────────
            # WHY: CSV stores each image as one row.
            #      FER-2013 format uses space-separated integers.
            #      e.g. "70 80 82 75 69 ..."  (2304 values for 48×48)
            pixel_string = " ".join(map(str, resized.flatten().tolist()))

            rows.append({
                "emotion": label,
                "pixels":  pixel_string,
                "Usage":   usage_label,
            })

    return rows

# ─────────────────────────────────────────────────────────────
print("=" * 55)
print("  🔄  Converting image folders → fer2013.csv")
print("=" * 55)

print(f"\n📁 Loading TRAIN images from: {TRAIN_DIR}")
train_rows = load_images_from_split(TRAIN_DIR, "Training")

print(f"\n📁 Loading TEST  images from: {TEST_DIR}")
test_rows  = load_images_from_split(TEST_DIR,  "PublicTest")

all_rows = train_rows + test_rows

if not all_rows:
    print("\n❌ No images were loaded. Check your TRAIN_DIR / TEST_DIR paths.")
    exit(1)

# ─────────────────────────────────────────────────────────────
# Build DataFrame and save
df = pd.DataFrame(all_rows, columns=["emotion", "pixels", "Usage"])

print(f"\n{'='*55}")
print(f"  ✅  Total rows : {len(df)}")
print(f"  🏋️  Training   : {len(df[df.Usage == 'Training'])}")
print(f"  🧪  PublicTest : {len(df[df.Usage == 'PublicTest'])}")
print(f"\n  Emotion distribution:")
emotion_names = {v: k.capitalize() for k, v in EMOTION_MAP.items()}
for label, count in sorted(df['emotion'].value_counts().items()):
    print(f"    {label} ({emotion_names[label]:10s}): {count}")

df.to_csv(OUTPUT_CSV, index=False)
print(f"\n  💾  Saved to: {OUTPUT_CSV}")
print("=" * 55)
