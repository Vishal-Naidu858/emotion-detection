# scripts/realtime.py

import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ── Config ─────────────────────────────────────────────
import os
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "emotion_model.h5")
IMG_SIZE    = 48
EMOTIONS    = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# Color for each emotion label box (BGR format)
EMOTION_COLORS = {
    'Angry':    (0,   0,   255),   # Red
    'Disgust':  (0,   128, 0),     # Dark Green
    'Fear':     (128, 0,   128),   # Purple
    'Happy':    (0,   255, 255),   # Yellow
    'Sad':      (255, 0,   0),     # Blue
    'Surprise': (0,   165, 255),   # Orange
    'Neutral':  (200, 200, 200),   # Grey
}

# ── Load model ──────────────────────────────────────────
print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded. Starting webcam...")

# ── Load Haar Cascade face detector ─────────────────────
# OpenCV includes this XML file — we find its path automatically
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# ── Start Webcam ─────────────────────────────────────────
cap = cv2.VideoCapture(0)   # 0 = default camera

if not cap.isOpened():
    print("❌ ERROR: Could not open webcam.")
    print("   Try changing VideoCapture(0) to VideoCapture(1)")
    exit()

print("✅ Webcam running. Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # ── Step 1: Convert to grayscale ─────────────────────
    # WHY: The model was trained on grayscale 48x48 images.
    #      Face detection also works better on grayscale.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── Step 2: Detect faces ──────────────────────────────
    # scaleFactor=1.1 : How much the image is shrunk per scale pass
    # minNeighbors=5  : Higher = fewer false positives
    # minSize=(30,30) : Ignore faces smaller than 30×30 pixels
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # ── Step 3: Process each detected face ───────────────
    for (x, y, w, h) in faces:

        # Extract the face region from the grayscale frame
        face_roi = gray[y:y+h, x:x+w]

        # Resize to 48×48 — the size the model expects
        face_resized = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))

        # Normalize pixels to 0.0–1.0
        face_normalized = face_resized / 255.0

        # Reshape to (1, 48, 48, 1):
        # 1 = batch size (one image), 48×48 = dimensions, 1 = grayscale channel
        face_input = face_normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        # ── Step 4: Predict emotion ───────────────────────
        predictions  = model.predict(face_input, verbose=0)[0]  # shape: (7,)
        emotion_idx  = np.argmax(predictions)                    # index of highest prob
        emotion_label= EMOTIONS[emotion_idx]
        confidence   = predictions[emotion_idx] * 100           # as percentage

        # ── Step 5: Draw on frame ─────────────────────────
        color = EMOTION_COLORS[emotion_label]

        # Rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Label background
        label_text = f"{emotion_label}: {confidence:.1f}%"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x, y-th-10), (x+tw+6, y), color, -1)

        # Label text
        cv2.putText(frame, label_text, (x+3, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # ── Show probability bar chart on frame ──────────
        bar_x, bar_y = 10, 10
        for i, (emo, prob) in enumerate(zip(EMOTIONS, predictions)):
            bar_len = int(prob * 150)
            bar_color = color if i == emotion_idx else (180, 180, 180)
            cv2.rectangle(frame,
                          (bar_x, bar_y + i*22),
                          (bar_x + bar_len, bar_y + i*22 + 16),
                          bar_color, -1)
            cv2.putText(frame, f"{emo[:3]} {prob*100:.0f}%",
                        (bar_x + 155, bar_y + i*22 + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

    # ── Display the frame ─────────────────────────────────
    cv2.putText(frame, "Press Q to quit", (frame.shape[1]-170, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    cv2.imshow("Emotion Detection", frame)

    # ── Quit on pressing 'q' ──────────────────────────────
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── Cleanup ──────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
print("Webcam released. Program ended.")