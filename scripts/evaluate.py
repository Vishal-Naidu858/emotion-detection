# scripts/evaluate.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)

# ── Config ─────────────────────────────────────────────
DATA_DIR   = "datasets/processed"
MODEL_PATH = "models/emotion_model.h5"
EMOTIONS   = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# ── Load data & model ───────────────────────────────────
print("Loading test data and model...")
X_test = np.load(f"{DATA_DIR}/X_test.npy")
y_test = np.load(f"{DATA_DIR}/y_test.npy")
model  = load_model(MODEL_PATH)
print("Model loaded successfully.")

# ── Predict ─────────────────────────────────────────────
y_pred_probs = model.predict(X_test)          # shape: (N, 7) probabilities
y_pred       = np.argmax(y_pred_probs, axis=1) # pick highest probability class
y_true       = y_test                          # integer labels

# ── Overall Accuracy ────────────────────────────────────
acc = accuracy_score(y_true, y_pred)
print(f"\n✅ Test Accuracy: {acc * 100:.2f}%")

# ── Per-class Report ────────────────────────────────────
# Shows Precision, Recall, F1-Score for each emotion
# Precision: Of all times we said "Happy", how many were actually happy?
# Recall:    Of all actual happy faces, how many did we catch?
# F1-Score:  Harmonic mean of both — the balanced single number
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=EMOTIONS))

# ── Confusion Matrix ─────────────────────────────────────
# Rows = actual emotion, Columns = predicted emotion
# Diagonal = correct predictions; off-diagonal = mistakes
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=EMOTIONS, yticklabels=EMOTIONS)
plt.title('Confusion Matrix', fontsize=16)
plt.ylabel('Actual Emotion',    fontsize=12)
plt.xlabel('Predicted Emotion', fontsize=12)
plt.tight_layout()
plt.savefig("models/confusion_matrix.png")
plt.show()
print("Confusion matrix saved to models/confusion_matrix.png")

# ── Visualize Sample Predictions ─────────────────────────
fig, axes = plt.subplots(3, 5, figsize=(15, 9))
axes = axes.ravel()
indices = np.random.choice(len(X_test), 15, replace=False)

for i, idx in enumerate(indices):
    img = X_test[idx].reshape(48, 48)
    axes[i].imshow(img, cmap='gray')
    actual    = EMOTIONS[y_true[idx]]
    predicted = EMOTIONS[y_pred[idx]]
    color     = 'green' if actual == predicted else 'red'
    axes[i].set_title(f"A: {actual}\nP: {predicted}",
                      color=color, fontsize=9)
    axes[i].axis('off')

plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)', fontsize=13)
plt.tight_layout()
plt.savefig("models/sample_predictions.png")
plt.show()