# app.py — Flask backend for Emotion Detection System
# Run with: python app.py
# Access at: http://localhost:5000

import os
import cv2
import numpy as np
import base64
import json
from flask import Flask, render_template, request, jsonify, Response
from tensorflow.keras.models import load_model

# ── Config ─────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "emotion_model.h5")
IMG_SIZE   = 48
EMOTIONS   = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

EMOTION_COLORS = {
    'Angry':    '#FF4444',
    'Disgust':  '#4CAF50',
    'Fear':     '#9C27B0',
    'Happy':    '#FFD700',
    'Sad':      '#2196F3',
    'Surprise': '#FF9800',
    'Neutral':  '#9E9E9E',
}

EMOTION_EMOJI = {
    'Angry':    '😠',
    'Disgust':  '🤢',
    'Fear':     '😨',
    'Happy':    '😊',
    'Sad':      '😢',
    'Surprise': '😮',
    'Neutral':  '😐',
}

# ── Flask App ───────────────────────────────────────────────
app = Flask(__name__)

# ── Load model once at startup ──────────────────────────────
print("Loading emotion model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully ✅")

# ── Load Haar Cascade ───────────────────────────────────────
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# ── Core Detection Function ─────────────────────────────────
def detect_emotions_in_frame(frame):
    """
    Takes a BGR numpy array frame.
    Returns annotated frame and list of detection results.
    """
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces   = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    results = []

    for (x, y, w, h) in faces:
        face_roi        = gray[y:y+h, x:x+w]
        face_resized    = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
        face_normalized = face_resized / 255.0
        face_input      = face_normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        predictions   = model.predict(face_input, verbose=0)[0]
        emotion_idx   = np.argmax(predictions)
        emotion_label = EMOTIONS[emotion_idx]
        confidence    = float(predictions[emotion_idx] * 100)

        # Draw bounding box (green)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Draw label background
        label     = f"{emotion_label} {confidence:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x, y - th - 12), (x + tw + 8, y), (0, 255, 0), -1)
        cv2.putText(frame, label, (x + 4, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        results.append({
            'emotion':    emotion_label,
            'confidence': round(confidence, 1),
            'emoji':      EMOTION_EMOJI[emotion_label],
            'color':      EMOTION_COLORS[emotion_label],
            'all_probs':  {
                EMOTIONS[i]: round(float(predictions[i] * 100), 1)
                for i in range(len(EMOTIONS))
            }
        })

    return frame, results

# ── Webcam Stream Generator ─────────────────────────────────
camera = None

def generate_frames():
    global camera
    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if not success:
            break

        annotated, _ = detect_emotions_in_frame(frame)
        _, buffer     = cv2.imencode('.jpg', annotated)
        frame_bytes   = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'
               + frame_bytes + b'\r\n')

# ── Routes ──────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/live')
def live():
    return render_template('live.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/video_feed')
def video_feed():
    """Streams webcam frames as MJPEG to the browser."""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Releases the webcam."""
    global camera
    if camera and camera.isOpened():
        camera.release()
        camera = None
    return jsonify({'status': 'stopped'})

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    """
    Receives a base64 webcam frame from browser,
    runs detection, returns results as JSON.
    Used for live emotion label overlay in sidebar.
    """
    data       = request.json.get('frame', '')
    img_data   = base64.b64decode(data.split(',')[1])
    nparr      = np.frombuffer(img_data, np.uint8)
    frame      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    _, results = detect_emotions_in_frame(frame)
    return jsonify({'results': results})

@app.route('/predict_image', methods=['POST'])
def predict_image():
    """
    Receives an uploaded image file,
    runs detection, returns annotated image + results.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file      = request.files['image']
    file_bytes= np.frombuffer(file.read(), np.uint8)
    frame     = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({'error': 'Could not read image'}), 400

    annotated, results = detect_emotions_in_frame(frame)

    # Encode annotated image to base64 for sending back
    _, buffer  = cv2.imencode('.jpg', annotated)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'image':   f'data:image/jpeg;base64,{img_base64}',
        'results': results,
        'count':   len(results)
    })

# ── Run ─────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n🚀 Emotion Detection System running at:")
    print("   http://localhost:5000\n")
    app.run(debug=True, host='0.0.0.0', port=5000)