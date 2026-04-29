import cv2
import numpy as np
from PIL import Image

from constants import IMG_SIZE

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if FACE_CASCADE.empty():
    raise RuntimeError("Failed to load Haar face detector.")


def detect_primary_face(pil_image: Image.Image):
    rgb = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48),
    )
    if len(faces) == 0:
        return pil_image.convert("RGB"), None

    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    pad = int(0.2 * max(w, h))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(rgb.shape[1], x + w + pad)
    y1 = min(rgb.shape[0], y + h + pad)
    return Image.fromarray(rgb[y0:y1, x0:x1]), (x0, y0, x1, y1)


def preprocess_pil_for_model(pil_img: Image.Image):
    model_img = pil_img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.expand_dims(np.array(model_img, dtype=np.float32) / 255.0, axis=0)
    return arr


def top_k_indices(probabilities, k=3):
    return np.argsort(probabilities)[::-1][: min(k, len(probabilities))]
