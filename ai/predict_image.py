import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

IMG_SIZE = 128
ARTIFACTS_DIR = Path("artifacts")
MODEL_DIR = ARTIFACTS_DIR / "model"


def detect_primary_face(pil_image: Image.Image):
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        raise RuntimeError("Failed to load Haar face detector.")

    rgb = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

    if len(faces) == 0:
        return pil_image.convert("RGB"), None

    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    pad = int(0.2 * max(w, h))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(rgb.shape[1], x + w + pad)
    y1 = min(rgb.shape[0], y + h + pad)
    return Image.fromarray(rgb[y0:y1, x0:x1]), (x0, y0, x1, y1)


def preprocess_image(image_path: Path):
    original = Image.open(image_path).convert("RGB")
    face_img, bbox = detect_primary_face(original)
    resized = face_img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.expand_dims(np.array(resized, dtype=np.float32) / 255.0, axis=0)
    return arr, bbox


def load_model_path():
    best = MODEL_DIR / "best_model.keras"
    final = MODEL_DIR / "final_model.keras"
    if best.exists():
        return best
    if final.exists():
        return final
    raise FileNotFoundError("No model found. Expected artifacts/model/best_model.keras or final_model.keras")


def predict(image_path: Path):
    class_names_path = MODEL_DIR / "class_names.json"
    if not class_names_path.exists():
        raise FileNotFoundError(f"Missing class names file: {class_names_path}")

    class_names = json.loads(class_names_path.read_text(encoding="utf-8"))
    model_path = load_model_path()
    model = tf.keras.models.load_model(model_path)

    arr, bbox = preprocess_image(image_path)
    probs = model.predict(arr, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    top_indices = np.argsort(probs)[::-1][: min(3, len(probs))]

    return {
        "input_image": str(image_path.resolve()),
        "model_path": str(model_path.resolve()),
        "face_bbox": bbox,
        "predicted_character": class_names[pred_idx],
        "confidence": float(probs[pred_idx]),
        "top_predictions": [
            {"character": class_names[int(i)], "score": float(probs[int(i)])}
            for i in top_indices
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to input image")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    result = predict(image_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
