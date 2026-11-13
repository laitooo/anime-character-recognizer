"""Rebuilds the full .keras model files from a weights-only checkpoint.

predict_image.py loads a full saved model (artifacts/model/*.keras), but the
Colab notebook's "Save Weights" cell only saves weights. This reconstructs the
same architecture defined in colab_anime_classifier.ipynb, loads the given
weights file into it, and saves it as the full model files predict_image.py
expects.
"""

import argparse
import json
from pathlib import Path

import tensorflow as tf

IMG_SIZE = 128
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "model"


def build_augmentation():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.08, 0.08),
        tf.keras.layers.RandomContrast(0.15),
    ], name="augmentation")


def conv_block(filters: int):
    return [
        tf.keras.layers.Conv2D(filters, (3, 3), padding="same", use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(filters, (3, 3), padding="same", use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),
    ]


def build_model(num_classes: int):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        build_augmentation(),
        *conv_block(32),
        *conv_block(64),
        *conv_block(128),
        tf.keras.layers.SpatialDropout2D(0.1),
        *conv_block(256),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ], name="anime_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("weights_path", help="Path to a .weights.h5 file")
    args = parser.parse_args()

    class_names = json.loads((MODEL_DIR / "class_names.json").read_text(encoding="utf-8"))
    model = build_model(num_classes=len(class_names))
    model.load_weights(args.weights_path)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_DIR / "final_model.keras")
    model.save(MODEL_DIR / "best_model.keras")
    print(f"Rebuilt model from {args.weights_path} for {len(class_names)} classes.")
    print(f"Saved to {MODEL_DIR / 'final_model.keras'} and {MODEL_DIR / 'best_model.keras'}")


if __name__ == "__main__":
    main()
