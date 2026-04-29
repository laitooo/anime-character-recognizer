import tensorflow as tf

from constants import IMG_SIZE


def build_model(num_classes: int):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def ann_structure_text():
    return (
        "ANN/CNN structure used in this project:\n\n"
        "Input: 128x128x3 RGB image\n"
        "-> Input normalization in tf.data pipeline (pixel / 255)\n"
        "-> Conv2D(32, 3x3, relu)\n"
        "-> MaxPooling2D(2x2)\n"
        "-> Conv2D(64, 3x3, relu)\n"
        "-> MaxPooling2D(2x2)\n"
        "-> Conv2D(128, 3x3, relu)\n"
        "-> MaxPooling2D(2x2)\n"
        "-> Flatten\n"
        "-> Dense(256, relu)\n"
        "-> Dropout(0.3)\n"
        "-> Dense(num_classes, softmax)\n"
    )
