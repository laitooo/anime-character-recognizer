from pathlib import Path

import kagglehub
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from constants import (
    BATCH_SIZE,
    DATASET_SLUG,
    IMAGE_EXTS,
    IMG_SIZE,
    SEED,
)


def download_dataset_path() -> Path:
    return Path(kagglehub.dataset_download(DATASET_SLUG))


def list_class_dirs(source_dir: Path):
    class_dirs = [d for d in source_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    return sorted(class_dirs, key=lambda p: p.name.lower())


def list_images(class_dir: Path):
    return sorted([f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS])


def resolve_source_dir(dataset_path: Path) -> Path:
    class_dirs = list_class_dirs(dataset_path)
    if class_dirs:
        return dataset_path

    nested_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if len(nested_dirs) == 1 and list_class_dirs(nested_dirs[0]):
        return nested_dirs[0]

    return dataset_path


def split_dataset_records(class_dirs):
    class_names = [d.name for d in class_dirs]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    train_records, val_records, test_records = [], [], []
    split_stats = {"train": {}, "val": {}, "test": {}}

    for class_dir in class_dirs:
        images = list_images(class_dir)
        if len(images) < 3:
            raise ValueError(f"Class '{class_dir.name}' has less than 3 images")

        train_files, remaining = train_test_split(
            images,
            test_size=0.30,
            random_state=SEED,
            shuffle=True,
        )
        val_files, test_files = train_test_split(
            remaining,
            test_size=0.50,
            random_state=SEED,
            shuffle=True,
        )

        label = class_to_idx[class_dir.name]
        train_records.extend((path, label) for path in train_files)
        val_records.extend((path, label) for path in val_files)
        test_records.extend((path, label) for path in test_files)

        split_stats["train"][class_dir.name] = len(train_files)
        split_stats["val"][class_dir.name] = len(val_files)
        split_stats["test"][class_dir.name] = len(test_files)

    return class_names, train_records, val_records, test_records, split_stats


def decode_and_resize(image_path, label):
    image_bytes = tf.io.read_file(image_path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def make_dataset(records, shuffle=True):
    paths = [str(path) for path, _ in records]
    labels = [label for _, label in records]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle and records:
        ds = ds.shuffle(buffer_size=len(records), seed=SEED, reshuffle_each_iteration=True)
    return (
        ds.map(decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
