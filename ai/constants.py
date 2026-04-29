from pathlib import Path

SEED = 42
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15

DATASET_SLUG = "anuragraj03/anime-face-dataset"

ARTIFACTS_DIR = Path("artifacts")
MODEL_DIR = ARTIFACTS_DIR / "model"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
