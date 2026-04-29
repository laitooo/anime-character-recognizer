# Anime Character Classifier (Notebook-Only)

This project trains a neural network to classify anime character images downloaded with `kagglehub`.

## Dataset source

The notebook downloads the dataset using:

`kagglehub.dataset_download("anuragraj03/anime-face-dataset")`

Each character folder in the downloaded dataset is used as the class label.

## Run in Google Colab

Open `colab_anime_classifier.ipynb` and run all cells in order.

The notebook includes:

1. Install dependencies and shared imports.
2. Read input dataset.
3. Split into train/validation/test variables (in memory).
4. Create model.
5. Create ANN structure text.
6. Train model.
7. Test model.
8. Upload image and predict character.
9. Save trained model weights.

## Project structure

- `constants.py`: shared constants and paths.
- `data_utils.py`: dataset download, folder discovery, in-memory splits, and tf.data builders.
- `model_utils.py`: CNN model factory and architecture text helper.
- `inference_utils.py`: face detection/cropping and single-image preprocessing helpers.
