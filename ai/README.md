# Anime Character Classifier (Notebook-Only)

A convolutional neural network (CNN) that classifies anime characters from images, trained and run entirely from a single Colab notebook.

## Results

| Metric | Value |
| --- | --- |
| Classes | 31 anime characters |
| Test accuracy | **83.5%** |
| Test loss | 0.53 |
| Best validation accuracy | 86.2% |
| Train / val / test split | 2,604 / 558 / 558 images |
| Total training | 60+ epochs, across multiple save/load sessions |

Evaluated on a held-out test split the model never saw during training or early stopping, so this is a true measure of generalization, not training accuracy. The numbers above reflect a model trained over several sessions (saved with `best_model.keras` and reloaded to continue), not a single run.

## Dataset

Downloaded automatically via `kagglehub`:

```python
kagglehub.dataset_download("anuragraj03/anime-face-dataset")
```

Source: [Kaggle — anime-face-dataset](https://www.kaggle.com/datasets/anuragraj03/anime-face-dataset)

Each character folder in the dataset becomes a class label (31 characters, 120 images each).

## Run in Google Colab

Open `colab_anime_classifier.ipynb` and run all cells in order:

1. **Install & imports** — install dependencies, import shared libraries.
2. **Read dataset** — download and index the dataset.
3. **Train/val/test split** — stratified 70/15/15 split, kept in memory.
4. **Build model** — construct the CNN architecture.
5. **CNN structure** — write a text summary of the architecture.
6. **Load saved model (optional)** — resume from `best_model.keras` if one exists, otherwise start fresh.
7. **Train** — train with early stopping and LR scheduling (not saved to disk automatically).
8. **Evaluate** — test the trained model on the held-out test split.
9. **Predict** — upload an image and get a character prediction.
10. **Save model (optional)** — explicitly save the current model as `best_model.keras` for next time.

## Notebook structure

Constants and helper functions live in their own dedicated cells:

- a constants cell (image size, batch size, epochs, dataset slug, artifact paths)
- a helper-functions cell (dataset loading, augmentation, model architecture)

## Training details

- Early stopping on `val_accuracy` (patience 8, restores best weights)
- Learning-rate reduction on `val_loss` plateau (factor 0.5, patience 3)
- Saving/loading is entirely optional and manual — training never silently overwrites a saved model
