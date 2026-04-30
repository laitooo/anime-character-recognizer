# Anime Character Recognizer Website

Single-page Next.js (TypeScript) app that uploads an anime image and predicts the character.

The UI is implemented in Next.js, and the API route runs `../ai/predict_image.py` so it uses your saved ANN model from `../ai/artifacts/model`.

## Requirements

- Node.js 18+ (recommended)
- Python environment with AI dependencies installed for `../ai/predict_image.py`
  - `tensorflow`
  - `numpy`
  - `pillow`
  - `opencv-python`

## Run

1. Install JavaScript dependencies:
   - `npm install`
2. Start the development server:
   - `npm run dev`
3. Open:
   - `http://localhost:3000`

## Optional configuration

- If your Python executable is not `python3`, run:
  - `PYTHON_EXECUTABLE=/path/to/python npm run dev`
