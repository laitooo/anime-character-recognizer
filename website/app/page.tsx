"use client";

import { FormEvent, useMemo, useState } from "react";

type TopPrediction = {
  character: string;
  score: number;
};

type PredictResponse = {
  character: string;
  confidence: number;
  top_predictions: TopPrediction[];
};

export default function HomePage() {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState("");
  const [result, setResult] = useState<PredictResponse | null>(null);

  const previewUrl = useMemo(() => {
    if (!file) {
      return null;
    }
    return URL.createObjectURL(file);
  }, [file]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file) {
      setStatus("Please select an image first.");
      return;
    }

    setStatus("Predicting...");
    setResult(null);

    const formData = new FormData();
    formData.append("image", file);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });
      const data = (await response.json()) as PredictResponse | { error: string };

      if (!response.ok || "error" in data) {
        setStatus("error" in data ? data.error : "Prediction failed.");
        return;
      }

      setResult(data);
      setStatus("Prediction complete.");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown request error.";
      setStatus(`Request failed: ${message}`);
    }
  }

  return (
    <main className="container">
      <h1>Anime Character Recognizer</h1>
      <p>Upload an anime image and identify the character with your trained CNN model.</p>

      <form className="form" onSubmit={handleSubmit}>
        <input
          type="file"
          accept=".jpg,.jpeg,.png,.webp,image/*"
          onChange={(event) => setFile(event.target.files?.[0] ?? null)}
          required
        />
        <button type="submit">Predict Character</button>
      </form>

      {previewUrl && <img className="preview" src={previewUrl} alt="Selected preview" />}

      {result && (
        <section className="result">
          <h2>Prediction</h2>
          <p>
            <strong>Character:</strong> {result.character}
          </p>
          <p>
            <strong>Confidence:</strong> {(result.confidence * 100).toFixed(2)}%
          </p>

          <h3>Top Matches</h3>
          <ul>
            {result.top_predictions.map((item) => (
              <li key={item.character}>
                {item.character}: {(item.score * 100).toFixed(2)}%
              </li>
            ))}
          </ul>
        </section>
      )}

      <p className="status">{status}</p>
    </main>
  );
}
