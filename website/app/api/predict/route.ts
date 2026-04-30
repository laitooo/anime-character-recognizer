import { existsSync, promises as fs } from "fs";
import path from "path";
import { spawn } from "child_process";

type PythonPrediction = {
  predicted_character: string;
  confidence: number;
  top_predictions: Array<{ character: string; score: number }>;
};

const REPO_ROOT = path.resolve(process.cwd(), "..");
const AI_DIR = path.join(REPO_ROOT, "ai");
const PREDICT_SCRIPT = path.join(AI_DIR, "predict_image.py");
const TEMP_DIR = path.join(process.cwd(), "tmp");

function resolvePythonExecutable(): string {
  if (process.env.PYTHON_EXECUTABLE) {
    return process.env.PYTHON_EXECUTABLE;
  }

  const candidates = [
    path.join(process.cwd(), "myenv", "bin", "python"),
    path.join(process.cwd(), ".venv", "bin", "python"),
    "python3",
    "python",
  ];

  for (const executable of candidates) {
    if (executable === "python3" || executable === "python" || existsSync(executable)) {
      return executable;
    }
  }

  return "python3";
}

function runPrediction(imagePath: string): Promise<PythonPrediction> {
  return new Promise((resolve, reject) => {
    const pythonExecutable = resolvePythonExecutable();
    const child = spawn(pythonExecutable, [PREDICT_SCRIPT, imagePath], {
      cwd: AI_DIR,
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk: Buffer) => {
      stdout += chunk.toString("utf-8");
    });

    child.stderr.on("data", (chunk: Buffer) => {
      stderr += chunk.toString("utf-8");
    });

    child.on("error", (error) => {
      reject(new Error(`Failed to start predictor process: ${error.message}`));
    });

    child.on("close", (code) => {
      if (code !== 0) {
        const stderrText = stderr.trim();
        reject(
          new Error(
            stderrText ||
              `Predictor exited with code ${code}. If dependencies are missing, install them in the Python env used by PYTHON_EXECUTABLE.`
          )
        );
        return;
      }

      try {
        const parsed = JSON.parse(stdout) as PythonPrediction;
        resolve(parsed);
      } catch (error) {
        const message = error instanceof Error ? error.message : "Invalid JSON";
        reject(new Error(`Could not parse predictor output: ${message}`));
      }
    });
  });
}

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const image = formData.get("image");

    if (!(image instanceof File)) {
      return Response.json({ error: "No image file was uploaded." }, { status: 400 });
    }

    const extension = path.extname(image.name).toLowerCase();
    if (![".jpg", ".jpeg", ".png", ".webp"].includes(extension)) {
      return Response.json(
        { error: "Only JPG, JPEG, PNG, and WEBP files are supported." },
        { status: 400 }
      );
    }

    await fs.mkdir(TEMP_DIR, { recursive: true });
    const tempFilePath = path.join(TEMP_DIR, `${Date.now()}-${Math.random()}${extension}`);

    try {
      const arrayBuffer = await image.arrayBuffer();
      await fs.writeFile(tempFilePath, Buffer.from(arrayBuffer));

      const prediction = await runPrediction(tempFilePath);

      return Response.json({
        character: prediction.predicted_character,
        confidence: prediction.confidence,
        top_predictions: prediction.top_predictions,
      });
    } finally {
      await fs.unlink(tempFilePath).catch(() => undefined);
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown server error";
    return Response.json({ error: `Prediction failed: ${message}` }, { status: 500 });
  }
}
