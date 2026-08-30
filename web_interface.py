from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, jsonify, render_template_string, request

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "outputs" / "models" / "cnn_full.pt"

app = Flask(__name__)


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = F.relu(self.b1(self.c1(x)))
        x = F.relu(self.b2(self.c2(x)))
        return F.max_pool2d(x, 2)


class LandslideCNN(nn.Module):
    def __init__(self, in_ch=8, dropout=0.5):
        super().__init__()
        self.blocks = nn.Sequential(
            ConvBlock(in_ch, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.head(self.blocks(x)).squeeze(1)


def load_model(model_path: Path = MODEL_PATH, device: str = "cpu") -> LandslideCNN:
    model = LandslideCNN(in_ch=8, dropout=0.5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


MODEL = None
try:
    MODEL = load_model()
except FileNotFoundError:
    MODEL = None


def validate_patch(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)

    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]

    if x.shape == (64, 64, 8):
        x = np.transpose(x, (2, 0, 1))

    if x.shape != (8, 64, 64):
        raise ValueError(
            "Input patch must have shape (8, 64, 64) or (64, 64, 8). "
            f"Received {x.shape}."
        )

    return x


def predict_probability(patch: np.ndarray, model: LandslideCNN) -> float:
    x = validate_patch(patch)
    x = torch.from_numpy(x[np.newaxis, :, :, :]).float()
    with torch.inference_mode():
        prob = torch.sigmoid(model(x)).item()
    return float(prob)


HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Landslide Risk Prediction</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      background: #f4f7fb;
      color: #1f2a37;
      margin: 0;
      padding: 32px;
    }
    .container {
      max-width: 720px;
      margin: 0 auto;
      background: white;
      border-radius: 12px;
      box-shadow: 0 8px 20px rgba(0,0,0,0.08);
      padding: 28px;
    }
    h1 { margin-top: 0; }
    .result {
      margin-top: 16px;
      padding: 14px 18px;
      border-radius: 8px;
      font-weight: bold;
      background: #edf6ff;
      border: 1px solid #cfe3ff;
    }
    .danger { background: #fff1f1; border-color: #f7c6c6; }
    .safe { background: #edfdf3; border-color: #c8ecd1; }
    form { display: grid; gap: 12px; }
    input[type="file"], textarea, button {
      width: 100%;
      padding: 10px 12px;
      border-radius: 8px;
      border: 1px solid #d9e2ec;
      font-size: 14px;
    }
    button {
      background: #2563eb;
      color: white;
      border: none;
      cursor: pointer;
      font-weight: bold;
    }
    .small {
      color: #5b6472;
      font-size: 13px;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>Landslide Risk Prediction</h1>
    <p class="small">
      Upload a single 8-channel patch with shape (8, 64, 64) or (64, 64, 8).
      The model returns a probability between 0 and 1, where higher values mean a greater landslide risk.
    </p>

    <form id="uploadForm" enctype="multipart/form-data">
      <input type="file" name="file" accept=".npy,.npz,.json" />
      <textarea name="jsonData" rows="6" placeholder='Optional JSON array: [[...], [...], ...] or raw NumPy-like 8x64x64 array'></textarea>
      <button type="submit">Predict</button>
    </form>

    <div id="result" class="result">Waiting for input...</div>
  </div>

  <script>
    const form = document.getElementById('uploadForm');
    const result = document.getElementById('result');

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      result.textContent = 'Processing...';
      result.className = 'result';

      const formData = new FormData(form);
      const response = await fetch('/predict', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) {
        result.classList.add('danger');
        result.textContent = data.error || 'Prediction failed.';
        return;
      }

      const probability = Number(data.probability);
      const isHigh = probability >= 0.5;
      result.classList.add(isHigh ? 'danger' : 'safe');
      result.textContent = `${data.label} — probability ${probability.toFixed(4)} (threshold 0.5)`;
    });
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/predict", methods=["POST"])
def predict_route():
    if MODEL is None:
        return jsonify({"error": "Model not found. Ensure outputs/models/cnn_full.pt exists."}), 404

    file = request.files.get("file")
    payload = request.form.get("jsonData", "")

    try:
        if file is not None and file.filename:
            name = (file.filename or "").lower()
            if name.endswith(".npy"):
                arr = np.load(file)
            elif name.endswith(".npz"):
                arr = np.load(file)["arr_0"]
            elif name.endswith(".json"):
                arr = np.array(json.load(file))
            else:
                raise ValueError("Unsupported file type. Use .npy, .npz, or .json.")
        elif payload.strip():
            arr = json.loads(payload)
            arr = np.asarray(arr, dtype=np.float32)
        else:
            raise ValueError("Please upload a patch or enter JSON data.")

        prob = predict_probability(arr, MODEL)
        label = "Landslide likely" if prob >= 0.5 else "Low landslide risk"
        return jsonify({"probability": prob, "label": label, "threshold": 0.5})

    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
