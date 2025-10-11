# src/app.py
from __future__ import annotations
import json
import joblib
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify

APP_VERSION = "v0.1"
MODEL_DIR = Path("model")

app = Flask(__name__)

# --- Load artifacts at startup ---
try:
    model = joblib.load(MODEL_DIR / "model.pkl")
    with open(MODEL_DIR / "feature_names.json", "r", encoding="utf-8") as f:
        feature_names = json.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load model artifacts: {e}")


@app.route("/health", methods=["GET"])
def health():
    """Simple health check."""
    return jsonify({"status": "ok", "model_version": APP_VERSION})


@app.route("/predict", methods=["POST"])
def predict():
    """Return predicted progression score."""
    try:
        data = request.get_json(force=True)
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")

        # Ensure all expected features are present
        missing = [f for f in feature_names if f not in data]
        if missing:
            return (
                jsonify(
                    {"error": f"Missing features: {missing}", "status": "failed"}
                ),
                400,
            )

        # Order features according to training order
        X = np.array([[data[f] for f in feature_names]])
        y_pred = float(model.predict(X)[0])

        return jsonify({"prediction": y_pred, "status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)