# tests/test_training_smoke.py
from pathlib import Path
import json

from src.train import train


def test_training_creates_artifacts():
    # Run training (uses default paths)
    metrics = train(seed=42)

    # Check files
    assert Path("model/model.pkl").exists()
    assert Path("model/scaler.pkl").exists()
    assert Path("model/model_v0.2.pkl").exists()
    assert Path("model/scaler_v0.2.pkl").exists()
    assert Path("model/feature_names.json").exists()
    assert Path("metrics.json").exists()
    assert Path("metrics_v0.2.json").exists()

    # Check metrics content
    assert isinstance(metrics["rmse"], float)
    assert metrics["version"] == "v0.2"
    assert metrics["model"] == "Ridge"
    assert metrics["scaler"] == "StandardScaler"

    # metrics.json is readable and consistent
    on_disk = json.loads(Path("metrics.json").read_text())
    assert on_disk["version"] == metrics["version"]
    assert "rmse" in on_disk
