# src/train.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ARTIFACT_DIR = Path("model")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_PATH = Path("metrics.json")

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "linear": {
        "version": "v0.1",
        "estimator": LinearRegression,
        "estimator_kwargs": {},
        "description": "StandardScaler + LinearRegression",
    },
    "ridge": {
        "version": "v0.2",
        "estimator": Ridge,
        "estimator_kwargs": {"alpha": 1.0},
        "description": "StandardScaler + Ridge(alpha=1.0)",
    },
    "random_forest": {
        "version": "v0.3",
        "estimator": RandomForestRegressor,
        "estimator_kwargs": {"n_estimators": 100, "random_state": 42},
        "description": "StandardScaler + RandomForestRegressor(n_estimators=100)",
    },
}


def train(seed: int = 42, model_name: str = "ridge") -> dict:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {', '.join(MODEL_REGISTRY)}"
        )

    config = MODEL_REGISTRY[model_name]
    version = config["version"]

    # Load dataset
    Xy = load_diabetes(as_frame=True)
    X = Xy.frame.drop(columns=["target"])
    y = Xy.frame["target"]

    feature_names = list(X.columns)

    # Deterministic split
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.2, random_state=seed
    )

    # Scale (fit on train, transform both)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Baseline model
    estimator_cls = config["estimator"]
    estimator_kwargs = config.get("estimator_kwargs", {})
    model = estimator_cls(**estimator_kwargs)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    preds = model.predict(X_test_scaled)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

    # Persist artifacts
    model_path = ARTIFACT_DIR / f"model_{version}.pkl"
    scaler_path = ARTIFACT_DIR / f"scaler_{version}.pkl"
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    # keep latest pointers for backwards compatibility / Docker build
    joblib.dump(model, ARTIFACT_DIR / "model.pkl")
    joblib.dump(scaler, ARTIFACT_DIR / "scaler.pkl")
    with open(ARTIFACT_DIR / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    # Persist metrics
    metrics = {
        "version": version,
        "rmse": rmse,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "random_state": seed,
        "model": model.__class__.__name__,
        "model_description": config["description"],
        "scaler": "StandardScaler",
    }

    # include baseline comparison if available
    baseline_metrics_path = Path("metrics_v0.1.json")
    if version != "v0.1" and baseline_metrics_path.exists():
        try:
            baseline_metrics = json.loads(baseline_metrics_path.read_text())
            baseline_rmse = baseline_metrics.get("rmse")
            if isinstance(baseline_rmse, (int, float)):
                metrics["baseline_rmse_v0_1"] = float(baseline_rmse)
                metrics["rmse_delta_vs_v0_1"] = rmse - float(baseline_rmse)
        except json.JSONDecodeError:
            pass

    metrics_path_versioned = Path(f"metrics_{version}.json")
    with open(metrics_path_versioned, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(
        f"[train] {version} ({model.__class__.__name__}) RMSE: {rmse:.4f} (seed={seed})"
    )
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train diabetes progression model")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument(
        "--model",
        choices=sorted(MODEL_REGISTRY.keys()),
        default="ridge",
        help="Model to train (default: ridge)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(seed=args.seed, model_name=args.model)
