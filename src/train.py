# src/train.py
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ARTIFACT_DIR = Path("model")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_PATH = Path("metrics.json")


def train(seed: int = 42) -> dict:
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
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Evaluate
    preds = model.predict(X_test_scaled)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

    # Persist artifacts
    joblib.dump(model, ARTIFACT_DIR / "model.pkl")
    joblib.dump(scaler, ARTIFACT_DIR / "scaler.pkl")
    with open(ARTIFACT_DIR / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    # Persist metrics
    metrics = {
        "version": "v0.1",
        "rmse": rmse,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "random_state": seed,
        "model": "LinearRegression",
        "scaler": "StandardScaler",
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[train] v0.1 RMSE: {rmse:.4f} (seed={seed})")
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train v0.1 baseline model")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(seed=args.seed)