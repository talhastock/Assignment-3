# Assignment 3 – MLOps: Virtual Diabetes Clinic Triage

This repository implements a small ML service that predicts short-term diabetes disease progression (higher = worse) using the scikit-learn Diabetes regression dataset. The service is packaged in Docker, exposed via a simple HTTP API, and built/released via GitHub Actions.

## Context & Requirements

- **Users & Flow**
  - Triage nurse sees a dashboard sorted by predicted progression (descending).
  - ML Service exposes `/predict` per patient.
  - MLOps team (this repo) owns training, packaging, testing, releasing.

- **Iterations**
  - **v0.1**: Baseline – `StandardScaler + LinearRegression`, report RMSE, ship working API & Docker image.
  - **v0.2**: Improvement – try `Ridge` or `RandomForestRegressor` or better preprocessing. Show metric deltas (RMSE; and precision/recall if adding a high-risk flag) in `CHANGELOG.md`.

- **Non-functional**
  - **Portability**: Self-contained Docker image with baked model.
  - **Observability**: JSON errors on bad input.
  - **Reproducibility**: Deterministic training; pinned env; GitHub Actions retrains/builds consistently.

## Professor Clarifications

- Use **Docker Compose**.
- `/predict` may assume **already scaled inputs** (no scaling service required).
- **No MLflow** required; focus on pipeline.
- "Reproduce locally" = runnable in Docker; OK to train locally then bake model into the image released via Actions.

## Dataset

We use the open scikit-learn Diabetes regression dataset:

```python
from sklearn.datasets import load_diabetes

Xy = load_diabetes(as_frame=True)
X = Xy.frame.drop(columns=["target"])
y = Xy.frame["target"]  # progression index (higher = worse)
```

## API Contract (to be implemented in later steps)

**GET /health** → `{"status":"ok","model_version":"<semver>"}`

**POST /predict** with JSON features (scaled values):

```json
{
  "age": 0.02,
  "sex": -0.044,
  "bmi": 0.06,
  "bp": -0.03,
  "s1": -0.02,
  "s2": 0.03,
  "s3": -0.02,
  "s4": 0.02,
  "s5": 0.02,
  "s6": -0.001
}
```

Response:

```json
{"prediction": <float>}
```

Exact field names & response shape will be finalized and documented with the implementation.

## CI/CD Expectations (later steps)

- **PR/push workflow**: lint, tests, (optional) quick training smoke, upload artifacts.
- **Tag workflow (v*)**: build Docker image, run container smoke tests, push to GHCR, publish a GitHub Release with metrics & changelog.

## Submission

Submit a PDF containing the public GitHub repository URL.

The GitHub Actions tab must show:
- PR/push workflow runs.
- Tag workflow (v0.1, v0.2) that builds, tests, pushes image to GHCR, and creates a Release with metrics/changelog.

## Local Dev Quickstart (will be expanded later)

```bash
# create/activate venv (example)
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

pip install -r requirements.txt
# (training & API commands will be added in later steps)
```

## Repository Structure (will evolve)

```
src/              # training + API (to be implemented)
model/            # trained artifacts (baked into image for releases)
tests/            # unit/integration tests
.github/workflows # CI/CD workflows
```