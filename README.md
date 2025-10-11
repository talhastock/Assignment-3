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

## Local Training (v0.1)

Train the baseline model (StandardScaler + LinearRegression), save artifacts, and write metrics:

```bash
# (optional) create/activate venv
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

pip install -r requirements.txt

# run training
python src/train.py --seed 42

# outputs:
# - model/model.pkl
# - model/scaler.pkl
# - model/feature_names.json
# - metrics.json (at repo root)
```

`metrics.json` includes rmse and run metadata:

```json
{
  "version": "v0.1",
  "rmse": 53.85,
  "n_train": 353,
  "n_test": 89,
  "random_state": 42,
  "model": "LinearRegression",
  "scaler": "StandardScaler"
}
```

Run smoke tests:

```bash
pytest -q
```

## API Service (Flask)

Run the API locally (requires that training artifacts already exist):

```bash
python src/app.py
```

### Endpoints

**GET /health**

Health check

Response:
```json
{"status": "ok", "model_version": "v0.1"}
```

**POST /predict**

Send scaled feature values (from scikit-learn Diabetes dataset order):

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":0.02,"sex":-0.044,"bmi":0.06,"bp":-0.03,"s1":-0.02,"s2":0.03,"s3":-0.02,"s4":0.02,"s5":0.02,"s6":-0.001}'
```

Response:
```json
{"prediction": 183.42, "status": "ok"}
```

If input is invalid or missing features, returns HTTP 400 with:
```json
{"error": "Missing features: ['bmi']", "status": "failed"}
```

## Docker Deployment (v0.1)

The service can be run entirely in Docker.

### Build and run manually
```bash
docker build -t diabetes-api:v0.1 .
docker run -d -p 8000:8000 --name diabetes_api diabetes-api:v0.1
```

### Or using Docker Compose
```bash
docker-compose up --build
```

### Verify health
```bash
curl http://localhost:8000/health
# {"status":"ok","model_version":"v0.1"}
```

### Example prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":0.02,"sex":-0.044,"bmi":0.06,"bp":-0.03,"s1":-0.02,"s2":0.03,"s3":-0.02,"s4":0.02,"s5":0.02,"s6":-0.001}'
```
Response:
```json
{"prediction": 157.75, "status": "ok"}
```

## CI/CD Pipelines (GitHub Actions)

Two workflows automate the project lifecycle:

### Continuous Integration (`ci.yml`)
Runs on every push/PR:
- Linting (`flake8`)
- Unit tests (`pytest`)
- Smoke training (`python src/train.py --seed 42`)
- Uploads artifacts (model and metrics)

### Release Pipeline (`release.yml`)
Triggered on tag push (`v0.1`, `v0.2`):
- Builds Docker image
- Pushes to GHCR (`ghcr.io/<username>/Assignment-3:<tag>`)
- Smoke tests container (`/health`)
- Creates GitHub Release with metrics and CHANGELOG summary

Example tag:
```bash
git tag v0.1
git push origin v0.1
```
This automatically publishes the Docker image and Release.

## Repository Structure (will evolve)

```
src/              # training + API (to be implemented)
model/            # trained artifacts (baked into image for releases)
tests/            # unit/integration tests
.github/workflows # CI/CD workflows
```