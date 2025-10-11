# Changelog

All notable changes to this project will be documented here.

## [Unreleased]

## [v0.2] - 2025-10-11

### Model Improvement: Ridge Regression

**Motivation**: Ridge regression adds L2 regularization to linear regression, which can reduce overfitting and improve generalization, especially with multicollinear features like those in the diabetes dataset.

**Changes**:
- Upgraded from `LinearRegression` to `Ridge(alpha=1.0)` 
- Extended training script to support multiple model types (`--model ridge|linear`)
- Updated API to serve v0.2 model with backward compatibility fallback
- Added versioned artifact persistence (`model_v0.2.pkl`, `scaler_v0.2.pkl`, `metrics_v0.2.json`)
- Enhanced metrics tracking with baseline comparison

**Performance Improvement**:

| Version | Model | RMSE | Delta vs v0.1 | Notes |
|---------|-------|------|---------------|-------|
| v0.1 | LinearRegression | 53.853 | - | Baseline |
| v0.2 | Ridge(alpha=1.0) | 53.778 | **-0.076** | **1.4% improvement** |

**Justification**: While the improvement is modest (~1.4%), Ridge regression provides:
- Better numerical stability with regularization
- Reduced risk of overfitting on new patient data
- Foundation for future hyperparameter tuning (alpha optimization)

## [v0.1] - 2025-10-11

- Baseline: StandardScaler + LinearRegression
- Deterministic training (random_state=42), dependency versions pinned
- Saved artifacts: model.pkl, scaler.pkl, feature_names.json
- Logged metrics: metrics.json (includes RMSE and run metadata)
- Added Flask API service with /health and /predict endpoints
- Implemented JSON error handling and feature-order validation
- Added smoke tests for API routes
- Added Dockerfile and docker-compose.yml
- Container exposes port 8000 with built-in healthcheck
- Verified service runs via Docker Compose
- Added CI/CD GitHub Actions:
  - ci.yml for lint/tests/artifacts on push
  - release.yml for Docker build/publish and GitHub Release creation