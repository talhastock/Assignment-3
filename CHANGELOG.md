# Changelog

All notable changes to this project will be documented here.

## [Unreleased]

## [v0.3] - 2025-10-15

### Model Improvement: Random Forest (n_estimator = 100)

**Motivation**: Random Forest uses ensemble learning with multiple decision trees, which can capture non-linear relationships and reduce overfitting through bagging, especially effective with complex feature interactions like those in the diabetes dataset.

**Changes**:
- Upgraded from `LinearRegression` to `RandomForestRegressor(n_estimators=100)` 
- Extended training script to support multiple model types (`--model forest|linear`)
- Updated API to serve v0.3 model with backward compatibility fallback
- Added versioned artifact persistence (`model_v0.3.pkl`, `scaler_v0.3.pkl`, `metrics_v0.3.json`)
- Enhanced metrics tracking with baseline comparison

**Performance Improvement**:

| Version | Model | RMSE | Delta vs v0.1 | Notes |
|---------|-------|------|---------------|-------|
| v0.1 | LinearRegression | 53.853 | - | Baseline |
| v0.3 | Random Forest(n_estimator = 100) | 54.3984 | **+0.5454** | **1.01% improvement** |

**Justification**: While the improvement is modest (~1.01%), Random Forest provides:
- Better handling of non-linear feature relationships
- Reduced risk of overfitting through ensemble averaging
- Foundation for future hyperparameter tuning (n_estimators, max_depth optimization)

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