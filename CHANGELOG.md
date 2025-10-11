# Changelog

All notable changes to this project will be documented here.

## [Unreleased]

## [v0.1] - 2025-10-11

- Baseline: StandardScaler + LinearRegression
- Deterministic training (random_state=42), dependency versions pinned
- Saved artifacts: model.pkl, scaler.pkl, feature_names.json
- Logged metrics: metrics.json (includes RMSE and run metadata)
- Added Flask API service with /health and /predict endpoints
- Implemented JSON error handling and feature-order validation
- Added smoke tests for API routes

## [v0.2] - Improvement (planned)

- Try Ridge or RandomForestRegressor / better preprocessing
- Show metric deltas (RMSE; precision/recall if high-risk flag added)
- Update CHANGELOG with rationale