# tests/test_api_smoke.py
import json
from src.app import app, feature_names


def test_health_route():
    client = app.test_client()
    res = client.get("/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "ok"
    assert "model_version" in data


def test_predict_route():
    client = app.test_client()

    # create dummy input (scaled-like values)
    payload = {name: 0.0 for name in feature_names}
    res = client.post("/predict", data=json.dumps(payload), content_type="application/json")
    assert res.status_code == 200
    out = res.get_json()
    assert "prediction" in out
    assert isinstance(out["prediction"], float)
