from fastapi.testclient import TestClient

from ml_inference_service.api import app
from ml_inference_service.model import MODEL_VERSION, predict


client = TestClient(app)


def test_health_and_readiness():
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == {"status": "ready", "model_version": MODEL_VERSION}


def test_model_info_exposes_versioned_contract():
    response = client.get("/v1/model-info")
    body = response.json()
    assert response.status_code == 200
    assert body["model_version"] == MODEL_VERSION
    assert "requests_last_hour" in body["input_schema"]


def test_prediction_is_bounded_and_explainable():
    response = client.post(
        "/v1/predict",
        json={
            "requests_last_hour": 28,
            "active_drivers": 12,
            "avg_pickup_eta_minutes": 9.5,
            "rainfall_mm": 4,
            "hour": 18,
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert 0 <= body["demand_score"] <= 1
    assert 0 <= body["confidence"] <= 1
    assert body["model_version"] == MODEL_VERSION
    assert body["feature_contributions"]
    assert body["latency_ms"] >= 0


def test_model_is_deterministic():
    args = (28, 12, 9.5, 4, 18)
    assert predict(*args) == predict(*args)
