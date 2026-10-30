import pytest
from fastapi.testclient import TestClient

from house_price.api.app import create_app


@pytest.fixture()
def client(trained_predictor, settings):
    with TestClient(create_app(predictor=trained_predictor, settings=settings)) as test_client:
        yield test_client


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    response = client.get("/api/v1/model")
    assert response.status_code == 200
    assert "test_metrics" in response.json()["metadata"]


def test_predict_success(client, valid_house):
    response = client.post("/api/v1/predict", json=valid_house)
    assert response.status_code == 200
    assert response.json()["predicted_price_usd"] > 0


def test_predict_batch(client, valid_house):
    response = client.post("/api/v1/predict/batch", json={"houses": [valid_house, valid_house]})
    body = response.json()
    assert response.status_code == 200
    assert body["count"] == 2 and len(body["predictions_usd"]) == 2


def test_predict_rejects_out_of_range_value(client, valid_house):
    response = client.post("/api/v1/predict", json={**valid_house, "Latitude": 80})
    assert response.status_code == 422


def test_predict_rejects_missing_field(client, valid_house):
    payload = {k: v for k, v in valid_house.items() if k != "MedInc"}
    assert client.post("/api/v1/predict", json=payload).status_code == 422


def test_batch_rejects_empty_list(client):
    assert client.post("/api/v1/predict/batch", json={"houses": []}).status_code == 422


def test_service_unavailable_without_model(settings, valid_house):
    # model_path não existe: a API sobe "degradada" e responde 503 nas previsões
    with TestClient(create_app(settings=settings)) as test_client:
        assert test_client.get("/health").json() == {"status": "degraded", "model_loaded": False}
        assert test_client.post("/api/v1/predict", json=valid_house).status_code == 503
