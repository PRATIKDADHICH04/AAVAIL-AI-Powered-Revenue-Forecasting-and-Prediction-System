import logging
import numpy as np
from app.app import create_app

class FakeModel:
    def predict(self, frame):
        return np.array([100.0] * len(frame))

def test_health():
    client = create_app(FakeModel(), {"countries": ["USA"]}, logging.getLogger("test")).test_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json["model_loaded"] is True

def test_country_prediction():
    client = create_app(FakeModel(), {"countries": ["USA"]}, logging.getLogger("test")).test_client()
    response = client.post("/predict", json={"target_date": "2024-01-01", "country": "USA"})
    assert response.status_code == 200
    assert response.json["predicted_revenue"] == 100.0

def test_all_country_prediction():
    client = create_app(FakeModel(), {"countries": ["USA", "Germany"]}, logging.getLogger("test")).test_client()
    response = client.post("/predict", json={"target_date": "2024-01-01"})
    assert response.status_code == 200
    assert response.json["total_predicted_revenue"] == 200.0

def test_missing_date_rejected():
    client = create_app(FakeModel(), {"countries": ["USA"]}, logging.getLogger("test")).test_client()
    assert client.post("/predict", json={}).status_code == 400
