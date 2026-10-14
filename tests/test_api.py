from fastapi.testclient import TestClient
from app.main import app, FEATURES

client = TestClient(app)

def test_health():
    assert client.get("/health").status_code == 200

def test_predict_valid():
    payload = {"features": {f: 1.0 for f in FEATURES}}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert r.json()["label"] in ("malignant", "benign")

def test_predict_missing_feature():
    r = client.post("/predict", json={"features": {"mean radius": 1.0}})
    assert r.status_code == 422