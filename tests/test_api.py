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

def test_index_page():
    r = client.get("/")
    assert r.status_code == 200
    assert "Breast Cancer Classifier" in r.text


def test_features_endpoint():
    assert client.get("/features").json() == FEATURES


def test_predict_batch():
    rows = [{f: 1.0 for f in FEATURES}] * 3
    r = client.post("/predict_batch", json={"rows": rows})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 3


def test_predict_batch_empty():
    assert client.post("/predict_batch", json={"rows": []}).status_code == 422
