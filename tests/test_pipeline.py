import json, os

def test_metrics_exist():
    assert os.path.exists("metrics.json")

def test_threshold_valid():
    thr = json.load(open("models/threshold.json"))["threshold"]
    assert 0.0 < thr < 1.0

def test_features_count():
    assert len(json.load(open("models/features.json"))) == 30