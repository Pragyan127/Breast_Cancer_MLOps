import json, time
from pathlib import Path
import joblib, pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL = joblib.load("models/model.joblib")
THRESHOLD = json.load(open("models/threshold.json"))["threshold"]
FEATURES = json.load(open("models/features.json"))
LOG = Path("logs/predictions.jsonl")
LOG.parent.mkdir(exist_ok=True)

app = FastAPI(title="Breast cancer demo API (educational only)")

class Request(BaseModel):
    features: dict[str, float]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: Request):
    missing = [f for f in FEATURES if f not in req.features]
    if missing:
        raise HTTPException(422, f"Missing features, e.g. {missing[:3]}")
    X = pd.DataFrame([[req.features[f] for f in FEATURES]], columns=FEATURES)
    prob = float(MODEL.predict_proba(X)[0, 1])
    label = "malignant" if prob >= THRESHOLD else "benign"
    with open(LOG, "a") as fh:
        fh.write(json.dumps({"ts": time.time(), **req.features, "prob": prob}) + "\n")
    return {"label": label, "probability_malignant": round(prob, 4),
            "disclaimer": "Educational demo. Not for medical use."}