import json, os, time
from pathlib import Path
import joblib, pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

MODEL = joblib.load("models/model.joblib")
THRESHOLD = json.load(open("models/threshold.json"))["threshold"]
FEATURES = json.load(open("models/features.json"))
LOG = Path(os.environ.get("LOG_PATH", "logs/predictions.jsonl"))
LOG.parent.mkdir(exist_ok=True)

app = FastAPI(title="Breast cancer demo API (educational only)")

class Request(BaseModel):
    features: dict[str, float]

@app.get("/health")
def health():
    return {"status": "ok"}

def _score(feats: dict[str, float]) -> dict:
    missing = [f for f in FEATURES if f not in feats]
    if missing:
        raise HTTPException(422, f"Missing features, e.g. {missing[:3]}")
    X = pd.DataFrame([[feats[f] for f in FEATURES]], columns=FEATURES)
    prob = float(MODEL.predict_proba(X)[0, 1])
    label = "malignant" if prob >= THRESHOLD else "benign"
    with open(LOG, "a") as fh:
        fh.write(json.dumps({"ts": time.time(), **feats, "prob": prob}) + "\n")
    return {"label": label, "probability_malignant": round(prob, 4)}


@app.post("/predict")
def predict(req: Request):
    return {**_score(req.features),
            "disclaimer": "Educational demo. Not for medical use."}


class BatchRequest(BaseModel):
    rows: list[dict[str, float]]


@app.post("/predict_batch")
def predict_batch(req: BatchRequest):
    if not 0 < len(req.rows) <= 1000:
        raise HTTPException(422, "Send between 1 and 1000 rows")
    return {"results": [_score(r) for r in req.rows],
            "disclaimer": "Educational demo. Not for medical use."}


@app.get("/features")
def features():
    return FEATURES


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    return (Path(__file__).parent / "index.html").read_text()
