import json
from pathlib import Path
import joblib, mlflow, pandas as pd
from sklearn.metrics import (confusion_matrix, precision_score,
                             recall_score, roc_auc_score, brier_score_loss)

test = pd.read_csv("data/processed/test.csv")
X, y = test.drop(columns="malignant"), test["malignant"]
model = joblib.load("models/model.joblib")
thr = json.load(open("models/threshold.json"))["threshold"]

proba = model.predict_proba(X)[:, 1]
pred = (proba >= thr).astype(int)
tn, fp, fn, tp = confusion_matrix(y, pred).ravel()

metrics = {
    "recall_malignant": recall_score(y, pred),
    "precision_malignant": precision_score(y, pred),
    "roc_auc": roc_auc_score(y, proba),
    "brier": brier_score_loss(y, proba),
    "false_negatives": int(fn),
    "false_positives": int(fp),
}
json.dump(metrics, open("metrics.json", "w"), indent=2)
print(metrics)

# Log test-set results to the MLflow run that trained this model
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("breast-cancer")
run_file = Path("models/run_id.txt")
run_id = run_file.read_text().strip() if run_file.exists() else None
try:
    mlflow.get_run(run_id)
except Exception:
    run_id = None   # training run not in this MLflow store; start a fresh run
with mlflow.start_run(run_id=run_id, run_name=None if run_id else "evaluate"):
    mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
