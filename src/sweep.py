"""Train several configurations with the same CV protocol as train.py and log each to MLflow.

Does not write models/ and does not use the test set, so the DVC pipeline is unaffected.
"""
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss, precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

params = yaml.safe_load(open("params.yaml"))
seed, target_recall = params["prepare"]["seed"], params["train"]["target_recall"]
train = pd.read_csv("data/processed/train.csv")
X, y = train.drop(columns="malignant"), train["malignant"]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

configs = (
    [{"model": "logreg", "C": c} for c in (0.01, 0.1, 0.5, 1.0, 10.0)]
    + [{"model": "rf", "n_estimators": n, "max_depth": d}
       for n, d in ((100, 3), (200, 5), (300, 8))]
)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("breast-cancer")
for cfg in configs:
    if cfg["model"] == "logreg":
        clf = LogisticRegression(C=cfg["C"], max_iter=1000)
        name = f"logreg_C{cfg['C']}"
    else:
        clf = RandomForestClassifier(
            n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"], random_state=seed)
        name = f"rf_n{cfg['n_estimators']}_d{cfg['max_depth']}"
    model = make_pipeline(StandardScaler(), clf)
    oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    _, rec, thr = precision_recall_curve(y, oof)
    ok = np.where(rec[:-1] >= target_recall)[0]
    threshold = float(thr[ok[-1]]) if len(ok) else 0.5
    pred = oof >= threshold
    with mlflow.start_run(run_name=name):
        mlflow.set_tag("source", "sweep")
        mlflow.log_params({**cfg, "target_recall": target_recall,
                           "threshold": round(threshold, 4)})
        mlflow.log_metrics({
            "cv_roc_auc": roc_auc_score(y, oof),
            "cv_recall": recall_score(y, pred),
            "cv_precision": precision_score(y, pred),
            "cv_brier": brier_score_loss(y, oof),
        })
    print(f"{name:18s} auc={roc_auc_score(y, oof):.4f} "
          f"recall={recall_score(y, pred):.3f} precision={precision_score(y, pred):.3f}")
