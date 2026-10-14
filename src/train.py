import json
from pathlib import Path
import joblib, mlflow, mlflow.sklearn, numpy as np, pandas as pd, yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

params = yaml.safe_load(open("params.yaml"))
p, seed = params["train"], params["prepare"]["seed"]

train = pd.read_csv("data/processed/train.csv")
X, y = train.drop(columns="malignant"), train["malignant"]

if p["model"] == "logreg":
    clf = LogisticRegression(C=p["C"], max_iter=1000)
else:
    clf = RandomForestClassifier(
        n_estimators=p["n_estimators"], max_depth=p["max_depth"], random_state=seed
    )
model = make_pipeline(StandardScaler(), clf)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

prec, rec, thr = precision_recall_curve(y, oof)
ok = np.where(rec[:-1] >= p["target_recall"])[0]
threshold = float(thr[ok[-1]]) if len(ok) else 0.5   # highest threshold meeting the recall target

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("breast-cancer")
with mlflow.start_run():
    mlflow.log_params({**p, "threshold": round(threshold, 4)})
    mlflow.log_metric("cv_roc_auc", roc_auc_score(y, oof))
    model.fit(X, y)
    mlflow.sklearn.log_model(model, "model")

Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/model.joblib")
json.dump({"threshold": threshold}, open("models/threshold.json", "w"))
json.dump(list(X.columns), open("models/features.json", "w"))