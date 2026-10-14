import sys
from pathlib import Path
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

ref = pd.read_csv("data/processed/train.csv").drop(columns="malignant")
cur = pd.read_csv("data/processed/test.csv").drop(columns="malignant")

# Simulate drift: larger tumors, as if a new clinic with different equipment
for c in [c for c in cur.columns if "radius" in c or "area" in c or "perimeter" in c]:
    cur[c] = cur[c] * 1.3

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref, current_data=cur)
Path("reports").mkdir(exist_ok=True)
report.save_html("reports/drift.html")

share = report.as_dict()["metrics"][0]["result"]["share_of_drifted_columns"]
print(f"Drifted share: {share:.2f}")
sys.exit(1 if share > 0.3 else 0)