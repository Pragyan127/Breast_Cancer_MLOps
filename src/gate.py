import json, sys, yaml

g = yaml.safe_load(open("params.yaml"))["gate"]
m = json.load(open("metrics.json"))
checks = {
    "recall_malignant": m["recall_malignant"] >= g["min_recall"],
    "precision_malignant": m["precision_malignant"] >= g["min_precision"],
    "roc_auc": m["roc_auc"] >= g["min_roc_auc"],
}
print(checks)
sys.exit(0 if all(checks.values()) else 1)