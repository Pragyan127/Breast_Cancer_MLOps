from pathlib import Path
import pandas as pd, yaml
from sklearn.model_selection import train_test_split

p = yaml.safe_load(open("params.yaml"))["prepare"]
df = pd.read_csv("data/raw/wdbc.csv")
df["malignant"] = 1 - df.pop("target")   # malignant is now the positive class

train, test = train_test_split(
    df, test_size=p["test_size"], stratify=df["malignant"], random_state=p["seed"]
)
Path("data/processed").mkdir(parents=True, exist_ok=True)
train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)