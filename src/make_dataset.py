from pathlib import Path
from sklearn.datasets import load_breast_cancer

Path("data/raw").mkdir(parents=True, exist_ok=True)
df = load_breast_cancer(as_frame=True).frame
df.to_csv("data/raw/wdbc.csv", index=False)
print(df.shape) 