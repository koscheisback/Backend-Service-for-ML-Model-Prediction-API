"""One-off helper that produces the *provided* model artifact (model/model.joblib).

NOT part of the service: the service never trains, it only loads the file.
"""
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

data = load_iris()
model = LogisticRegression(max_iter=500).fit(data.data, data.target)
out = Path(__file__).resolve().parent.parent / "model" / "model.joblib"
joblib.dump({"model": model, "classes": list(data.target_names)}, out)
print(f"wrote {out}")
