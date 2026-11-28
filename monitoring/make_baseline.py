import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.train import FEATURES, load_data, time_split

df = load_data()
X_train, X_test, y_train, y_test = time_split(df)

feats = {}
for f in FEATURES:
    col = X_train[f].dropna()
    feats[f] = {"mean": float(col.mean()), "std": float(col.std())}

out = {"source": "X_train (time_split src/train.py)", "n_train": int(len(X_train)), "features": feats}
(ROOT / "monitoring" / "baseline.json").write_text(json.dumps(out, indent=2))
print("n_train =", len(X_train))
for k, v in feats.items():
    print(f"{k:20s} mean={v['mean']:.4f} std={v['std']:.4f}")
