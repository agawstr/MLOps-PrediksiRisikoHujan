import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train import CONFIGS, FEATURES, TARGET, TIME_COL, evaluate, time_split


def make_df(n=300):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    df[TIME_COL] = pd.date_range("2026-10-01", periods=n, freq="h")
    df[TARGET] = (rng.random(n) > 0.7).astype(int)
    return df


def test_time_split_tanpa_kebocoran():
    X_tr, X_te, _, _ = time_split(make_df())
    assert X_tr.index.max() < X_te.index.min()
    assert len(X_tr) + len(X_te) == 300


def test_fitur_tidak_memuat_kolom_bocor():
    assert "tp" not in FEATURES and "weather_desc" not in FEATURES


def test_evaluate_metrik_valid():
    X_tr, X_te, y_tr, y_te = time_split(make_df())
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X_tr, y_tr)
    m = evaluate(model, X_te, y_te, "test")
    for k in ("test_f1", "test_recall", "test_precision", "test_pr_auc"):
        assert 0.0 <= m[k] <= 1.0


def test_konfigurasi_hyperparameter_lengkap():
    assert len(CONFIGS) >= 1
    for c in CONFIGS:
        assert {"n_estimators", "max_depth", "min_samples_leaf"} <= set(c)
