"""Isi deskripsi + tag metadata untuk satu versi model di Registry (LK-07).
Pakai: python src/registry_describe.py <versi> "<catatan>" """
import sys

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
c = MlflowClient()
NAME = "rain-risk-classifier"
ver, note = sys.argv[1], sys.argv[2]

if ver == "1":  # deskripsi level model, cukup sekali
    c.update_registered_model(NAME, description=(
        "Klasifikasi biner risiko hujan (is_rain_risk) untuk 30 kelurahan Kota Malang. "
        "Data: BMKG API (LK-04) diproses ke weather_processed.csv, diversi DVC (LK-05), "
        "eksperimen dilacak MLflow (LK-06). Split berbasis waktu 70/30, metrik utama test_f1."))
    c.set_registered_model_tag(NAME, "task", "binary-classification")
    c.set_registered_model_tag(NAME, "domain", "weather-rain-risk")

v = c.get_model_version(NAME, ver)
r = c.get_run(v.run_id)
p, m = r.data.params, r.data.metrics
c.update_model_version(NAME, ver, description=(
    f"{note}. RandomForest n_estimators={p['n_estimators']}, max_depth={p['max_depth']}, "
    f"min_samples_leaf={p['min_samples_leaf']}, class_weight=balanced. "
    f"Test: f1={m['test_f1']:.3f}, recall={m['test_recall']:.3f}, "
    f"precision={m['test_precision']:.3f}, pr_auc={m['test_pr_auc']:.3f}. "
    f"Data: {p['data_path']} (DVC md5 {p['data_dvc_md5']}), commit {p['git_commit']}."))
tags = {
    "run_name": r.info.run_name, "algorithm": "RandomForest",
    "data_dvc_md5": p["data_dvc_md5"], "git_commit": p["git_commit"],
    "test_f1": f"{m['test_f1']:.3f}", "test_recall": f"{m['test_recall']:.3f}",
}
for k, val in tags.items():
    c.set_model_version_tag(NAME, ver, k, val)
print(f"OK v{ver}:", r.info.run_name, f"test_f1={m['test_f1']:.3f}")