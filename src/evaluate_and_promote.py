"""evaluate_and_promote.py - Tahap 3 dan 4 LK-8: gerbang validasi, lalu registry Staging."""
import os
import sys
from pathlib import Path

import mlflow
import yaml
from mlflow.tracking import MlflowClient

from train import EXPERIMENT_NAME, MODEL_NAME, TRACKING_URI, git_commit

cfg = yaml.safe_load(Path("config/thresholds.yaml").read_text())
mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()
sha = git_commit()

runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME],
                          filter_string=f"params.git_commit = '{sha}'",
                          order_by=["metrics.test_f1 DESC"])
if runs.empty:
    sys.exit(f"Tidak ada run untuk commit {sha}. Apakah tahap training berjalan?")
best = runs.iloc[0]
f1, recall = best["metrics.test_f1"], best["metrics.test_recall"]
gap = best["metrics.train_f1"] - f1

checks = [
    ("test_recall", recall, f">= {cfg['min_test_recall']}", recall >= cfg["min_test_recall"]),
    ("test_f1", f1, f">= {cfg['min_test_f1']}", f1 >= cfg["min_test_f1"]),
    ("gap F1 (train-test)", gap, f"<= {cfg['max_gap_f1']}", gap <= cfg["max_gap_f1"]),
]
try:  # bandingkan dengan Production kalau sudah ada
    prod = client.get_model_version_by_alias(MODEL_NAME, "production")
    prod_f1 = client.get_run(prod.run_id).data.metrics["test_f1"]
    checks.append(("test_f1 vs Production", f1, f">= {prod_f1 - cfg['max_regression']:.3f}",
                   f1 >= prod_f1 - cfg["max_regression"]))
except Exception:
    print("Belum ada model Production, perbandingan dilewati.")

passed = all(c[3] for c in checks)
rows = "\n".join(f"| {n} | {v:.3f} | {r} | {'LULUS' if ok else 'GAGAL'} |" for n, v, r, ok in checks)
report = (f"### Evaluasi run `{best['tags.mlflow.runName']}` (commit {sha})\n\n"
          f"| Kriteria | Nilai | Syarat | Hasil |\n|---|---|---|---|\n{rows}\n\n"
          f"**Keputusan: {'LOLOS' if passed else 'DITOLAK'}**\n")
print(report)
if os.getenv("GITHUB_STEP_SUMMARY"):
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
        f.write(report)
if not passed:
    sys.exit(1)
if os.getenv("GITHUB_EVENT_NAME") == "pull_request":
    print("PR: lolos validasi, registry tidak diubah (hanya untuk push ke main).")
    sys.exit(0)

ver = str(mlflow.register_model(best["tags.model_uri"], MODEL_NAME).version)
client.update_model_version(MODEL_NAME, ver, description=(
    f"Dilatih otomatis oleh CI (commit {sha}), run {best['tags.mlflow.runName']}: "
    f"test_f1={f1:.3f}, test_recall={recall:.3f}. Lolos config/thresholds.yaml."))
tags = {"git_commit": sha, "ci_run": os.getenv("GITHUB_RUN_NUMBER", "local"),
        "evaluation_passed": "true",
        "data_dvc_md5": str(best.get("params.data_dvc_md5", "unknown")),
        "stage": "Staging"}
for k, v in tags.items():
    client.set_model_version_tag(MODEL_NAME, ver, k, v)
try:
    client.transition_model_version_stage(MODEL_NAME, ver, stage="Staging")
except Exception as e:
    print("[info] stage API tidak tersedia:", type(e).__name__)
client.set_registered_model_alias(MODEL_NAME, "staging", ver)
print(f"Terdaftar: {MODEL_NAME} v{ver} -> Staging")
