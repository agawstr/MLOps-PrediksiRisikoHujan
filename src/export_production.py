"""Ekspor model Production ke models/production + tulis metadata lineage (LK-07)."""
from pathlib import Path

import mlflow
import yaml
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
c = MlflowClient()
NAME = "rain-risk-classifier"

v = c.get_model_version_by_alias(NAME, "production")
r = c.get_run(v.run_id)
p, m = r.data.params, r.data.metrics

dst = Path("models/production")
dst.mkdir(parents=True, exist_ok=True)
try:
    mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{NAME}@production", dst_path=str(dst))
except Exception:
    mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{v.run_id}/model", dst_path=str(dst))

meta = {
    "model_name": NAME, "version": int(v.version), "alias": "production",
    "mlflow_run_id": v.run_id, "run_name": r.info.run_name,
    "params": {k: p[k] for k in ("n_estimators", "max_depth", "min_samples_leaf")},
    "metrics": {k: round(m[k], 4) for k in ("test_f1", "test_recall", "test_precision", "test_pr_auc")},
    "data_path": p["data_path"], "data_dvc_md5": p["data_dvc_md5"], "git_commit": p["git_commit"],
}
Path("models/production_meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
print(yaml.safe_dump(meta, sort_keys=False))