"""train_v2.py - Latih kandidat v2 (parameter sedikit berbeda) lalu daftarkan sebagai versi baru."""
import mlflow
from train import (DATA_PATH, EXPERIMENT_NAME, MODEL_NAME, TARGET, TRACKING_URI,
                   dvc_data_hash, git_commit, load_data, run_experiment, time_split)

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

df = load_data()
data = time_split(df)
data_info = {
    "data_path": str(DATA_PATH), "n_rows": len(df),
    "n_train": len(data[0]), "n_test": len(data[1]),
    "label_positive_ratio": round(float(df[TARGET].mean()), 3),
    "git_commit": git_commit(), "data_dvc_md5": dvc_data_hash(),
}

params = {"n_estimators": 150, "max_depth": 6, "min_samples_leaf": 3}
run_id = run_experiment(params, data, data_info)
v = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
print(f"Terdaftar: {MODEL_NAME} v{v.version} dari run {run_id}")