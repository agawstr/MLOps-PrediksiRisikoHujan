"""train.py - Pelatihan model Prediksi Risiko Hujan + pelacakan eksperimen MLflow (LK-06).

Alur:
  data/processed/weather_processed.csv (hasil preprocess LK-04, versi DVC LK-05)
  -> split berbasis waktu -> latih RandomForest dengan beberapa konfigurasi
  -> catat param / metrik / model ke MLflow -> (opsional) daftarkan model terbaik.

Cara pakai:
  python src/train.py                 # jalankan semua konfigurasi di CONFIGS
  python src/train.py --register      # + daftarkan run terbaik ke Model Registry
  mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import argparse
import subprocess
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)

# ----------------------------- Konfigurasi -----------------------------
DATA_PATH = Path("data/processed/weather_processed.csv")
TIME_COL = "local_datetime"
TARGET = "is_rain_risk"
# Jangan masukkan tp / weather_desc: label dibuat dari keduanya (data leakage).
FEATURES = [
    "t", "hu", "ws", "tcc",
    "sin_hour", "cos_hour",
    "t_lag1", "hu_lag1", "ws_lag1",
    "t_rolling_mean_3h", "hu_rolling_mean_3h",
]
TEST_SIZE = 0.3  # 30% data terbaru (~5 hari) agar mencakup hari hujan dan kering
SEED = 42
EXPERIMENT_NAME = "prediksi-risiko-hujan"
MODEL_NAME = "rain-risk-classifier"
TRACKING_URI = "sqlite:///mlflow.db"

# Variasi hyperparameter (minimal 3 run).
CONFIGS = [
    {"n_estimators": 50, "max_depth": 3, "min_samples_leaf": 5},
    {"n_estimators": 100, "max_depth": 5, "min_samples_leaf": 3},
    {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 2},
    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
]


# ------------------------------- Fungsi --------------------------------
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Baca dataset terproses dan pastikan kolom yang dibutuhkan ada."""
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + [TARGET, TIME_COL] if c not in df.columns]
    if missing:
        raise KeyError(f"Kolom tidak ada: {missing}. Kolom tersedia: {list(df.columns)}")
    return df.dropna(subset=FEATURES + [TARGET]).sort_values(TIME_COL).reset_index(drop=True)


def time_split(df: pd.DataFrame, test_size: float = TEST_SIZE):
    """Split berurutan waktu (data lama = train, data baru = test) agar tidak bocor."""
    cut = int(len(df) * (1 - test_size))
    train, test = df.iloc[:cut], df.iloc[cut:]
    return train[FEATURES], test[FEATURES], train[TARGET], test[TARGET]


def git_commit() -> str:
    """Hash commit saat ini, untuk melacak versi kode (reproduksibilitas)."""
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def dvc_data_hash(path: str = "data/processed.dvc") -> str:
    """Hash DVC dataset terproses = identitas versi data (LK-05)."""
    try:
        for line in Path(path).read_text().splitlines():
            if "md5:" in line:
                return line.split("md5:")[1].strip()
    except OSError:
        pass
    return "unknown"


def evaluate(model, X, y, prefix: str) -> dict:
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        f"{prefix}_accuracy": accuracy_score(y, pred),
        f"{prefix}_precision": precision_score(y, pred, zero_division=0),
        f"{prefix}_recall": recall_score(y, pred, zero_division=0),
        f"{prefix}_f1": f1_score(y, pred, zero_division=0),
        f"{prefix}_roc_auc": roc_auc_score(y, proba),
        f"{prefix}_pr_auc": average_precision_score(y, proba),
    }


def run_experiment(params: dict, data: tuple, data_info: dict) -> str:
    X_train, X_test, y_train, y_test = data
    run_name = f"rf_n{params['n_estimators']}_d{params['max_depth']}_leaf{params['min_samples_leaf']}"

    with mlflow.start_run(run_name=run_name) as run:
        # 1) Parameter
        model_params = {**params, "class_weight": "balanced", "random_state": SEED}
        mlflow.log_params(model_params)
        mlflow.log_params({"model_type": "RandomForestClassifier", "features": ",".join(FEATURES)})
        mlflow.log_params(data_info)

        # 2) Latih
        model = RandomForestClassifier(**model_params, n_jobs=-1)
        model.fit(X_train, y_train)

        # 3) Metrik (train + test untuk melihat overfitting)
        metrics = {**evaluate(model, X_train, y_train, "train"), **evaluate(model, X_test, y_test, "test")}
        mlflow.log_metrics(metrics)
        # Pembanding: model "selalu tebak hujan" (recall=1, precision=rasio hujan di test)
        pos = float(y_test.mean())
        mlflow.log_metrics({"test_positive_ratio": pos, "test_baseline_f1": 2 * pos / (1 + pos)})

        # 4) Artefak: feature importance + model
        importance = dict(zip(FEATURES, model.feature_importances_.round(4).tolist()))
        mlflow.log_dict(importance, "feature_importance.json")
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=X_train.head(3),
            skops_trusted_types=["sklearn.tree._tree.Tree"],  # model dilatih sendiri
        )

        print(
            f"{run_name}: test_f1={metrics['test_f1']:.3f} test_recall={metrics['test_recall']:.3f} "
            f"test_pr_auc={metrics['test_pr_auc']:.3f} baseline_f1={2 * pos / (1 + pos):.3f}"
        )
        return run.info.run_id


def register_best(experiment_name: str = EXPERIMENT_NAME, metric: str = "test_f1"):
    """Daftarkan run dengan metrik terbaik ke Model Registry."""
    best = mlflow.search_runs(
        experiment_names=[experiment_name], order_by=[f"metrics.{metric} DESC"], max_results=1
    ).iloc[0]
    uri = f"runs:/{best.run_id}/model"
    version = mlflow.register_model(uri, MODEL_NAME)
    print(f"Terdaftar: {MODEL_NAME} v{version.version} dari run {best.run_id} ({metric}={best[f'metrics.{metric}']:.3f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--register", action="store_true", help="daftarkan run terbaik ke Model Registry")
    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_data()
    data = time_split(df)
    data_info = {
        "data_path": str(DATA_PATH),
        "n_rows": len(df),
        "n_train": len(data[0]),
        "n_test": len(data[1]),
        "label_positive_ratio": round(float(df[TARGET].mean()), 3),
        "git_commit": git_commit(),
        "data_dvc_md5": dvc_data_hash(),
    }

    for params in CONFIGS:
        run_experiment(params, data, data_info)

    if args.register:
        register_best()


if __name__ == "__main__":
    main()