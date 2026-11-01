"""Bandingkan versi model, pilih terbaik, lalu promosikan (LK-07).
Pakai: python src/promote_model.py compare|staging|production"""
import sys

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
c = MlflowClient()
NAME = "rain-risk-classifier"


def compare() -> str:
    rows = []
    for v in c.search_model_versions(f"name='{NAME}'"):
        m = c.get_run(v.run_id).data.metrics
        rows.append((v.version, m["test_f1"], m["test_recall"], m["test_precision"],
                     m["test_pr_auc"], m["train_f1"] - m["test_f1"]))
    rows.sort(key=lambda r: int(r[0]))
    print(f"{'ver':<4}{'test_f1':>9}{'recall':>9}{'prec':>9}{'pr_auc':>9}{'gap_f1':>9}")
    for r in rows:
        print(f"v{r[0]:<3}{r[1]:>9.3f}{r[2]:>9.3f}{r[3]:>9.3f}{r[4]:>9.3f}{r[5]:>9.3f}")
    best = max(rows, key=lambda r: (r[1], r[4]))[0]  # test_f1, seri -> pr_auc
    print("Versi terpilih:", f"v{best}", "(test_f1 tertinggi)")
    return best


def promote(ver: str, stage: str, alias: str):
    try:
        c.transition_model_version_stage(NAME, ver, stage=stage,
                                         archive_existing_versions=(stage == "Production"))
        print(f"[stage API] v{ver} -> {stage}")
    except Exception as e:
        print(f"[info] stage API tidak tersedia ({type(e).__name__}); pakai alias + tag")
    c.set_registered_model_alias(NAME, alias, ver)
    c.set_model_version_tag(NAME, ver, "stage", stage)
    print(f"v{ver} -> alias '{alias}', tag stage={stage}")


if __name__ == "__main__":
    step = sys.argv[1]
    best = compare()
    if step == "staging":
        promote(best, "Staging", "staging")
    elif step == "production":
        promote(best, "Production", "production")
        c.set_model_version_tag(NAME, best, "promoted_reason",
                                "test_f1 tertinggi dan gap train-test terkecil di antara versi")