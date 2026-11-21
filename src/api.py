import os
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException

app = FastAPI(title="Prediksi Risiko Hujan")
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
MODEL_URI = os.environ.get("MODEL_URI", "models:/rain-risk-classifier@staging")
model = None


@app.on_event("startup")
def load_model():
    global model
    try:
        model = mlflow.pyfunc.load_model(MODEL_URI)
        print(f"Model berhasil dimuat dari {MODEL_URI}")
    except Exception as e:
        print(f"Gagal memuat model: {e}")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None,
            "tracking_uri": os.environ["MLFLOW_TRACKING_URI"]}


@app.post("/predict")
def predict(features: dict):
    if model is None:
        raise HTTPException(503, "Model belum dimuat")
    return {"prediction": model.predict(pd.DataFrame([features])).tolist()}