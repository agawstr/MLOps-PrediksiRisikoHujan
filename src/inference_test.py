"""Uji kesiapan inferensi: muat model Production dari Registry lalu prediksi data baru (LK-07)."""
import sys
import time

sys.path.insert(0, "src")
import mlflow
import pandas as pd
from train import FEATURES, TIME_COL, load_data

mlflow.set_tracking_uri("sqlite:///mlflow.db")
URI = "models:/rain-risk-classifier@production"

t0 = time.perf_counter()
model = mlflow.pyfunc.load_model(URI)
print(f"Load dari registry : {time.perf_counter() - t0:.2f} s")
print("Fitur signature    :", [c.name for c in model.metadata.get_input_schema().inputs])

df = load_data()
test = df.iloc[int(len(df) * 0.7):]                      # bagian data uji (split sama dengan train.py)
new = test.sample(10, random_state=42).sort_values(TIME_COL)   # simulasi 10 data baru

t0 = time.perf_counter()
pred = model.predict(new[FEATURES])
ms = (time.perf_counter() - t0) * 1000
out = new[[TIME_COL, "is_rain_risk"]].assign(pred_is_rain_risk=pred)
print(out.to_string(index=False))
print(f"Inferensi {len(new)} baris: {ms:.1f} ms")