import json, math, os, time
from collections import deque

import httpx
from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

UPSTREAM_URL = os.getenv("UPSTREAM_URL", "http://nginx:80")
WINDOW = int(os.getenv("DRIFT_WINDOW", "200"))
MIN_N = int(os.getenv("DRIFT_MIN_N", "30"))
BASELINE_PATH = os.getenv("BASELINE_PATH", "/app/baseline.json")

REQ = Counter("rr_requests_total", "Request ke /invocations per status HTTP", ["status"])
LAT = Histogram("rr_request_latency_seconds", "Latensi end-to-end /invocations (detik)",
                buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5))
PRED = Counter("rr_predictions_total", "Jumlah prediksi per label", ["label"])
UPS = Counter("rr_upstream_requests_total", "Request per replika (alamat upstream)", ["upstream"])
POS_W = Gauge("rr_positive_rate_window", "Rasio prediksi 1 pada jendela terbaru")
POS_B = Gauge("rr_positive_rate_baseline", "Rasio prediksi 1 baseline")
WIN_N = Gauge("rr_window_size", "Jumlah baris pada jendela")
F_W = Gauge("rr_feature_window_mean", "Rata-rata fitur pada jendela", ["feature"])
F_B = Gauge("rr_feature_baseline_mean", "Rata-rata fitur baseline (X_train)", ["feature"])
F_D = Gauge("rr_feature_drift_d", "(mean jendela - mean baseline) / std baseline", ["feature"])


def load_baseline():
    try:
        with open(BASELINE_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


BASE = load_baseline()
FEATS = BASE.get("features", {})
for _n, _s in FEATS.items():
    F_B.labels(_n).set(_s["mean"])
if "positive_rate" in BASE:
    POS_B.set(BASE["positive_rate"])

rows = deque(maxlen=WINDOW)
preds_win = deque(maxlen=WINDOW)
client = httpx.AsyncClient(timeout=60)
app = FastAPI()


def record(body_bytes, resp_json):
    try:
        df = json.loads(body_bytes)["dataframe_split"]
        for row in df["data"]:
            rows.append(dict(zip(df["columns"], row)))
        for p in resp_json["predictions"]:
            PRED.labels(str(int(p))).inc()
            preds_win.append(int(p))
    except Exception:
        pass


def refresh():
    WIN_N.set(len(rows))
    if preds_win:
        POS_W.set(sum(preds_win) / len(preds_win))
    for name, s in FEATS.items():
        vals = [r[name] for r in rows
                if isinstance(r.get(name), (int, float)) and not math.isnan(r[name])]
        if len(vals) < MIN_N or not s.get("std"):
            continue
        m = sum(vals) / len(vals)
        F_W.labels(name).set(m)
        F_D.labels(name).set((m - s["mean"]) / s["std"])


@app.post("/invocations")
async def invocations(request: Request):
    body = await request.body()
    start = time.perf_counter()
    status = "502"
    try:
        r = await client.post(f"{UPSTREAM_URL}/invocations", content=body,
                              headers={"Content-Type": request.headers.get("content-type", "application/json")})
        status = str(r.status_code)
        UPS.labels(r.headers.get("x-upstream", "unknown")).inc()
        if r.status_code == 200:
            record(body, r.json())
        return Response(content=r.content, status_code=r.status_code,
                        media_type=r.headers.get("content-type", "application/json"))
    except httpx.HTTPError:
        return Response(content='{"error":"upstream unreachable"}', status_code=502,
                        media_type="application/json")
    finally:
        REQ.labels(status).inc()
        LAT.observe(time.perf_counter() - start)


@app.get("/ping")
async def ping():
    try:
        r = await client.get(f"{UPSTREAM_URL}/ping")
        return Response(content=r.content, status_code=r.status_code)
    except httpx.HTTPError:
        return Response(status_code=502)


@app.get("/metrics")
def metrics():
    refresh()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
