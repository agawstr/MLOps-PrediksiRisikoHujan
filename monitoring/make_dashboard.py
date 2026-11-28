import json
from pathlib import Path

OUT = Path(__file__).parent / "grafana/provisioning/dashboards/rain-risk.json"
n = [0]
G, Y, R = "green", "yellow", "red"

def P(title, kind, exprs, x, y, w, h, unit=None, desc="", steps=None, mn=None, mx=None, stack=False):
    n[0] += 1
    d = {}
    if unit: d["unit"] = unit
    if mn is not None: d["min"] = mn
    if mx is not None: d["max"] = mx
    if steps:
        d["thresholds"] = {"mode": "absolute", "steps": [{"color": c, "value": v} for c, v in steps]}
    c = {}
    if kind == "timeseries":
        if steps: c["thresholdsStyle"] = {"mode": "line"}
        if stack: c.update({"stacking": {"mode": "normal"}, "fillOpacity": 40})
        d["custom"] = c
    p = {"id": n[0], "title": title, "type": kind, "description": desc,
         "gridPos": {"x": x, "y": y, "w": w, "h": h},
         "fieldConfig": {"defaults": d, "overrides": []},
         "targets": [{"expr": e, "legendFormat": l, "refId": chr(65 + i)} for i, (e, l) in enumerate(exprs)]}
    if kind == "stat":
        p["options"] = {"colorMode": "background", "reduceOptions": {"calcs": ["lastNotNull"]}}
    else:
        p["options"] = {"legend": {"displayMode": "table", "placement": "bottom", "calcs": ["lastNotNull", "max"]}}
    return p

RATE = "sum(rate(rr_requests_total[1m]))"
ERR = '(sum(rate(rr_requests_total{status!="200"}[1m])) or vector(0)) / ' + RATE + " * 100"
HQ = lambda q: f"histogram_quantile({q}, sum(rate(rr_request_latency_seconds_bucket[1m])) by (le))"
DIFF = "abs(rr_positive_rate_window - rr_positive_rate_baseline) and on() (rr_window_size >= 30)"
DS = [(G, None), (Y, 0.25), (R, 0.5)]
DP = [(G, None), (Y, 0.10), (R, 0.15)]

panels = [
 P("Throughput (req/s)", "stat", [(RATE, "req/s")], 0, 0, 6, 4, "reqps", "Request per detik ke /invocations (rata-rata 1 menit)"),
 P("Error rate (%)", "stat", [(ERR, "error")], 6, 0, 6, 4, "percent", "Persen request non-200", [(G, None), (Y, 1), (R, 5)]),
 P("Drift fitur: max |d|", "stat", [("max(abs(rr_feature_drift_d))", "max|d|")], 12, 0, 6, 4, None,
   "d = (mean jendela - mean baseline) / std baseline. WARNING > 0,25; CRITICAL > 0,5", DS),
 P("Drift prediksi: |rasio 1 - baseline|", "stat", [(DIFF, "selisih")], 18, 0, 6, 4, None,
   "Selisih rasio prediksi kelas 1 (jendela vs baseline). WARNING > 0,10; CRITICAL > 0,15", DP),
 P("Latensi inferensi p50 / p95 / p99", "timeseries", [(HQ(0.5), "p50"), (HQ(0.95), "p95"), (HQ(0.99), "p99")], 0, 4, 12, 8, "s",
   "Latensi end-to-end gateway -> nginx -> replika, dari histogram"),
 P("Throughput (req/s)", "timeseries", [(RATE, "total")], 12, 4, 12, 8, "reqps"),
 P("Request per replika (distribusi trafik)", "timeseries", [("sum by (upstream)(rate(rr_upstream_requests_total[1m]))", "{{upstream}}")], 0, 12, 12, 8, "reqps",
   "Dari header X-Upstream nginx; tiga alamat = tiga replika", stack=True),
 P("Error rate (%)", "timeseries", [(ERR, "error %")], 12, 12, 12, 8, "percent"),
 P("Distribusi prediksi (per label)", "timeseries", [("sum by (label)(rate(rr_predictions_total[1m]))", "label {{label}}")], 0, 20, 12, 8, "ops",
   "Laju prediksi per kelas (0 = tidak berisiko, 1 = risiko hujan)", stack=True),
 P("Rasio prediksi 1: jendela vs baseline", "timeseries",
   [("rr_positive_rate_window and on() (rr_window_size >= 30)", "jendela"), ("rr_positive_rate_baseline", "baseline")], 12, 20, 12, 8, "percentunit", mn=0, mx=1),
 P("Drift per fitur |d| (WARNING 0,25 / CRITICAL 0,5)", "timeseries", [("abs(rr_feature_drift_d)", "{{feature}}")], 0, 28, 12, 8, None,
   "Pergeseran mean fitur input dalam satuan std baseline (jendela 200 baris terakhir)", DS, mn=0),
 P("CPU gateway (proses)", "timeseries", [('rate(process_cpu_seconds_total{job="gateway"}[1m])', "gateway")], 12, 28, 6, 8, "percentunit"),
 P("RAM gateway (proses)", "timeseries", [('process_resident_memory_bytes{job="gateway"}', "gateway")], 18, 28, 6, 8, "bytes"),
]
dash = {"uid": "rain-risk-obs", "title": "Rain Risk - Observability (LK-11)", "tags": ["lk11"],
        "refresh": "5s", "time": {"from": "now-15m", "to": "now"}, "schemaVersion": 39, "panels": panels}
OUT.write_text(json.dumps(dash, indent=2))
print("ditulis:", OUT, "| panel:", len(panels))
