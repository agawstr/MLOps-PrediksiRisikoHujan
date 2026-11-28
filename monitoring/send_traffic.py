import argparse, json, statistics, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.train import FEATURES, load_data, time_split

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8090/invocations")
ap.add_argument("--source", choices=["train", "test"], default="train")
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--concurrency", type=int, default=1)
ap.add_argument("--shift", action="append", default=[], help="fitur=delta, boleh diulang")
ap.add_argument("--save-positive-rate", action="store_true")
a = ap.parse_args()

df = load_data()
X_train, X_test, _, _ = time_split(df)
X = (X_train if a.source == "train" else X_test)[FEATURES].dropna()
S = X.sample(n=min(a.n, len(X)), random_state=a.seed).copy()
for s in a.shift:
    k, v = s.split("=")
    S[k] += float(v)

def send(row):
    body = json.dumps({"dataframe_split": {"columns": FEATURES, "data": [[float(x) for x in row]]}}).encode()
    req = urllib.request.Request(a.url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            code, out = r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        code, out = e.code, {}
    except Exception:
        code, out = 0, {}
    return code, time.perf_counter() - t0, out.get("predictions", [None])[0]

t0 = time.perf_counter()
with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
    res = list(ex.map(send, S.values.tolist()))
dur = time.perf_counter() - t0

codes = {}
for c, _, _ in res:
    codes[c] = codes.get(c, 0) + 1
lat = sorted(l for c, l, _ in res if c == 200)
preds = [p for c, _, p in res if c == 200 and p is not None]
pos = sum(preds) / len(preds) if preds else float("nan")
q = lambda p: lat[min(len(lat) - 1, int(p * len(lat)))] * 1000 if lat else float("nan")
print(f"n={len(res)} status={codes} durasi={dur:.1f}s req/s={len(res)/dur:.1f}")
print(f"latensi klien p50={q(0.5):.1f}ms p95={q(0.95):.1f}ms")
print(f"rasio prediksi 1 = {pos:.4f} shift={a.shift or '-'} seed={a.seed} source={a.source}")

if a.save_positive_rate:
    p = ROOT / "monitoring" / "baseline.json"
    d = json.loads(p.read_text())
    d["positive_rate"] = pos
    d["positive_rate_source"] = f"trafik referensi: {a.source}, n={len(preds)}, seed={a.seed}"
    p.write_text(json.dumps(d, indent=2))
    print("positive_rate disimpan ke baseline.json")
