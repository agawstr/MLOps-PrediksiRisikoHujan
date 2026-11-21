# Sistem Prediksi Risiko Hujan Berbasis Machine Learning untuk Mendukung Pengambilan Keputusan Event Outdoor di Kota Malang

## Deskripsi Proyek

Proyek ini merupakan perancangan sistem MLOps untuk memprediksi **risiko hujan** (klasifikasi biner) menggunakan data cuaca yang bersifat dinamis dari **BMKG Web API**, guna mendukung pengambilan keputusan event organizer saat mempersiapkan kegiatan outdoor di Kota Malang.

Sistem menggunakan pendekatan **Continual Learning** melalui proses monitoring data dan performa model secara berkala. Apabila terdeteksi data drift atau penurunan performa model, proses retraining akan dilakukan agar model dapat beradaptasi terhadap perubahan pola cuaca.

## Tujuan

- Mengambil data cuaca secara dinamis melalui BMKG Web API
- Melakukan preprocessing dan feature engineering pada data cuaca (temperatur, kelembapan, tutupan awan, dll)
- Membangun model klasifikasi biner untuk memprediksi risiko hujan
- Melakukan eksperimen dan pencatatan model menggunakan MLflow
- Menerapkan monitoring terhadap data dan performa model
- Menerapkan mekanisme retraining ketika terjadi data drift atau penurunan performa

## Model Machine Learning

Model baseline yang digunakan dalam proyek ini adalah **Random Forest Classifier**, digunakan untuk memprediksi risiko hujan (berisiko / tidak berisiko) pada periode berikutnya berdasarkan fitur dari data cuaca historis dan hasil feature engineering.

Metrik evaluasi yang digunakan meliputi:

- Accuracy
- Precision
- **Recall** (prioritas utama — gagal deteksi hujan lebih berisiko dibanding false alarm)
- F1-Score
- API Latency

## Struktur Direktori

```
MLOps-PrediksiRisikoHujan/
├── .devcontainer/
│   └── devcontainer.json
├── config/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── models/
├── notebooks/
├── src/
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Penjelasan Direktori

| Direktori/File | Fungsi |
|---|---|
| `.devcontainer/` | Konfigurasi environment GitHub Codespaces |
| `config/` | Menyimpan konfigurasi proyek |
| `data/raw/` | Data mentah hasil fetch dari BMKG API |
| `data/processed/` | Data yang sudah dibersihkan & siap untuk training |
| `data/external/` | Data pendukung dari sumber luar |
| `models/` | Menyimpan model machine learning hasil training |
| `notebooks/` | Notebook untuk eksplorasi (EDA) dan eksperimen |
| `src/` | Source code utama |
| `requirements.txt` | Daftar dependency Python yang digunakan |
| `.gitignore` | Menentukan file yang tidak perlu disimpan dalam Git |
| `LICENSE` | Lisensi proyek (MIT) |
| `README.md` | Dokumentasi proyek |

## Environment Development

Proyek dikembangkan menggunakan GitHub Codespaces dengan environment Python yang telah dikonfigurasi melalui Dev Container.

**Versi Python:** 3.11

**Dependency utama:**
- pandas
- numpy
- scikit-learn
- requests
- matplotlib
- seaborn
- mlflow
- dvc

## Cara Menjalankan di GitHub Codespaces

1. Buka repository ini di GitHub
2. Pilih **Code** → **Codespaces**
3. Buat atau buka Codespace pada branch `main`
4. Environment akan otomatis mengikuti konfigurasi pada `.devcontainer/devcontainer.json`
5. Install dependency dengan perintah:
   ```bash
   python -m pip install -r requirements.txt
   ```

## Branching Strategy (GitHub Flow)

- `main` selalu stabil
- Eksperimen/fitur baru dikerjakan di branch terpisah, contoh: `feat/initial-eda`
- Setelah tervalidasi, buka Pull Request ke `main`, review, lalu merge

```bash
git checkout -b feat/initial-eda
# ...kerja & commit...
git push -u origin feat/initial-eda
# buka Pull Request di GitHub, lalu merge ke main
```

## LK-04: Data Ingestion Dinamis & Otomasi Prapemrosesan

### Cara Menjalankan
```bash
python -m pip install -r requirements.txt

# 1. Ambil data terbaru dari BMKG API
python src/data/ingest_data.py

# 2. Bersihkan semua data mentah
python src/data/preprocess.py
```

### Penjelasan Skrip
| File | Fungsi |
|---|---|
| `src/data/ingest_data.py` | Mengambil prakiraan cuaca tiap kelurahan (kode ADM4) dan menyimpannya ke `data/raw/bmkg_<adm4>_<timestamp>.json`. Data lama tidak ditimpa. Jika koneksi gagal, skrip mencetak pesan dan lanjut ke lokasi berikutnya. |
| `src/data/preprocess.py` | Membaca semua file di `data/raw/`, membersihkan (tipe data, nilai tidak valid, duplikat), resampling 3 jam ke 1 jam, feature engineering, dan menyimpan `data/processed/weather_processed.csv`. |

### Pengambilan Berkala
- Simulasi lokal: ubah `JUMLAH_PUTARAN` dan `JEDA_DETIK` di `ingest_data.py`.
- Otomatis: workflow `.github/workflows/ingest.yml` menjalankan kedua skrip setiap hari (cron `0 0 * * *`). Sejak LK-05, data tidak lagi di-commit ke Git, melainkan di-versioning dengan DVC (lihat section LK-05).

### Catatan
Lokasi utama tetap Kel. Mojolangu (`35.73.05.1009`, LK-03). Lokasi lain
ditambahkan untuk memenuhi jumlah data (>= 500 baris).

## LK-05: Data Versioning dengan DVC

### Mengapa data perlu di-versioning?
Model ML dibentuk dari kode, parameter, dan data. Git hanya cocok untuk
file teks kecil dan tidak dirancang untuk file besar yang terus berubah,
sedangkan data cuaca BMKG di proyek ini bertambah setiap ingestion
dijalankan. Tanpa versioning, kita tidak tahu data mana yang dipakai untuk
melatih suatu model, sehingga hasil training tidak bisa direproduksi,
bug pada prediksi historis sulit ditelusuri, dan perubahan data tidak bisa
diaudit. Dengan DVC, setiap commit Git menunjuk ke versi dataset tertentu
lewat file pointer `data/raw.dvc` (berisi hash md5), sehingga eksperimen
dapat diulang dengan data yang sama persis, performa model dapat
dibandingkan antar versi data, dan kita bisa rollback bila data baru
menurunkan performa. Repositori Git tetap ringan karena data asli
disimpan di remote storage.

### Data lineage (silsilah data)
BMKG Web API → `src/data/ingest_data.py` → `data/raw/*.json`
→ `src/data/preprocess.py` → `data/processed/weather_processed.csv`

Setiap versi data terikat ke satu commit Git dan tag (`data-v1`,
`data-v2`), sehingga asal-usul dan perubahan data dapat ditelusuri.

### Alur penambahan versi data
1. Ambil data baru dan bersihkan:
   `python src/data/ingest_data.py && python src/data/preprocess.py`
2. Catat versi baru: `dvc add data/raw data/processed`
3. Commit pointer: `git add data/raw.dvc data/processed.dvc data/.gitignore`
   lalu `git commit -m "data: <keterangan>"` dan `git tag data-vN`
4. Kirim data ke remote: `dvc push`

### Membandingkan dan memulihkan versi
- Bandingkan versi: `dvc diff data-v1 data-v2`
- Riwayat versi: `git log --oneline --decorate -- data/raw.dvc`
- Pulihkan versi: `git checkout data-v1 && dvc pull`
- Ambil data di mesin lain: `git clone <repo> && dvc pull`

### Ringkasan versi
| Versi | Tag | Commit | File raw | Ukuran | md5 (8 awal) |
|---|---|---|---|---|---|
| v1 | data-v1 | c5abe44 | 480 | 8,9 MB | 5b2e2e42 |
| v2 | data-v2 | ee3e615 | 510 | 9,4 MB | f6a64621 |

Transisi v1 ke v2: 30 file raw ditambahkan dan `weather_processed.csv`
berubah (`dvc diff data-v1 data-v2`).

### Catatan
- Remote storage memakai folder lokal (`../dvc-storage`) untuk demo. Untuk produksi dapat diganti S3/GCS/MinIO dengan `dvc remote modify`.
- Workflow `ingest.yml` tidak lagi meng-commit data ke Git; versioning data dilakukan dengan DVC.
- Karena remote berupa folder lokal, perintah `dvc pull` dari mesin lain baru bisa dipakai setelah remote diganti ke storage yang dapat diakses bersama (S3/GCS/MinIO).

## LK-06: Eksperimen dan Pelacakan Metrik dengan MLflow

- `python src/train.py` melatih RandomForest dengan 5 konfigurasi hyperparameter (4 dari LK-06, 1 ditambahkan pada LK-08, split berbasis waktu 70/30, `class_weight=balanced`) dan mencatat parameter, metrik train/test, feature importance, serta model ke MLflow (`sqlite:///mlflow.db`, experiment `prediksi-risiko-hujan`).
- Setiap run mencatat `git_commit` dan `data_dvc_md5` agar hasilnya dapat direproduksi.
- Lihat hasil: `mlflow ui --backend-store-uri sqlite:///mlflow.db`
- `python src/train.py --register` mendaftarkan run dengan test_f1 tertinggi ke Model Registry sebagai `rain-risk-classifier`.

## LK-07: Model Registry, Versioning, dan Kesiapan Inferensi

### Model aktif untuk inferensi
- Model: `rain-risk-classifier` **Version 1** (alias `production`), run `rf_n100_d5_leaf3`
- Alasan: test_f1 tertinggi (0.567 vs 0.536), recall tertinggi (0.706 vs 0.617, selaras prioritas recall proyek), dan gap train-test terkecil (0.125 vs 0.213, v2 lebih overfit)
- Data: `data/processed` (DVC md5 `c010bd78c012e47ba41a93c303eefd32.dir`), commit kode `9793307`
- Muat model: `mlflow.pyfunc.load_model("models:/rain-risk-classifier@production")`
- Metadata: `models/production.dvc`, `models/production_meta.yaml`

### Perbandingan versi
| Versi | n_estimators / max_depth / min_samples_leaf | test_f1 | recall | precision | PR-AUC | gap F1 | Status |
|---|---|---|---|---|---|---|---|
| v1 | 100 / 5 / 3 | 0.567 | 0.706 | 0.474 | 0.513 | 0.125 | Production |
| v2 | 150 / 6 / 3 | 0.536 | 0.617 | 0.474 | 0.506 | 0.213 | None |

Transisi status v1: None -> Staging -> Production (stage API MLflow 3.16.1 yang berstatus deprecated, ditambah alias `staging`/`production` dan tag `stage`).

### Alur menjalankan
1. `python src/train.py --register` : melatih 4 run dan mendaftarkan terbaik sebagai v1
2. `python src/registry_describe.py 1 "<catatan>"` : deskripsi dan tag metadata
3. `python src/train_v2.py` lalu `python src/registry_describe.py 2 "<catatan>"` : kandidat v2
4. `python src/promote_model.py compare`, lalu `staging`, lalu `production`
5. `python src/export_production.py` lalu `dvc add models/production` : metadata lineage ke DVC
6. `python src/inference_test.py` : uji pemuatan model dari registry dan prediksi data baru

### Catatan
- Data pelatihan: snapshot data-v2 (10.836 baris setelah dropna, 30 kelurahan). Pembaruan data = ingest ulang -> preprocess -> `dvc add` -> retrain (versi model baru).
- Registry memakai SQLite lokal (`mlflow.db`, tidak di-commit); salinan model tersedia di `models/production` via DVC.

## LK-08: Otomatisasi End-to-End Pipeline (CI/CD/CT)

Workflow `.github/workflows/mlops-automation.yaml` berjalan otomatis pada **push ke `main`** dan **pull request ke `main`**:

| Job | Tahap | Isi |
|---|---|---|
| `testing` | 1. Automated Testing | `pytest` (data sintetis) pada `tests/` |
| `training` | 2. Automated Training | `dvc pull data/processed.dvc` dari remote DagsHub, lalu `python src/train.py` (5 konfigurasi, dicatat ke MLflow) |
| `registry` | 3-4. Validasi dan Registry | `src/evaluate_and_promote.py`: model terbaik dibandingkan dengan ambang di `config/thresholds.yaml`; hanya jika lolos dan pada push ke `main`, didaftarkan ke MLflow Model Registry berstatus **Staging** |

**Ambang** (`config/thresholds.yaml`, ditetapkan dari baseline LK-06 karena LK-01 menunda angka sampai baseline ada): `test_recall >= 0.60`, `test_f1 >= 0.50`, `gap F1 (train-test) <= 0.20`, dan tidak boleh turun lebih dari 0.01 F1 dari model Production (bila ada).

**Secrets** (GitHub Settings, tanpa nilai di kode): `DAGSHUB_USER`, `DAGSHUB_TOKEN`, `MLFLOW_TRACKING_URI`.

**Layanan eksternal:** remote DVC dan MLflow tracking/registry memakai DagsHub, karena runner GitHub tidak dapat mengakses storage lokal Codespace dan registry di dalam runner hilang setelah job selesai.

**Bukti:** run yang lolos sampai Staging (push ke `main`) dan run yang sengaja gagal pada PR dengan ambang F1 0.90 (job `registry` menolak model).

## LK-09: Orkestrasi dengan Docker Compose

Sistem terdiri dari dua layanan dalam satu custom bridge network (`ml-net`):

| Layanan | Fungsi | Port |
|---|---|---|
| `mlflow-server` | Tracking server + Model Registry (SQLite di volume `mlflow-data`) | 5000 |
| `api-service` | API inferensi FastAPI, memuat model `rain-risk-classifier@staging` | 8000 |

`api-service` menunggu `mlflow-server` berstatus *healthy* (`depends_on` dengan `condition: service_healthy`) dan mengakses model lewat nama layanan: `http://mlflow-server:5000`. Data registry disimpan di volume `mlflow-data`, sehingga tidak hilang saat kontainer dimatikan.

### Prasyarat

- Docker dan Docker Compose v2
- Python 3.11
- Dependensi training: `pip install -r requirements.txt`
- Data terbaru: `dvc pull`

### Menjalankan sistem (pertama kali)

Registry lokal masih kosong pada percobaan pertama, jadi model perlu dilatih dan didaftarkan sekali. Setelahnya data tersimpan di volume.

```bash
# 1. Jalankan MLflow server
docker compose up -d mlflow-server

# 2. Latih model ke registry lokal
export MLFLOW_TRACKING_URI=http://localhost:5000
python src/train.py

# 3. Beri alias "staging" pada versi terbaru rain-risk-classifier
python - <<'EOF'
from mlflow.tracking import MlflowClient
c = MlflowClient()
v = max(c.search_model_versions("name='rain-risk-classifier'"),
        key=lambda x: int(x.version))
c.set_registered_model_alias("rain-risk-classifier", "staging", v.version)
print("alias staging -> versi", v.version)
EOF

# 4. Jalankan seluruh sistem
docker compose up -d --build
```

### Menjalankan sistem (berikutnya)

Setelah model terdaftar, seluruh sistem cukup dijalankan dengan satu perintah:

```bash
docker compose up -d --build
```

### Verifikasi

```bash
docker compose ps
curl -s localhost:8000/health; echo
```

`/health` harus menampilkan `"model_loaded": true`.

Buat contoh input dari data latih, lalu uji prediksi:

```bash
MLFLOW_TRACKING_URI=http://localhost:5000 python -c "
import json, mlflow, pandas as pd
m = mlflow.pyfunc.load_model('models:/rain-risk-classifier@staging')
cols = m.metadata.get_input_schema().input_names()
df = pd.read_csv('data/processed/weather_processed.csv')
row = df[cols].dropna().iloc[0].to_dict()
json.dump(row, open('/tmp/sample.json', 'w'), default=float)
"

curl -s -X POST localhost:8000/predict \
  -H "Content-Type: application/json" -d @/tmp/sample.json; echo
```

Hasil: `{"prediction":[0]}` (atau `[1]`).

Uji komunikasi antar kontainer (service discovery):

```bash
docker compose exec api-service python -c "import urllib.request; print(urllib.request.urlopen('http://mlflow-server:5000/health').read())"
```

Hasil: `b'OK'`.

### Menghentikan

```bash
docker compose down        # data tetap ada di volume
docker compose down -v     # menghapus volume (registry ikut hilang)
```

### Catatan

- Batas memori: `mlflow-server` 2 GB, `api-service` 1 GB. MLflow dijalankan dengan `--workers 1` agar hemat memori.
- Log API dapat memuat satu peringatan MLflow tentang `pytest`. Paket itu tercatat sebagai dependensi lingkungan training tetapi tidak dipasang di image API. Peringatan ini tidak memengaruhi inferensi.
- Jika API memberi `model_loaded: false`, pastikan alias `staging` sudah dibuat (langkah 3), lalu jalankan `docker compose restart api-service`.