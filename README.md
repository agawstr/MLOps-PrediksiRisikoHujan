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