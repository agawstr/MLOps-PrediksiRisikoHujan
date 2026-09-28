import glob
import json
import os

import numpy as np
import pandas as pd

NUMERIC_COLS = ["t", "hu", "ws", "tcc", "tp"]

# 1. Baca SEMUA file JSON di data/raw/ (bukan cuma satu file)
records = []
for path in sorted(glob.glob("data/raw/*.json")):
    with open(path, encoding="utf-8") as f:
        raw_data = json.load(f)

    # 2. Unpack JSON bersarang data[0]['cuaca'], tambah info lokasi
    adm4 = raw_data["lokasi"]["adm4"]
    desa = raw_data["lokasi"]["desa"]
    for item in raw_data["data"][0]["cuaca"]:
        for forecast in item:
            forecast["adm4"] = adm4
            forecast["kelurahan"] = desa
            records.append(forecast)

df = pd.DataFrame(records)
print(f"Data mentah: {len(df)} baris dari {df['adm4'].nunique()} lokasi")

# 3. Pembersihan data
df["local_datetime"] = pd.to_datetime(df["local_datetime"])
df["analysis_date"] = pd.to_datetime(df["analysis_date"])
for col in NUMERIC_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# nilai yang tidak masuk akal dianggap kosong (NaN)
df.loc[~df["hu"].between(0, 100), "hu"] = np.nan
df.loc[df["ws"] < 0, "ws"] = np.nan
df.loc[df["tp"] < 0, "tp"] = np.nan

# waktu yang sama bisa muncul di banyak file -> ambil rilis terbaru
df = df.sort_values(["adm4", "local_datetime", "analysis_date"])
df = df.drop_duplicates(subset=["adm4", "local_datetime"], keep="last")
print(f"Setelah hapus duplikat: {len(df)} baris")
print("Missing value per kolom:")
print(df[NUMERIC_COLS].isna().sum())


# 4. Fungsi resampling + feature engineering untuk SATU kelurahan
def proses_lokasi(df_lokasi):
    df_lokasi = df_lokasi.set_index("local_datetime")

    # Resampling 3 jam -> 1 jam, sekaligus isi nilai yang kosong
    df_num = df_lokasi[NUMERIC_COLS].resample("1h").interpolate(method="linear")
    df_cat = df_lokasi[["weather_desc"]].resample("1h").ffill()
    hasil = df_num.join(df_cat).reset_index()

    # Feature Engineering
    hour = hasil["local_datetime"].dt.hour
    hasil["sin_hour"] = np.sin(2 * np.pi * hour / 24)
    hasil["cos_hour"] = np.cos(2 * np.pi * hour / 24)

    for col in ["t", "hu", "ws"]:
        hasil[f"{col}_lag1"] = hasil[col].shift(1)

    hasil["t_rolling_mean_3h"] = hasil["t"].rolling(3).mean()
    hasil["hu_rolling_mean_3h"] = hasil["hu"].rolling(3).mean()

    hasil["is_rain_risk"] = (
        (hasil["tp"] > 0)
        | hasil["weather_desc"].str.contains("Hujan|Petir", case=False, na=False)
    ).astype(int)
    return hasil


# 5. Proses tiap kelurahan sendiri-sendiri, lalu gabungkan
# (supaya nilai lag/rolling tidak tercampur antar kelurahan)
hasil_semua = []
for adm4, df_lokasi in df.groupby("adm4"):
    hasil = proses_lokasi(df_lokasi)
    hasil.insert(0, "kelurahan", df_lokasi["kelurahan"].iloc[0])
    hasil.insert(0, "adm4", adm4)
    hasil_semua.append(hasil)

df_clean = pd.concat(hasil_semua, ignore_index=True)

# 6. Simpan dataset olahan ke data/processed/
os.makedirs("data/processed", exist_ok=True)
output_csv = "data/processed/weather_processed.csv"
df_clean.to_csv(output_csv, index=False)

print("Preprocessing selesai!")
print(f"Data terproses disimpan di: {output_csv}")
print(f"Total: {len(df_clean)} baris dari {df_clean['adm4'].nunique()} lokasi")
print(df_clean.head())
