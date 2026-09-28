import json
import os
import time
from datetime import datetime

import requests

# 1. Daftar kelurahan Kota Malang (kode ADM4) yang akan diambil datanya.
#    Mojolangu = lokasi utama dari LK-03. Kode lain dari daftar wilayah Kemendagri

ADM4_LIST = [
    "35.73.05.1009",  # Mojolangu (utama)
    "35.73.05.1001",  # Tunggulwulung
    "35.73.05.1002",  # Merjosari
    "35.73.05.1003",  # Tlogomas
    "35.73.05.1004",  # Dinoyo
    "35.73.05.1005",  # Sumbersari
    "35.73.05.1006",  # Ketawanggede
    "35.73.05.1007",  # Jatimulyo
    "35.73.05.1008",  # Tunjungsekar
    "35.73.05.1010",  # Tulusrejo
    "35.73.05.1011",  # Lowokwaru
    "35.73.05.1012",  # Tasikmadu
    "35.73.02.1001",  # Klojen
    "35.73.02.1003",  # Samaan
    "35.73.02.1004",  # Kiduldalem
    "35.73.02.1005",  # Sukoharjo
    "35.73.02.1008",  # Oro-oro Dowo
    "35.73.02.1009",  # Bareng
    "35.73.01.1005",  # Blimbing
    "35.73.01.1008",  # Bunulrejo
    "35.73.01.1009",  # Kesatrian
    "35.73.03.1001",  # Kotalama
    "35.73.03.1003",  # Bumiayu
    "35.73.03.1005",  # Buring
    "35.73.03.1006",  # Kedungkandang
    "35.73.03.1008",  # Sawojajar
    "35.73.03.1010",  # Cemorokandang
    "35.73.04.1001",  # Ciptomulyo
    "35.73.04.1003",  # Kebonsari
    "35.73.04.1010",  # Mulyorejo
]

JUMLAH_PUTARAN = 1   # ubah jadi 3 untuk simulasi pengambilan berkala
JEDA_DETIK = 60      # jeda antar putaran (kalau JUMLAH_PUTARAN > 1)


def fetch_weather_data(adm4):
    """Ambil data cuaca 1 kelurahan. Return None kalau gagal."""
    url = f"https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4={adm4}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[{adm4}] Gagal mengambil data: {e}")
        return None


def save_raw_data(adm4, data):
    """Simpan data mentah dengan nama file berisi timestamp."""
    os.makedirs("data/raw", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"data/raw/bmkg_{adm4}_{timestamp}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"[{adm4}] Tersimpan: {filepath}")


def main():
    for putaran in range(JUMLAH_PUTARAN):
        print(f"=== Putaran {putaran + 1} dari {JUMLAH_PUTARAN} ===")
        berhasil = 0
        for adm4 in ADM4_LIST:
            data = fetch_weather_data(adm4)
            if data is not None:
                save_raw_data(adm4, data)
                berhasil += 1
            time.sleep(1.5)  # jeda supaya tidak kena batas request BMKG
        print(f"Selesai: {berhasil} dari {len(ADM4_LIST)} lokasi berhasil")
        if putaran < JUMLAH_PUTARAN - 1:
            time.sleep(JEDA_DETIK)


if __name__ == "__main__":
    main()
