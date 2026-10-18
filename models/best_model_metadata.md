# Model Metadata: rain-risk-classifier (v1)

Model terbaik hasil eksperimen LK-06, dipersiapkan untuk tahap deployment berikutnya.

## Identitas
| Item | Nilai |
|---|---|
| Registered model | rain-risk-classifier, Version 1 |
| Run name | rf_n100_d5_leaf3 |
| Run ID | 7a9ff9fafd7f4a00bf9c024e2e7726c8 |
| Logged model ID | m-716dd21947184f7a9933ec30e6850f99 |
| Experiment | prediksi-risiko-hujan |
| Tracking URI | sqlite:///mlflow.db |
| Tugas | Klasifikasi biner risiko hujan (`is_rain_risk`) |

## Parameter terbaik
| Parameter | Nilai |
|---|---|
| model_type | RandomForestClassifier |
| n_estimators | 100 |
| max_depth | 5 |
| min_samples_leaf | 3 |
| class_weight | balanced |
| random_state | 42 |

## Fitur input (11)
t, hu, ws, tcc, sin_hour, cos_hour, t_lag1, hu_lag1, ws_lag1, t_rolling_mean_3h, hu_rolling_mean_3h

(`tp` dan `weather_desc` tidak dipakai karena menjadi sumber label, untuk mencegah kebocoran data.)

## Performa (data uji, split berbasis waktu 70/30)
| Metrik | Nilai |
|---|---|
| test_f1 | 0.567 |
| test_recall | 0.706 |
| test_precision | 0.474 |
| test_accuracy | 0.702 |
| test_roc_auc | 0.721 |
| test_pr_auc | 0.513 |
| baseline F1 (selalu tebak hujan) | 0.433 |
| train_f1 | 0.692 (selisih dengan test_f1 = 0.125) |

## Alasan dipilih
F1 dan recall uji tertinggi di antara 4 run, dengan selisih train-test yang masih wajar.
Model yang lebih dalam (max_depth 10 dan None) overfitting (train_f1 0.906 dan 0.998, test_f1 turun).

## Versi data dan kode
| Item | Nilai |
|---|---|
| Dataset | data/processed/weather_processed.csv |
| Versi data | data-v2 (DVC md5 c010bd78c012e47ba41a93c303eefd32.dir) |
| Jumlah baris | 10.836 (train 7.585, test 3.251) |
| Git branch | feat/lk06-mlflow |
| Git commit (tercatat di run) | 9793307 |
| Git commit kode final (train.py) | ead98e4 |

## Cara reproduksi
1. `git checkout feat/lk06-mlflow`
2. `pip install -r requirements.txt`
3. `dvc pull`
4. `python src/train.py`
5. `mlflow ui --backend-store-uri sqlite:///mlflow.db`

## Catatan untuk deployment
- Metrik dihitung pada satu split waktu (±5 hari), jadi perlu validasi lebih lanjut (misalnya TimeSeriesSplit) dan pemantauan data drift.
- Threshold prediksi 0.5; dapat disetel bila ingin recall lebih tinggi.