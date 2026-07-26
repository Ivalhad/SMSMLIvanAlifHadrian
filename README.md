# 🤖 SMSML — Ivan Alif Hadrian

> **Submission Dicoding | Membangun Sistem Machine Learning**  
> Tahap 2: Membangun Model + Tahap 3: Serving, Monitoring & Logging

---

## 📋 Deskripsi

Repository ini adalah **pusat utama proyek Sistem Machine Learning** yang mencakup dua tahap kritis:
1. **Membangun Model** — Training Random Forest dengan MLflow tracking (baseline + hyperparameter tuning)
2. **Monitoring & Logging** — Serving model via MLflow, monitoring metrik dengan Prometheus, dan visualisasi dashboard di Grafana

- 🔗 **Link GitHub Eksperimen:** [github.com/Ivalhad/EksperimenSMLIvanAlifHadrian](https://github.com/Ivalhad/EksperimenSMLIvanAlifHadrian)
- 🔗 **Link GitHub Workflow CI:** [github.com/Ivalhad/Workflow-CI](https://github.com/Ivalhad/Workflow-CI)

---

## 🗂️ Struktur Folder

```
SMSML_IvanAlifHadrian/
│
├── .gitignore
├── Eksperimen_SML_Nama-siswa.txt      # Link repository eksperimen
├── Workflow-CI.txt                    # Link repository Workflow CI
│
├── Membangun_model/
│   ├── requirements.txt               # Dependencies Python
│   ├── modelling.py                   # Training baseline (RandomForest + MLflow autolog)
│   ├── modelling_tuning.py            # Training + GridSearchCV + manual MLflow logging
│   ├── screenshoot_artifak.png        # Bukti MLflow artifacts
│   ├── screenshoot_dashboard.png      # Bukti MLflow dashboard
│   ├── heart_disease_preprocessing/
│   │   ├── train.csv                  # Data training (dari Eksperimen folder)
│   │   └── test.csv                   # Data testing (dari Eksperimen folder)
│   └── mlruns/                        # MLflow experiment tracking (lokal)
│
└── Monitoring_dan_Logging/
    ├── 2.prometheus.yml               # Konfigurasi Prometheus scraping
    ├── 3.prometheus_exporter.py       # Custom exporter metrik ke Prometheus
    ├── 7.Inference.py                 # Script inferensi manual ke model endpoint
    ├── 1.bukti_serving/
    │   └── bukti_serving.png          # Screenshot MLflow model serving
    ├── 4.bukti monitoring Prometheus/
    │   ├── 1.monitoring_predictions_total.png
    │   ├── 2.monitoring_latency.png
    │   ├── 3.monitoring_endpoint_up.png
    │   ├── 4.monitoring_error_rate.png
    │   └── 5.monitoring_accuracy.png
    ├── 5.bukti monitoring Grafana/
    │   ├── 1.monitoring_predictions_total.png
    │   ├── 2.monitoring_latency.png
    │   ├── 3.monitoring_endpoint_up.png
    │   ├── 4.monitoring_error_rate.png
    │   └── 5.monitoring_accuracy.png
    └── 6.bukti alerting Grafana/
        ├── 1.rules_error_rate.png
        └── 2.notifikasi_error_rate.png
```

---

## 🧠 Bagian 1: Membangun Model (`Membangun_model/`)

### Dependencies (`requirements.txt`)

```
mlflow==2.19.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.26.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

### A. Baseline Model (`modelling.py`)

Training pertama menggunakan **RandomForestClassifier** dengan `MLflow autolog` untuk mencatat semua parameter dan metrik secara otomatis.

| Konfigurasi | Nilai |
|---|---|
| **Algorithm** | Random Forest Classifier |
| **n_estimators** | 100 |
| **random_state** | 42 |
| **MLflow Experiment** | `heart-disease-baseline` |
| **MLflow Run Name** | `baseline-random-forest` |
| **Logging Mode** | `mlflow.sklearn.autolog()` |

**Metrik yang dicatat:** Accuracy, F1 Score

### B. Model dengan Hyperparameter Tuning (`modelling_tuning.py`)

Training lanjutan menggunakan **GridSearchCV** dengan manual MLflow logging dan berbagai artifact visual.

| Konfigurasi | Nilai |
|---|---|
| **Algorithm** | Random Forest + GridSearchCV |
| **cv** | 5-fold cross-validation |
| **scoring** | F1 Score |
| **MLflow Experiment** | `heart-disease-tuning` |
| **MLflow Run Name** | `rf-gridsearch-tuning` |
| **Logging Mode** | Manual `mlflow.log_*` |

**Grid Hyperparameter yang Diuji:**

| Parameter | Nilai yang Diuji |
|---|---|
| `n_estimators` | 100, 200, 300 |
| `max_depth` | None, 10, 20 |
| `min_samples_split` | 2, 5 |

**Metrik yang dicatat:** Accuracy, Precision, Recall, F1 Score, ROC-AUC, Best CV Score

**Artifact yang di-log ke MLflow:**
- 📊 `confusion_matrix.png` — Heatmap confusion matrix
- 📈 `roc_curve.png` — Kurva ROC dengan nilai AUC
- 📄 `classification_report.json` — Laporan klasifikasi lengkap
- 🤖 `model/` — Model artifact (sklearn format)

### Menjalankan Training

```bash
cd Membangun_model

# Install dependencies
pip install -r requirements.txt

# Jalankan baseline
python modelling.py

# Jalankan tuning
python modelling_tuning.py

# Buka MLflow UI
mlflow ui
# Akses di: http://127.0.0.1:5000
```

---

## 📡 Bagian 2: Monitoring & Logging (`Monitoring_dan_Logging/`)

### Arsitektur Monitoring

```
MLflow Model Serving          Prometheus Exporter           Prometheus          Grafana
  (port 8080)          ←──    (port 8001)             ←──  (scrape /metrics)   (dashboard)
  /invocations               3.prometheus_exporter.py        2.prometheus.yml    alerting
```

### A. Konfigurasi Prometheus (`2.prometheus.yml`)

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'ml-model-monitoring'
    static_configs:
      - targets: ['host.docker.internal:8001']
    metrics_path: /metrics
```

Prometheus scrape metrik dari exporter setiap **15 detik** melalui endpoint `http://host.docker.internal:8001/metrics`.

### B. Prometheus Exporter (`3.prometheus_exporter.py`)

Custom exporter yang mengirimkan request ke model endpoint secara berkala dan melaporkan metrik ke Prometheus.

**Metrik yang Diekspor:**

| Metrik | Tipe | Deskripsi |
|---|---|---|
| `model_predictions_total` | Counter | Total prediksi yang dilakukan |
| `model_prediction_latency_seconds` | Histogram | Latency request dalam detik |
| `model_endpoint_up` | Gauge | Status endpoint (1=up, 0=down) |
| `model_prediction_error_rate` | Gauge | Persentase error dari total request |
| `model_accuracy_score` | Gauge | Estimasi akurasi model real-time |

**Cara kerja:**
1. Baca data dari `test.csv` secara acak setiap 5 detik
2. Kirim request POST ke `http://127.0.0.1:8080/invocations`
3. Hitung latency, akurasi kumulatif, dan error rate
4. Update semua gauge/counter Prometheus
5. Expose metrik di `http://localhost:8001/metrics`

### C. Script Inferensi (`7.Inference.py`)

Script sederhana untuk menguji model endpoint secara manual dengan mengirim 5 sample dari `test.csv`.

```bash
# Jalankan inference
python 7.Inference.py

# Output:
# Status Code : 200
# Predictions : {"predictions": [1, 0, 1, 1, 0]}
```

### Menjalankan Stack Monitoring

**1. Serve model MLflow:**
```bash
mlflow models serve -m "models:/heart-disease-model/1" -p 8080 --no-conda
```

**2. Jalankan Prometheus Exporter:**
```bash
pip install prometheus-client requests pandas
python Monitoring_dan_Logging/3.prometheus_exporter.py
```

**3. Jalankan Prometheus (Docker):**
```bash
docker run -d -p 9090:9090 \
  -v $(pwd)/Monitoring_dan_Logging/2.prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus
```

**4. Jalankan Grafana (Docker):**
```bash
docker run -d -p 3000:3000 grafana/grafana
# Login: admin/admin → http://localhost:3000
```

---

## 📸 Bukti Monitoring

### Prometheus Monitoring (5 Panel)
| Panel | Screenshot |
|---|---|
| Total Predictions | `4.bukti monitoring Prometheus/1.monitoring_predictions_total.png` |
| Latency | `4.bukti monitoring Prometheus/2.monitoring_latency.png` |
| Endpoint Status | `4.bukti monitoring Prometheus/3.monitoring_endpoint_up.png` |
| Error Rate | `4.bukti monitoring Prometheus/4.monitoring_error_rate.png` |
| Accuracy | `4.bukti monitoring Prometheus/5.monitoring_accuracy.png` |

### Grafana Dashboard (5 Panel)
| Panel | Screenshot |
|---|---|
| Total Predictions | `5.bukti monitoring Grafana/1.monitoring_predictions_total.png` |
| Latency | `5.bukti monitoring Grafana/2.monitoring_latency.png` |
| Endpoint Status | `5.bukti monitoring Grafana/3.monitoring_endpoint_up.png` |
| Error Rate | `5.bukti monitoring Grafana/4.monitoring_error_rate.png` |
| Accuracy | `5.bukti monitoring Grafana/5.monitoring_accuracy.png` |

### Grafana Alerting
| Alert | Screenshot |
|---|---|
| Alert Rules Error Rate | `6.bukti alerting Grafana/1.rules_error_rate.png` |
| Notifikasi Error Rate | `6.bukti alerting Grafana/2.notifikasi_error_rate.png` |

---

## 🔗 Keterkaitan dengan Folder Lain

```
Eksperimen_SML_IvanAlifHadrian   ──→   SMSML_IvanAlifHadrian (ini)   ──→   Workflow-CI
     (EDA & Preprocessing)               (Modelling & Serving)               (CI Otomatis)

train.csv / test.csv  ──────────────────→ Input untuk modelling.py & exporter
Model (MLflow runs) ─────────────────────────────────────────────→ Dipaket ulang di Workflow-CI
```

---

## 👤 Author

| Info | Detail |
|---|---|
| **Nama** | Ivan Alif Hadrian |
| **Program** | Dicoding — Membangun Sistem Machine Learning |
| **Tahap** | 2 — Membangun Model & 3 — Monitoring & Logging |
