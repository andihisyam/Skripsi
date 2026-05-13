"""
config.py
=========
Pusat konfigurasi untuk MPC Portfolio Optimization.
Semua parameter yang perlu di-tune untuk eksperimen skripsi ada di sini.
Ubah nilai di file ini — JANGAN hardcode di file lain.
"""

import os

# ============================================================
# PATH & DIREKTORI
# ============================================================
PREDICTION_BASE_DIR = r"D:\Kuliah\Skripsi\Code\predictions_forward"
ACTUAL_BASE_DIR     = r"D:\Kuliah\Skripsi\Data"
HISTORICAL_BASE_DIR = r"D:\Kuliah\Skripsi\Data\Saham_Horizon_22"
OUTPUT_DIR          = "mpc_results"
TRAINING_SUMMARY_XLSX = r"D:\Kuliah\Skripsi\Code\FIX_MODELS\ALL_training_summary.xlsx"

CONFIDENCE_CSV = os.path.join(PREDICTION_BASE_DIR, "all_predictions.csv")


# ============================================================
# PARAMETER PORTOFOLIO
# ============================================================

# Modal awal investasi (Rp)
TOTAL_CAPITAL = 100_000_000

# Horizon prediksi yang dijalankan (hari)
# Contoh eksperimen: [5, 10, 22] atau [22] saja
HORIZONS = [22]

# Jumlah saham yang dipilih dari seluruh prediksi (top-N)
# Ubah untuk eksperimen: 3, 5, 10, None (semua)
N_SELECTED_ASSETS = 5


# ============================================================
# PARAMETER MPC OPTIMIZER
# ============================================================
# Tuning tip untuk skripsi:
#   - Q_SCALE tinggi  → optimizer lebih agresif kejar return
#   - R_SCALE tinggi  → optimizer lebih konservatif (penalti perubahan bobot)
#   - Rasio Q/R yang umum: 5:1 (agresif), 1:1 (netral), 1:5 (konservatif)

# Bobot matriks state-cost (Q) — penalti deviasi dari target return
Q_SCALE = 0.5

# Bobot matriks control-cost (R) — penalti perubahan bobot portofolio
R_SCALE = 0.1

# Batas bawah bobot per saham (0.0 = boleh tidak diinvestasikan)
MIN_WEIGHT = 0.0

# Batas atas bobot per saham (0.20 = maks 20% per saham)
# Catatan: batas ini bersifat adaptif terhadap confidence score
# max_w_i = MAX_WEIGHT * (0.5 + 0.5 * confidence_i)
MAX_WEIGHT = 0.20

# Skalar lambda untuk penalti return dalam objective function
# Nilai lebih tinggi → optimizer lebih fokus pada expected return
LAMBDA_RETURN = 1.0


# ============================================================
# PARAMETER CONFIDENCE SCORE
# ============================================================

# Aktifkan/nonaktifkan penggunaan confidence score
USE_CONFIDENCE = True

# Saham dengan confidence < nilai ini akan di-skip
# Ubah untuk eksperimen: 0.0 (semua masuk), 0.3, 0.5, 0.7
MIN_CONFIDENCE = 0.0

# Pangkat confidence dalam penalty: adj_return = raw_return * (conf ^ power)
# power = 1.0 → penalti linear
# power > 1.0 → saham confidence rendah dihukum lebih berat
# power < 1.0 → penalti lebih lunak
CONFIDENCE_POWER = 1.0


# ============================================================
# PARAMETER RISK ANALYSIS
# ============================================================

# Jumlah hari historis untuk menghitung volatility
# Rekomendasi: 22 (1 bulan), 60 (3 bulan), 252 (1 tahun)
LOOKBACK_DAYS = 60

# Confidence level untuk Value-at-Risk
# 0.95 → VaR 95%, 0.99 → VaR 99%
CONFIDENCE_LEVEL = 0.95

# Risk-free rate harian (untuk Sharpe Ratio)
# Default: 0.0001 ≈ 2.5% per tahun
RISK_FREE_RATE = 0.0001
