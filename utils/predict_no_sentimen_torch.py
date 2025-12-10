# utils/predict_no_sentimen_torch.py

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from typing import List, Dict, Tuple
from utils.preprocess_no_sentimen import add_indicators, prepare_sequences
from utils.kalman import apply_kalman_filter
from config import TARGET_COL, H_1M


# ======================================================================
# 1️⃣ KONFIGURASI FOLDER DATA REAL / PEMBANDING
# ======================================================================

REAL_DATA_DIR = "../Data/Saham_Horizon_22"


# ======================================================================
# 2️⃣ MODEL LSTM
# ======================================================================

class LSTMRegressor(nn.Module):
    """
    Arsitektur LSTM harus sama dengan saat training.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dense_units: int, output_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_dim, dense_units)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(dense_units, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]      # ambil step terakhir
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        return self.fc_out(out)


def build_lstm_model_for_predict(num_features: int) -> nn.Module:
    """
    Membuat model sesuai jumlah fitur.
    """
    if num_features < 10:
        hidden_dim = 64
        dense_units = 512
    else:
        hidden_dim = 128
        dense_units = 64

    return LSTMRegressor(
        input_dim=num_features,
        hidden_dim=hidden_dim,
        dense_units=dense_units,
        output_dim=H_1M,   # horizon prediksi
    )


# ======================================================================
# 3️⃣ SCALER UTILITIES
# ======================================================================

def resolve_scaler_path(model_path: str) -> str:
    base = os.path.basename(model_path)

    if base.endswith("_best_price.pt"):
        scaler_file = base.replace("_best_price.pt", "_scaler.pkl")
    else:
        scaler_file = base.replace(".pt", "_scaler.pkl")

    return os.path.join(os.path.dirname(model_path), scaler_file)


def inverse_transform_target(y_scaled: np.ndarray, scaler_dict: Dict[str, object]):
    """
    Mengubah hasil prediksi kembali ke Rupiah.
    """
    if "target_close" not in scaler_dict:
        raise RuntimeError("Scaler 'target_close' tidak ditemukan.")

    scaler = scaler_dict["target_close"]

    if y_scaled.ndim == 1:
        y_scaled = y_scaled.reshape(-1, 1)

    return scaler.inverse_transform(y_scaled).flatten()


# ======================================================================
# 4️⃣ LOAD MODEL & SEQUENCE
# ======================================================================

def prepare_last_sequence(df, feature_subset, scaler_dict):
    """
    Mengambil sequence terakhir (X saja) untuk prediksi.
    Mengikuti signature asli prepare_sequences().
    """
    X_all, _, scaler_dict, cols = prepare_sequences(
        df,
        TARGET_COL,      # parameter ke-2 = target_col_name
        H_1M,            # parameter ke-3 = horizon
        feature_subset,  # parameter ke-4
        scaler_dict      # parameter ke-5
        # step_size diabaikan (default=1)
    )

    if X_all.shape[0] == 0:
        raise ValueError("Data terlalu pendek!")

    return X_all[-1:], scaler_dict



def load_model(model_path: str, num_features: int, device):
    model = build_lstm_model_for_predict(num_features).to(device)

    state = torch.load(model_path, map_location=device)

    # handle berbagai bentuk state_dict
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state)
    model.eval()
    return model


# ======================================================================
# 5️⃣ LOAD REAL PRICE (untuk plot pembanding)
# ======================================================================

def load_real_price_segment(ticker: str, horizon: int):
    """
    Baca data real:
    - ambil kolom Price
    - urutannya dibalik
    - ambil horizon terakhir
    """
    real_path = os.path.join(REAL_DATA_DIR, f"{ticker}.csv")

    if not os.path.exists(real_path):
        print("⚠️ Real price tidak ditemukan → plot tetap jalan.")
        return None

    df_real = pd.read_csv(real_path)

    if "Price" not in df_real.columns:
        print("⚠️ Kolom Price tidak ada → real price dilewati.")
        return None

    prices = df_real["Price"].values[::-1]   # balik urutan

    if len(prices) < horizon:
        print("⚠️ Real data lebih pendek dari horizon → dilewati.")
        return None

    return prices[-horizon:]


# ======================================================================
# 6️⃣ PLOTTING OUTPUT
# ======================================================================

def plot_predictions(ticker, raw_pred, kalman_pred, real_segment, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"{ticker}_prediction.png")

    plt.figure(figsize=(9, 4))

    # real price
    if real_segment is not None:
        plt.plot(real_segment, label="Real Price", marker="x", linewidth=1.2)

    plt.plot(raw_pred, label="Raw Prediction", marker="o")
    plt.plot(kalman_pred, label="Kalman Smoothed", linestyle="--", marker="s")

    plt.title(f"{ticker} — Prediksi {H_1M} Hari")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"🖼️ Grafik disimpan → {out_png}")


# ======================================================================
# 7️⃣ FUNGSI UTAMA: predict_next()
# ======================================================================

def predict_next(prices_path: str, model_path: str, feature_subset: List[str], out_dir: str = "predictions"):
    """
    Dipanggil oleh main.py → DO NOT CHANGE THE API.
    """

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ticker = os.path.splitext(os.path.basename(prices_path))[0]

    print(f"\n📈 Prediksi: {ticker} (device={device})")

    # 1. Load harga + indikator
    df = pd.read_csv(prices_path)
    df = add_indicators(df)

    # 2. Load scaler
    scaler_path = resolve_scaler_path(model_path)
    scaler_dict = joblib.load(scaler_path)

    # 3. Siapkan sequence
    X_last, scaler_dict = prepare_last_sequence(df, feature_subset, scaler_dict)
    num_features = X_last.shape[-1]

    # 4. Load model
    model = load_model(model_path, num_features, device)

    # 5. Predict scaled
    X_t = torch.tensor(X_last, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_scaled = model(X_t)
        y_scaled = y_scaled.detach().cpu().numpy().reshape(-1, 1)

    # 6. Inverse → Rupiah
    raw_pred = inverse_transform_target(y_scaled, scaler_dict)

    # 7. Kalman
    try:
        kalman_pred = apply_kalman_filter(pd.Series(raw_pred)).values
    except:
        kalman_pred = raw_pred

    # 8. Load real price (opsional)
    real_segment = load_real_price_segment(ticker, H_1M)

    # 9. Save CSV
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{ticker}_prediction.csv")
    pd.DataFrame({
        "Day": np.arange(1, H_1M + 1),
        "Raw": raw_pred,
        "Kalman": kalman_pred
    }).to_csv(csv_path, index=False)

    print(f"💾 CSV disimpan → {csv_path}")

    # 10. Plot output
    plot_predictions(ticker, raw_pred, kalman_pred, real_segment, out_dir)

    print("✨ Prediksi selesai!\n")
    return raw_pred, kalman_pred
