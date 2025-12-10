# ================================================================
# predict_multi_no_sentimen.py (FINAL REFACTORED VERSION)
# ================================================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

import torch
import torch.nn as nn

from utils.kalman import apply_kalman_filter

# PREPROCESS MULTI (BENAR)
from utils.preprocess_multi_no_sentimen import (
    load_multi_csv,
    add_simple_returns_multi,
    add_indicators_multi,
    prepare_sequences_multi,
    inverse_transform_target_multi,
)

from config import TARGET_COL, H_1M, N_STEPS


# ==============================================================
# 1. MODEL LSTM (HARUS IDENTIK DENGAN TRAINING MULTI)
# ==============================================================

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, dense_units, output_dim):
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
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        return self.fc_out(out)


def build_lstm_model(num_features: int):
    """Pemilihan hidden dim mengikuti aturan training."""
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
        output_dim=H_1M
    )


# ==============================================================
# 2. Load Real Price Horizon
# ==============================================================

REAL_DATA_DIR = "../Data/Saham_Horizon_22"


def load_real_price_segment(ticker: str, horizon: int):
    path = os.path.join(REAL_DATA_DIR, f"{ticker}.csv")

    if not os.path.exists(path):
        print(f"⚠️ Real price tidak ditemukan untuk {ticker}")
        return None

    df = pd.read_csv(path)
    if "Price" not in df.columns:
        print(f"⚠️ Kolom Price tidak ditemukan pada {ticker}")
        return None

    prices = df["Price"].values[::-1]  # dibalik karena dataset dari belakang
    if len(prices) < horizon:
        print(f"⚠️ Real price {ticker} kurang panjang")
        return None

    return prices[-horizon:]


# ==============================================================
# 3. Plot hasil prediksi
# ==============================================================

def plot_multi_prediction(ticker, raw_pred, kalman_pred, real_segment, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"{ticker}_multi_pred.png")

    plt.figure(figsize=(9, 4))

    if real_segment is not None:
        plt.plot(real_segment, label="Real Price", marker="x")

    plt.plot(raw_pred, label="Raw Prediction", marker="o")
    plt.plot(kalman_pred, label="Kalman Smoothed", linestyle="--", marker="s")

    plt.title(f"{ticker} — Multi-Emiten Model Prediction ({H_1M} Hari)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"🖼️ Grafik {ticker} disimpan → {out_png}")


# ==============================================================
# 4. PREDIKSI MULTI-EMITEN (VERSI FINAL)
# ==============================================================

def predict_next_multi(
    tickers: List[str],
    data_dir: str,
    model_path: str,
    scaler_path: str,
    feature_subset: List[str],
    out_dir: str = "predictions_multi"
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔥 Prediksi Multi-Emiten (device={device})")

    # ---------------------------------------------------------
    # LOAD MODEL
    # ---------------------------------------------------------
    state_dict = torch.load(model_path, map_location=device)

    # ---------------------------------------------------------
    # LOAD SCALER MULTI
    # ---------------------------------------------------------
    scaler_dict = joblib.load(scaler_path)
    if "target_close_multi" not in scaler_dict:
        raise RuntimeError("❌ Scaler multi tidak memiliki 'target_close_multi'!")

    print("   ✅ Scaler multi loaded")

    # ---------------------------------------------------------
    # LOAD SEMUA CSV SEKALIGUS (WAJIB → SAMAKAN PREPROCESS TRAINING)
    # ---------------------------------------------------------
    csv_paths = [os.path.join(data_dir, f"{t}.csv") for t in tickers]

    df_all = load_multi_csv(csv_paths, tickers)
    df_all = add_simple_returns_multi(df_all)
    df_all = add_indicators_multi(df_all)

    # ---------------------------------------------------------
    # PREDIKSI PER-TICKER
    # ---------------------------------------------------------
    results = {}

    for ticker in tickers:
        print(f"\n📌 Prediksi untuk: {ticker}")

        df_t = df_all[df_all["Ticker"] == ticker].copy()
        if df_t.shape[0] < N_STEPS:
            print(f"⚠️ Data {ticker} terlalu pendek")
            continue

        # prepare sequence transform mode
        X_all, _, _, cols = prepare_sequences_multi(
            df_t,
            TARGET_COL,
            feature_subset,
            scaler_dict,
            horizon=H_1M
        )

        X_last = X_all[-1:]     # 1-window terakhir
        num_features = X_last.shape[-1]

        # buat model sesuai jumlah fitur
        model = build_lstm_model(num_features).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        # prediksi scaled
        X_t = torch.tensor(X_last, dtype=torch.float32).to(device)

        with torch.no_grad():
            y_scaled = model(X_t).detach().cpu().numpy().reshape(-1, 1)

        # inverse transform scaled → harga asli
        raw_pred = inverse_transform_target_multi(y_scaled, scaler_dict)

        try:
            kalman_pred = apply_kalman_filter(pd.Series(raw_pred)).values
        except:
            kalman_pred = raw_pred

        # real price
        real_segment = load_real_price_segment(ticker, H_1M)

        # simpan csv
        out_ticker_dir = os.path.join(out_dir, ticker)
        os.makedirs(out_ticker_dir, exist_ok=True)

        csv_out = os.path.join(out_ticker_dir, f"{ticker}_prediction.csv")
        pd.DataFrame({
            "Day": np.arange(1, H_1M + 1),
            "Raw": raw_pred,
            "Kalman": kalman_pred
        }).to_csv(csv_out, index=False)

        print(f"💾 Hasil CSV {ticker} → {csv_out}")

        # plot
        plot_multi_prediction(
            ticker=ticker,
            raw_pred=raw_pred,
            kalman_pred=kalman_pred,
            real_segment=real_segment,
            out_dir=out_ticker_dir
        )

        results[ticker] = (raw_pred, kalman_pred)

    print("\n🎉 Prediksi Multi-Emiten SELESAI!")
    return results
