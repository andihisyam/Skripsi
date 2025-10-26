# utils/train_no_sentimen.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path
from typing import Tuple
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.metrics import mean_absolute_percentage_error, r2_score, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


from utils.io_utils import load_prices_from_folder
from utils.preprocess_no_sentimen import add_simple_returns, add_indicators, prepare_sequences
from utils.summarize import summarize_training_results

from config import DATE_COL, TARGET_COL, N_STEPS, H_1M, TRAIN_END, VAL_END

USE_RETURN_AS_TARGET = False   # True -> target "Return"; False -> "Close"

# Semua kombinasi fitur (selalu termasuk 'Close')
# ======================================================
# Semua kombinasi fitur (versi dengan Close_lag1-3)
# ======================================================
FEATURE_COMBINATIONS = {
    # Single (selalu + lag dari Close)
    "Open_LAG": ["Open", "Close_lag1", "Close_lag2", "Close_lag3"],
    "High_LAG": ["High", "Close_lag1", "Close_lag2", "Close_lag3"],
    "Low_LAG": ["Low", "Close_lag1", "Close_lag2", "Close_lag3"],
    "Volume_LAG": ["Volume", "Close_lag1", "Close_lag2", "Close_lag3"],
    "Close_LAG": ["Close_lag1", "Close_lag2", "Close_lag3"],

    # Pairwise
    "Open_High_LAG": ["Open", "High", "Close_lag1", "Close_lag2", "Close_lag3"],
    "Open_Low_LAG": ["Open", "Low", "Close_lag1", "Close_lag2", "Close_lag3"],
    "Open_Volume_LAG": ["Open", "Volume", "Close_lag1", "Close_lag2", "Close_lag3"],
    "High_Low_LAG": ["High", "Low", "Close_lag1", "Close_lag2", "Close_lag3"],
    "High_Volume_LAG": ["High", "Volume", "Close_lag1", "Close_lag2", "Close_lag3"],
    "Low_Volume_LAG": ["Low", "Volume", "Close_lag1", "Close_lag2", "Close_lag3"],

    # Triplet
    "Open_High_Low_LAG": ["Open", "High", "Low", "Close_lag1", "Close_lag2", "Close_lag3"],
    "Open_High_Volume_LAG": ["Open", "High", "Volume", "Close_lag1", "Close_lag2", "Close_lag3"],
    "High_Low_Volume_LAG": ["High", "Low", "Volume", "Close_lag1", "Close_lag2", "Close_lag3"],

    # Full Base
    "OHLVC_LAG": ["Open", "High", "Low", "Volume", "Close_lag1", "Close_lag2", "Close_lag3"],

    # Full Base + MA
    "OHLVC_LAG_MA": [
        "Open", "High", "Low", "Volume",
        "Close_lag1", "Close_lag2", "Close_lag3",
        "MA5", "MA10", "MA20", "MA30", "MA50"
    ],

    # Full Base + EMA
    "OHLVC_LAG_EMA": [
        "Open", "High", "Low", "Volume",
        "Close_lag1", "Close_lag2", "Close_lag3",
        "EMA5", "EMA12", "EMA26", "EMA30", "EMA50"
    ],

    # Full Base + RSI
    "OHLVC_LAG_RSI": [
        "Open", "High", "Low", "Volume",
        "Close_lag1", "Close_lag2", "Close_lag3",
        "RSI7", "RSI14", "RSI21"
    ],

    # Full Base + MACD
    "OHLVC_LAG_MACD": [
        "Open", "High", "Low", "Volume",
        "Close_lag1", "Close_lag2", "Close_lag3",
        "MACD_5_20", "MACD_10_30", "MACD_12_26"
    ],

    # Campuran indikator
    "OHLVC_LAG_MA_EMA": [
        "Open", "High", "Low", "Volume",
        "Close_lag1", "Close_lag2", "Close_lag3",
        "MA5", "MA10", "MA20", "MA30", "MA50",
        "EMA5", "EMA12", "EMA26", "EMA30", "EMA50"
    ],
    "OHLVC_LAG_MA_RSI": [
        "Open", "High", "Low", "Volume",
        "Close_lag1", "Close_lag2", "Close_lag3",
        "MA5", "MA10", "MA20", "MA30", "MA50",
        "RSI7", "RSI14", "RSI21"
    ],
    "OHLVC_LAG_MA_MACD": [
        "Open", "High", "Low", "Volume",
        "Close_lag1", "Close_lag2", "Close_lag3",
        "MA5", "MA10", "MA20", "MA30", "MA50",
        "MACD_5_20", "MACD_10_30", "MACD_12_26"
    ],
    "OHLVC_LAG_EMA_RSI": [
        "Open", "High", "Low", "Volume",
        "Close_lag1", "Close_lag2", "Close_lag3",
        "EMA5", "EMA12", "EMA26", "EMA30", "EMA50",
        "RSI7", "RSI14", "RSI21"
    ],
    "OHLVC_LAG_EMA_MACD": [
        "Open", "High", "Low", "Volume",
        "Close_lag1", "Close_lag2", "Close_lag3",
        "EMA5", "EMA12", "EMA26", "EMA30", "EMA50",
        "MACD_5_20", "MACD_10_30", "MACD_12_26"
    ],
    "OHLVC_LAG_RSI_MACD": [
        "Open", "High", "Low", "Volume",
        "Close_lag1", "Close_lag2", "Close_lag3",
        "RSI7", "RSI14", "RSI21",
        "MACD_5_20", "MACD_10_30", "MACD_12_26"
    ],

    # Kombinasi 3 indikator
    "OHLVC_LAG_MA_EMA_RSI": [
        "Open", "High", "Low", "Volume",
        "Close_lag1", "Close_lag2", "Close_lag3",
        "MA5", "MA10", "MA20", "MA30", "MA50",
        "EMA5", "EMA12", "EMA26", "EMA30", "EMA50",
        "RSI7", "RSI14", "RSI21"
    ],
    "OHLVC_LAG_MA_RSI_MACD": [
        "Open", "High", "Low", "Volume",
        "Close_lag1", "Close_lag2", "Close_lag3",
        "MA5", "MA10", "MA20", "MA30", "MA50",
        "RSI7", "RSI14", "RSI21",
        "MACD_5_20", "MACD_10_30", "MACD_12_26"
    ],
    "OHLVC_LAG_EMA_RSI_MACD": [
        "Open", "High", "Low", "Volume",
        "Close_lag1", "Close_lag2", "Close_lag3",
        "EMA5", "EMA12", "EMA26", "EMA30", "EMA50",
        "RSI7", "RSI14", "RSI21",
        "MACD_5_20", "MACD_10_30", "MACD_12_26"
    ],

    # Kombinasi semua indikator
    "OHLVC_LAG_MA_EMA_RSI_MACD": [
        "Open", "High", "Low", "Volume",
        "Close_lag1", "Close_lag2", "Close_lag3",
        "MA5", "MA10", "MA20", "MA30", "MA50",
        "EMA5", "EMA12", "EMA26", "EMA30", "EMA50",
        "RSI7", "RSI14", "RSI21",
        "MACD_5_20", "MACD_10_30", "MACD_12_26"
    ]
}

# ======================================================
# Fungsi bantu
# ======================================================
def determine_test_size(df: pd.DataFrame) -> Tuple[int, str]:
    """Menentukan ukuran test adaptif tergantung panjang data."""
    START_DATE_EXPECTED = date(2015, 3, 2)
    MIN_NEEDED = N_STEPS + H_1M
    start_actual = df[DATE_COL].min().date()
    total_rows = len(df)

    # === [UPDATE] Versi lebih adaptif ===
    adaptive_size = max(int(total_rows * 0.15), MIN_NEEDED + 20)

    # Kasus jika datanya lengkap (misal saham lama seperti BBRI)
    if start_actual == START_DATE_EXPECTED and total_rows > 1000:
        return adaptive_size, "adaptive_15%"
    # Kasus jika data pendek (IPO baru)
    elif total_rows <= MIN_NEEDED * 3:
        return MIN_NEEDED + 20, "min_window_safe"
    else:
        return adaptive_size, "adaptive_default"



def build_lstm_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    """Bangun arsitektur model LSTM standar."""
    units = 64 if input_shape[-1] < 10 else 128
    dense_units = 32 if input_shape[-1] < 10 else 64
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(units, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(dense_units, activation="relu"),
        layers.Dense(H_1M)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="mse", metrics=["mae"])
    return model


def plot_loss_curve(history, t, comb_name, fold, best_epoch, pdf):
    """Plot loss curve untuk 1 fold dan simpan ke PDF."""
    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.axvline(best_epoch, color='r', linestyle='--', label='Best Epoch')
    plt.title(f"{t} - {comb_name} (Fold {fold+1})")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    pdf.savefig()
    plt.close()


# ======================================================
# Fungsi utama training
# ======================================================
def train_each(data_dir="../Data/Saham", out_dir="models", tickers_filter=None,subset_filter=None):
    DATA_DIR = Path(data_dir)
    if not DATA_DIR.exists():
        raise SystemExit(f"❌ Folder tidak ditemukan: {DATA_DIR.resolve()}")

    # 1️⃣ Load data dan pra-pemrosesan
    prices = load_prices_from_folder(str(DATA_DIR))
    prices = add_simple_returns(prices)
    prices = add_indicators(prices)

    tickers = prices["Ticker"].unique()
    if tickers_filter:
        tickers = [t for t in tickers if t in tickers_filter]
    print(f"📊 Jumlah emiten terbaca: {len(tickers)} — {tickers}")

    logs = []

    # 2️⃣ Loop kombinasi fitur
    for comb_name, feature_subset in FEATURE_COMBINATIONS.items():
        if subset_filter and comb_name not in subset_filter:
            continue
        print(f"\n================= Mulai training kombinasi fitur: {comb_name} =================")

        # Buat PDF terpisah per kombinasi fitur
        pdf_path = os.path.join(out_dir, f"loss_curves_{comb_name}.pdf")
        pdf = PdfPages(pdf_path)
        pdf.infodict().update({
            'Title': f"Loss Curves for {comb_name}",
            'Author': "Fadhlan Siregar",
            'Subject': "Training Loss Curves per Fold",
            'Keywords': "LSTM, Stock Prediction, Training, Loss Curve"
        })

        for t in tickers:
            p_i = prices[prices["Ticker"] == t].copy()
            need_cols = [TARGET_COL] + (["Return"] if USE_RETURN_AS_TARGET else [])
            p_i = p_i.dropna(subset=need_cols)

            test_size, mode_split = determine_test_size(p_i)
            tscv = TimeSeriesSplit(n_splits=5, test_size=test_size)
            print(f"[{t}] Split mode={mode_split}, test_size={test_size}")

            best_model = None
            best_val_loss_global = float("inf")
            best_fold_info = None

            for fold, (train_idx, val_idx) in enumerate(tscv.split(p_i)):
                tr, va = p_i.iloc[train_idx], p_i.iloc[val_idx]
                print(f"[{t}] Fold {fold+1}: Train={len(tr)}, Val={len(va)}")

                try:
                    Xtr, ytr, scaler, _ = prepare_sequences(tr, TARGET_COL, H_1M, feature_subset)
                    Xva, yva, _, _ = prepare_sequences(va, TARGET_COL, H_1M, feature_subset, scaler=scaler)
                except Exception as e:
                    print(f"⚠️ Skip {t} (fold={fold+1}, fitur={comb_name}): {e}")
                    continue

                model = build_lstm_model((N_STEPS, Xtr.shape[-1]))
                history = model.fit(Xtr, ytr, validation_data=(Xva, yva),
                                    epochs=100, batch_size=32, verbose=1)

                # Evaluasi
                yva_pred = model.predict(Xva, verbose=0)
                val_mape = mean_absolute_percentage_error(yva, yva_pred)
                val_rmse = root_mean_squared_error(yva, yva_pred)
                val_r2 = r2_score(yva, yva_pred)

                val_losses = history.history["val_loss"]
                best_epoch = np.argmin(val_losses)
                best_val_loss = val_losses[best_epoch]
                best_val_mae = history.history["val_mae"][best_epoch]

                print(f"[{t}] Fold {fold+1}: Best Epoch={best_epoch+1}, "
                    f"Val_Loss={best_val_loss:.6f}, Val_MAE={best_val_mae:.6f}")

                logs.append({
                    "Ticker": t,
                    "Fitur": comb_name,
                    "Fold": fold + 1,
                    "Best_Epoch": int(best_epoch + 1),
                    "Val_Loss": float(best_val_loss),
                    "Val_MAE": float(best_val_mae),
                    "Val_MAPE": float(val_mape),
                    "Val_RMSE": float(val_rmse),
                    "Val_R2": float(val_r2),
                })

                # Plot Loss Curve
                plot_loss_curve(history, t, comb_name, fold, best_epoch, pdf)

                # Simpan model terbaik global
                if best_val_loss < best_val_loss_global:
                    best_val_loss_global = best_val_loss
                    best_model = model
                    best_fold_info = {
                        "Fold": fold + 1,
                        "Best_Epoch": best_epoch + 1,
                        "Val_Loss": best_val_loss,
                        "Val_MAE": best_val_mae,
                    }

            # Setelah semua fold
            if best_model is not None:
                save_dir = os.path.join(out_dir, comb_name)
                os.makedirs(save_dir, exist_ok=True)
                out_name = os.path.join(save_dir, f"{t}_best_price.keras")
                best_model.save(out_name)
                print(f"[{t}] ✅ Model terbaik (Fold {best_fold_info['Fold']}) disimpan ke: {out_name}")

        # Tutup PDF untuk kombinasi ini
        pdf.close()
        print(f"[LOG] Semua loss curve untuk fitur {comb_name} disimpan di: {pdf_path}")

    # 3️⃣ Simpan log ke CSV dan buat summary Excel
    log_path = os.path.join(out_dir, "training_summary.csv")
    pd.DataFrame(logs).to_csv(log_path, index=False)
    print(f"[LOG] Ringkasan hasil training disimpan ke: {log_path}")

    try:
        summarize_training_results(
            csv_path=log_path,
            out_path=os.path.join(out_dir, "training_results_with_epoch.xlsx"),
        )
        print("[LOG] File Excel summary berhasil dibuat otomatis ✅")
    except Exception as e:
        print(f"[WARNING] Gagal membuat summary Excel: {e}")

