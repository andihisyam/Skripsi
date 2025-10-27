# utils/predict_no_sentimen.py (versi final - hybrid scaler + Kalman + auto plot)
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from utils.preprocess_no_sentimen import add_indicators, prepare_sequences
from utils.kalman import apply_kalman_filter  
from config import TARGET_COL, H_1M, N_STEPS


# ======================================================
# Inverse transform hanya fitur harga
# ======================================================
def inverse_transform_price_only(y_scaled: np.ndarray, scaler_dict: dict):
    """
    Mengembalikan hasil prediksi ke skala harga asli (Rupiah)
    menggunakan scaler 'price' dari scaler_dict.
    """
    if "price" not in scaler_dict:
        raise ValueError("Scaler untuk 'price' tidak ditemukan di scaler_dict.")
    scaler = scaler_dict["price"]

    if y_scaled.ndim == 1:
        y_scaled = y_scaled.reshape(-1, 1)

    y_real = scaler.inverse_transform(y_scaled)
    return y_real


# ======================================================
# Fungsi utama prediksi
# ======================================================
def predict_next(prices_path: str, model_path: str, feature_subset: list, out_dir: str = "predictions"):
    """
    Melakukan prediksi harga saham menggunakan model dan scaler hasil training.
    Termasuk smoothing hasil prediksi dengan Kalman filter dan visualisasi.
    """
    ticker = os.path.splitext(os.path.basename(prices_path))[0]
    print(f"📈 Mulai prediksi untuk {ticker}")

    # --- 1️⃣ Load data saham dan tambahkan indikator teknikal ---
    df = pd.read_csv(prices_path)
    df = add_indicators(df)

    # --- 2️⃣ Load model & scaler ---
    model = load_model(model_path)
    scaler_path = model_path.replace("_best_price.keras", "_scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file tidak ditemukan: {scaler_path}")

    scaler_dict = joblib.load(scaler_path)
    print(f"✅ Model & scaler berhasil dimuat untuk {ticker}")

    # --- 3️⃣ Siapkan sequence terakhir (N_STEPS terakhir) ---
    try:
        X_all, _, _, _ = prepare_sequences(
            df, TARGET_COL, H_1M, feature_subset=feature_subset, scaler_dict=scaler_dict
        )
    except Exception as e:
        raise RuntimeError(f"Gagal menyiapkan data untuk prediksi: {e}")

    X_latest = X_all[-1:]  # window terakhir (misal 60 hari terakhir)

    # --- 4️⃣ Prediksi ---
    y_pred_scaled = model.predict(X_latest)
    y_pred_real = inverse_transform_price_only(y_pred_scaled.reshape(-1, 1), scaler_dict).flatten()

    # --- 5️⃣ Terapkan Kalman filter ---
    try:
        y_smooth = apply_kalman_filter(y_pred_real)
        print("✨ Kalman smoothing berhasil diterapkan.")
    except Exception as e:
        print(f"⚠️ Gagal menerapkan Kalman filter: {e}")
        y_smooth = y_pred_real  # fallback ke hasil asli

    # --- 6️⃣ Simpan hasil ke CSV ---
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{ticker}_prediction.csv")

    result_df = pd.DataFrame({
        "Ticker": [ticker] * len(y_pred_real),
        "Predicted_Close_Raw": y_pred_real,
        "Predicted_Close_Smoothed": y_smooth
    })
    result_df.to_csv(out_csv, index=False)

    print(f"💾 Hasil prediksi disimpan ke: {out_csv}")

    # --- 7️⃣ Plot hasil prediksi ---
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, H_1M + 1), y_pred_real, marker="o", label="Prediksi (Raw)")
    plt.plot(range(1, H_1M + 1), y_smooth, marker="s", label="Prediksi (Kalman)", linestyle="--")
    plt.title(f"Prediksi Harga {ticker} {H_1M} Hari Kedepan")
    plt.xlabel("Hari ke-")
    plt.ylabel("Harga Prediksi (Rupiah)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_png = os.path.join(out_dir, f"{ticker}_prediction.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"🖼️ Grafik prediksi disimpan ke: {out_png}")
    print(f"✅ Prediksi selesai untuk {ticker} — Output: {len(y_pred_real)} hari (raw + smoothed)")

    return result_df
