import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from utils.io_utils import load_price_single
from utils.preprocess_no_sentimen import add_simple_returns, add_indicators, prepare_sequences
from utils.kalman import apply_kalman_filter
from config import TARGET_COL, N_STEPS, H_1M


def predict_next(prices_path: str, model_path: str, feature_subset, out_dir="predictions"):
    """
    Prediksi harga horizon H_1M ke depan untuk 1 emiten.
    prices_path : path CSV data harga emiten
    model_path  : path model .keras yang sudah dilatih
    feature_subset : list fitur yang dipakai model
    """
    if not os.path.exists(model_path):
        raise SystemExit(f"Model {model_path} tidak ditemukan!")

    # 1) Load data
    df = load_price_single(prices_path)
    df = add_simple_returns(df)
    df = add_indicators(df)

    # 2) Load model
    model = tf.keras.models.load_model(model_path)

    # 3) Siapkan X_pred (pakai 60 hari terakhir)
    X_pred, _, _, _ = prepare_sequences(
        df.tail(N_STEPS + H_1M),
        target_col_name=TARGET_COL,
        horizon=H_1M,
        feature_subset=feature_subset
    )
    X_pred = X_pred[-1:]  # hanya sequence terakhir

    # 4) Prediksi
    y_pred = model.predict(X_pred).flatten()
    y_pred_smooth = apply_kalman_filter(pd.Series(y_pred.flatten()))

    # 5) Simpan hasil
    os.makedirs(out_dir, exist_ok=True)
    ticker = os.path.basename(prices_path).replace(".csv", "")
    out_csv = os.path.join(out_dir, f"{ticker}_prediction.csv")
    pd.DataFrame({"Day": range(1, H_1M + 1), "Predicted_Close": y_pred, "Smoothed_Close": y_pred_smooth}).to_csv(out_csv, index=False)

    # 6) Plot
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, H_1M + 1), y_pred, marker="o", label="Prediksi")
    plt.title(f"Prediksi Harga {ticker} {H_1M} Hari Kedepan")
    plt.xlabel("Hari ke-")
    plt.ylabel("Harga Prediksi")
    plt.legend()
    out_png = os.path.join(out_dir, f"{ticker}_prediction.png")
    plt.savefig(out_png)
    plt.close()

    print(f"[DONE] Prediksi {ticker} disimpan di:\n- {out_csv}\n- {out_png}")
