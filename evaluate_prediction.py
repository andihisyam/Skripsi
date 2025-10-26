# evaluate_predictions.py
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
import numpy as np

def evaluate_predictions(ticker, pred_dir="predictions", horizon_dir="D:/Kuliah/Skripsi/Data/Saham_Horizon_22"):
    # Path file prediksi dan aktual
    pred_path = os.path.join(pred_dir, f"{ticker}_prediction.csv")
    actual_path = os.path.join(horizon_dir, f"{ticker}.csv")

    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"❌ File prediksi tidak ditemukan: {pred_path}")
    if not os.path.exists(actual_path):
        raise FileNotFoundError(f"❌ File aktual tidak ditemukan: {actual_path}")

    # Baca file
    df_pred = pd.read_csv(pred_path)
    df_actual = pd.read_csv(actual_path)

    # Normalisasi nama kolom tanggal
    tanggal_pred = "Tanggal" if "Tanggal" in df_pred.columns else "Date"
    tanggal_actual = "Date" if "Date" in df_actual.columns else df_actual.columns[0]

    # Konversi kolom tanggal ke datetime
    df_pred[tanggal_pred] = pd.to_datetime(df_pred[tanggal_pred], errors="coerce")
    df_actual[tanggal_actual] = pd.to_datetime(df_actual[tanggal_actual], errors="coerce")

    # Gabungkan berdasarkan tanggal
    df_merge = pd.merge(df_pred, df_actual, left_on=tanggal_pred, right_on=tanggal_actual, how="inner")

    if len(df_merge) == 0:
        raise ValueError("⚠️ Tidak ada tanggal yang cocok antara prediksi dan data aktual.")

    # Urutkan berdasarkan tanggal naik (ascending)
    df_merge = df_merge.sort_values(by=tanggal_pred).reset_index(drop=True)

    # Gunakan kolom 'Price' atau 'Close' sebagai harga aktual
    harga_kolom = None
    for kol in ["Price", "Close"]:
        if kol in df_merge.columns:
            harga_kolom = kol
            break
    if harga_kolom is None:
        raise ValueError("⚠️ Tidak ditemukan kolom harga ('Price' atau 'Close') di data aktual.")

    # 💡 Konversi kolom harga ke float (hapus tanda koma atau simbol ribuan)
    df_merge[harga_kolom] = (
        df_merge[harga_kolom]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(".", "", regex=False)
        .astype(float)
    )

    # Pilih kolom prediksi yang sesuai
    pred_raw_col = "Predicted_Close" if "Predicted_Close" in df_merge.columns else None
    pred_smooth_col = "Smoothed_Close" if "Smoothed_Close" in df_merge.columns else None

    y_true = df_merge[harga_kolom].values
    y_pred_raw = df_merge[pred_raw_col].astype(float).values if pred_raw_col else None
    y_pred_smooth = df_merge[pred_smooth_col].astype(float).values if pred_smooth_col else None

    # Hitung metrik
    def metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return mae, mape, rmse, r2

    print(f"\n📊 Evaluasi hasil prediksi untuk {ticker}")
    print("-" * 60)

    if y_pred_raw is not None:
        mae, mape, rmse, r2 = metrics(y_true, y_pred_raw)
        print(f"🔹 Model Asli (Predicted_Close)")
        print(f"   MAE  : {mae:.2f}")
        print(f"   MAPE : {mape:.4f}")
        print(f"   RMSE : {rmse:.2f}")
        print(f"   R²   : {r2:.4f}\n")

    if y_pred_smooth is not None:
        mae, mape, rmse, r2 = metrics(y_true, y_pred_smooth)
        print(f"🔹 Setelah Kalman Filter (Smoothed_Close)")
        print(f"   MAE  : {mae:.2f}")
        print(f"   MAPE : {mape:.4f}")
        print(f"   RMSE : {rmse:.2f}")
        print(f"   R²   : {r2:.4f}")

    # Visualisasi
    plt.figure(figsize=(10, 5))
    plt.plot(df_merge[tanggal_pred], y_true, label="Aktual", linewidth=2)
    if y_pred_raw is not None:
        plt.plot(df_merge[tanggal_pred], y_pred_raw, label="Prediksi (Raw)", linestyle="--", alpha=0.6)
    if y_pred_smooth is not None:
        plt.plot(df_merge[tanggal_pred], y_pred_smooth, label="Prediksi (Kalman)", linewidth=2)
    plt.title(f"Hasil Prediksi vs Aktual ({ticker})")
    plt.xlabel("Tanggal")
    plt.ylabel("Harga Saham")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # (Opsional) simpan hasil merge buat dicek manual
    save_path = os.path.join(pred_dir, f"{ticker}_merged_evaluation.csv")
    df_merge.to_csv(save_path, index=False)
    print(f"\n📝 File hasil merge disimpan di: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluasi hasil prediksi saham")
    parser.add_argument("--ticker", required=True, help="Ticker emiten, contoh: ARTO")
    parser.add_argument("--pred_dir", default="predictions", help="Folder hasil prediksi")
    parser.add_argument("--horizon_dir", default="D:/Kuliah/Skripsi/Data/Saham_Horizon_22", help="Folder data aktual horizon 22")
    args = parser.parse_args()

    evaluate_predictions(args.ticker, args.pred_dir, args.horizon_dir)
