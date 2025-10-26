# utils/summarize_results.py
import pandas as pd
import os

def summarize_training_results(
    csv_path="models/training_summary.csv",
    out_path="models/training_results_with_epoch.xlsx"
):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File tidak ditemukan: {csv_path}")

    print(f"[LOG] Membaca hasil training dari: {csv_path}")
    df = pd.read_csv(csv_path)

    # Pastikan kolom wajib ada
    expected_cols = {"Ticker", "Fitur", "Fold", "Best_Epoch", "Val_Loss", "Val_MAE", "Val_MAPE", "Val_RMSE", "Val_R2"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Kolom berikut tidak ditemukan di CSV: {missing}")

    # 1️⃣ Hitung rata-rata tiap metrik per kombinasi fitur
    summary_mean = (
        df.groupby(["Ticker", "Fitur"])
          [["Val_Loss", "Val_MAE", "Val_MAPE", "Val_RMSE", "Val_R2", "Best_Epoch"]]
          .mean()
          .reset_index()
          .rename(columns={"Best_Epoch": "Best_Epoch_Avg"})
    )

    # 2️⃣ Tambahkan deviasi standar untuk stabilitas model
    summary_std = (
        df.groupby(["Ticker", "Fitur"])
          [["Val_Loss", "Val_MAE"]]
          .std()
          .reset_index()
          .rename(columns={"Val_Loss": "Val_Loss_std", "Val_MAE": "Val_MAE_std"})
    )

    # 3️⃣ Ambil fold terbaik (Val_Loss terkecil)
    best_fold_info = (
        df.loc[df.groupby(["Ticker", "Fitur"])["Val_Loss"].idxmin(), 
               ["Ticker", "Fitur", "Fold", "Best_Epoch", "Val_Loss", "Val_MAE"]]
          .rename(columns={
              "Fold": "Best_Fold",
              "Best_Epoch": "Best_Epoch_Min_Loss",
              "Val_Loss": "Best_Val_Loss",
              "Val_MAE": "Best_Val_MAE"
          })
    )

    # 4️⃣ Gabungkan semua hasil
    summary_final = (
        summary_mean
        .merge(summary_std, on=["Ticker", "Fitur"], how="left")
        .merge(best_fold_info, on=["Ticker", "Fitur"], how="left")
    )

    # 5️⃣ Simpan ke satu file Excel dengan dua sheet
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="PerFold", index=False)
        summary_final.to_excel(writer, sheet_name="Summary", index=False)

    print(f"[LOG] Hasil lengkap disimpan ke: {out_path}")
    print(f" - Sheet 'PerFold'  : hasil setiap fold")
    print(f" - Sheet 'Summary'  : rata-rata, deviasi, dan epoch terbaik per fitur\n")

if __name__ == "__main__":
    summarize_training_results()
