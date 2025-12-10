import pandas as pd
import os


# ==========================================================
# 1. VALIDASI KOLUMN
# ==========================================================
def validate_columns(df, required_cols):
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Kolom berikut tidak ditemukan di CSV: {missing}")


# ==========================================================
# 2. SUMMARY RATA-RATA
# ==========================================================
def compute_summary_mean(df):
    return (
        df.groupby(["Ticker", "Fitur"])[
            ["Val_Loss", "Val_MAE", "Val_MAPE", "Val_RMSE", "Val_R2", "Best_Epoch"]
        ]
        .mean()
        .reset_index()
        .rename(columns={"Best_Epoch": "Best_Epoch_Avg"})
    )


# ==========================================================
# 3. SUMMARY DEV STD
# ==========================================================
def compute_summary_std(df):
    return (
        df.groupby(["Ticker", "Fitur"])[["Val_Loss", "Val_MAE"]]
        .std()
        .reset_index()
        .rename(columns={
            "Val_Loss": "Val_Loss_Std",
            "Val_MAE": "Val_MAE_Std"
        })
    )


# ==========================================================
# 4. FOLD TERBAIK PER FITUR
# ==========================================================
def compute_best_fold(df):
    idx = df.groupby(["Ticker", "Fitur"])["Val_Loss"].idxmin()
    return (
        df.loc[idx, ["Ticker", "Fitur", "Fold", "Dense_Units", "Best_Epoch",
                     "Val_Loss", "Val_MAE", "Val_MAPE", "Val_RMSE"]]
          .rename(columns={
              "Fold": "Best_Fold",
              "Best_Epoch": "Best_Epoch_Min_Loss",
              "Val_Loss": "Best_Val_Loss"
          })
    )


# ==========================================================
# 5. BEST MODEL PER TICKER (RMSE TERENDAH)
# ==========================================================
def compute_best_model_per_ticker(df):
    idx = df.groupby("Ticker")["Val_RMSE"].idxmin()
    return (
        df.loc[idx, ["Ticker", "Fitur", "Dense_Units", "Fold",
                     "Best_Epoch", "Val_RMSE", "Val_MAPE", "Val_R2"]]
          .sort_values("Ticker")
          .reset_index(drop=True)
    )


# ==========================================================
# 6. BEST FEATURE GLOBAL
# ==========================================================
def compute_best_feature(df):
    return (
        df.groupby("Fitur")["Val_RMSE"]
          .mean()
          .sort_values()
          .reset_index()
          .rename(columns={"Val_RMSE": "Avg_Val_RMSE"})
    )


# ==========================================================
# 7. LEADERBOARD (GLOBAL RANKING)
# ==========================================================
def compute_leaderboard(df):
    return (
        df.sort_values("Val_RMSE")
          .reset_index(drop=True)
    )


# ==========================================================
# 8. SHEET PER DENSE UNIT
# ==========================================================
def build_dense_sheets(df):
    dense_dict = {}
    for d in sorted(df["Dense_Units"].unique()):
        dense_dict[f"Dense_{d}"] = df[df["Dense_Units"] == d]
    return dense_dict


# ==========================================================
# 9. FUNGSI UTAMA
# ==========================================================
def summarize_training_results_advanced(
    csv_path="models/training_summary.csv",
    out_path="models/training_results_complete.xlsx"
):

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File tidak ditemukan: {csv_path}")

    print(f"[LOG] Membaca hasil training dari: {csv_path}")
    df = pd.read_csv(csv_path)

    # Validasi kolom
    required_cols = {
        "Ticker", "Fitur", "Fold", "Dense_Units", "Best_Epoch",
        "Val_Loss", "Val_MAE", "Val_MAPE", "Val_RMSE", "Val_R2"
    }
    validate_columns(df, required_cols)

    # Summary utama
    summary_mean = compute_summary_mean(df)
    summary_std = compute_summary_std(df)
    summary_best_fold = compute_best_fold(df)
    summary_final = (
        summary_mean
        .merge(summary_std, on=["Ticker", "Fitur"], how="left")
        .merge(summary_best_fold, on=["Ticker", "Fitur"], how="left")
    )

    # Sheet tambahan
    dense_sheets = build_dense_sheets(df)
    best_per_ticker = compute_best_model_per_ticker(df)
    best_feature = compute_best_feature(df)
    leaderboard = compute_leaderboard(df)

    # Tulis semua sheet
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="PerFold", index=False)
        summary_final.to_excel(writer, sheet_name="Summary", index=False)
        best_per_ticker.to_excel(writer, sheet_name="Best_Model_Per_Ticker", index=False)
        best_feature.to_excel(writer, sheet_name="Best_Feature", index=False)
        leaderboard.to_excel(writer, sheet_name="Leaderboard", index=False)

        # Sheet dense per dense_units
        for sheet_name, df_dense in dense_sheets.items():
            df_dense.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\n[LOG] File Excel lengkap dibuat di: {out_path}")
    print("Sheet yang dihasilkan:")
    print(" - PerFold")
    print(" - Summary")
    print(" - Best_Model_Per_Ticker")
    print(" - Best_Feature")
    print(" - Leaderboard")
    for key in dense_sheets.keys():
        print(f" - {key}")


if __name__ == "__main__":
    summarize_training_results_advanced()
