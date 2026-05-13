import os
import json
import random
from pathlib import Path
from typing import Optional, Sequence, Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import joblib

from sklearn.metrics import r2_score, root_mean_squared_error

from utils.io_utils import load_prices_from_folder
from utils.optuna import (
    train_model_cv,
    run_optuna_cv,
    prepare_val_with_context,
    inverse_target,
    build_lstm_model_by_backend,
    model_forward_by_backend,
)
from utils.preprocess_sentimen import (
    add_simple_returns,
    add_return_lags,
    add_indicators,
    prepare_sequences,
    load_sentimen,
    merge_sentimen,
    SENTIMEN_COLS,
)
from utils.debug_training import debug_trial, debug_sequences

from config import DATE_COL, TARGET_COL, N_STEPS, PRICE_COL


# ======================================================
# Target forward-H return
# ======================================================
def add_forward_return(df: pd.DataFrame, H: int, out_col: str, log: bool = True) -> pd.DataFrame:
    df = df.sort_values(["Ticker", DATE_COL]).copy()
    g  = df.groupby("Ticker", group_keys=False)[PRICE_COL]
    future = g.shift(-H)
    now    = df[PRICE_COL].astype(float)
    if log:
        df[out_col] = np.log(future / now)
    else:
        df[out_col] = (future / now) - 1.0
    return df


# ======================================================
# Metrics
# ======================================================
def eval_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return {
        "RMSE": float(root_mean_squared_error(y_true, y_pred)),
        "MAE":  float(np.mean(np.abs(y_true - y_pred))),
        "R2":   float(r2_score(y_true, y_pred)),
    }


def directional_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def information_coefficient(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_multistep_metrics(y_true, y_pred):
    """
    Return:
      - overall metrics (flatten semua step)
      - metrics per step (list of dict)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    overall = eval_metrics(y_true.reshape(-1), y_pred.reshape(-1))
    overall["DA"] = directional_accuracy(y_true.reshape(-1), y_pred.reshape(-1))
    overall["IC"] = information_coefficient(y_true.reshape(-1), y_pred.reshape(-1))

    by_step = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        m = eval_metrics(yt, yp)
        m["DA"] = directional_accuracy(yt, yp)
        m["IC"] = information_coefficient(yt, yp)
        m["Step"] = i + 1
        by_step.append(m)

    return overall, by_step


def save_loss_pdf(history: dict, pdf_path: str):
    train_loss = history.get("train_loss", [])
    val_loss   = history.get("val_loss", [])
    if not train_loss and not val_loss:
        return
    plt.figure()
    if train_loss:
        plt.plot(train_loss, label="Training Loss")
    if val_loss:
        plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Huber)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.close()


def save_test_pred_pdf(
    y_true, y_pred, pdf_path: str,
    dates=None,
    close_ref=None,
    y_hist=None, dates_hist=None,
    close_hist=None,
):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if close_ref is not None:
        close_ref    = np.asarray(close_ref).reshape(-1)[:len(y_true)]
        actual_close = close_ref * np.exp(y_true)
        pred_close   = close_ref * np.exp(y_pred)
        ylabel       = "Harga (Close)"
    else:
        actual_close = y_true
        pred_close   = y_pred
        ylabel       = "Log Return"

    plt.figure(figsize=(14, 5))

    if close_hist is not None and dates_hist is not None:
        dates_hist = pd.to_datetime(dates_hist).values
        close_hist = np.asarray(close_hist).reshape(-1)[:len(dates_hist)]
        plt.plot(dates_hist, close_hist, label="Historis (aktual)",
                 linewidth=1, color="steelblue", alpha=0.4)

    if dates is not None:
        dates = pd.to_datetime(dates).values[:len(actual_close)]
        plt.plot(dates, actual_close, label="Aktual (test)",
                 linewidth=1.5, color="steelblue")
        plt.plot(dates, pred_close,   label="Prediksi (test)",
                 linewidth=1.5, color="orange", linestyle="--")
        plt.axvline(x=dates[0], color="gray", linestyle=":", linewidth=1)
    else:
        plt.plot(actual_close, label="Aktual (test)",   linewidth=1.5)
        plt.plot(pred_close,   label="Prediksi (test)", linewidth=1.5, linestyle="--")

    plt.xlabel("Tanggal")
    plt.ylabel(ylabel)
    plt.title("Prediksi vs Aktual — Full Timeline")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gcf().autofmt_xdate(rotation=45)
    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.close()


def save_cv_fold_loss_curves(best_fold_histories, cv_plot_dir: str):
    if not best_fold_histories:
        return
    os.makedirs(cv_plot_dir, exist_ok=True)
    for k, hist in enumerate(best_fold_histories, start=1):
        save_loss_pdf(hist, os.path.join(cv_plot_dir, f"fold_{k}_loss_curve.pdf"))


# ======================================================
# Checkpoint helpers
# ======================================================
def append_row_to_csv(row_dict: dict, csv_path: str):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df_row = pd.DataFrame([row_dict])
    if os.path.exists(csv_path):
        try:
            old_df = pd.read_csv(csv_path)
            df_all = pd.concat([old_df, df_row], ignore_index=True)
        except Exception:
            df_all = df_row.copy()
    else:
        df_all = df_row.copy()
    dedup_cols = [c for c in ["H_FWD", "Ticker", "Fitur"] if c in df_all.columns]
    if dedup_cols:
        df_all = df_all.drop_duplicates(subset=dedup_cols, keep="last")
    df_all.to_csv(csv_path, index=False)


def append_fold_row_to_csv(row_dict: dict, csv_path: str):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df_row = pd.DataFrame([row_dict])
    if os.path.exists(csv_path):
        try:
            old_df = pd.read_csv(csv_path)
            df_all = pd.concat([old_df, df_row], ignore_index=True)
        except Exception:
            df_all = df_row.copy()
    else:
        df_all = df_row.copy()
    dedup_cols = [c for c in ["H_FWD", "Ticker", "Fitur", "Fold"] if c in df_all.columns]
    if dedup_cols:
        df_all = df_all.drop_duplicates(subset=dedup_cols, keep="last")
    df_all.to_csv(csv_path, index=False)


def build_excel_from_checkpoints(out_dir: str):
    all_summary, all_folds = [], []
    if not os.path.exists(out_dir):
        return
    for folder in sorted(os.listdir(out_dir)):
        horizon_dir = os.path.join(out_dir, folder)
        if not os.path.isdir(horizon_dir):
            continue
        for ckpt, store in [
            ("training_summary_checkpoint.csv", all_summary),
            ("training_folds_checkpoint.csv",   all_folds),
        ]:
            path = os.path.join(horizon_dir, ckpt)
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    if not df.empty:
                        store.append(df)
                except Exception as e:
                    print(f"[WARN] {path}: {e}")

    df_sum   = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    df_folds = pd.concat(all_folds,   ignore_index=True) if all_folds   else pd.DataFrame()

    df_sum.to_csv(os.path.join(out_dir, "ALL_training_summary.csv"), index=False)
    df_folds.to_csv(os.path.join(out_dir, "ALL_training_folds_metrics.csv"), index=False)

    excel_path = os.path.join(out_dir, "ALL_training_results.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_sum.to_excel(writer,   sheet_name="ALL_Summary", index=False)
        df_folds.to_excel(writer, sheet_name="ALL_Folds",   index=False)
    print(f"[SUMMARY] Excel rebuilt: {excel_path}")


# ======================================================
# Config
# ======================================================
N_TRIALS_OPTUNA = 20
MAX_EPOCHS      = 200

# ── PILIHAN MODE ──────────────────────────────────────
# H_OUTPUT = None -> otomatis mengikuti H_FWD (disarankan untuk multi-horizon)
# H_OUTPUT = 1    -> single-step (log return total H hari)
# H_OUTPUT = N>1  -> multi-step fixed N untuk semua horizon
H_OUTPUT: Optional[int] = None

HORIZONS: List[int] = [22, 44]
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10

# Path file sentimen
SENTIMEN_PATH = "../Data/sentimen.xlsx"

RET_1D     = "ret_1d"
R1, R2, R3 = f"{RET_1D}_lag1", f"{RET_1D}_lag2", f"{RET_1D}_lag3"

# ── FEATURE COMBINATIONS dengan sentimen ─────────────
# Sentimen (Positif, Negatif, Netral) bisa dimasukkan
# ke subset manapun sebagai fitur tambahan
FEATURE_COMBINATIONS: Dict[str, Sequence[str]] = {
    "Close_LAG_SENT": [R1, R2, R3, "Positif", "Negatif", "Netral"],
    "OHLVC_LAG_SENT": ["Open", "High", "Low", "Volume", R1, R2, R3,
                        "Positif", "Negatif", "Netral"],
    "OHLVC_LAG_MA_SENT": [
        "Open", "High", "Low", "Volume", R1, R2, R3,
        "MA5", "MA10", "MA20", "MA30", "MA50","Positif", "Negatif", "Netral"
    ],

    "OHLVC_LAG_EMA_SENT": [
        "Open", "High", "Low", "Volume", R1, R2, R3,
        "EMA5", "EMA12", "EMA26", "EMA30", "EMA50","Positif", "Negatif", "Netral"
    ],

    "OHLVC_LAG_RSI_SENT": [
        "Open", "High", "Low", "Volume", R1, R2, R3,
        "RSI7", "RSI14", "RSI21",
        "Positif", "Negatif", "Netral"
    ],

    "OHLVC_LAG_MACD_SENT": [
        "Open", "High", "Low", "Volume", R1, R2, R3,
        "MACD_5_20", "MACD_10_30", "MACD_12_26","Positif", "Negatif", "Netral"
    ],

    "OHLVC_LAG_MA_EMA_SENT": [
        "Open", "High", "Low", "Volume", R1, R2, R3,
        "MA5", "MA10", "MA20", "MA30", "MA50",
        "EMA5", "EMA12", "EMA26", "EMA30", "EMA50","Positif", "Negatif", "Netral"
    ],

    "OHLVC_LAG_RSI_MACD_SENT": [
        "Open", "High", "Low", "Volume", R1, R2, R3,
        "RSI7", "RSI14", "RSI21",
        "MACD_5_20", "MACD_10_30", "MACD_12_26",
        "Positif", "Negatif", "Netral"
    ],

    "OHLVC_LAG_MA_EMA_RSI_MACD_SENT": [
        "Open", "High", "Low", "Volume", R1, R2, R3,
        "MA5", "MA10", "MA20", "MA30", "MA50",
        "EMA5", "EMA12", "EMA26", "EMA30", "EMA50",
        "RSI7", "RSI14", "RSI21",
        "MACD_5_20", "MACD_10_30", "MACD_12_26",
        "Positif", "Negatif", "Netral"
    ],
}


MIN_DATA_REQUIRED = 700


# ======================================================
# Main training
# ======================================================
def train_each(
    data_dir: str = "../Data/Saham",
    out_dir:  str = "train_result_multiH",
    tickers_filter: Optional[Sequence[str]] = None,
    subset_filter:  Optional[Sequence[str]] = None,
    horizons:       Optional[Sequence[int]] = None,
    overwrite: bool = False,
    save_cv_curves: bool = True,
    sentimen_path: str = SENTIMEN_PATH,
    h_output: Optional[int] = H_OUTPUT,
    random_seed: int = 42,
    model_backend: str = "seq2seq",
):
    horizons = list(horizons) if horizons is not None else HORIZONS

    set_global_seed(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")
    print(f"[SEED] {random_seed}")
    if h_output is None:
        print("[MODE] H_output mengikuti H_FWD (dynamic per horizon)")
    else:
        print(f"[MODE] H_output fixed={h_output} ({'multi-step' if h_output > 1 else 'single-step'})")
    print(f"[MODEL] Backend={model_backend}")

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    base = load_prices_from_folder(str(data_path))

    # ── Build fitur dasar ──
    print("[STEP] Build base features ...")
    base = add_simple_returns(base, out_col=RET_1D)
    base = add_return_lags(base, return_col=RET_1D, lags=(1, 2, 3))
    base = add_indicators(base)

    # ── Merge sentimen ──
    df_sentimen = None
    if os.path.exists(sentimen_path):
        print(f"[SENTIMEN] Load dari: {sentimen_path}")
        df_sentimen = load_sentimen(sentimen_path)
        base        = merge_sentimen(base, df_sentimen)
        print(f"[SENTIMEN] Kolom sentimen ditambahkan: {SENTIMEN_COLS}")
    else:
        print(f"[WARN] File sentimen tidak ditemukan: {sentimen_path}")
        print("[WARN] Training tanpa fitur sentimen.")

    tickers_all = base["Ticker"].unique()
    tickers     = [t for t in tickers_all if (not tickers_filter or t in tickers_filter)]
    print(f"[INFO] Emiten ({len(tickers)}): {tickers}")

    os.makedirs(out_dir, exist_ok=True)

    for H_FWD in horizons:
        h_out_cur = int(H_FWD) if h_output is None else int(h_output)

        # ── Untuk H_output=1: target = log return total H hari ──
        # ── Untuk H_output>1: target = return harian (RET_1D) ──
        if h_out_cur == 1:
            TARGET_TRAIN = f"return_fwd_{H_FWD}"
        else:
            # multi-step: target adalah return harian
            # prepare_sequences akan ambil H_output step berurutan
            TARGET_TRAIN = RET_1D

        print(f"\n{'#'*100}")
        print(f"[HORIZON] H={H_FWD} | H_output={h_out_cur} | target={TARGET_TRAIN}")
        print(f"{'#'*100}")

        prices = base.copy()

        if h_out_cur == 1:
            prices = add_forward_return(prices, H=H_FWD, out_col=TARGET_TRAIN, log=True)

        prices = prices.dropna().reset_index(drop=True)

        horizon_dir      = os.path.join(out_dir, f"FWD_{H_FWD}")
        os.makedirs(horizon_dir, exist_ok=True)

        summary_ckpt_csv = os.path.join(horizon_dir, "training_summary_checkpoint.csv")
        folds_ckpt_csv   = os.path.join(horizon_dir, "training_folds_checkpoint.csv")
        logs, fold_logs  = [], []

        for comb_name, fitur in FEATURE_COMBINATIONS.items():
            if subset_filter and comb_name not in subset_filter:
                print(f"[SKIP] Subset '{comb_name}' tidak termasuk filter.")
                continue

            # skip subset _SENT kalau tidak ada sentimen
            if df_sentimen is None and any(s in fitur for s in SENTIMEN_COLS):
                print(f"[SKIP] {comb_name}: data sentimen tidak tersedia.")
                continue

            print(f"\n{'='*90}\n[SUBSET] {comb_name} | H={H_FWD}\n{'='*90}")

            for t in tickers:
                df_t = prices[prices["Ticker"] == t].copy().dropna(subset=[TARGET_TRAIN])
                if df_t.empty:
                    print(f"[SKIP] {t}: data kosong.")
                    continue

                if len(df_t) < MIN_DATA_REQUIRED:
                    print(f"[SKIP] {t}: data terlalu sedikit ({len(df_t)} baris, minimal {MIN_DATA_REQUIRED}).")
                    continue

                last_dt = pd.to_datetime(df_t[DATE_COL]).max()
                data_sig = f"n{len(df_t)}_d{last_dt:%Y%m%d}"
                study_name = f"{t}__{comb_name}__{model_backend}__FWD{H_FWD}__Hout{h_out_cur}__{data_sig}_split"

                save_dir    = os.path.join(horizon_dir, comb_name)
                os.makedirs(save_dir, exist_ok=True)

                model_path  = os.path.join(save_dir, f"{t}_best_{TARGET_TRAIN}_Hout{h_out_cur}.pt")
                scaler_path = os.path.join(save_dir, f"{t}_scaler_{TARGET_TRAIN}_Hout{h_out_cur}.pkl")
                pdf_path    = os.path.join(save_dir, f"{t}_loss_curve.pdf")

                if (not overwrite) and os.path.exists(model_path) and os.path.exists(scaler_path):
                    print(f"[SKIP] {t} | {comb_name} | H={H_FWD}: sudah ada.")
                    continue

                print(f"\n[OPTIMIZING] {t} | {comb_name} | H={H_FWD} | H_output={h_out_cur}")

                # ── STEP 1: Optuna ──
                try:
                    best_params, best_avg_loss, best_fold_histories, best_fold_metrics = run_optuna_cv(
                        df_t=df_t,
                        fitur=list(fitur),
                        device=device,
                        H_output=h_out_cur,         # ← pass H_output ke Optuna
                        target_col_name=TARGET_TRAIN,
                        horizon=1,
                        trials=N_TRIALS_OPTUNA,
                        MAX_EPOCHS=MAX_EPOCHS,
                        return_best_fold_histories=True,
                        return_best_fold_metrics=True,
                        study_db_path=os.path.join(out_dir, "optuna_studies.db"),
                        study_name=study_name,
                        train_ratio=TRAIN_RATIO,
                        val_ratio=VAL_RATIO,
                        h_output=h_out_cur,         # ← pass ke prepare_sequences
                        model_backend=model_backend,
                    )
                except Exception as e:
                    print(f"[ERROR] Optuna gagal: {t} ({comb_name}) H={H_FWD} -> {e}")
                    continue

                print(f"✔ Optuna done | Best loss: {best_avg_loss:.6f}")
                print(f"  Params: {best_params}")

                best_num_layers = int(best_params.get("num_layers", 1))
                split_mode      = f"ratio_{TRAIN_RATIO}_{VAL_RATIO}"

                if best_fold_metrics:
                    for m in best_fold_metrics:
                        fold_row = {
                            "H_FWD": H_FWD, "H_output": h_out_cur,
                            "Ticker": t, "Fitur": comb_name,
                            "SplitMode": split_mode,
                            "Fold": m.get("Fold"),
                            "CV_ValLoss_RMSE": m.get("ValLoss_RMSE"),
                            "CV_RMSE": m.get("RMSE"), "CV_MAE": m.get("MAE"), "CV_R2": m.get("R2"),
                            "NumLayers": best_num_layers,
                            "Optuna_Params": str(best_params),
                            "Avg_CV_Loss": float(best_avg_loss),
                        }
                        fold_logs.append(fold_row)
                        append_fold_row_to_csv(fold_row, folds_ckpt_csv)

                # ── STEP 2: Final retrain ──
                n       = len(df_t)
                n_train = int(TRAIN_RATIO * n)
                n_val   = int(VAL_RATIO * n)

                train_final_df = df_t.iloc[:n_train]
                val_final_df   = df_t.iloc[n_train:n_train + n_val]
                test_final_df  = df_t.iloc[n_train + n_val:]
                test_size      = len(test_final_df)

                if len(train_final_df) < N_STEPS + 5 or len(val_final_df) < N_STEPS + 2:
                    print(f"[SKIP] {t}: data terlalu sedikit.")
                    continue

                try:
                    Xtr, ytr, Xva, yva, scaler_dict, _ = prepare_val_with_context(
                        tr_df=train_final_df,
                        va_df=val_final_df,
                        target_col_name=TARGET_TRAIN,
                        horizon=1,
                        feature_subset=list(fitur),
                        n_steps=N_STEPS,
                        context_extra=50,
                        h_output=h_out_cur,
                    )
                except Exception as e:
                    print(f"[ERROR] prepare_val_with_context gagal: {e}")
                    continue

                if Xtr.shape[0] < 10 or Xva.shape[0] < 5:
                    print(f"[SKIP] {t}: sequence terlalu sedikit.")
                    continue

                debug_sequences(
                    Xtr, ytr, Xva, yva,
                    target_col_name=TARGET_TRAIN,
                    prefix=f"[FINAL {t} | {comb_name} | H={H_FWD}] "
                )

                print(f"[RETRAIN] Xtr={Xtr.shape} | Xva={Xva.shape} | ytr={ytr.shape}")

                try:
                    best_val_loss, best_state, best_pred_va, history, best_epoch = train_model_cv(
                        Xtr, ytr, Xva, yva,
                        device=device,
                        params=best_params,
                        H_output=h_out_cur,
                        MAX_EPOCHS=MAX_EPOCHS,
                        verbose=True,
                        prefix=f"[FINAL {t} | {comb_name} | H={H_FWD}] ",
                        model_backend=model_backend,
                    )
                except Exception as e:
                    print(f"[ERROR] train_model_cv gagal: {e}")
                    continue

                print(f"[RETRAIN DONE] Best Val Loss: {best_val_loss:.6f} (epoch {best_epoch+1})")

                debug_trial(
                    trial_num=0, fold=0,
                    ytr=ytr.reshape(-1), yva=yva.reshape(-1),
                    best_pred_va=best_pred_va.reshape(-1),
                    epoch_train_losses=history["train_loss"],
                    epoch_val_losses=history["val_loss"],
                    prefix=f"[FINAL {t} | {comb_name} | H={H_FWD}] ",
                )

                # Eval final (original scale)
                target_key = f"target_{TARGET_TRAIN}"
                val_metrics_by_step = []
                try:
                    scaler_tgt = scaler_dict[target_key]

                    y_true_real_2d = scaler_tgt.inverse_transform(
                        yva.reshape(-1, 1)
                    ).reshape(yva.shape)
                    y_pred_real_2d = scaler_tgt.inverse_transform(
                        best_pred_va.reshape(-1, 1)
                    ).reshape(best_pred_va.shape)

                    metrics, val_metrics_by_step = compute_multistep_metrics(y_true_real_2d, y_pred_real_2d)

                    print("[EVAL FINAL - ORIGINAL SCALE]")
                    print(
                        f"  RMSE={metrics['RMSE']:.6f} | "
                        f"MAE={metrics['MAE']:.6f} | "
                        f"R2={metrics['R2']:.6f} | "
                        f"DA={metrics['DA']:.4f} | IC={metrics['IC']:.4f}"
                    )
                    if h_out_cur > 1 and val_metrics_by_step:
                        first_m = val_metrics_by_step[0]
                        last_m = val_metrics_by_step[-1]
                        print(
                            f"  Step t+1 RMSE={first_m['RMSE']:.6f} | "
                            f"Step t+{h_out_cur} RMSE={last_m['RMSE']:.6f}"
                        )
                except Exception as e:
                    scaler_tgt = None
                    metrics = {"RMSE": None, "MAE": None, "R2": None, "DA": None, "IC": None}
                    print(f"[WARN] Gagal hitung metrik: {e}")

                # STEP 3: Evaluasi test set
                test_metrics = {"RMSE": None, "MAE": None, "R2": None, "DA": None, "IC": None}
                test_metrics_by_step = []

                if len(test_final_df) >= N_STEPS + 2 and scaler_tgt is not None:
                    try:
                        context_rows = val_final_df.iloc[-N_STEPS:].copy().reset_index(drop=True)
                        te_with_ctx  = pd.concat(
                            [context_rows, test_final_df.reset_index(drop=True)],
                            ignore_index=True
                        )

                        Xte_full, yte_full, _, _ = prepare_sequences(
                            df=te_with_ctx,
                            target_col_name=TARGET_TRAIN,
                            horizon=1,
                            feature_subset=list(fitur),
                            scaler_dict=scaler_dict,
                            is_train=False,
                            H_output=h_out_cur,
                            verbose=False,
                        )

                        # Semua window di Xte_full valid untuk test karena context hanya sebagai input historis.
                        Xte = Xte_full
                        yte = yte_full

                        if Xte.shape[0] >= 2:
                            model_te = build_lstm_model_by_backend(
                                Xtr=Xte,
                                params={
                                    "dense_units": int(best_params["dense_units"]),
                                    "hidden_units": int(best_params["hidden_units"]),
                                    "dropout": float(best_params["dropout"]),
                                    "num_layers": int(best_params.get("num_layers", 1)),
                                },
                                H_output=h_out_cur,
                                model_backend=model_backend,
                            ).to(device)
                            model_te.load_state_dict(best_state)
                            model_te.eval()

                            with torch.no_grad():
                                Xte_tensor = torch.tensor(Xte, dtype=torch.float32).to(device)
                                pred_te = model_forward_by_backend(
                                    model=model_te,
                                    x=Xte_tensor,
                                    y=None,
                                    model_backend=model_backend,
                                    is_train=False,
                                ).cpu().numpy()

                            y_true_te_2d = scaler_tgt.inverse_transform(
                                yte.reshape(-1, 1)
                            ).reshape(yte.shape)
                            y_pred_te_2d = scaler_tgt.inverse_transform(
                                pred_te.reshape(-1, 1)
                            ).reshape(pred_te.shape)

                            test_metrics, test_metrics_by_step = compute_multistep_metrics(y_true_te_2d, y_pred_te_2d)

                            print("[EVAL TEST SET - ORIGINAL SCALE]")
                            print(
                                f"  RMSE={test_metrics['RMSE']:.6f} | "
                                f"MAE={test_metrics['MAE']:.6f} | "
                                f"R2={test_metrics['R2']:.6f} | "
                                f"DA={test_metrics['DA']:.4f} | "
                                f"IC={test_metrics['IC']:.4f}"
                            )
                            if h_out_cur > 1 and test_metrics_by_step:
                                first_m = test_metrics_by_step[0]
                                last_m = test_metrics_by_step[-1]
                                print(
                                    f"  Step t+1 RMSE={first_m['RMSE']:.6f} | "
                                    f"Step t+{h_out_cur} RMSE={last_m['RMSE']:.6f}"
                                )

                            # Plot untuk multi-step memakai step-1 agar alignment tanggal/close jelas.
                            if h_out_cur > 1:
                                y_true_plot = y_true_te_2d[:, 0]
                                y_pred_plot = y_pred_te_2d[:, 0]
                                pdf_suffix = "_step1"
                            else:
                                y_true_plot = y_true_te_2d.reshape(-1)
                                y_pred_plot = y_pred_te_2d.reshape(-1)
                                pdf_suffix = ""

                            test_dates = test_final_df[DATE_COL].iloc[N_STEPS:].reset_index(drop=True)
                            test_dates = test_dates.iloc[:len(y_true_plot)]

                            close_ref  = test_final_df[PRICE_COL].iloc[N_STEPS:].reset_index(drop=True)
                            close_ref  = close_ref.iloc[:len(y_true_plot)].values
                            hist_df    = pd.concat([train_final_df, val_final_df], ignore_index=True)
                            dates_hist = hist_df[DATE_COL].reset_index(drop=True)
                            close_hist = hist_df[PRICE_COL].values

                            test_pdf_path = os.path.join(save_dir, f"{t}_test_pred_{TARGET_TRAIN}{pdf_suffix}.pdf")
                            try:
                                save_test_pred_pdf(
                                    y_true_plot, y_pred_plot, test_pdf_path,
                                    dates=test_dates, close_ref=close_ref,
                                    dates_hist=dates_hist, close_hist=close_hist,
                                )
                                print(f"[SAVED] Test pred PDF: {test_pdf_path}")
                            except Exception as e:
                                print(f"[WARN] Gagal simpan test pred PDF: {e}")

                    except Exception as e:
                        print(f"[WARN] Gagal evaluasi test set: {e}")
                # ── Save artifacts ──
                if (not overwrite) and os.path.exists(model_path) and os.path.exists(scaler_path):
                    print(f"[SKIP] Sudah ada model+scaler.")
                else:
                    torch.save(best_state, model_path)
                    joblib.dump(scaler_dict, scaler_path)
                    print(f"[SAVED] Model : {model_path}")
                    print(f"[SAVED] Scaler: {scaler_path}")

                try:
                    save_loss_pdf(history, pdf_path)
                    print(f"[SAVED] Loss PDF: {pdf_path}")
                except Exception as e:
                    pdf_path = None

                log_row = {
                    "H_FWD": H_FWD, "H_output": h_out_cur,
                    "ModelBackend": model_backend,
                    "Ticker": t, "Fitur": comb_name,
                    "SplitMode": split_mode, "TestSize": test_size,
                    "NumLayers": best_num_layers,
                    "Optuna_Params": str(best_params),
                    "Avg_CV_Loss": float(best_avg_loss),
                    "Final_BestValLoss_RMSE": float(best_val_loss),
                    "Final_BestEpoch": int(best_epoch) + 1,
                    "Val_RMSE": metrics["RMSE"], "Val_MAE": metrics["MAE"],
                    "Val_R2": metrics["R2"], "Val_DA": metrics.get("DA"),
                    "Val_IC": metrics.get("IC"),
                    "Val_ByStep_JSON": json.dumps(val_metrics_by_step, ensure_ascii=False),
                    "Test_RMSE": test_metrics["RMSE"], "Test_MAE": test_metrics["MAE"],
                    "Test_R2": test_metrics["R2"], "Test_DA": test_metrics.get("DA"),
                    "Test_IC": test_metrics.get("IC"),
                    "Test_ByStep_JSON": json.dumps(test_metrics_by_step, ensure_ascii=False),
                    "ModelPath": model_path, "ScalerPath": scaler_path, "LossPDF": pdf_path,
                }
                logs.append(log_row)
                append_row_to_csv(log_row, summary_ckpt_csv)

        pd.DataFrame(logs).to_csv(
            os.path.join(horizon_dir, "training_summary.csv"), index=False
        )
        pd.DataFrame(fold_logs).to_csv(
            os.path.join(horizon_dir, "training_folds_metrics.csv"), index=False
        )
        print(f"\n[SUMMARY H={H_FWD}] CSV saved.")

        try:
            build_excel_from_checkpoints(out_dir)
        except Exception as e:
            print(f"[WARN] Gagal rebuild Excel: {e}")

    try:
        build_excel_from_checkpoints(out_dir)
    except Exception as e:
        print(f"[WARN] Gagal rekap gabungan: {e}")
