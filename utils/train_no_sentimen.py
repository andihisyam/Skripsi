# utils/train_no_sentimen.py

import os
from pathlib import Path
from datetime import date
from typing import Optional, Sequence, Tuple, Dict, Any
import logging

import numpy as np
import pandas as pd
import torch
import joblib

from sklearn.metrics import (
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)

from sklearn.model_selection import TimeSeriesSplit

from utils.io_utils import load_prices_from_folder
from utils.preprocess_no_sentimen import (
    add_simple_returns,
    add_indicators,
    prepare_sequences,
)
from utils.model_utils import build_lstm_model
from utils.summarize import summarize_training_results_advanced
from utils.optuna_objective import run_optuna_for_fold

from config import DATE_COL, TARGET_COL, N_STEPS, H_1M, TRAIN_END, VAL_END
from sklearn.model_selection import GridSearchCV


# ======================================================
# ⚙ Hyperparameter Tuning Config (Optuna)
# ======================================================
N_TRIALS_OPTUNA = 30   # jumlah trial Optuna 
MAX_EPOCHS = 100        # Balanced mode

# Parameter GridSearch
param_grid = {
    'dense_units': [32, 64, 128],
    'hidden_units': [32, 64, 128],
    'dropout': [0.1, 0.3, 0.5],
    'lr': [1e-4, 1e-3, 1e-2],
    'batch_size': [16, 32],
    'optimizer': ['Adam', 'AdamW']
}

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)

# ======================================================
# 1️⃣ Kombinasi fitur (tetap seperti versi awal)
# ======================================================
FEATURE_COMBINATIONS: Dict[str, Sequence[str]] = {
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
    ],
}

# Grid Search Function
def run_grid_search_for_fold(Xtr, ytr):
    model = build_lstm_model(input_shape=Xtr.shape, dense_units=64, hidden_units=128, dropout=0.3, output_dim=H_1M)
    
    # Setup GridSearchCV untuk LSTM model
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(Xtr, ytr)
    return grid_search.best_params_


# ======================================================
# 📌 Penentuan Test Size (dynamic)
# ======================================================
def determine_test_size(df: pd.DataFrame) -> Tuple[int, str]:
    """
    Menentukan ukuran test set yang adaptif berdasarkan panjang data
    dan tanggal mulai (khusus dataset mulai 2015-03-02).
    """
    START_DATE_EXPECTED = date(2015, 3, 2)
    MIN_NEEDED = N_STEPS + H_1M

    start_actual = df[DATE_COL].min().date()
    total_rows = len(df)

    adaptive_size = max(int(total_rows * 0.15), MIN_NEEDED + 20)

    if start_actual == START_DATE_EXPECTED and total_rows > 1000:
        return adaptive_size, "adaptive_15%"
    elif total_rows <= MIN_NEEDED * 3:
        return MIN_NEEDED + 20, "min_window_safe"
    else:
        return adaptive_size, "adaptive_default"


# ======================================================
# 📌 Training Utama dengan Optuna
# ======================================================
def train_each(
    data_dir: str = "../Data/Saham",
    out_dir: str = "models",
    tickers_filter: Optional[Sequence[str]] = None,
    subset_filter: Optional[Sequence[str]] = None,
):
    """
    Training LSTM per emiten dan per subset fitur dengan:
    - TimeSeriesSplit (CV)
    - Hyperparameter tuning menggunakan Optuna dan Grid Search
    - Penyimpanan model terbaik per emiten + log hasil ke CSV & Excel
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"[DEVICE] {device}")

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # Load & preprocessing global
    prices = load_prices_from_folder(str(data_path))
    prices = add_simple_returns(prices)
    prices = add_indicators(prices)

    tickers_all = prices["Ticker"].unique()
    if tickers_filter:
        tickers = [t for t in tickers_all if t in tickers_filter]
    else:
        tickers = list(tickers_all)

    logging.info(f"[INFO] Emiten terbaca: {tickers}")

    os.makedirs(out_dir, exist_ok=True)

    logs = []

    # ======================================================
    # LOOP SUBSET FITUR
    # ======================================================
    for comb_name, fitur in FEATURE_COMBINATIONS.items():
        if subset_filter and comb_name not in subset_filter:
            continue

        logging.info(f"\n==============================================")
        logging.info(f"[START] Training subset fitur: {comb_name}")
        logging.info(f"==============================================")

        # LOOP TICKER
        for t in tickers:
            logging.info(f"\n----------------------------------------------")
            logging.info(f"[TICKER] Mulai training untuk: {t}")
            logging.info(f"----------------------------------------------")

            df_t = prices[prices["Ticker"] == t].copy()
            df_t = df_t.dropna(subset=[TARGET_COL])

            if df_t.empty:
                logging.warning(f"  ⚠ Data untuk {t} kosong setelah dropna {TARGET_COL}. Skip.")
                continue

            test_size, split_mode = determine_test_size(df_t)
            logging.info(f"  ▶ Split mode: {split_mode}, test_size={test_size}")

            tscv = TimeSeriesSplit(n_splits=5, test_size=test_size)

            best_model_state_global = None
            best_val_loss_global = float("inf")
            best_scaler_dict_global = None

            # LOOP FOLD
            for fold, (tr_idx, va_idx) in enumerate(tscv.split(df_t)):
                logging.info(f"\n  ▶ [Fold {fold+1}/5] Train={len(tr_idx)}, Val={len(va_idx)}")

                tr = df_t.iloc[tr_idx]
                va = df_t.iloc[va_idx]

                try:
                    Xtr, ytr, scaler_dict, _ = prepare_sequences(
                        tr, TARGET_COL, H_1M, fitur
                    )
                    Xva, yva, _, _ = prepare_sequences(
                        va, TARGET_COL, H_1M, fitur, scaler_dict=scaler_dict
                    )
                except Exception as e:
                    logging.warning(f"  ⚠ SKIP Fold {fold+1} ({t}, {comb_name}): {e}")
                    continue

                if Xtr.shape[0] < 10 or Xva.shape[0] < 5:
                    logging.warning(f"  ⚠ SKIP Fold {fold+1}: data sequence terlalu sedikit.")
                    continue

                input_shape = (N_STEPS, Xtr.shape[-1])

                # ==============================================
                # 🔍 Optuna tuning untuk 1 fold
                # ==============================================
                logging.info(f"    ▶ Mulai tuning dengan Optuna untuk Fold {fold+1}...")
                best_result = run_optuna_for_fold(
                    input_shape=input_shape,
                    Xtr=Xtr,
                    ytr=ytr,
                    Xva=Xva,
                    yva=yva,
                    device=device,
                    H_output=H_1M,
                    trials=N_TRIALS_OPTUNA,
                    MAX_EPOCHS=MAX_EPOCHS,
                )

                fold_best_loss = float(best_result["loss"])
                fold_best_epoch = int(best_result["epoch"])
                fold_best_pred = best_result["pred"]
                fold_best_params = best_result["params"]
                fold_best_state = best_result.get("state_dict", None)

                logging.info(f"    ✔ Fold {fold+1} selesai — Loss terbaik={fold_best_loss:.6f} di Epoch {fold_best_epoch+1}")
                logging.info(f"      Param terbaik: {fold_best_params}")

                # ==============================================
                # 🔍 Grid Search tuning untuk 1 fold
                # ==============================================
                logging.info(f"    ▶ Mulai tuning dengan Grid Search untuk Fold {fold+1}...")
                best_params_grid_search = run_grid_search_for_fold(Xtr, ytr)

                logging.info(f"    ✔ Grid Search result: {best_params_grid_search}")

                # ==============================================
                # 🔢 Hitung metrik evaluasi per fold
                # ==============================================
                y_true = yva.reshape(-1)
                y_pred = np.array(fold_best_pred).reshape(-1)

                val_rmse = root_mean_squared_error(y_true, y_pred)
                val_mape = mean_absolute_percentage_error(y_true, y_pred)
                val_r2 = r2_score(y_true, y_pred)
                val_mae = float(np.mean(np.abs(y_true - y_pred)))
                val_loss = float(fold_best_loss)  # sama dengan RMSELoss terbaik

                logs.append(
                    {
                        "Ticker": t,
                        "Fitur": comb_name,
                        "Fold": fold + 1,
                        "Dense_Units": int(fold_best_params["dense_units"]),
                        "Hidden_Units": int(fold_best_params["hidden_units"]),
                        "Dropout": float(fold_best_params["dropout"]),
                        "LR": float(fold_best_params["lr"]),
                        "Batch_Size": int(fold_best_params["batch_size"]),
                        "Optimizer": str(fold_best_params["optimizer"]),
                        "Best_Epoch": fold_best_epoch + 1,
                        "Val_Loss": val_loss,
                        "Val_MAE": val_mae,
                        "Val_MAPE": float(val_mape),
                        "Val_RMSE": float(val_rmse),
                        "Val_R2": float(val_r2),
                    }
                )

                # Simpan model terbaik global per ticker & fitur
                if fold_best_state is not None and fold_best_loss < best_val_loss_global:
                    best_val_loss_global = fold_best_loss
                    best_model_state_global = fold_best_state
                    best_scaler_dict_global = scaler_dict

            # Save model terbaik per ticker & subset fitur
            if best_model_state_global is not None:
                save_dir = os.path.join(out_dir, comb_name)
                os.makedirs(save_dir, exist_ok=True)

                model_path = os.path.join(save_dir, f"{t}_best_price.pt")
                scaler_path = os.path.join(save_dir, f"{t}_scaler.pkl")

                torch.save(best_model_state_global, model_path)
                joblib.dump(best_scaler_dict_global, scaler_path)

                logging.info(f"\n  💾 [SAVED] Model terbaik untuk {t}: {model_path}")
            else:
                logging.warning(f"\n  ⚠ Tidak ada model valid yang disimpan untuk {t} ({comb_name}).")

    # ======================================================
    # Simpan log seluruh training ke CSV + Excel summary
    # ======================================================
    summary_path = os.path.join(out_dir, "training_summary.csv")
    pd.DataFrame(logs).to_csv(summary_path, index=False)

    try:
        summarize_training_results_advanced(
            summary_path,
            os.path.join(out_dir, "training_results_complete.xlsx"),
        )
        logging.info("\n[LOG] Excel summary lengkap dibuat.")
    except Exception as e:
        logging.warning("[WARN] Gagal membuat summary Excel:", e)
