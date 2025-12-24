# utils/train_no_sentimen.py
import os
from pathlib import Path
from typing import Tuple
from datetime import date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import joblib

from sklearn.metrics import (
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import TimeSeriesSplit

import torch
import torch.nn as nn
import torch.optim as optim

from utils.io_utils import load_prices_from_folder
from utils.preprocess_no_sentimen import (
    add_simple_returns,
    add_indicators,
    prepare_sequences,
)
from utils.summarize import summarize_training_results_advanced

from config import DATE_COL, TARGET_COL, N_STEPS, H_1M, TRAIN_END, VAL_END




# ======================================================
# 📌 RMSE Loss Function 
# ======================================================
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))

# ======================================================
# 1️⃣ Kombinasi fitur (tetap sama seperti versi Keras)
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
# 📌 Penentuan Test Size (same logic)
# ======================================================
def determine_test_size(df: pd.DataFrame):
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
# 📌 Model LSTM base
# ======================================================
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, dense_units, output_dim, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, dense_units)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(dense_units, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc_out(out)
        return out

# ======================================================
# 📌 Build LSTM Model 
# ======================================================
def build_lstm_model(input_shape, hidden_units, dense_units, dropout, output_dim):
    num_features = input_shape[-1]
    return LSTMRegressor(
        input_dim=num_features,
        hidden_dim=hidden_units,
        dense_units=dense_units,
        output_dim=output_dim,
        dropout=dropout,  # pastikan LSTMRegressor menerima dropout (lihat catatan di bawah)
    )


# ======================================================
# 📌 Train 1 Fold (RMSE + logging)
# ======================================================
def train_one_fold(
    model,
    Xtr, ytr,
    Xva, yva,
    device,
    num_epochs,
    batch_size,
    lr,
    optimizer_name="Adam",
    fold=None,
    ticker=None,
    comb_name=None,
):
    print(f"      ▶ Training dimulai | Ticker={ticker} | Subset={comb_name} | Fold={fold}")
    print(f"        Hyperparams: epochs={num_epochs}, batch={batch_size}, lr={lr}, opt={optimizer_name}")

    model = model.to(device)

    # Optimizer
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Optimizer tidak dikenali: {optimizer_name}")

    criterion = RMSELoss()

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32).to(device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32).to(device)
    Xva_t = torch.tensor(Xva, dtype=torch.float32).to(device)
    yva_t = torch.tensor(yva, dtype=torch.float32).to(device)

    n_train = Xtr_t.size(0)

    train_losses, val_losses, val_maes = [], [], []

    best_val_loss = float("inf")
    best_epoch = 0
    best_pred = None
    best_state = None

    for epoch in range(num_epochs):
        # TRAIN
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            batch_X = Xtr_t[idx]
            batch_y = ytr_t[idx]

            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_X.size(0)

        epoch_loss /= n_train
        train_losses.append(epoch_loss)

        # VALIDATION
        model.eval()
        with torch.no_grad():
            pred_va = model(Xva_t)
            vloss = criterion(pred_va, yva_t).item()
            val_losses.append(vloss)

            mae = torch.mean(torch.abs(pred_va - yva_t)).item()
            val_maes.append(mae)

            if vloss < best_val_loss:
                best_val_loss = vloss
                best_epoch = epoch
                best_pred = pred_va.detach().cpu().numpy()
                best_state = model.state_dict()

        if (epoch + 1) % 10 == 0:
            print(
                f"        [Fold {fold}] Epoch {epoch+1}/{num_epochs} "
                f"| Train={epoch_loss:.6f} | Val(RMSE)={vloss:.6f} | MAE={mae:.6f}"
            )

    print(f"      ✔ Fold {fold} selesai — Best Val(RMSE)={best_val_loss:.6f} di Epoch {best_epoch+1}")

    return train_losses, val_losses, val_maes, best_epoch, best_pred, best_state

# ======================================================
# 📌 Training Utama
# ======================================================
def train_each(data_dir="../Data/Saham", out_dir="models",
               tickers_filter=None, subset_filter=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    DATA_DIR = Path(data_dir)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    prices = load_prices_from_folder(str(DATA_DIR))
    prices = add_simple_returns(prices)
    prices = add_indicators(prices)

    tickers = prices["Ticker"].unique()
    if tickers_filter:
        tickers = [t for t in tickers if t in tickers_filter]

    print(f"[INFO] Emiten terbaca: {tickers}")

    os.makedirs(out_dir, exist_ok=True)
    logs = []

    # ======================================================
    # LOOP SUBSET FITUR
    # ======================================================
    for comb_name, fitur in FEATURE_COMBINATIONS.items():
        if subset_filter and comb_name not in subset_filter:
            continue

        print("\n==============================================")
        print(f"[START] Training subset fitur: {comb_name}")
        print("==============================================")

        pdf_path = os.path.join(out_dir, f"loss_curves_{comb_name}.pdf")
        pdf = PdfPages(pdf_path)

        # LOOP TICKER
        for t in tickers:
            print("\n----------------------------------------------")
            print(f"[TICKER] Mulai training untuk: {t}")
            print("----------------------------------------------")

            df_t = prices[prices["Ticker"] == t].copy()
            df_t = df_t.dropna(subset=[TARGET_COL])

            test_size, split_mode = determine_test_size(df_t)
            tscv = TimeSeriesSplit(n_splits=5, test_size=test_size)

            best_model_state = None
            best_val_loss_global = float("inf")
            best_scaler_dict = None

            # LOOP FOLD
            for fold, (tr_idx, va_idx) in enumerate(tscv.split(df_t)):
                print(f"\n  ▶ [Fold {fold+1}/5] Train={len(tr_idx)}, Val={len(va_idx)}")

                tr = df_t.iloc[tr_idx]
                va = df_t.iloc[va_idx]

                try:
                    Xtr, ytr, scaler_dict, _ = prepare_sequences(tr, TARGET_COL, H_1M, fitur)
                    Xva, yva, _, _ = prepare_sequences(va, TARGET_COL, H_1M, fitur, scaler_dict=scaler_dict)
                except Exception as e:
                    print(f"  ⚠ SKIP Fold {fold+1}: {e}")
                    continue

                fold_best_loss = float("inf")
                fold_best_dense = None
                fold_best_pred = None
                fold_best_epoch = None
                fold_best_state = None

                # ======================================================
                # DENSE UNITS TUNING
                # ======================================================
                for dense_units in DENSE_CANDIDATES:
                    model = build_lstm_model((N_STEPS, Xtr.shape[-1]), dense_units)

                    (
                        train_losses,
                        val_losses,
                        val_maes,
                        best_epoch,
                        yva_pred_best,
                    ) = train_one_fold(
                        model, Xtr, ytr, Xva, yva, device=device,
                        dense_units=dense_units, fold=fold+1, ticker=t
                    )

                    vloss_best = val_losses[best_epoch]
                    mae_best = val_maes[best_epoch]

                    if vloss_best < fold_best_loss:
                        fold_best_loss = vloss_best
                        fold_best_dense = dense_units
                        fold_best_pred = yva_pred_best
                        fold_best_epoch = best_epoch
                        fold_best_state = model.state_dict()
                    # =============================
                    # LOG SEMUA DENSE UNITS
                    # =============================
                    logs.append({
                        "Ticker": t,
                        "Fitur": comb_name,
                        "Fold": fold+1,
                        "Dense_Units": dense_units,
                        "Best_Epoch": best_epoch+1,
                        "Val_Loss": float(vloss_best),
                        "Val_MAE": float(mae_best),
                        "Val_RMSE": float(root_mean_squared_error(yva, yva_pred_best)),
                        "Val_MAPE": float(mean_absolute_percentage_error(yva, yva_pred_best)),
                        "Val_R2": float(r2_score(yva, yva_pred_best)),
})

                print(f"  ✔ Fold {fold+1} selesai — Dense terbaik={fold_best_dense}, Loss={fold_best_loss:.6f}")

                plt.figure()
                plt.plot(train_losses, label="Train")
                plt.plot(val_losses, label="Val")
                plt.axvline(fold_best_epoch, linestyle="--", color="red", label="Best")
                plt.title(f"{t} - {comb_name} Fold {fold+1} | Dense={fold_best_dense}")
                plt.legend()
                pdf.savefig()
                plt.close()

                if fold_best_loss < best_val_loss_global:
                    best_val_loss_global = fold_best_loss
                    best_model_state = fold_best_state
                    best_scaler_dict = scaler_dict

            # Save model terbaik per ticker
            if best_model_state is not None:
                save_dir = os.path.join(out_dir, comb_name)
                os.makedirs(save_dir, exist_ok=True)

                model_path = os.path.join(save_dir, f"{t}_best_price.pt")
                scaler_path = os.path.join(save_dir, f"{t}_scaler.pkl")

                torch.save(best_model_state, model_path)
                joblib.dump(best_scaler_dict, scaler_path)

                print(f"\n  💾 [SAVED] Model terbaik untuk {t}: {model_path}")

        pdf.close()

    summary_path = os.path.join(out_dir, "training_summary.csv")
    pd.DataFrame(logs).to_csv(summary_path, index=False)

    try:
        summarize_training_results_advanced(
            summary_path,
            os.path.join(out_dir, "training_results_with_epoch.xlsx"),
        )
        print("\n[LOG] Excel summary dibuat.")
    except Exception as e:
        print("[WARN] Gagal membuat summary Excel.",e)
