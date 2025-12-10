# utils/train_multi_no_sentimen.py
import os
from pathlib import Path
import matplotlib.pyplot as plt
import logging
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.backends.backend_pdf import PdfPages

from utils.preprocess_multi_no_sentimen import (
    add_indicators_multi,
    add_simple_returns_multi,
    prepare_sequences_multi
)
from config import TARGET_COL, N_STEPS, H_1M

# Import model dari training single-stock
from utils.train_no_sentimen_torch import (
    FEATURE_COMBINATIONS
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error
)
logger = logging.getLogger(__name__)

# ======================================================
# MODEL LSTM PYTORCH
# ======================================================
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dense_units: int, output_dim: int):
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
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        out = out[:, -1, :]      # timestep terakhir
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc_out(out)   # (batch, output_dim = H_1M)
        return out


def build_lstm_model_pytorch(num_features: int) -> nn.Module:
    """
    Meniru aturan di Keras:
      - hidden_dim = 64 jika fitur < 10, else 128
      - dense_units = 512 jika fitur < 10, else 64
    """
    if num_features < 10:
        hidden_dim = 64
        dense_units = 512
    else:
        hidden_dim = 128
        dense_units = 64

    model = LSTMRegressor(
        input_dim=num_features,
        hidden_dim=hidden_dim,
        dense_units=dense_units,
        output_dim=H_1M,
    )
    return model
# ===================================================================
# TRAIN MULTI-EMITEN dengan 80/20 SPLIT (Scaling Global)
# ===================================================================
def train_multi(
    data_dir: str = "../Data/Saham",
    out_dir: str = "models_multi",
    tickers=None,
    subset_features=None,
    epochs: int = 100,
    batch_size: int = 32,
):
    logger.info("=== MODE: TRAIN MULTI-EMITEN (80/20 SPLIT) ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[DEVICE] using: {device}")

    # ---------------------------------------------------------------
    # VALIDASI INPUT
    # ---------------------------------------------------------------
    DATA_DIR = Path(data_dir)
    if not DATA_DIR.exists():
        raise SystemExit(f"❌ Folder tidak ditemukan: {DATA_DIR.resolve()}")

    if not tickers or len(tickers) < 2:
        raise SystemExit("❌ Minimal pilih 2 emiten untuk training multi-stock!")

    logger.info(f"[INFO] Emiten yang digunakan: {tickers}")

    # ---------------------------------------------------------------
    # Jika subset tidak diberikan, gunakan SEMUA subset fitur
    # ---------------------------------------------------------------
    if subset_features is None:
        subset_features = list(FEATURE_COMBINATIONS.keys())
        logger.info(f"[INFO] Training SEMUA subset fitur: total {len(subset_features)} subset")
    else:
        logger.info(f"[INFO] Training subset fitur tertentu: {subset_features}")

    os.makedirs(out_dir, exist_ok=True)

    summary_logs = []

    # ===============================================================
    # LOOP SETIAP SUBSET FITUR
    # ===============================================================
    for comb_name in subset_features:

        fitur = FEATURE_COMBINATIONS.get(comb_name)
        if fitur is None:
            logger.warning(f"[SKIP] Subset '{comb_name}' tidak ditemukan di FEATURE_COMBINATIONS!")
            continue

        logger.info("====================================================================")
        logger.info(f"[START] Training subset fitur: {comb_name}")
        logger.info(f"[FITUR] {fitur}")
        logger.info("====================================================================")

        # ===================================================================
        # STEP 1 — Load hanya file ticker yang dipilih lalu gabungkan
        # ===================================================================
        dfs = []
        for t in tickers:
            fpath = DATA_DIR / f"{t}.csv"
            if not fpath.exists():
                raise FileNotFoundError(f"File tidak ditemukan: {fpath}")
            df_t = pd.read_csv(fpath)
            df_t["Ticker"] = t
            dfs.append(df_t)
            logger.info(f"[LOAD] {t}: {len(df_t)} baris")

        df_all = pd.concat(dfs, ignore_index=True)
        logger.info(f"[INFO] Total baris gabungan: {len(df_all)}")

        # Tambahkan indikator (per-Ticker)
        df_all = add_simple_returns_multi(df_all)
        df_all = add_indicators_multi(df_all)
        df_all = df_all.dropna(subset=[TARGET_COL]).reset_index(drop=True)

        # ===================================================================
        # STEP 2 — Fit SCALER GLOBAL SEKALI saja di df_all
        # ===================================================================
        try:
            _, _, scaler_global, _ = prepare_sequences_multi(
                df_all,
                TARGET_COL,
                fitur,
                scaler_dict=None,      # fit scaler global
                horizon=H_1M,
            )
        except Exception as e:
            logger.warning(f"[SKIP] Gagal fit scaler global untuk subset {comb_name}: {e}")
            continue

        # ===================================================================
        # STEP 3 — Windowing per-EMITEN dengan scaler_global (transform only)
        # ===================================================================
        X_list, y_list = [], []

        for t in tickers:
            df_t = df_all[df_all["Ticker"] == t].copy()
            if df_t.shape[0] < (N_STEPS + H_1M):
                logger.warning(f"[SKIP] {t}: data terlalu pendek untuk windowing.")
                continue

            try:
                X_t, y_t, _, _ = prepare_sequences_multi(
                    df_t,
                    TARGET_COL,
                    fitur,
                    scaler_dict=scaler_global,  # pakai scaler global (fit=False)
                    horizon=H_1M,
                )
            except Exception as e:
                logger.warning(f"[SKIP] Error ketika windowing {t}: {e}")
                continue

            logger.info(f"[{t}] Sequence: {len(X_t)}")
            X_list.append(X_t)
            y_list.append(y_t)

        if len(X_list) == 0:
            logger.warning(f"[SKIP] Tidak ada sequence untuk subset {comb_name}")
            continue

        X_all = np.concatenate(X_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)

        n_samples = X_all.shape[0]
        logger.info(f"[INFO] Total sequence training (gabungan): {n_samples}")

        if n_samples < 20:
            logger.warning(f"[SKIP] Sequence terlalu sedikit (<20) untuk subset {comb_name}")
            continue

        # ===================================================================
        # STEP 4 — 80/20 Split (berdasarkan urutan sequence)
        # ===================================================================
        split_idx = int(n_samples * 0.8)
        X_train = X_all[:split_idx]
        y_train = y_all[:split_idx]
        X_val = X_all[split_idx:]
        y_val = y_all[split_idx:]

        logger.info(f"[SPLIT] Train={X_train.shape[0]} seq, Val={X_val.shape[0]} seq")

        # ===================================================================
        # STEP 5 — Build model & siapkan tensor
        # ===================================================================
        num_features = X_all.shape[-1]
        model = build_lstm_model_pytorch(num_features).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        X_tr_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_tr_t = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_va_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_va_t = torch.tensor(y_val, dtype=torch.float32).to(device)

        n_train = X_tr_t.size(0)

        logger.info(f"[TRAIN] Mulai training PyTorch untuk subset {comb_name} (epochs={epochs})")

        best_val_loss = float("inf")
        best_state = None
        best_epoch = -1
        history_train = []
        history_val = []

        # ===================================================================
        # STEP 6 — TRAINING LOOP
        # ===================================================================
        for epoch in range(epochs):

            perm = torch.randperm(n_train)
            X_tr_epoch = X_tr_t[perm]
            y_tr_epoch = y_tr_t[perm]

            batch_losses = []

            for i in range(0, n_train, batch_size):
                batch_X = X_tr_epoch[i:i + batch_size]
                batch_y = y_tr_epoch[i:i + batch_size]

                optimizer.zero_grad()
                out = model(batch_X)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

                batch_losses.append(loss.item())

            epoch_train_loss = float(np.mean(batch_losses))

            with torch.no_grad():
                out_val = model(X_va_t)
                epoch_val_loss = float(criterion(out_val, y_va_t).item())

            history_train.append(epoch_train_loss)
            history_val.append(epoch_val_loss)

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_state = model.state_dict()
                best_epoch = epoch + 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"[{comb_name}] Epoch {epoch+1}/{epochs} → "
                    f"Train_Loss={epoch_train_loss:.6f}, Val_Loss={epoch_val_loss:.6f}"
                )

        logger.info(
            f"[DONE] Training selesai untuk subset {comb_name}, "
            f"best_epoch={best_epoch}, best_val_loss={best_val_loss:.6f}"
        )

        # ===================================================================
        # STEP 7 — EVALUASI METRIK PADA MODEL TERBAIK
        # ===================================================================
        model.load_state_dict(best_state)

        with torch.no_grad():
            y_val_pred_t = model(X_va_t)
            y_val_pred = y_val_pred_t.cpu().numpy()
            y_val_true = y_va_t.cpu().numpy()

        val_mae = mean_absolute_error(y_val_true, y_val_pred)
        val_mape = mean_absolute_percentage_error(y_val_true, y_val_pred)
        val_rmse = root_mean_squared_error(y_val_true, y_val_pred)
        val_r2 = r2_score(y_val_true, y_val_pred)

        logger.info(
            f"[METRIC] {comb_name} → "
            f"MAE={val_mae:.6f}, MAPE={val_mape:.6f}, "
            f"RMSE={val_rmse:.6f}, R2={val_r2:.6f}"
        )

        summary_logs.append({
            "Subset": comb_name,
            "Tickers": ",".join(tickers),
            "Num_Seq_Total": int(n_samples),
            "Train_Seq": int(X_train.shape[0]),
            "Val_Seq": int(X_val.shape[0]),
            "Best_Epoch": int(best_epoch),
            "Val_Loss": float(best_val_loss),
            "Val_MAE": float(val_mae),
            "Val_MAPE": float(val_mape),
            "Val_RMSE": float(val_rmse),
            "Val_R2": float(val_r2),
        })

        # ===================================================================
        # STEP 8 — SAVE MODEL + SCALER + LOSS CURVE
        # ===================================================================
        save_dir = os.path.join(out_dir, comb_name)
        os.makedirs(save_dir, exist_ok=True)

        model_path = os.path.join(save_dir, "multi_best_price.pt")
        scaler_path = os.path.join(save_dir, "multi_scaler.pkl")
        pdf_path = os.path.join(save_dir, f"loss_curve_{comb_name}.pdf")

        torch.save(best_state, model_path)
        joblib.dump(scaler_global, scaler_path)

        logger.info(f"[SAVE] Model → {model_path}")
        logger.info(f"[SAVE] Scaler → {scaler_path}")

        try:
            with PdfPages(pdf_path) as pdf:
                plt.figure()
                plt.plot(history_train, label="Train Loss")
                plt.plot(history_val, label="Val Loss")
                plt.axvline(best_epoch - 1, linestyle="--", label="Best Epoch")
                plt.xlabel("Epoch")
                plt.ylabel("MSE Loss")
                plt.title(f"Loss Curve Multi-Emiten ({comb_name})")
                plt.legend()
                plt.grid(alpha=0.3)
                pdf.savefig()
                plt.close()
            logger.info(f"[SAVE] Loss curve PDF → {pdf_path}")
        except Exception as e:
            logger.warning(f"[WARNING] Gagal menyimpan loss curve PDF untuk {comb_name}: {e}")

    # ===================================================================
    # STEP 9 — SIMPAN SUMMARY GLOBAL
    # ===================================================================
    if len(summary_logs) > 0:
        summary_df = pd.DataFrame(summary_logs)
        csv_path = os.path.join(out_dir, "summary_multi_training.csv")
        xlsx_path = os.path.join(out_dir, "summary_multi_training.xlsx")

        summary_df.to_csv(csv_path, index=False)
        try:
            summary_df.to_excel(xlsx_path, index=False)
        except Exception as e:
            logger.warning(f"[WARNING] Gagal menyimpan summary ke Excel: {e}")

        logger.info(f"[SUMMARY] CSV → {csv_path}")
        logger.info(f"[SUMMARY] XLSX → {xlsx_path}")
    else:
        logger.warning("[SUMMARY] Tidak ada subset yang berhasil ditraining. Summary tidak dibuat.")

    logger.info("=== SEMUA TRAINING MULTI-EMITEN (80/20) SELESAI ===")