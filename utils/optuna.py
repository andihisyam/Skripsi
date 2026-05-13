import optuna
import pandas as pd
from optuna.pruners import MedianPruner
import torch
from datetime import date
from typing import Tuple, Optional, List, Dict
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils.model_utils_seq2seq import build_lstm_model as build_lstm_model_seq2seq
from utils.model_utils_sklearn import LSTMSklearnNet
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, root_mean_squared_error
from utils.debug_training import debug_trial, debug_sequences
from config import DATE_COL, TARGET_COL, N_STEPS
from utils.preprocess_sentimen import prepare_sequences


def build_lstm_model_by_backend(
    Xtr,
    params,
    H_output: int,
    model_backend: str = "seq2seq",
):
    backend = str(model_backend).lower()
    if backend == "lstm_sklearn":
        return LSTMSklearnNet(
            input_size=Xtr.shape[2],
            dense_units=params["dense_units"],
            hidden_units=params["hidden_units"],
            dropout=params["dropout"],
            output_dim=H_output,
            num_layers=params.get("num_layers", 1),
        )

    return build_lstm_model_seq2seq(
        Xtr,
        dense_units=params["dense_units"],
        hidden_units=params["hidden_units"],
        dropout=params["dropout"],
        output_dim=H_output,
        num_layers=params.get("num_layers", 1),
    )


def model_forward_by_backend(model, x, y=None, model_backend: str = "seq2seq", is_train: bool = False):
    backend = str(model_backend).lower()
    if backend == "lstm_sklearn":
        return model(x)

    if is_train and y is not None:
        return model(
            x,
            teacher_forcing_targets=y.view(y.size(0), -1),
            teacher_forcing_ratio=0.5,
        )
    return model(x, teacher_forcing_targets=None, teacher_forcing_ratio=0.0)


# ======================================================
# 📌 Penentuan Test Size (dynamic)
# ======================================================
def determine_test_size(df: pd.DataFrame, horizon: int) -> Tuple[int, str]:
    START_DATE_EXPECTED = date(2015, 3, 2)
    MIN_NEEDED = N_STEPS + horizon

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
# ✅ Helper: Validation dengan konteks (tail train + val)
# ======================================================
def prepare_val_with_context(
    tr_df: pd.DataFrame,
    va_df: pd.DataFrame,
    target_col_name: str,
    horizon: int,
    feature_subset: List[str],
    n_steps: int = N_STEPS,
    context_extra: int = 50,
    h_output: int = 1,          # ← BARU: pass ke prepare_sequences
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict, List[str]]:
    """
    Scaler di-fit HANYA pada tr_df, val di-transform dengan scaler train.
    h_output=1  → target shape (N, 1)
    h_output>1  → target shape (N, h_output) — multi-step
    """
    Xtr, ytr, scaler_dict, cols = prepare_sequences(
        df=tr_df,
        target_col_name=target_col_name,
        horizon=horizon,
        feature_subset=feature_subset,
        scaler_dict=None,
        is_train=True,
        H_output=h_output,
        verbose=False,
    )
 
    context_rows = tr_df.iloc[-context_extra:] if context_extra > 0 else pd.DataFrame()
    va_with_ctx  = pd.concat([context_rows, va_df], ignore_index=True)
 
    Xva_full, yva_full, _, _ = prepare_sequences(
        df=va_with_ctx,
        target_col_name=target_col_name,
        horizon=horizon,
        feature_subset=feature_subset,
        scaler_dict=scaler_dict,
        is_train=False,
        H_output=h_output,
        verbose=False,
    )
 
    n_ctx_windows = max(0, context_extra - n_steps + 1)
    Xva = Xva_full[n_ctx_windows:]
    yva = yva_full[n_ctx_windows:]
 
    return Xtr, ytr, Xva, yva, scaler_dict, cols


# ======================================================
# ✅ Helper: inverse target agar metrik di skala asli
# ======================================================
def inverse_target(
    y_scaled: np.ndarray,
    scaler_dict: Dict,
    target_col_name: str,
) -> np.ndarray:
    """
    Kembalikan prediksi ke skala asli (log return).
 
    FIX Bug 4: key scaler menggunakan f"target_{target_col_name}"
    yang konsisten dengan cara fit di prepare_sequences.
    """
    # FIX Bug 4: key harus sama persis dengan yang dipakai saat fit
    target_key = f"target_{target_col_name}"
 
    if target_key not in scaler_dict:
        raise KeyError(
            f"[inverse_target] key '{target_key}' tidak ditemukan di scaler_dict.\n"
            f"Keys yang tersedia: {list(scaler_dict.keys())}"
        )
 
    scaler  = scaler_dict[target_key]
    y_real  = scaler.inverse_transform(
        np.asarray(y_scaled).reshape(-1, 1)
    ).flatten()
 
    return y_real


# ======================================================
# Training 1 fold
# ======================================================
def train_model_cv(
    Xtr, ytr, Xva, yva,
    device,
    params,
    H_output,
    MAX_EPOCHS=80,
    verbose=True,
    prefix="",
    trial = None,
    model_backend: str = "seq2seq",
):
    """
    Train LSTM dengan mini-batch + shuffle per epoch.
 
    Fix yang diterapkan:
      1. Shuffle index setiap epoch  → model tidak hafal urutan batch
      2. batch_y.view(pred.shape)    → tidak ada silent broadcasting di HuberLoss
      3. CosineAnnealingLR scheduler → lr turun bertahap sepanjang epoch
    """
    model = build_lstm_model_by_backend(
        Xtr=Xtr,
        params=params,
        H_output=H_output,
        model_backend=model_backend,
    ).to(device)
 
    optimizer_name = str(params.get("optimizer", "AdamW")).lower()
    lr = float(params["lr"])
    weight_decay = float(params.get("weight_decay", 1e-3))

    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = nn.HuberLoss(delta=float(params.get("huber_delta", 1.0)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max = MAX_EPOCHS,
        eta_min = 1e-6,
    )
 
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
    yva_t = torch.tensor(yva, dtype=torch.float32, device=device)
 
    n_train    = Xtr_t.size(0)
    batch_size = params["batch_size"]
 
    best_val_loss = float("inf")
    best_state    = None
    best_pred_va  = None
    best_epoch    = -1
 
    history = {"train_loss": [], "val_loss": []}
 
    for epoch in range(MAX_EPOCHS):
        model.train()
        train_losses = []
 
        perm         = torch.randperm(n_train, device=device)
        Xtr_shuffled = Xtr_t[perm]
        ytr_shuffled = ytr_t[perm]
 
        for i in range(0, n_train, batch_size):
            optimizer.zero_grad(set_to_none=True)
 
            end     = min(i + batch_size, n_train)
            batch_X = Xtr_shuffled[i:end]
            batch_y = ytr_shuffled[i:end]

            pred = model_forward_by_backend(
                model=model,
                x=batch_X,
                y=batch_y,
                model_backend=model_backend,
                is_train=True,
            )
            loss = criterion(pred, batch_y.view(pred.shape))
            loss.backward()
 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
 
        epoch_train_loss = float(np.mean(train_losses)) if train_losses else np.nan
 
        model.eval()
        with torch.no_grad():
            pred_va = model_forward_by_backend(
                model=model,
                x=Xva_t,
                y=None,
                model_backend=model_backend,
                is_train=False,
            )
            vloss   = criterion(pred_va, yva_t.view(pred_va.shape)).item()
 
        scheduler.step()
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(vloss)

    
        if vloss < best_val_loss:
            best_val_loss = vloss
            best_epoch    = epoch
            best_state    = {k: v.cpu() for k, v in model.state_dict().items()}
            best_pred_va  = pred_va.cpu().numpy()
 
        if trial is not None:
            trial.report(vloss, step=epoch)
            if trial.should_prune():
                if verbose:
                    print(f"{prefix} Pruned at epoch {epoch+1}")
                raise optuna.TrialPruned()
            
        if verbose and (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"{prefix} Ep {epoch+1:03d} | "
                f"Tr={epoch_train_loss:.6f} | "
                f"Val={vloss:.6f} | "
                f"Best={best_val_loss:.6f} (ep {best_epoch+1}) | "
                f"LR={current_lr:.2e}"
            )
 
    return best_val_loss, best_state, best_pred_va, history, best_epoch


# ======================================================
# Optuna + TimeSeries CV
# ======================================================
def run_optuna_cv(
    df_t,
    fitur,
    device,
    H_output,
    target_col_name: str = TARGET_COL,
    horizon: int = 1,
    trials: int = 30,
    MAX_EPOCHS: int = 50,
    verbose: bool = True,
    return_best_fold_histories: bool = False,
    return_best_fold_metrics: bool = False,
    study_db_path: str = "optuna_studies.db",
    study_name: Optional[str] = None,
    context_extra: int = 50,
    metrics_in_original_scale: bool = True,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    h_output: int = 1,
    model_backend: str = "seq2seq",
):

    def eval_metrics_np(y_true, y_pred):
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)
        return {
            "RMSE": float(root_mean_squared_error(y_true, y_pred)),
            "MAE":  float(np.mean(np.abs(y_true - y_pred))),
            "R2":   float(r2_score(y_true, y_pred)),
        }
 
    # ======================================================
    # Split sekali, fixed — tidak berubah antar trial
    # ======================================================
    n       = len(df_t)
    n_train = int(train_ratio * n)
    n_val   = int(val_ratio * n)
 
    tr_df = df_t.iloc[:n_train].copy()
    va_df = df_t.iloc[n_train:n_train + n_val].copy()
 
    if verbose:
        print(
            f"[SPLIT] train={len(tr_df)} | val={len(va_df)} | "
            f"test={n - n_train - n_val} rows "
            f"({train_ratio:.0%} / {val_ratio:.0%} / {1-train_ratio-val_ratio:.0%})"
        )
 
    # ======================================================
    # Prepare sequences — scaler fit SEKALI di tr_df
    # ======================================================
    try:
        Xtr, ytr, Xva, yva, scaler, cols = prepare_val_with_context(
            tr_df=tr_df,
            va_df=va_df,
            target_col_name=target_col_name,
            horizon=horizon,
            feature_subset=fitur,
            n_steps=N_STEPS,
            context_extra=context_extra,
            h_output=h_output,
        )
    except Exception as e:
        raise RuntimeError(f"[run_optuna_cv] prepare_val_with_context gagal: {e}")
 
    if len(Xtr) == 0 or len(Xva) == 0:
        raise ValueError(
            f"[run_optuna_cv] Sequence kosong — "
            f"Xtr={len(Xtr)}, Xva={len(Xva)}. Data mungkin terlalu sedikit."
        )
 
    debug_sequences(Xtr, ytr, Xva, yva, target_col_name, prefix="[OPTUNA] ")
 
    if verbose:
        print(f"[SEQUENCES] Xtr={Xtr.shape} | Xva={Xva.shape}")
 
    # ======================================================
    # Best-trial collectors
    # ======================================================
    best_trial_value   = float("inf")
    best_trial_history = None
    best_trial_metrics = None
 
    # ======================================================
    # Objective — 1 trial = 1 training run
    # ======================================================
    def objective(trial):
        nonlocal best_trial_value, best_trial_history, best_trial_metrics
 
        params = {
            "hidden_units": trial.suggest_categorical("hidden_units", [32, 64, 96, 128]),
            "dense_units":  trial.suggest_categorical("dense_units",  [16, 32, 64]),
            "dropout":      trial.suggest_float("dropout", 0.1, 0.4),
            "lr":           trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "batch_size":   trial.suggest_categorical("batch_size", [32, 64, 128]),
            "optimizer":    trial.suggest_categorical("optimizer", ["AdamW"]),
            "num_layers":   trial.suggest_categorical("num_layers", [1, 2]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "huber_delta":  trial.suggest_categorical("huber_delta", [0.5, 1.0, 1.5]),
        }
 
        try:
            val_loss, _, best_pred_va, history, best_epoch = train_model_cv(
                Xtr, ytr, Xva, yva,
                device=device,
                params=params,
                H_output=H_output,
                MAX_EPOCHS=MAX_EPOCHS,
                verbose=verbose,
                prefix=f"[T{trial.number}] ",
                trial=trial,
                model_backend=model_backend,
            )
 
            debug_trial(
                trial_num=trial.number,
                fold=1,
                ytr=ytr,
                yva=yva,
                best_pred_va=best_pred_va,
                epoch_train_losses=history["train_loss"],
                epoch_val_losses=history["val_loss"],
                prefix=f"[T{trial.number}] ",
            )
 
            if val_loss < best_trial_value:
                best_trial_value   = val_loss
                best_trial_history = [history]
 
                if return_best_fold_metrics:
                    if metrics_in_original_scale:
                        y_true = inverse_target(yva,          scaler, target_col_name)
                        y_pred = inverse_target(best_pred_va, scaler, target_col_name)
                    else:
                        y_true = yva
                        y_pred = best_pred_va
 
                    m = eval_metrics_np(y_true, y_pred)
                    m["Fold"]         = 1
                    m["BestEpoch"]    = int(best_epoch) + 1
                    m["ValLoss_RMSE"] = float(val_loss)
                    best_trial_metrics = [m]
 
            return float(val_loss)
 
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"[ERROR] Trial {trial.number}: {e}")
            return float("inf")
 
    # ======================================================
    # Study — suffix _split agar tidak bentrok dengan study CV lama
    # ======================================================
    resolved_study_name = (
        study_name
        if study_name is not None
        else f"study_H{horizon}_{target_col_name}_split"
    )
 
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        storage=f"sqlite:///{study_db_path}",
        study_name=resolved_study_name,
        load_if_exists=True,
    )
 
    remaining = trials - len(study.trials)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    else:
        print(
            f"[INFO] Study '{resolved_study_name}' sudah punya "
            f"{len(study.trials)} trials, skip optimize."
        )
 
    if return_best_fold_histories or return_best_fold_metrics:
        return (
            study.best_params,
            study.best_value,
            best_trial_history,
            best_trial_metrics,
        )
 
    return study.best_params, study.best_value
 

