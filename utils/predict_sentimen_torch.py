import ast
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from config import DATE_COL, PRICE_COL, N_STEPS
from utils.io_utils import load_price_single
from utils.kalman import apply_kalman_filter_level_trend
from utils.model_utils_seq2seq import build_lstm_model as build_lstm_model_seq2seq
from utils.model_utils_sklearn import LSTMSklearnNet
from utils.preprocess_sentimen import (
    add_indicators,
    add_return_lags,
    add_simple_returns,
    load_sentimen,
    merge_sentimen,
    prepare_sequences,
)
from utils.train_optuna_sentimen import FEATURE_COMBINATIONS, RET_1D


LOWER_IS_BETTER = {
    "Test_RMSE",
    "Test_MAE",
    "Avg_CV_Loss",
    "Final_BestValLoss_RMSE",
    "Val_RMSE",
    "Val_MAE",
    "TestSize",
}


def _minmax_series(s: pd.Series) -> pd.Series:
    s_num = pd.to_numeric(s, errors="coerce")
    mn, mx = s_num.min(), s_num.max()
    if pd.isna(mn) or pd.isna(mx) or np.isclose(mx, mn):
        return pd.Series([0.5] * len(s_num), index=s_num.index)
    return (s_num - mn) / (mx - mn)


def _compute_da_ic_score(df: pd.DataFrame, da_col: str = "Test_DA", ic_col: str = "Test_IC") -> pd.Series:
    """
    Combined score berbasis DA + IC (higher is better).
    Bobot default:
      - DA: 0.7
      - IC: 0.3
    """
    da = _minmax_series(df[da_col])
    ic = _minmax_series(df[ic_col])
    score = 0.7 * da + 0.3 * ic
    return score.astype(float).clip(0.0, 1.0)


def _load_summary_table(summary_file: str) -> pd.DataFrame:
    p = Path(summary_file)
    if not p.exists():
        raise FileNotFoundError(f"Summary file tidak ditemukan: {summary_file}")

    if p.suffix.lower() in {".xlsx", ".xls"}:
        xls = pd.ExcelFile(p)
        sheet = "ALL_Summary" if "ALL_Summary" in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(p, sheet_name=sheet)
    else:
        df = pd.read_csv(p)

    needed = {"H_FWD", "Ticker", "Fitur", "Optuna_Params"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Kolom wajib tidak ada di summary: {missing}")

    return df


def _is_better_metric_higher(metric: str) -> bool:
    return metric not in LOWER_IS_BETTER


def select_best_models(
    summary_file: str,
    horizons: Sequence[int],
    metric: str = "DA_IC",
    tickers_filter: Optional[Sequence[str]] = None,
    subset_filter: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    df = _load_summary_table(summary_file)
    df = df[df["H_FWD"].isin(horizons)].copy()

    if tickers_filter:
        wanted = {t.upper() for t in tickers_filter}
        df = df[df["Ticker"].str.upper().isin(wanted)].copy()
    if subset_filter:
        wanted_subset = {s.strip() for s in subset_filter}
        df = df[df["Fitur"].astype(str).isin(wanted_subset)].copy()

    # Paksa pemilihan model berbasis kombinasi DA + IC.
    if str(metric).upper() != "DA_IC":
        print(f"[WARN] Metric '{metric}' diabaikan. Pemilihan model dipaksa pakai DA_IC.")

    needed_da_ic = {"Test_DA", "Test_IC"}
    missing_da_ic = needed_da_ic - set(df.columns)
    if missing_da_ic:
        raise ValueError(f"Kolom wajib DA+IC tidak ada di summary: {missing_da_ic}")

    df["Test_DA"] = pd.to_numeric(df["Test_DA"], errors="coerce")
    df["Test_IC"] = pd.to_numeric(df["Test_IC"], errors="coerce")

    # DA+IC wajib valid. IC <= 0 tidak dipakai agar confidence lebih bermakna.
    df = df.dropna(subset=["Test_DA", "Test_IC", "Ticker", "Fitur"]).copy()
    df = df[df["Test_IC"] > 0].copy()
    if df.empty:
        raise ValueError("Tidak ada baris valid untuk DA+IC (pastikan Test_IC > 0 tersedia).")

    # Skor dihitung per horizon agar adil antar H.
    da_norm = df.groupby("H_FWD")["Test_DA"].transform(_minmax_series)
    ic_norm = df.groupby("H_FWD")["Test_IC"].transform(_minmax_series)
    df["DA_IC_Score"] = (0.7 * da_norm + 0.3 * ic_norm).clip(0.0, 1.0)
    df["Confidence_Score"] = df["DA_IC_Score"].clip(0.0, 1.0)

    best_rows = []
    for (h, tkr), g in df.groupby(["H_FWD", "Ticker"], as_index=False):
        g2 = g.sort_values("DA_IC_Score", ascending=False)
        best_rows.append(g2.iloc[0])

    best_df = pd.DataFrame(best_rows).reset_index(drop=True)
    print(f"[SELECT] {len(best_df)} model terbaik terpilih berdasarkan DA_IC.")
    return best_df


def _parse_optuna_params(v) -> Dict:
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            return ast.literal_eval(v)
        except Exception as e:
            raise ValueError(f"Gagal parse Optuna_Params: {e}")
    raise ValueError("Optuna_Params format tidak dikenali.")


def _build_model_from_backend(
    backend: str,
    X_last: np.ndarray,
    params: Dict,
    h_output: int,
    device: torch.device,
):
    backend_norm = str(backend).lower()
    if backend_norm == "lstm_sklearn":
        model = LSTMSklearnNet(
            input_size=X_last.shape[2],
            dense_units=int(params["dense_units"]),
            hidden_units=int(params["hidden_units"]),
            dropout=float(params["dropout"]),
            output_dim=int(h_output),
            num_layers=int(params.get("num_layers", 1)),
        ).to(device)
        return model

    return build_lstm_model_seq2seq(
        X_last,
        dense_units=int(params["dense_units"]),
        hidden_units=int(params["hidden_units"]),
        dropout=float(params["dropout"]),
        output_dim=int(h_output),
        num_layers=int(params.get("num_layers", 1)),
    ).to(device)


def _resolve_artifact_path(raw_path: str, summary_file: str, fallback_path: Path) -> Path:
    raw = Path(str(raw_path))
    summary_path = Path(summary_file).resolve()
    summary_dir = summary_path.parent

    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(Path.cwd() / raw)
        candidates.append(summary_dir / raw)
        candidates.append(summary_dir.parent / raw)

    candidates.append(fallback_path)

    for c in candidates:
        if c.exists():
            return c

    return fallback_path


def _prepare_input(
    ticker: str,
    data_dir: str,
    sentimen_path: str,
    feature_subset: List[str],
    scaler_dict: Dict,
    h_output: int,
) -> Tuple[np.ndarray, float, pd.Timestamp]:
    csv_path = Path(data_dir) / f"{ticker}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV saham tidak ditemukan: {csv_path}")

    df = load_price_single(str(csv_path))
    df_sent = load_sentimen(sentimen_path)
    df = merge_sentimen(df, df_sent)

    df = add_simple_returns(df, out_col=RET_1D)
    df = add_return_lags(df, return_col=RET_1D, lags=(1, 2, 3))
    df = add_indicators(df)
    df = df.dropna().reset_index(drop=True)

    min_needed = N_STEPS + max(1, h_output)
    if len(df) < min_needed:
        raise ValueError(
            f"{ticker}: data kurang untuk inferensi, perlu >={min_needed}, tersedia {len(df)}"
        )

    X_all, _, _, _ = prepare_sequences(
        df=df,
        target_col_name=RET_1D,
        horizon=1,
        feature_subset=feature_subset,
        scaler_dict=scaler_dict,
        is_train=False,
        H_output=h_output,
        verbose=False,
    )
    if len(X_all) == 0:
        raise ValueError(f"{ticker}: tidak ada sequence inferensi yang terbentuk.")

    X_last = X_all[[-1]]
    last_close = float(df[PRICE_COL].iloc[-1])
    last_date = pd.to_datetime(df[DATE_COL].iloc[-1])
    return X_last, last_close, last_date


def _to_numeric_price(series: pd.Series) -> pd.Series:
    """
    Robust parser untuk kolom harga yang bisa berisi separator lokal.
    Fokus untuk kolom harga (bukan persen/volume).
    """
    s = series.astype(str).str.strip()
    s = s.str.replace(" ", "", regex=False)
    # Hapus pemisah ribuan umum.
    s = s.str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce")


def _load_actual_forward(
    ticker: str,
    horizon: int,
    horizon_data_base_dir: str,
) -> Optional[pd.DataFrame]:
    horizon_dir = Path(horizon_data_base_dir) / f"Saham_Horizon_{horizon}"
    path = horizon_dir / f"{ticker}.csv"
    if not path.exists():
        print(f"[WARN] Aktual tidak ditemukan: {path}")
        return None

    df = pd.read_csv(path)
    if "Price" in df.columns and PRICE_COL not in df.columns:
        df = df.rename(columns={"Price": PRICE_COL})

    if DATE_COL not in df.columns or PRICE_COL not in df.columns:
        print(f"[WARN] Format aktual tidak sesuai untuk {ticker} H={horizon}: kolom kurang.")
        return None

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[PRICE_COL] = _to_numeric_price(df[PRICE_COL])
    df = df.dropna(subset=[DATE_COL, PRICE_COL]).copy()
    # Urut ascending: day-1 forward ada di baris awal.
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    if df.empty:
        return None

    return df.iloc[:horizon].copy()


def _save_plot(
    ticker: str,
    horizon: int,
    pred_close_raw: np.ndarray,
    pred_close_kalman: np.ndarray,
    actual_close: Optional[np.ndarray],
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    days = np.arange(1, horizon + 1)

    plt.figure(figsize=(11, 5))
    plt.plot(days, pred_close_raw, label="Prediksi raw", linestyle="--", color="orange")
    plt.plot(days, pred_close_kalman, label="Prediksi kalman", color="darkorange")
    if actual_close is not None and len(actual_close) > 0:
        d_act = np.arange(1, len(actual_close) + 1)
        plt.plot(d_act, actual_close, label="Aktual", color="steelblue", marker="x", linewidth=1.6)
    plt.xlabel("Hari ke depan")
    plt.ylabel("Harga (Close)")
    plt.title(f"{ticker} - Prediksi {horizon} Hari")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    pdf_path = out_dir / f"{ticker}_H{horizon}_forward_pred.pdf"
    plt.savefig(pdf_path)
    plt.close()


def _save_csv(
    ticker: str,
    horizon: int,
    last_close: float,
    pred_ret: np.ndarray,
    pred_close_raw: np.ndarray,
    pred_close_kalman: np.ndarray,
    actual_close: Optional[np.ndarray],
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    days = np.arange(1, horizon + 1)

    df_out = pd.DataFrame(
        {
            "Hari_ke": days,
            "Pred_LogRet": pred_ret,
            "Pred_Close_Raw": pred_close_raw,
            "Pred_Close_Kalman": pred_close_kalman,
        }
    )
    df_out["Pred_Return_Raw_Pct"] = (df_out["Pred_Close_Raw"] / last_close - 1.0) * 100.0
    df_out["Pred_Return_Kalman_Pct"] = (df_out["Pred_Close_Kalman"] / last_close - 1.0) * 100.0
    if actual_close is not None and len(actual_close) > 0:
        act = np.full(horizon, np.nan, dtype=float)
        n = min(horizon, len(actual_close))
        act[:n] = actual_close[:n]
        df_out["Actual_Close"] = act
        df_out["Actual_Return_Pct"] = (df_out["Actual_Close"] / last_close - 1.0) * 100.0
    else:
        df_out["Actual_Close"] = np.nan
        df_out["Actual_Return_Pct"] = np.nan

    csv_path = out_dir / f"{ticker}_H{horizon}_forward_pred.csv"
    df_out.to_csv(csv_path, index=False)


def predict_forward_sentiment(
    summary_file: str,
    data_dir: str,
    sentimen_path: str,
    out_dir: str = "predictions_forward",
    horizons: Sequence[int] = (22, 44),
    horizon_data_base_dir: str = "../Data",
    tickers_filter: Optional[Sequence[str]] = None,
    subset_filter: Optional[Sequence[str]] = None,
    metric: str = "DA_IC",
    model_backend_override: str = "auto",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    best_df = select_best_models(
        summary_file=summary_file,
        horizons=horizons,
        metric=metric,
        tickers_filter=tickers_filter,
        subset_filter=subset_filter,
    )

    summary_path = Path(summary_file).resolve()
    summary_dir = summary_path.parent

    all_rows = []

    for _, row in best_df.iterrows():
        ticker = str(row["Ticker"]).upper()
        h_fwd = int(row["H_FWD"])
        fitur = str(row["Fitur"])

        if fitur not in FEATURE_COMBINATIONS:
            print(f"[SKIP] {ticker} H={h_fwd}: subset fitur '{fitur}' tidak ditemukan.")
            continue

        feature_subset = list(FEATURE_COMBINATIONS[fitur])

        h_output = int(row["H_output"]) if "H_output" in row and not pd.isna(row["H_output"]) else h_fwd
        if h_output < h_fwd:
            print(
                f"[SKIP] {ticker} H={h_fwd}: model H_output={h_output} belum cukup untuk horizon {h_fwd}."
            )
            continue

        fallback_model = summary_dir / f"FWD_{h_fwd}" / fitur / f"{ticker}_best_{RET_1D}_Hout{h_output}.pt"
        fallback_scaler = summary_dir / f"FWD_{h_fwd}" / fitur / f"{ticker}_scaler_{RET_1D}_Hout{h_output}.pkl"

        model_path = _resolve_artifact_path(str(row.get("ModelPath", "")), summary_file, fallback_model)
        scaler_path = _resolve_artifact_path(str(row.get("ScalerPath", "")), summary_file, fallback_scaler)

        if not model_path.exists() or not scaler_path.exists():
            print(f"[SKIP] {ticker} H={h_fwd}: artifact tidak ditemukan.")
            print(f"       model : {model_path}")
            print(f"       scaler: {scaler_path}")
            continue

        print(
            f"\n[PREDICT] {ticker} | H={h_fwd} | fitur={fitur} | "
            f"DA_IC={float(row.get('DA_IC_Score', np.nan)):.6f} | "
            f"Confidence={float(row.get('Confidence_Score', np.nan)):.6f}"
        )

        scaler_dict = joblib.load(scaler_path)

        try:
            X_last, last_close, last_date = _prepare_input(
                ticker=ticker,
                data_dir=data_dir,
                sentimen_path=sentimen_path,
                feature_subset=feature_subset,
                scaler_dict=scaler_dict,
                h_output=h_output,
            )
        except Exception as e:
            print(f"[SKIP] {ticker} H={h_fwd}: gagal prepare input -> {e}")
            continue

        target_key = f"target_{RET_1D}"
        if target_key not in scaler_dict:
            print(f"[SKIP] {ticker} H={h_fwd}: scaler target key '{target_key}' tidak ada.")
            continue
        scaler_tgt = scaler_dict[target_key]

        try:
            params = _parse_optuna_params(row["Optuna_Params"])
            model_backend = (
                str(row.get("ModelBackend", "seq2seq"))
                if str(model_backend_override).lower() == "auto"
                else str(model_backend_override)
            )
            model = _build_model_from_backend(
                backend=model_backend,
                X_last=X_last,
                params=params,
                h_output=h_output,
                device=device,
            )
            state = torch.load(model_path, map_location=device)
            model.load_state_dict(state)
            model.eval()
        except Exception as e:
            print(f"[SKIP] {ticker} H={h_fwd}: gagal load model -> {e}")
            continue

        with torch.no_grad():
            x_t = torch.tensor(X_last, dtype=torch.float32).to(device)
            pred_scaled = model(x_t).cpu().numpy().reshape(-1)

        pred_ret_all = scaler_tgt.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)
        pred_ret = pred_ret_all[:h_fwd]
        pred_close_raw = last_close * np.exp(np.cumsum(pred_ret))
        pred_close_kalman = apply_kalman_filter_level_trend(pd.Series(pred_close_raw)).values
        actual_df = _load_actual_forward(
            ticker=ticker,
            horizon=h_fwd,
            horizon_data_base_dir=horizon_data_base_dir,
        )
        actual_close = actual_df[PRICE_COL].to_numpy(dtype=float) if actual_df is not None else None

        h_dir = Path(out_dir) / f"H{h_fwd}"
        _save_plot(ticker, h_fwd, pred_close_raw, pred_close_kalman, actual_close, h_dir)
        _save_csv(ticker, h_fwd, last_close, pred_ret, pred_close_raw, pred_close_kalman, actual_close, h_dir)

        actual_end_close = None
        actual_end_ret_pct = None
        if actual_close is not None and len(actual_close) > 0:
            actual_end_close = float(actual_close[min(h_fwd, len(actual_close)) - 1])
            actual_end_ret_pct = float((actual_end_close / last_close - 1.0) * 100.0)

        all_rows.append(
            {
                "H_FWD": h_fwd,
                "H_output": h_output,
                "ModelBackend": str(row.get("ModelBackend", "seq2seq")),
                "Ticker": ticker,
                "Fitur": fitur,
                "Last_Date": str(last_date.date()),
                "Last_Close": last_close,
                "Pred_End_Close_Raw": float(pred_close_raw[-1]),
                "Pred_End_Close_Kalman": float(pred_close_kalman[-1]),
                "Pred_End_Return_Raw_Pct": float((pred_close_raw[-1] / last_close - 1.0) * 100.0),
                "Pred_End_Return_Kalman_Pct": float((pred_close_kalman[-1] / last_close - 1.0) * 100.0),
                "Actual_End_Close": actual_end_close,
                "Actual_End_Return_Pct": actual_end_ret_pct,
                # Metric utama berbasis DA+IC + confidence siap pakai untuk MPC.
                "Selected_Test_DA": row.get("Test_DA"),
                "Selected_Test_IC": row.get("Test_IC"),
                "DA_IC_Score": row.get("DA_IC_Score"),
                "Confidence_Score": row.get("Confidence_Score"),
                "Selected_Metric": "DA_IC",
                "ModelPath_Used": str(model_path),
                "ScalerPath_Used": str(scaler_path),
            }
        )

    if not all_rows:
        print("[DONE] Tidak ada prediksi yang berhasil dibuat.")
        return []

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(out_root / "all_predictions.csv", index=False)

    for h in sorted(set(all_df["H_FWD"])):
        dfh = all_df[all_df["H_FWD"] == h].copy()
        dfh = dfh.sort_values("Pred_End_Return_Kalman_Pct", ascending=False).reset_index(drop=True)
        dfh.index += 1
        dfh.to_csv(out_root / f"ranking_emiten_H{h}.csv", index_label="Rank")

    print(f"[DONE] Prediksi selesai. Output di: {out_root.resolve()}")
    return all_rows
