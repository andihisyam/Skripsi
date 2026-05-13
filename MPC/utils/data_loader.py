"""
utils/data_loader.py
====================
Fungsi-fungsi untuk load dan merge data:
  - load_price_csv         : baca file CSV harga saham
  - load_actual_prices     : ambil harga aktual H hari ke depan
  - load_historical_volatility : hitung volatility dari data historis
  - load_confidence_scores : baca confidence score per ticker
  - load_predicted_returns : baca hasil prediksi LSTM + terapkan confidence
  - select_top_n_assets    : pilih top-N saham berdasarkan adjusted return

Semua fungsi ini hanya bertanggung jawab untuk I/O dan transformasi data.
Tidak ada logika optimasi atau evaluasi di sini.
"""

import os
import glob
import numpy as np
import pandas as pd
from functools import reduce
from typing import Dict, List, Optional, Tuple

from config import (
    ACTUAL_BASE_DIR,
    HISTORICAL_BASE_DIR,
    PREDICTION_BASE_DIR,
    CONFIDENCE_CSV,
    TRAINING_SUMMARY_XLSX,
    LOOKBACK_DAYS,
    USE_CONFIDENCE,
    MIN_CONFIDENCE,
    CONFIDENCE_POWER,
    N_SELECTED_ASSETS,
)


# ============================================================
# HELPER INTERNAL
# ============================================================

def _merge_on_date(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Merge list of DataFrames pada kolom Date secara outer join."""
    return reduce(
        lambda left, right: pd.merge(left, right, on="Date", how="outer"),
        dfs
    )


# ============================================================
# LOAD HARGA SAHAM
# ============================================================

def load_price_csv(path: str) -> Optional[pd.DataFrame]:
    """
    Load file CSV harga saham (kolom Date + Close/Price).

    Return DataFrame dengan kolom:
        Date, Close, Return_Daily, Cumulative_Return
    Return None jika file tidak ditemukan atau data tidak valid.
    """
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)

    # Deteksi kolom harga (Close atau Price, fallback ke kolom ke-2)
    price_col = next(
        (c for c in ["Close", "Price"] if c in df.columns),
        df.columns[1] if len(df.columns) > 1 else None
    )
    if price_col is None:
        return None

    df[price_col] = pd.to_numeric(
        df[price_col].astype(str).str.replace(",", "", regex=False),
        errors="coerce"
    )
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = (
        df.dropna(subset=["Date", price_col])
          .sort_values("Date")
          .reset_index(drop=True)
          .rename(columns={price_col: "Close"})
    )

    df["Return_Daily"]      = np.log(df["Close"] / df["Close"].shift(1))
    df["Cumulative_Return"] = (df["Close"] / df["Close"].iloc[0]) - 1

    return df[["Date", "Close", "Return_Daily", "Cumulative_Return"]].dropna()


def load_actual_prices(ticker: str, H: int) -> Optional[pd.DataFrame]:
    """
    Load harga aktual H hari ke depan dari folder Saham_Horizon_{H}.

    Baris pertama di-skip karena merupakan harga awal sebelum periode.
    Cumulative_Return dihitung ulang dari baris pertama yang tersisa.
    """
    path = os.path.join(ACTUAL_BASE_DIR, f"Saham_Horizon_{H}", f"{ticker}.csv")
    df   = load_price_csv(path)

    if df is not None and len(df) > 1:
        df = df.iloc[1:].reset_index(drop=True)
        df["Cumulative_Return"] = (df["Close"] / df["Close"].iloc[0]) - 1

    return df


def load_historical_volatility(
    ticker: str,
    lookback_days: int = LOOKBACK_DAYS,
) -> Optional[float]:
    """
    Hitung volatility harian dari data historis ticker.

    Menggunakan lookback_days hari terakhir dari data di HISTORICAL_BASE_DIR.
    Return None jika data tidak tersedia atau tidak cukup (< 5 baris).
    """
    path = os.path.join(HISTORICAL_BASE_DIR, f"{ticker}.csv")
    df   = load_price_csv(path)

    if df is None or len(df) < 5:
        print(f"[WARN] Data historis {ticker} tidak cukup")
        return None

    return float(df["Return_Daily"].tail(lookback_days).std())


# ============================================================
# LOAD CONFIDENCE SCORE
# ============================================================

def load_confidence_scores(
    conf_csv: str,
    H: int,
) -> Dict[str, float]:
    """
    Load confidence score per ticker untuk horizon H dari CSV.

    Kolom yang dibutuhkan: H_FWD, Ticker, Confidence_Score.
    Return dict kosong jika file tidak ada atau kolom tidak lengkap.
    """
    if not os.path.exists(conf_csv):
        print(f"[WARN] Confidence CSV tidak ditemukan: {conf_csv}")
        return {}

    df = pd.read_csv(conf_csv)

    needed = {"H_FWD", "Ticker"}
    if not needed.issubset(df.columns):
        print(f"[WARN] Kolom confidence minimal tidak lengkap: {list(df.columns)}")
        return {}

    df = df[df["H_FWD"] == H].copy()
    if df.empty:
        return {}

    conf_col = next(
        (c for c in ["Confidence_Score", "Selected_Test_DA", "Test_DA"] if c in df.columns),
        None,
    )
    if conf_col is None:
        print("[WARN] Kolom confidence tidak ditemukan. Default confidence = 1.0")
        df["Confidence_Score"] = 1.0
    else:
        df["Confidence_Score"] = (
            pd.to_numeric(df[conf_col], errors="coerce")
              .fillna(1.0)
              .clip(0.0, 1.0)
        )
    return dict(zip(df["Ticker"], df["Confidence_Score"]))


def load_prediction_summary(H: int, pred_base_dir: str = PREDICTION_BASE_DIR) -> pd.DataFrame:
    """
    Load all_predictions.csv untuk horizon H.

    Mendukung dua skema:
      - Baru: Pred_End_Return_Kalman_Pct / Selected_Test_DA
      - Lama: Pred_Return_Pct / Confidence_Score
    """
    path = os.path.join(pred_base_dir, "all_predictions.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    df = pd.read_csv(path)
    needed = {"H_FWD", "Ticker", "Fitur"}
    if not needed.issubset(df.columns):
        raise ValueError(
            f"Kolom wajib all_predictions tidak lengkap. Minimal butuh {needed}, "
            f"kolom tersedia: {list(df.columns)}"
        )

    df = df[df["H_FWD"] == H].copy()
    if df.empty:
        raise ValueError(f"Tidak ada data prediksi untuk H_FWD={H} di {path}")

    ret_col = next(
        (c for c in ["Pred_End_Return_Kalman_Pct", "Pred_Return_Pct", "Pred_End_Return_Raw_Pct"] if c in df.columns),
        None,
    )
    if ret_col is None:
        raise ValueError(
            "Kolom return prediksi tidak ditemukan. Butuh salah satu dari: "
            "Pred_End_Return_Kalman_Pct, Pred_Return_Pct, Pred_End_Return_Raw_Pct"
        )

    conf_col = next(
        (c for c in ["Confidence_Score", "Selected_Test_DA", "Test_DA"] if c in df.columns),
        None,
    )
    if conf_col is None:
        df["Confidence_Score"] = 1.0
    else:
        df["Confidence_Score"] = (
            pd.to_numeric(df[conf_col], errors="coerce")
              .fillna(1.0)
              .clip(0.0, 1.0)
        )

    df["Pred_Return_Pct_Selected"] = pd.to_numeric(df[ret_col], errors="coerce")
    df = df.dropna(subset=["Pred_Return_Pct_Selected"]).copy()
    if df.empty:
        raise ValueError("Semua nilai return prediksi kosong/tidak valid di all_predictions.csv")

    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Fitur"] = df["Fitur"].astype(str).str.strip()
    return df.reset_index(drop=True)


def load_training_summary(H: int, summary_path: str = TRAINING_SUMMARY_XLSX) -> pd.DataFrame:
    """Load ALL_training_summary.xlsx untuk horizon H."""
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"File training summary tidak ditemukan: {summary_path}")

    df = pd.read_excel(summary_path)
    needed = {"H_FWD", "Ticker", "Fitur", "Test_DA", "Test_IC"}
    if not needed.issubset(df.columns):
        raise ValueError(
            f"Kolom training summary tidak lengkap. Butuh {needed}, "
            f"kolom tersedia: {list(df.columns)}"
        )

    df = df[df["H_FWD"] == H].copy()
    if df.empty:
        raise ValueError(f"Tidak ada data training summary untuk H_FWD={H}")

    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Fitur"] = df["Fitur"].astype(str).str.strip()
    df["Test_DA"] = pd.to_numeric(df["Test_DA"], errors="coerce").fillna(-np.inf)
    df["Test_IC"] = pd.to_numeric(df["Test_IC"], errors="coerce").fillna(-np.inf)
    return df.reset_index(drop=True)


def get_best_subset_from_training(
    ticker: str,
    training_df: pd.DataFrame,
) -> str:
    """Ambil fitur terbaik per ticker dari training summary (urut Test_DA, lalu Test_IC)."""
    tkr = str(ticker).upper().strip()
    df_t = training_df[training_df["Ticker"] == tkr].copy()
    if df_t.empty:
        raise ValueError(f"Ticker {tkr} tidak ada di training summary")

    best = (
        df_t.sort_values(["Test_DA", "Test_IC"], ascending=[False, False])
            .iloc[0]
    )
    return str(best["Fitur"])


def get_subset_options_for_ticker(
    ticker: str,
    training_df: pd.DataFrame,
    pred_summary_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Ambil daftar fitur kandidat untuk 1 ticker.

    Jika pred_summary_df diberikan, hanya menampilkan fitur yang tersedia
    di all_predictions (supaya bisa langsung dipakai forward).
    """
    tkr = str(ticker).upper().strip()
    df_t = training_df[training_df["Ticker"] == tkr].copy()
    if df_t.empty:
        return pd.DataFrame(columns=["Fitur", "Test_DA", "Test_IC"])

    ranked = (
        df_t.sort_values(["Test_DA", "Test_IC"], ascending=[False, False])
            .drop_duplicates(subset=["Fitur"], keep="first")
            .reset_index(drop=True)
    )

    if pred_summary_df is None:
        return ranked[["Fitur", "Test_DA", "Test_IC"]]

    available = set(
        pred_summary_df.loc[pred_summary_df["Ticker"] == tkr, "Fitur"]
        .astype(str)
        .str.strip()
        .tolist()
    )
    if not available:
        return pd.DataFrame(columns=["Fitur", "Test_DA", "Test_IC"])

    ranked = ranked[ranked["Fitur"].isin(available)].copy().reset_index(drop=True)
    return ranked[["Fitur", "Test_DA", "Test_IC"]]


def load_predicted_returns_from_selected_pairs(
    H: int,
    ticker_feature_pairs: List[Tuple[str, Optional[str]]],
    auto_feature_from_training: bool = False,
    pred_base_dir: str = PREDICTION_BASE_DIR,
    training_df: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Load predicted log return untuk daftar ticker/subset pilihan user.

    Parameters
    ----------
    ticker_feature_pairs:
        List tuple (ticker, fitur_opsional).
        Jika fitur None dan auto_feature_from_training=True,
        fitur dipilih otomatis dari training summary.
    """
    pred_df = load_prediction_summary(H, pred_base_dir=pred_base_dir)
    if training_df is None and auto_feature_from_training:
        training_df = load_training_summary(H)

    tickers, fiturs, raw_ret, adj_ret, conf_list = [], [], [], [], []

    for ticker, fitur in ticker_feature_pairs:
        tkr = str(ticker).upper().strip()
        df_t = pred_df[pred_df["Ticker"] == tkr].copy()
        if df_t.empty:
            raise ValueError(f"Ticker {tkr} tidak ada di all_predictions untuk H={H}")

        used_fitur = fitur
        if used_fitur is None and auto_feature_from_training:
            used_fitur = get_best_subset_from_training(tkr, training_df)  # type: ignore[arg-type]

        if used_fitur is not None:
            df_match = df_t[df_t["Fitur"] == str(used_fitur).strip()].copy()
            if df_match.empty and auto_feature_from_training:
                # fallback agar mode 'emiten' tetap bisa jalan jika subset terbaik
                # belum tersedia di all_predictions forward
                fallback = (
                    df_t.sort_values(
                        ["Confidence_Score", "Pred_Return_Pct_Selected"],
                        ascending=[False, False],
                    )
                    .iloc[0]
                )
                print(
                    f"[WARN] {tkr}: subset terbaik '{used_fitur}' belum tersedia di forward. "
                    f"Fallback ke '{fallback['Fitur']}'."
                )
                row = fallback
                used_fitur = str(fallback["Fitur"])
            elif df_match.empty:
                raise ValueError(
                    f"Ticker {tkr} tidak punya subset '{used_fitur}' di all_predictions H={H}"
                )
            else:
                row = df_match.iloc[0]
                used_fitur = str(used_fitur).strip()
        else:
            row = (
                df_t.sort_values(
                    ["Confidence_Score", "Pred_Return_Pct_Selected"],
                    ascending=[False, False],
                )
                .iloc[0]
            )
            used_fitur = str(row["Fitur"])

        confidence = float(np.clip(row["Confidence_Score"], 0.0, 1.0))
        if confidence < MIN_CONFIDENCE:
            print(
                f"[INFO] Skip {tkr} — confidence {confidence:.3f} "
                f"< {MIN_CONFIDENCE:.3f}"
            )
            continue

        ret_simple = float(row["Pred_Return_Pct_Selected"]) / 100.0
        log_ret = float(np.log1p(ret_simple))
        adj_log_ret = log_ret * (confidence ** CONFIDENCE_POWER) if USE_CONFIDENCE else log_ret

        tickers.append(tkr)
        fiturs.append(used_fitur)
        raw_ret.append(log_ret)
        adj_ret.append(adj_log_ret)
        conf_list.append(confidence)

        print(
            f"  {tkr} | Fitur: {used_fitur} | "
            f"Pred Return: {ret_simple*100:+.2f}% | "
            f"Conf: {confidence:.3f} | Adj LogRet: {adj_log_ret:+.4f}"
        )

    if not tickers:
        raise ValueError("Tidak ada emiten valid dari pilihan user")

    return (
        np.array(raw_ret).reshape(-1, 1),
        np.array(adj_ret).reshape(-1, 1),
        np.array(conf_list).reshape(-1, 1),
        tickers,
        fiturs,
    )


# ============================================================
# LOAD RETURN PREDIKSI
# ============================================================

def load_predicted_returns(
    H: int,
    pred_base_dir: str = PREDICTION_BASE_DIR,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Baca file prediksi per emiten, hitung log return H hari,
    dan terapkan confidence penalty.

    Alur per ticker:
        1. Baca Pred_Close_Kalman dari CSV prediksi
        2. Hitung log return H hari
        3. Kalikan dengan (confidence ^ CONFIDENCE_POWER) → adjusted return
        4. Skip jika confidence < MIN_CONFIDENCE

    Return:
        r_pred_raw : (N,1) log return mentah dari prediksi
        r_pred_adj : (N,1) log return setelah penalty confidence
        conf_arr   : (N,1) confidence score
        tickers    : list nama ticker yang valid
    """
    pattern = os.path.join(pred_base_dir, f"H{H}", f"*_H{H}_forward_pred.csv")
    files   = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(
            f"Tidak ada file prediksi di: {os.path.dirname(pattern)}"
        )

    conf_map = load_confidence_scores(CONFIDENCE_CSV, H) if USE_CONFIDENCE else {}
    tickers, raw_ret, adj_ret, conf_list = [], [], [], []

    for f in sorted(files):
        ticker = os.path.basename(f).split(f"_H{H}")[0]
        df     = pd.read_csv(f)

        if "Pred_Close_Kalman" not in df.columns:
            print(f"[WARN] Lewati {ticker} — kolom Pred_Close_Kalman tidak ada")
            continue

        kalman = pd.to_numeric(df["Pred_Close_Kalman"], errors="coerce").dropna()
        if len(kalman) < H:
            print(f"[WARN] Lewati {ticker} — data Kalman hanya {len(kalman)} baris")
            continue

        # Harga awal: ambil dari Pred_Close_Raw jika tersedia
        if "Pred_Close_Raw" in df.columns:
            raw = pd.to_numeric(df["Pred_Close_Raw"], errors="coerce").dropna()
            last_close = float(raw.iloc[0]) / np.exp(
                np.log(float(raw.iloc[-1]) / float(raw.iloc[0])) / H
            )
        else:
            last_close = float(kalman.iloc[0])

        log_ret    = np.log(float(kalman.iloc[H - 1]) / last_close)
        confidence = float(np.clip(conf_map.get(ticker, 1.0), 0.0, 1.0))

        if confidence < MIN_CONFIDENCE:
            print(
                f"[INFO] Skip {ticker} — confidence {confidence:.3f} "
                f"< {MIN_CONFIDENCE:.3f}"
            )
            continue

        adj_log_ret = log_ret * (confidence ** CONFIDENCE_POWER)

        tickers.append(ticker)
        raw_ret.append(log_ret)
        adj_ret.append(adj_log_ret)
        conf_list.append(confidence)

        print(
            f"  {ticker} | Pred Close H={H}: {float(kalman.iloc[H-1]):,.2f} | "
            f"Raw LogRet: {log_ret:+.4f} | Conf: {confidence:.3f} | "
            f"Adj LogRet: {adj_log_ret:+.4f}"
        )

    if not tickers:
        raise ValueError(f"Tidak ada emiten valid untuk H={H}")

    return (
        np.array(raw_ret).reshape(-1, 1),
        np.array(adj_ret).reshape(-1, 1),
        np.array(conf_list).reshape(-1, 1),
        tickers,
    )


# ============================================================
# SELEKSI TOP-N ASET
# ============================================================

def select_top_n_assets(
    r_raw:   np.ndarray,
    r_adj:   np.ndarray,
    conf:    np.ndarray,
    tickers: List[str],
    top_n:   Optional[int] = N_SELECTED_ASSETS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Pilih top-N saham berdasarkan adjusted predicted return.

    Jika top_n adalah None atau >= jumlah ticker, semua saham dikembalikan.
    Tie-breaking: jika adjusted return sama, saham dengan confidence lebih
    tinggi diprioritaskan.
    """
    if top_n is None or top_n >= len(tickers):
        return r_raw, r_adj, conf, tickers

    df = (
        pd.DataFrame({
            "Ticker": tickers,
            "Raw":    r_raw.flatten(),
            "Adj":    r_adj.flatten(),
            "Conf":   conf.flatten(),
        })
        .sort_values(["Adj", "Conf"], ascending=[False, False])
        .iloc[:top_n]
        .reset_index(drop=True)
    )

    print(f"\n[SELECT TOP-{top_n}]")
    print(df[["Ticker", "Raw", "Adj", "Conf"]].to_string(index=False))

    return (
        df["Raw"].values.reshape(-1, 1),
        df["Adj"].values.reshape(-1, 1),
        df["Conf"].values.reshape(-1, 1),
        df["Ticker"].tolist(),
    )


# ============================================================
# BUILD PORTFOLIO TIME SERIES (dipakai oleh evaluation)
# ============================================================

def build_portfolio_timeseries(
    weights: np.ndarray,
    tickers: List[str],
    H:       int,
    capital: float,
) -> Tuple[Optional[pd.DataFrame], np.ndarray]:
    """
    Load data aktual per ticker, hitung weighted portfolio time series.

    Return:
        df_port        : DataFrame dengan kolom Date, Portfolio_Daily_Return,
                         Portfolio_Cumulative_Return, Portfolio_Value
        realized_arr   : array realized cumulative return per ticker
    """
    daily_dfs, cum_dfs, realized = [], [], []

    for i, t in enumerate(tickers):
        df = load_actual_prices(t, H)
        if df is None or df.empty:
            realized.append(np.nan)
            continue

        daily_dfs.append(
            df[["Date", "Return_Daily"]].rename(columns={"Return_Daily": t})
        )
        cum_dfs.append(
            df[["Date", "Cumulative_Return"]].rename(columns={"Cumulative_Return": t})
        )
        realized.append(float(df["Cumulative_Return"].iloc[-1]))

    realized_arr = np.array(realized)

    if not daily_dfs:
        return None, realized_arr

    df_daily = _merge_on_date(daily_dfs).sort_values("Date").fillna(0)
    df_cum   = _merge_on_date(cum_dfs).sort_values("Date").ffill().fillna(0)

    valid_tickers = [t for t in tickers if t in df_daily.columns]
    w_valid       = np.array([weights[tickers.index(t)] for t in valid_tickers])
    w_valid      /= w_valid.sum()

    w_daily = df_daily[valid_tickers].values @ w_valid
    w_cum   = df_cum[valid_tickers].values   @ w_valid

    df_port = pd.DataFrame({
        "Date":                        df_daily["Date"].values,
        "Portfolio_Daily_Return":      w_daily,
        "Portfolio_Cumulative_Return": w_cum,
        "Portfolio_Value":             capital * (1 + w_cum),
    })
    return df_port, realized_arr
