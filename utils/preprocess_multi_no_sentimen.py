# ================================================================
# preprocess_multi_no_sentimen.py
# Preprocessing khusus MULTI-EMITEN
# ================================================================

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from numpy.lib.stride_tricks import sliding_window_view

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from config import (
    DATE_COL,
    TARGET_COL,
    N_STEPS,
    H_1M,
    MA_PERIODS,
    EMA_PERIODS,
    RSI_PERIODS,
    MACD_CONFIGS
)


# =====================================================================
# 1. LOAD CSV MULTI
# =====================================================================
def load_multi_csv(csv_paths: List[str], tickers: List[str]) -> pd.DataFrame:
    dfs = []
    for path, t in zip(csv_paths, tickers):
        df = pd.read_csv(path)
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        df["Ticker"] = t
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.dropna(subset=[DATE_COL]).sort_values([ "Ticker", DATE_COL ])
    df_all = df_all.reset_index(drop=True)

    return df_all


# =====================================================================
# 2. SIMPLE RETURN per-EMITEN
# =====================================================================
def add_simple_returns_multi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Ticker", DATE_COL])
    df["Return"] = df.groupby("Ticker")[TARGET_COL].pct_change()
    return df


# =====================================================================
# 3. INDIKATOR TEKNIKAL (groupby Ticker)
# =====================================================================
def add_indicators_multi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # MA
    for p in MA_PERIODS:
        df[f"MA{p}"] = df.groupby("Ticker")[TARGET_COL].transform(lambda x: x.rolling(p).mean())

    # EMA
    for p in EMA_PERIODS:
        df[f"EMA{p}"] = df.groupby("Ticker")[TARGET_COL].transform(lambda x: x.ewm(span=p, adjust=False).mean())

    # RSI
    for p in RSI_PERIODS:
        def compute_rsi(series):
            delta = series.diff()
            gain = delta.clip(lower=0).rolling(p).mean()
            loss = -delta.clip(upper=0).rolling(p).mean()
            rs = gain / (loss + 1e-9)
            return 100 - (100 / (1 + rs))

        df[f"RSI{p}"] = df.groupby("Ticker")[TARGET_COL].transform(compute_rsi)

    # MACD
    for fast, slow in MACD_CONFIGS:
        def compute_macd(series):
            ema_fast = series.ewm(span=fast, adjust=False).mean()
            ema_slow = series.ewm(span=slow, adjust=False).mean()
            return ema_fast - ema_slow

        df[f"MACD_{fast}_{slow}"] = df.groupby("Ticker")[TARGET_COL].transform(compute_macd)

    # LAG close
    for lag in range(1, 4):
        df[f"{TARGET_COL}_lag{lag}"] = df.groupby("Ticker")[TARGET_COL].shift(lag)

    df = df.dropna().reset_index(drop=True)
    return df


# =====================================================================
# 4. FIT HYBRID SCALERS (SAMA PERSIS dengan single-emiten)
# =====================================================================
def fit_scale_multi(df_feat: pd.DataFrame,
                    scaler_dict: Optional[Dict[str, object]] = None,
                    fit: bool = True):

    scaler_dict = scaler_dict or {}
    df_scaled = df_feat.copy()

    price_cols = [c for c in df_feat.columns if any(k in c.lower() for k in ["open", "close", "high", "low"])]
    volume_cols = [c for c in df_feat.columns if "volume" in c.lower()]
    indicator_cols = [c for c in df_feat.columns if c not in price_cols + volume_cols]

    if fit:
        scaler_dict["price"] = StandardScaler()
        scaler_dict["volume"] = RobustScaler()
        scaler_dict["indicator"] = MinMaxScaler()

        if price_cols:
            df_scaled[price_cols] = scaler_dict["price"].fit_transform(df_feat[price_cols])
        if volume_cols:
            df_scaled[volume_cols] = scaler_dict["volume"].fit_transform(df_feat[volume_cols])
        if indicator_cols:
            df_scaled[indicator_cols] = scaler_dict["indicator"].fit_transform(df_feat[indicator_cols])

    else:
        if "price" in scaler_dict and price_cols:
            df_scaled[price_cols] = scaler_dict["price"].transform(df_feat[price_cols])
        if "volume" in scaler_dict and volume_cols:
            df_scaled[volume_cols] = scaler_dict["volume"].transform(df_feat[volume_cols])
        if "indicator" in scaler_dict and indicator_cols:
            df_scaled[indicator_cols] = scaler_dict["indicator"].transform(df_feat[indicator_cols])

    return df_scaled, scaler_dict


# =====================================================================
# 5. SCALING TARGET KHUSUS MULTI-EMITEN
# =====================================================================
def scale_target_multi(y: np.ndarray,
                       scaler_dict: Optional[Dict[str, object]],
                       fit: bool):

    if fit:
        scaler = StandardScaler()
        y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()
        scaler_dict = {"target_close_multi": scaler}
        return y_scaled, scaler_dict

    else:
        scaler = scaler_dict["target_close_multi"]
        y_scaled = scaler.transform(y.reshape(-1, 1)).flatten()
        return y_scaled, scaler_dict


# =====================================================================
# 6. WINDOWING MULTI-EMITEN
# =====================================================================
def prepare_sequences_multi(
    df: pd.DataFrame,
    target_col: str,
    feature_subset: Optional[List[str]],
    scaler_dict: Optional[Dict[str, object]] = None,
    horizon: int = H_1M,
    step_size: int = 1,
):

    # Ambil fitur numerik (selain date & target)
    feat_cols = df.drop(columns=[DATE_COL, target_col, "Ticker"], errors="ignore")
    num_df = feat_cols.select_dtypes(include=[np.number])

    # Pilih subset fitur
    if feature_subset:
        available = [c for c in feature_subset if c in num_df.columns]
        num_df = num_df[available]

    # Scaling fitur
    fit_mode = scaler_dict is None
    df_scaled, scaler_dict = fit_scale_multi(num_df, scaler_dict, fit=fit_mode)
    X_mat = df_scaled.values

    # Scaling target
    y = df[target_col].values
    y_scaled, scaler_dict = scale_target_multi(y, scaler_dict, fit=fit_mode)

    T, F = X_mat.shape
    needed = N_STEPS + horizon
    if T < needed:
        raise ValueError(f"Data terlalu pendek. Perlu ≥ {needed}, ada {T}")

    # Window X
    X_full = sliding_window_view(X_mat, (N_STEPS, F))[:, 0, :, :]

    # Window Y
    y_list = []
    max_start = T - (N_STEPS + horizon) + 1
    for s in range(0, max_start, step_size):
        y_list.append(y_scaled[s + N_STEPS : s + N_STEPS + horizon])

    X = X_full[:max_start:step_size]
    y = np.stack(y_list, axis=0)

    return X, y, scaler_dict, num_df.columns.tolist()


# =====================================================================
# 7. INVERSE TRANSFORM PREDIKSI
# =====================================================================
def inverse_transform_target_multi(y_scaled: np.ndarray,
                                   scaler_dict: Dict[str, object]):

    if y_scaled.ndim == 1:
        y_scaled = y_scaled.reshape(-1, 1)

    scaler = scaler_dict["target_close_multi"]
    return scaler.inverse_transform(y_scaled).flatten()
