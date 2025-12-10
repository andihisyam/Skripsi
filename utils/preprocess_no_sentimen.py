import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from numpy.lib.stride_tricks import sliding_window_view
from typing import Tuple, Optional, List, Dict
from config import (
    DATE_COL, TARGET_COL, N_STEPS, H_1M,
    TRAIN_RATIO, VAL_RATIO,
    MA_PERIODS, EMA_PERIODS, RSI_PERIODS, MACD_CONFIGS
)

# ================================================================
# 1. LOAD CSV
# ================================================================
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    return df


# ================================================================
# 2. SIMPLE RETURN
# ================================================================
def add_simple_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Ticker", DATE_COL])
    df["Return"] = df.groupby("Ticker")[TARGET_COL].pct_change()
    return df


# ================================================================
# 3. AMBIL FITUR NUMERIK
# ================================================================
def select_numeric_matrix(df: pd.DataFrame, target_col: str,
                          feature_subset: Optional[List[str]] = None):
    drop_cols = [DATE_COL, target_col]
    num_df = df.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number])

    if feature_subset:
        available = [f for f in feature_subset if f in num_df.columns]
        num_df = num_df[available]

    return num_df.values.astype(float), list(num_df.columns)


# ================================================================
# 4. FIT HYBRID SCALER UNTUK FITUR
# ================================================================
def fit_scale(data: pd.DataFrame,
              scaler_dict: Optional[Dict[str, object]] = None,
              fit: bool = True):

    scaler_dict = scaler_dict or {}
    df_scaled = data.copy()

    price_cols = [c for c in data.columns if any(k in c.lower() for k in ["open", "close", "high", "low"])]
    volume_cols = [c for c in data.columns if "volume" in c.lower()]
    indicator_cols = [c for c in data.columns if c not in price_cols + volume_cols]

    if fit:
        scaler_dict["price"] = StandardScaler()
        scaler_dict["volume"] = RobustScaler()
        scaler_dict["indicator"] = MinMaxScaler()

        if price_cols:
            df_scaled[price_cols] = scaler_dict["price"].fit_transform(data[price_cols])
        if volume_cols:
            df_scaled[volume_cols] = scaler_dict["volume"].fit_transform(data[volume_cols])
        if indicator_cols:
            df_scaled[indicator_cols] = scaler_dict["indicator"].fit_transform(data[indicator_cols])

    else:
        if "price" in scaler_dict and price_cols:
            df_scaled[price_cols] = scaler_dict["price"].transform(data[price_cols])
        if "volume" in scaler_dict and volume_cols:
            df_scaled[volume_cols] = scaler_dict["volume"].transform(data[volume_cols])
        if "indicator" in scaler_dict and indicator_cols:
            df_scaled[indicator_cols] = scaler_dict["indicator"].transform(data[indicator_cols])

    return df_scaled, scaler_dict


# ================================================================
# 5. SCALE TARGET CLOSE
# ================================================================
def scale_target(y: np.ndarray,
                 scaler_dict: dict,
                 fit: bool):

    if fit:
        scaler = StandardScaler()
        y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()
        scaler_dict["target_close"] = scaler
        return y_scaled, scaler_dict

    else:
        scaler = scaler_dict["target_close"]
        y_scaled = scaler.transform(y.reshape(-1, 1)).flatten()
        return y_scaled, scaler_dict
    


# ================================================================
# 6. WINDOWING (X, Y)
# ================================================================
def prepare_sequences(
    df: pd.DataFrame,
    target_col_name: str,
    horizon: int = H_1M,
    feature_subset: Optional[List[str]] = None,
    scaler_dict: Optional[Dict[str, object]] = None,
    step_size: int = 1,
):

    # 1. Ambil fitur numerik
    data_num, cols = select_numeric_matrix(df, target_col=target_col_name, feature_subset=feature_subset)
    data_df = pd.DataFrame(data_num, columns=cols)

    # 2. Scaling fitur
    fit_mode = scaler_dict is None
    data_scaled_df, scaler_dict = fit_scale(data_df, scaler_dict, fit=fit_mode)
    data_scaled = data_scaled_df.values

    # 3. TARGET asli & scaling target
    tgt = df[target_col_name].values
    tgt_scaled, scaler_dict = scale_target(tgt, scaler_dict, fit=fit_mode)

    T, F = data_scaled.shape
    needed = N_STEPS + horizon
    if T < needed:
        raise ValueError(f"Data terlalu pendek. Perlu ≥ {needed}, ada {T}")

    # 4. Buat window X
    X_full = sliding_window_view(data_scaled, (N_STEPS, F))[:, 0, :, :]

    # 5. Buat window Y (scaled)
    y_list = []
    max_start = T - (N_STEPS + horizon) + 1
    for s in range(0, max_start, step_size):
        y_list.append(tgt_scaled[s + N_STEPS : s + N_STEPS + horizon])

    X = X_full[:max_start:step_size]
    y = np.stack(y_list, axis=0)

    return X, y, scaler_dict, cols


# ================================================================
# 7. SPLIT (Hybrid)
# ================================================================
def time_split_hybrid(df: pd.DataFrame,
                      train_end: str, val_end: str,
                      start_date_expected="2015-03-02",
                      train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO,
                      min_needed=None):

    df = df.sort_values(DATE_COL).reset_index(drop=True)
    start_actual = df[DATE_COL].min().date()

    if min_needed is None:
        min_needed = N_STEPS + H_1M

    if str(start_actual) == start_date_expected:
        tr = df[df[DATE_COL] <= pd.Timestamp(train_end)]
        va = df[(df[DATE_COL] > pd.Timestamp(train_end)) & (df[DATE_COL] <= pd.Timestamp(val_end))]
        te = df[df[DATE_COL] > pd.Timestamp(val_end)]
        mode = "date"
    else:
        n = len(df)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)
        tr = df.iloc[:n_train]
        va = df.iloc[n_train:n_train+n_val]
        te = df.iloc[n_train+n_val:]
        mode = "ratio"

    return tr, va, te, mode


# ================================================================
# 8. INDIKATOR TEKNIKAL
# ================================================================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # MA
    for p in MA_PERIODS:
        df[f"MA{p}"] = df[TARGET_COL].rolling(p).mean()

    # EMA
    for p in EMA_PERIODS:
        df[f"EMA{p}"] = df[TARGET_COL].ewm(span=p, adjust=False).mean()

    # RSI
    for p in RSI_PERIODS:
        delta = df[TARGET_COL].diff()
        gain = delta.clip(lower=0).rolling(p).mean()
        loss = -delta.clip(upper=0).rolling(p).mean()
        rs = gain / (loss + 1e-9)
        df[f"RSI{p}"] = 100 - (100 / (1 + rs))

    # MACD
    for fast, slow in MACD_CONFIGS:
        ema_fast = df[TARGET_COL].ewm(span=fast, adjust=False).mean()
        ema_slow = df[TARGET_COL].ewm(span=slow, adjust=False).mean()
        df[f"MACD_{fast}_{slow}"] = ema_fast - ema_slow

    # LAG close
    for lag in range(1, 4):
        df[f"{TARGET_COL}_lag{lag}"] = df[TARGET_COL].shift(lag)

    return df.dropna().reset_index(drop=True)
