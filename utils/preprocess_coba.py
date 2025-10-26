import pandas as pd, numpy as np
from sklearn.preprocessing import MinMaxScaler
from numpy.lib.stride_tricks import sliding_window_view
from typing import Tuple, Optional, List
from config import DATE_COL, TEXT_SENT_COL, TARGET_COL, PRICE_FEATURES, SENT_FEATURES

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL)
    return df

def merge_sentiment(df_prices: pd.DataFrame, df_sent: pd.DataFrame, method="inner", tolerance_days=3) -> pd.DataFrame:
    prices = df_prices.drop_duplicates(subset=[DATE_COL]).sort_values(DATE_COL)
    sent   = df_sent.drop_duplicates(subset=[DATE_COL]).sort_values(DATE_COL)

    if method == "inner":
        merged = pd.merge(prices, sent[[DATE_COL] + SENT_FEATURES], on=DATE_COL, how="inner")
    elif method == "asof":
        merged = pd.merge_asof(
            prices, sent[[DATE_COL] + SENT_FEATURES].sort_values(DATE_COL),
            on=DATE_COL, direction="nearest", tolerance=pd.Timedelta(days=tolerance_days)
        )
        merged[SENT_FEATURES] = merged[SENT_FEATURES].ffill().bfill()
    else:
        raise ValueError("method harus 'inner' atau 'asof'.")
    return merged

def add_simple_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Tambah kolom Return per Ticker agar skala lintas emiten comparable."""
    df = df.sort_values(["Ticker", DATE_COL])
    df["Return"] = df.groupby("Ticker")[TARGET_COL].pct_change()
    return df

def select_numeric_matrix(df: pd.DataFrame, drop_text_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    drop_cols = [DATE_COL]
    if drop_text_cols: drop_cols += drop_text_cols
    num_df = df.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number])
    return num_df.values.astype(float), list(num_df.columns)

def fit_scale(data: np.ndarray, scaler: Optional[MinMaxScaler] = None):
    scaler = scaler or MinMaxScaler()
    return scaler.fit_transform(data), scaler

def prepare_sequences(df_merged: pd.DataFrame, n_steps: int, target_col_name: str, horizon: int, scaler: Optional[MinMaxScaler]=None, step_size:int=1):
    data_num, cols = select_numeric_matrix(df_merged, drop_text_cols=[TEXT_SENT_COL])
    if target_col_name not in cols:
        raise ValueError(f"Target {target_col_name} tidak ada di {cols}")
    t_idx = cols.index(target_col_name)

    data_scaled, scaler = fit_scale(data_num, scaler)
    T, F = data_scaled.shape
    needed = n_steps + horizon
    if T < needed: raise ValueError(f"Data terlalu pendek. Perlu ≥ {needed}, ada {T}")

    X_full = sliding_window_view(data_scaled, (n_steps, F))[:, 0, :, :]  # (T-n_steps+1, n_steps, F)
    tgt    = data_scaled[:, t_idx]

    y_list = []
    max_start = T - (n_steps + horizon) + 1
    for s in range(0, max_start, step_size):
        y_list.append(tgt[s + n_steps : s + n_steps + horizon])
    X = X_full[:max_start:step_size]
    y = np.stack(y_list, axis=0)
    return X, y, scaler, cols

def time_split_by_date(df: pd.DataFrame, train_end: str, val_end: str):
    tr = df[df[DATE_COL] <= pd.Timestamp(train_end)]
    va = df[(df[DATE_COL] > pd.Timestamp(train_end)) & (df[DATE_COL] <= pd.Timestamp(val_end))]
    te = df[df[DATE_COL] > pd.Timestamp(val_end)]
    return tr, va, te
