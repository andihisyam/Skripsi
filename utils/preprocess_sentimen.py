import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from numpy.lib.stride_tricks import sliding_window_view
from typing import Optional, List, Dict
from config import (
    DATE_COL, PRICE_COL, TARGET_COL, N_STEPS, H_1M,
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
# 2. LOAD SENTIMEN
# ================================================================
SENTIMEN_COLS = ["Positif", "Negatif", "Netral"]

def load_sentimen(path: str) -> pd.DataFrame:
    """
    Load file sentimen (Excel atau CSV).
    Kolom yang diharapkan: Date, Positif, Negatif, Netral (proporsi 0-1).

    Return: DataFrame dengan kolom Date + SENTIMEN_COLS, sudah di-sort ascending.
    """
    if path.endswith(".xlsx") or path.endswith(".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)

    # dukung alias kolom sentimen (Indonesia / Inggris)
    alias_map = {
        "Positif": ["Positif", "Positive"],
        "Negatif": ["Negatif", "Negative"],
        "Netral": ["Netral", "Neutral"],
    }
    rename_map = {}
    for standar, kandidat in alias_map.items():
        if standar in df.columns:
            continue
        for kolom in kandidat:
            if kolom in df.columns:
                rename_map[kolom] = standar
                break
    if rename_map:
        df = df.rename(columns=rename_map)

    # pastikan kolom sentimen ada dan numerik
    for col in SENTIMEN_COLS:
        if col not in df.columns:
            raise ValueError(
                f"[load_sentimen] Kolom '{col}' tidak ditemukan. "
                f"Kolom tersedia: {list(df.columns)}"
            )
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df[[DATE_COL] + SENTIMEN_COLS]


def merge_sentimen(df: pd.DataFrame, df_sentimen: pd.DataFrame) -> pd.DataFrame:
    """
    Merge sentimen ke data saham berdasarkan tanggal.
    Hari libur/weekend yang tidak ada di sentimen di-forward fill.
    Catatan: sengaja TANPA backward fill untuk menghindari lookahead leakage.
    """
    df = df.copy()
    df = df.merge(df_sentimen, on=DATE_COL, how="left")

    # Forward fill hanya dari masa lalu -> aman untuk time-series.
    if "Ticker" in df.columns:
        df[SENTIMEN_COLS] = (
            df.sort_values(["Ticker", DATE_COL])
              .groupby("Ticker")[SENTIMEN_COLS]
              .ffill()
        )
    else:
        df[SENTIMEN_COLS] = df.sort_values(DATE_COL)[SENTIMEN_COLS].ffill()

    # Sisa NaN (umumnya di awal seri) diisi netral.
    df[SENTIMEN_COLS] = df[SENTIMEN_COLS].fillna({"Positif": 0.33, "Negatif": 0.33, "Netral": 0.34})

    return df


# ================================================================
# 3. SIMPLE RETURN
# ================================================================
def add_simple_returns(df: pd.DataFrame, out_col: str = TARGET_COL) -> pd.DataFrame:
    df = df.sort_values(["Ticker", DATE_COL]).copy()
    df[out_col] = df.groupby("Ticker")[PRICE_COL].pct_change()
    return df


def add_return_lags(df: pd.DataFrame, lags=(1, 2, 3), return_col: str = TARGET_COL) -> pd.DataFrame:
    df = df.sort_values(["Ticker", DATE_COL]).copy()
    g = df.groupby("Ticker", group_keys=False)
    for lag in lags:
        df[f"{return_col}_lag{lag}"] = g[return_col].shift(lag)
    return df


def add_forward_return(df: pd.DataFrame, H: int, out_col: str, log: bool = True) -> pd.DataFrame:
    """
    Untuk H_output=1: shift target H hari ke depan (log return total).
    Untuk H_output=H: tidak dipakai — target multi-step dibuat di prepare_sequences.
    """
    df = df.sort_values(["Ticker", DATE_COL]).copy()
    g = df.groupby("Ticker", group_keys=False)[PRICE_COL]
    future = g.shift(-H)
    now    = df[PRICE_COL].astype(float)
    if log:
        df[out_col] = np.log(future / now)
    else:
        df[out_col] = (future / now) - 1.0
    return df


# ================================================================
# 4. AMBIL FITUR NUMERIK
# ================================================================
def select_numeric_matrix(df: pd.DataFrame, target_col: str, feature_subset=None):
    drop_cols = [DATE_COL, target_col, PRICE_COL]
    num_df = df.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number])
    if feature_subset:
        available = [f for f in feature_subset if f in num_df.columns]
        num_df = num_df[available]
    return num_df.values.astype(float), list(num_df.columns)


# ================================================================
# 5. FIT HYBRID SCALER UNTUK FITUR
# ================================================================
def fit_scale(
    data: pd.DataFrame,
    scaler_dict: Optional[Dict[str, object]] = None,
    fit: bool = True,
):
    scaler_dict = scaler_dict or {}
    df_scaled   = data.copy()

    price_cols     = [c for c in data.columns if any(k in c.lower() for k in ["open", "close", "high", "low"])]
    volume_cols    = [c for c in data.columns if "volume" in c.lower()]
    # sentimen masuk ke indicator_cols → MinMaxScaler cocok karena sudah proporsi 0-1
    indicator_cols = [c for c in data.columns if c not in price_cols + volume_cols]

    if fit:
        scaler_dict["price"]     = StandardScaler()
        scaler_dict["volume"]    = RobustScaler()
        scaler_dict["indicator"] = MinMaxScaler()

        if price_cols:
            df_scaled[price_cols]     = scaler_dict["price"].fit_transform(data[price_cols])
        if volume_cols:
            df_scaled[volume_cols]    = scaler_dict["volume"].fit_transform(data[volume_cols])
        if indicator_cols:
            df_scaled[indicator_cols] = scaler_dict["indicator"].fit_transform(data[indicator_cols])
    else:
        if "price" in scaler_dict and price_cols:
            df_scaled[price_cols]     = scaler_dict["price"].transform(data[price_cols])
        if "volume" in scaler_dict and volume_cols:
            df_scaled[volume_cols]    = scaler_dict["volume"].transform(data[volume_cols])
        if "indicator" in scaler_dict and indicator_cols:
            df_scaled[indicator_cols] = scaler_dict["indicator"].transform(data[indicator_cols])

    return df_scaled, scaler_dict


# ================================================================
# 6. SCALE TARGET
# ================================================================
def scale_target(
    y: np.ndarray,
    scaler_dict: dict,
    fit: bool,
    target_key: str = "target",
):
    if fit:
        scaler   = StandardScaler()
        y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()
        scaler_dict[target_key] = scaler
    else:
        if target_key not in scaler_dict:
            raise KeyError(
                f"[scale_target] key '{target_key}' tidak ada di scaler_dict. "
                f"Pastikan fit=True dipanggil lebih dulu dengan key yang sama."
            )
        scaler   = scaler_dict[target_key]
        y_scaled = scaler.transform(y.reshape(-1, 1)).flatten()
    return y_scaled, scaler_dict


# ================================================================
# 7. WINDOWING — support H_output=1 dan H_output>1 (multi-step)
# ================================================================
def prepare_sequences(
    df: pd.DataFrame,
    target_col_name: str,
    horizon: int = 1,
    feature_subset: Optional[List[str]] = None,
    scaler_dict: Optional[Dict[str, object]] = None,
    is_train: bool = True,
    step_size: int = 1,
    H_output: int = 1,          # ← BARU: jumlah step output
    verbose: bool = True,
):
    """
    is_train=True  → fit scaler (panggil untuk train set)
    is_train=False → transform saja (val/test)

    H_output=1  → target: 1 angka (log return total H hari) — mode lama
    H_output>1  → target: H_output angka berurutan (return harian hari 1..H_output)
                  dalam mode ini, 'horizon' diabaikan karena multi-step langsung
                  dari return harian (ret_1d)
    """
    # --- 1) AMBIL FITUR ---
    all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if feature_subset is None:
        cols = [c for c in all_numeric_cols if c != target_col_name]
    else:
        cols = [c for c in feature_subset if c in all_numeric_cols and c != target_col_name]

    data_df = df[cols].copy()
    if verbose:
        print(f"Fitur: {data_df.shape}, is_train={is_train}, H_output={H_output}")

    # --- 2) SCALING FITUR ---
    data_scaled_df, scaler_dict = fit_scale(data_df, scaler_dict, fit=is_train)
    data_scaled = data_scaled_df.values

    # --- 3) TARGET & SCALING ---
    tgt        = pd.to_numeric(df[target_col_name], errors="coerce").values.astype(float)
    target_key = f"target_{target_col_name}"
    tgt_scaled, scaler_dict = scale_target(tgt, scaler_dict, fit=is_train, target_key=target_key)
    tgt_scaled = tgt_scaled.flatten()

    # --- 4) WINDOWING ---
    T, F = data_scaled.shape

    if H_output == 1:
        # ── Mode lama: 1 target per window ──
        if T < N_STEPS + horizon:
            raise ValueError(f"Data terlalu pendek: perlu >= {N_STEPS + horizon}, ada {T}")

        X_full    = sliding_window_view(data_scaled, (N_STEPS, F))[:, 0, :, :]
        max_start = T - N_STEPS - horizon + 1

        y_list = []
        for s in range(0, max_start, step_size):
            target_idx = s + N_STEPS + horizon - 1
            y_list.append(tgt_scaled[target_idx])

        X = X_full[:max_start:step_size]
        y = np.asarray(y_list).reshape(-1, 1)

    else:
        # ── Mode baru: H_output target per window (multi-step) ──
        # Target adalah return harian hari ke-1 sampai H_output setelah window
        # tgt_scaled harus berisi return harian (ret_1d setelah scaling)
        if T < N_STEPS + H_output:
            raise ValueError(
                f"Data terlalu pendek untuk multi-step: "
                f"perlu >= {N_STEPS + H_output}, ada {T}"
            )

        X_full    = sliding_window_view(data_scaled, (N_STEPS, F))[:, 0, :, :]
        max_start = T - N_STEPS - H_output + 1

        y_list = []
        for s in range(0, max_start, step_size):
            # ambil H_output nilai target berurutan setelah window
            target_seq = tgt_scaled[s + N_STEPS: s + N_STEPS + H_output]
            y_list.append(target_seq)

        X = X_full[:max_start:step_size]
        y = np.asarray(y_list)   # shape: (N, H_output)

    if verbose:
        print(f"X={X.shape}, y={y.shape}, target={target_col_name}")

    return X, y, scaler_dict, cols


# ================================================================
# 8. SPLIT (Hybrid)
# ================================================================
def time_split_hybrid(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
    start_date_expected: str = "2015-03-02",
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    min_needed: int = None,
):
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
        n       = len(df)
        n_train = int(train_ratio * n)
        n_val   = int(val_ratio * n)
        tr      = df.iloc[:n_train]
        va      = df.iloc[n_train:n_train + n_val]
        te      = df.iloc[n_train + n_val:]
        mode    = "ratio"

    return tr, va, te, mode


# ================================================================
# 9. INDIKATOR TEKNIKAL
# ================================================================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Ticker", DATE_COL]).copy()
    g  = df.groupby("Ticker", group_keys=False)

    for p in MA_PERIODS:
        df[f"MA{p}"] = g[PRICE_COL].apply(lambda s: s.rolling(p).mean())

    for p in EMA_PERIODS:
        df[f"EMA{p}"] = g[PRICE_COL].apply(lambda s: s.ewm(span=p, adjust=False).mean())

    for p in RSI_PERIODS:
        def rsi(series, period=p):
            delta = series.diff()
            gain  = delta.clip(lower=0).rolling(period).mean()
            loss  = (-delta.clip(upper=0)).rolling(period).mean()
            rs    = gain / (loss + 1e-9)
            return 100 - (100 / (1 + rs))
        df[f"RSI{p}"] = g[PRICE_COL].apply(rsi)

    for fast, slow in MACD_CONFIGS:
        def macd(series, f=fast, s=slow):
            return series.ewm(span=f, adjust=False).mean() - series.ewm(span=s, adjust=False).mean()
        df[f"MACD_{fast}_{slow}"] = g[PRICE_COL].apply(macd)

    for lag in range(1, 4):
        df[f"{PRICE_COL}_lag{lag}"] = g[PRICE_COL].shift(lag)

    return df
