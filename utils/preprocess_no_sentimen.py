# utils/preprocess_no_sentimen.py (versi revisi)
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

# ========================
# 1. Load CSV
# ========================
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL)
    return df


# ========================
# 2. Tambahkan Return
# ========================
def add_simple_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Tambah kolom Return per Ticker agar skala lintas emiten comparable."""
    df = df.sort_values(["Ticker", DATE_COL])
    df["Return"] = df.groupby("Ticker")[TARGET_COL].pct_change()
    return df


# ========================
# 3. Ambil fitur numerik (tanpa target)
# ========================
def select_numeric_matrix(df: pd.DataFrame, target_col: str, feature_subset: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    """
    Ambil subset fitur numerik jadi matrix numpy.
    """
    drop_cols = [DATE_COL, target_col]
    num_df = df.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number])

    if feature_subset:
        available = [f for f in feature_subset if f in num_df.columns]
        num_df = num_df[available]

    return num_df.values.astype(float), list(num_df.columns)


# ========================
# 4. Scaling (Standard untuk harga, Robust untuk Volume)
# ========================
def fit_scale(data: pd.DataFrame, scaler_dict: Optional[Dict[str, object]] = None, fit: bool = True):
    """
    Scaling hybrid:
    - StandardScaler untuk fitur harga (Open, Close, High, Low dan lag-nya)
    - RobustScaler untuk Volume
    - MinMaxScaler fallback untuk indikator teknikal (RSI, MA, EMA, MACD, dll.)
    """

    scaler_dict = scaler_dict or {}
    df_scaled = data.copy()

    # =========================
    # 🔍 DEBUG 1 — tampilkan semua kolom masuk
    # =========================
    print("\n[DEBUG] 🔍 Masuk ke fit_scale()")
    print(f"[DEBUG] Kolom input data: {list(data.columns)}")

    # =========================
    #  Inisialisasi scaler
    # =========================
    if fit:
        scaler_dict["price"] = StandardScaler()
        scaler_dict["volume"] = RobustScaler()
        scaler_dict["indicator"] = MinMaxScaler()

        # ✅ Gunakan deteksi substring biar lag ikut terbaca
        price_cols = [c for c in data.columns if any(k in c.lower() for k in ["open", "close", "high", "low"])]
        volume_cols = [c for c in data.columns if "volume" in c.lower()]
        indicator_cols = [c for c in data.columns if c not in price_cols + volume_cols]

        # =========================
        # 🔍 DEBUG 2 — tampilkan grouping
        # =========================
        print(f"[DEBUG] Deteksi price_cols: {price_cols}")
        print(f"[DEBUG] Deteksi volume_cols: {volume_cols}")
        print(f"[DEBUG] Deteksi indicator_cols: {indicator_cols}")

        # Fit tiap kelompok fitur
        if price_cols:
            df_scaled[price_cols] = scaler_dict["price"].fit_transform(data[price_cols])
            print(f"[DEBUG] ✅ Scaler 'price' di-fit untuk {len(price_cols)} kolom.")
        else:
            print("[DEBUG] ⚠️ Tidak ada kolom harga terdeteksi untuk scaler 'price'.")

        if volume_cols:
            df_scaled[volume_cols] = scaler_dict["volume"].fit_transform(data[volume_cols])
            print(f"[DEBUG] ✅ Scaler 'volume' di-fit untuk {len(volume_cols)} kolom.")
        else:
            print("[DEBUG] ⚠️ Tidak ada kolom volume terdeteksi untuk scaler 'volume'.")

        if indicator_cols:
            df_scaled[indicator_cols] = scaler_dict["indicator"].fit_transform(data[indicator_cols])
            print(f"[DEBUG] ✅ Scaler 'indicator' di-fit untuk {len(indicator_cols)} kolom.")
        else:
            print("[DEBUG] ⚠️ Tidak ada kolom indikator terdeteksi untuk scaler 'indicator'.")

    else:
        # =========================
        # Transform pakai scaler lama
        # =========================
        price_cols = [c for c in data.columns if any(k in c.lower() for k in ["open", "close", "high", "low"])]
        volume_cols = [c for c in data.columns if "volume" in c.lower()]
        indicator_cols = [c for c in data.columns if c not in price_cols + volume_cols]

        print(f"[DEBUG] (Transform mode) price_cols: {price_cols}")
        print(f"[DEBUG] (Transform mode) volume_cols: {volume_cols}")
        print(f"[DEBUG] (Transform mode) indicator_cols: {indicator_cols}")

        if "price" in scaler_dict and price_cols:
            df_scaled[price_cols] = scaler_dict["price"].transform(data[price_cols])
        else:
            print("[DEBUG] ⚠️ Skip transform price — tidak ada kolom atau scaler belum di-fit.")

        if "volume" in scaler_dict and volume_cols:
            df_scaled[volume_cols] = scaler_dict["volume"].transform(data[volume_cols])
        else:
            print("[DEBUG] ⚠️ Skip transform volume — tidak ada kolom atau scaler belum di-fit.")

        if "indicator" in scaler_dict and indicator_cols:
            df_scaled[indicator_cols] = scaler_dict["indicator"].transform(data[indicator_cols])
        else:
            print("[DEBUG] ⚠️ Skip transform indicator — tidak ada kolom atau scaler belum di-fit.")

    print(f"[DEBUG] ✅ Total kolom output setelah scaling: {len(df_scaled.columns)}")
    print("[DEBUG] ✅ Selesai fit_scale()\n")

    return df_scaled, scaler_dict



# ========================
# 6. Split data (train/val/test)
# ========================
def time_split_hybrid(df: pd.DataFrame,
                      train_end: str, val_end: str,
                      start_date_expected="2015-03-02",
                      train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO,
                      min_needed: int = None):
    """
    Hybrid split:
    - Jika data dimulai tepat dari start_date_expected → pakai split tanggal.
    - Jika tidak → pakai split proporsi.
    - Jika hasil split terlalu pendek → fallback ke proporsi.
    """
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    start_actual = df[DATE_COL].min().date()
    print(f"\n[DEBUG] Mulai time_split_hybrid, start_actual={start_actual}, rows={len(df)}")

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

    print(f"[DEBUG] Split mode={mode}, train={len(tr)}, val={len(va)}, test={len(te)}")
    return tr, va, te, mode


# ========================
# 7. Indikator teknikal (MA, EMA, RSI, MACD + lag)
# ========================
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

    # Lag fitur
    for lag in range(1, 4):
        df[f"{TARGET_COL}_lag{lag}"] = df[TARGET_COL].shift(lag)

    df = df.dropna().reset_index(drop=True)
    return df
