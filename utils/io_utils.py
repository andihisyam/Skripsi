# utils/io_utils.py
import os
import glob
import pandas as pd
from config import DATE_COL, PRICE_FEATURES



def load_prices_from_folder(folder_path: str) -> pd.DataFrame:
    """
    Membaca semua CSV di folder, menambahkan kolom Ticker dari nama file,
    lalu menggabungkannya jadi satu DataFrame.
    
    Struktur CSV per file (contoh ACES.csv):
    Date,Close,High,Low,Open,Volume
    2015-03-02,652.68,668.41,644.82,668.41,23352800
    ...
    """
    all_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    if not all_files:
        raise FileNotFoundError(f"Tidak ada file CSV di folder: {folder_path}")

    frames = []
    for fp in all_files:
        ticker = os.path.splitext(os.path.basename(fp))[0].upper()  # "ACES.csv" -> "ACES"
        df = pd.read_csv(fp)

        if DATE_COL not in df.columns:
            raise ValueError(f"{fp}: kolom '{DATE_COL}' tidak ditemukan")

        # pastikan kolom harga lengkap
        missing = [c for c in PRICE_FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"{fp}: kolom wajib hilang: {missing}")

        # parsing tanggal + sort
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL)

        # ubah kolom harga jadi numerik
        for c in PRICE_FEATURES:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df["Ticker"] = ticker
        frames.append(df)

    prices = pd.concat(frames, ignore_index=True)
    prices = prices.sort_values(["Ticker", DATE_COL]).reset_index(drop=True)
    return prices


def load_sentiment(path: str) -> pd.DataFrame:
    """
    Membaca CSV sentimen pasar dengan format:
    Date,Positif,Negatif,Netral
    """
    df = pd.read_csv(path)
    if DATE_COL not in df.columns:
        raise ValueError(f"{path}: kolom '{DATE_COL}' tidak ditemukan")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL)

    for c in ["Positif", "Negatif", "Netral"]:
        if c not in df.columns:
            raise ValueError(f"{path}: kolom '{c}' tidak ditemukan")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # net sentiment
    df["S_net"] = df["Positif"] - df["Negatif"]
    return df

def load_price_single(path: str) -> pd.DataFrame:
    """
    Membaca satu CSV harga saham (bukan folder).
    Struktur CSV (contoh ACES.csv):
    Date,Close,High,Low,Open,Volume
    """
    import pandas as pd
    from config import DATE_COL, PRICE_FEATURES
    import os

    ticker = os.path.splitext(os.path.basename(path))[0].upper()
    df = pd.read_csv(path)

    if DATE_COL not in df.columns:
        raise ValueError(f"{path}: kolom '{DATE_COL}' tidak ditemukan")

    # pastikan kolom harga lengkap
    missing = [c for c in PRICE_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: kolom wajib hilang: {missing}")

    # parsing tanggal + sort
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL)

    # ubah kolom harga jadi numerik
    for c in PRICE_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Ticker"] = ticker
    return df
