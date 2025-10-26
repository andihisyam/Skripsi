import argparse
import os
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import DATE_COL,TARGET_COL,PRICE_FEATURES,SENT_FEATURES,TICKER_COL

# === import loader dari utils ===
from utils.io_utils import load_prices_from_folder

# =========================
# Utils
# =========================
def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def load_prices_auto(prices_path: str, date_col: str, ticker_col: str) -> pd.DataFrame:
    """
    Jika prices_path adalah folder -> gabungkan semua CSV per emiten.
    Jika prices_path adalah file CSV -> baca langsung (harus sudah ada kolom Ticker).
    """
    if os.path.isdir(prices_path):
        df = load_prices_from_folder(prices_path)
    else:
        df = pd.read_csv(prices_path)
        if date_col not in df.columns:
            raise ValueError(f"Kolom tanggal '{date_col}' tidak ditemukan di {prices_path}")
        if ticker_col not in df.columns:
            raise ValueError(f"Kolom ticker '{ticker_col}' tidak ditemukan di {prices_path}")
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values([ticker_col, date_col])
        for c in PRICE_FEATURES:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def basic_quality_checks(prices: pd.DataFrame, outdir: str, date_col: str, ticker_col: str) -> None:
    miss = prices.isna().sum().sort_values(ascending=False)
    miss.to_csv(os.path.join(outdir, "missing_counts_prices.csv"))
    dup = prices.duplicated(subset=[ticker_col, date_col]).sum()
    with open(os.path.join(outdir, "duplicates_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Duplikat (by {ticker_col},{date_col}): {dup}\n")
    cov = (prices.groupby(ticker_col)[date_col]
           .agg(['min', 'max', 'count'])
           .rename(columns={'min': 'first_date', 'max': 'last_date', 'count': 'n_rows'}))
    cov["n_days"] = cov["n_rows"]
    cov.to_csv(os.path.join(outdir, "coverage_per_ticker.csv"))
    prices.head(20).to_csv(os.path.join(outdir, "sample_prices_head20.csv"), index=False)

def add_returns(prices: pd.DataFrame, date_col: str, ticker_col: str, target_col: str) -> pd.DataFrame:
    prices = prices.sort_values([ticker_col, date_col]).copy()
    prices["Return"] = prices.groupby(ticker_col)[target_col].pct_change()
    return prices

def equal_weight_market_return(prices: pd.DataFrame, date_col: str, ticker_col: str) -> pd.DataFrame:
    ret = prices[[date_col, ticker_col, "Return"]].dropna()
    mkt = (ret.groupby(date_col)["Return"].mean()
           .rename("Mkt_Return").to_frame().reset_index())
    return mkt

def merge_prices_sentiment(prices: pd.DataFrame, sent: pd.DataFrame, date_col: str) -> pd.DataFrame:
    return pd.merge(prices, sent[[date_col] + SENT_FEATURES + ["S_net"]], on=date_col, how="inner")

def pivot_returns_wide(prices: pd.DataFrame, date_col: str, ticker_col: str) -> pd.DataFrame:
    return prices.pivot_table(index=date_col, columns=ticker_col, values="Return")

# =========================
# Plot helpers (1 chart per figure, no styles)
# =========================
def plot_time_series(df: pd.DataFrame, x: str, y: str, title: str, outpath: str) -> None:
    plt.figure()
    plt.plot(df[x], df[y])
    plt.title(title)
    plt.xlabel(x); plt.ylabel(y)
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()

def plot_hist(series: pd.Series, title: str, outpath: str, bins: int = 50) -> None:
    plt.figure()
    plt.hist(series.dropna().values, bins=bins)
    plt.title(title)
    plt.xlabel("Value"); plt.ylabel("Frequency")
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()

def plot_box_by_year(prices: pd.DataFrame, date_col: str, value_col: str, title: str, outpath: str) -> None:
    df = prices[[date_col, value_col]].dropna().copy()
    df["Year"] = df[date_col].dt.year
    data = [df[df["Year"] == y][value_col].values for y in sorted(df["Year"].unique())]
    plt.figure()
    plt.boxplot(data, labels=sorted(df["Year"].unique()))
    plt.title(title)
    plt.xlabel("Year"); plt.ylabel(value_col)
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()



def plot_scatter(x: pd.Series, y: pd.Series, title: str, xlabel: str, ylabel: str, outpath: str) -> None:
    plt.figure()
    plt.scatter(x, y, s=8)
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()

# =========================
# Main EDA pipeline
# =========================
def run_eda(prices_input: str, outdir: str,
            date_col: str, ticker_col: str, target_col: str) -> None:
    ensure_outdir(outdir)

    # 1) Load harga (folder atau file)
    prices = load_prices_auto(prices_input, date_col, ticker_col)

    # 2) Quality checks
    basic_quality_checks(prices, outdir, date_col, ticker_col)

    # 3) Returns
    prices = add_returns(prices, date_col, ticker_col, target_col)
    prices.to_csv(os.path.join(outdir, "prices_with_returns.csv"), index=False)

    # ========================= PLOTS PER EMITEN =========================
    all_tickers = prices[ticker_col].unique()

    for t in all_tickers:
        g = prices[prices[ticker_col] == t][[date_col, target_col]].dropna()
        if len(g) > 10:
            # Grafik harga
            plot_time_series(
                g, date_col, target_col,
                f"Harga {t} - {target_col}",
                os.path.join(outdir, f"ts_price_{t}.png")
            )

            # Histogram return
            gr = prices[prices[ticker_col] == t]["Return"]
            plot_hist(
                gr, f"Distribusi Return Harian {t}",
                os.path.join(outdir, f"hist_return_{t}.png")
            )

            # Boxplot return tahunan
            gt = prices[prices[ticker_col] == t][[date_col, "Return"]].dropna()
            if len(gt) > 50:
                plot_box_by_year(
                    gt, date_col, "Return",
                    f"Boxplot Return Tahunan {t}",
                    os.path.join(outdir, f"box_return_year_{t}.png")
                )

            # Summary statistik per ticker
            stats = g.copy()
            stats["Return"] = prices[prices[ticker_col] == t]["Return"]
            ret_stats = stats["Return"].agg(['mean','std','min','max','count']).to_frame().T
            ret_stats.to_csv(os.path.join(outdir, f"stats_{t}.csv"), index=False)

    # ===== Ringkasan angka global =====
    with open(os.path.join(outdir, "key_numbers.txt"), "w", encoding="utf-8") as f:
        f.write(f"Jumlah ticker: {prices[ticker_col].nunique()}\n")
        f.write(f"Rentang tanggal prices: {prices[date_col].min().date()} .. {prices[date_col].max().date()}\n")

    print(f"[EDA] Selesai. Output disimpan di: {outdir}")
