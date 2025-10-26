# EDA.py
import argparse
import os
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import DATE_COL,TARGET_COL,PRICE_FEATURES,SENT_FEATURES,TICKER_COL

# === import loader dari utils ===
from utils.io_utils import load_prices_from_folder, load_sentiment

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

def plot_heatmap_corr(corr: pd.DataFrame, title: str, outpath: str) -> None:
    plt.figure()
    plt.imshow(corr.values, aspect='auto', interpolation='nearest')
    plt.title(title); plt.colorbar()
    plt.xticks(ticks=np.arange(len(corr.columns)), labels=corr.columns, rotation=90)
    plt.yticks(ticks=np.arange(len(corr.index)), labels=corr.index)
    plt.tight_layout(); plt.savefig(outpath, dpi=200); plt.close()

def plot_scatter(x: pd.Series, y: pd.Series, title: str, xlabel: str, ylabel: str, outpath: str) -> None:
    plt.figure()
    plt.scatter(x, y, s=8)
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()

# =========================
# Main EDA pipeline
# =========================
def run_eda(prices_input: str, sentiment_path: str, outdir: str,
            date_col: str, ticker_col: str, target_col: str,
            sample_tickers: List[str] = None) -> None:
    ensure_outdir(outdir)

    # 1) Load (folder atau file)
    prices = load_prices_auto(prices_input, date_col, ticker_col)
    sent = load_sentiment(sentiment_path)

    # 2) Quality checks
    basic_quality_checks(prices, outdir, date_col, ticker_col)

    # 3) Returns
    prices = add_returns(prices, date_col, ticker_col, target_col)
    prices.to_csv(os.path.join(outdir, "prices_with_returns.csv"), index=False)

    # 4) Market equal-weight return
    mkt = equal_weight_market_return(prices, date_col, ticker_col)
    mkt.to_csv(os.path.join(outdir, "market_equal_weight_return.csv"), index=False)

    # 5) Merge sentiment (inner)
    merged = merge_prices_sentiment(prices, sent, date_col)
    merged.to_csv(os.path.join(outdir, "merged_prices_sentiment.csv"), index=False)

    # 6) Korelasi lintas emiten
    wide_ret = pivot_returns_wide(prices, date_col, ticker_col)
    corr = wide_ret.corr(min_periods=50)
    corr.to_csv(os.path.join(outdir, "corr_returns_between_tickers.csv"))

    # 7) Sentiment aggregates
    sent["S_net_rm7"] = sent["S_net"].rolling(7, min_periods=1).mean()
    sent["S_net_rm14"] = sent["S_net"].rolling(14, min_periods=1).mean()
    sent.to_csv(os.path.join(outdir, "sentiment_enriched.csv"), index=False)

    # 8) Sentimen (lag aman) vs market return
    df_mkt_sent = pd.merge(mkt, sent[[date_col, "S_net", "S_net_rm7", "S_net_rm14"]], on=date_col, how="inner").copy()
    df_mkt_sent["S_net_lag1"] = df_mkt_sent["S_net"].shift(1)
    df_mkt_sent["S_net_rm7_lag1"] = df_mkt_sent["S_net_rm7"].shift(1)
    df_mkt_sent["S_net_rm14_lag1"] = df_mkt_sent["S_net_rm14"].shift(1)
    df_mkt_sent.to_csv(os.path.join(outdir, "market_sentiment_lagged.csv"), index=False)

    # ========================= PLOTS =========================
    # (A) beberapa ticker contoh
    if sample_tickers is None:
        top5 = (prices.groupby(ticker_col)[date_col].count()
                .sort_values(ascending=False).head(5).index.tolist())
    else:
        top5 = sample_tickers

    for t in top5:
        g = prices[prices[ticker_col] == t][[date_col, target_col]].dropna()
        if len(g) > 10:
            plot_time_series(g, date_col, target_col, f"Harga {t} - {target_col}",
                             os.path.join(outdir, f"ts_price_{t}.png"))
            gr = prices[prices[ticker_col] == t]["Return"]
            plot_hist(gr, f"Distribusi Return Harian {t}", os.path.join(outdir, f"hist_return_{t}.png"))
            gt = prices[prices[ticker_col] == t][[date_col, "Return"]].dropna()
            if len(gt) > 50:
                plot_box_by_year(gt, date_col, "Return", f"Boxplot Return Tahunan {t}",
                                 os.path.join(outdir, f"box_return_year_{t}.png"))

    # (B) Heatmap korelasi (subset agar gambar tidak kelewat besar)
    few = wide_ret.dropna(axis=1, thresh=int(0.7 * len(wide_ret))).iloc[:, :40]
    if few.shape[1] >= 2:
        corr_few = few.corr(min_periods=50)
        plot_heatmap_corr(corr_few, "Korelasi Return Antar Emiten (subset)",
                          os.path.join(outdir, "heatmap_corr_returns.png"))

    # (C) Sentiment time series
    plot_time_series(sent, date_col, "S_net", "Net Sentiment Harian (Positif - Negatif)",
                     os.path.join(outdir, "ts_snet.png"))
    plot_time_series(sent, date_col, "S_net_rm7", "Net Sentiment Rolling 7D",
                     os.path.join(outdir, "ts_snet_rm7.png"))
    plot_time_series(sent, date_col, "S_net_rm14", "Net Sentiment Rolling 14D",
                     os.path.join(outdir, "ts_snet_rm14.png"))

    # (D) Scatter S_net(t-1) vs Mkt_Return(t)
    dsc = df_mkt_sent.dropna(subset=["Mkt_Return", "S_net_lag1"])
    if len(dsc) > 50:
        plot_scatter(dsc["S_net_lag1"], dsc["Mkt_Return"],
                     "S_net(t-1) vs Mkt_Return(t)", "S_net lag 1", "Market Return",
                     os.path.join(outdir, "scatter_snetlag1_vs_mktreturn.png"))

    # (E) Rolling correlation 60D
    roll = dsc[[DATE_COL, "S_net_lag1", "Mkt_Return"]].set_index(DATE_COL).copy()
    if len(roll) > 120:
        rc = roll["S_net_lag1"].rolling(60).corr(roll["Mkt_Return"]).rename("rolling_corr_60").reset_index()
        plot_time_series(rc.dropna(), DATE_COL, "rolling_corr_60",
                         "Rolling Corr 60D: S_net(t-1) vs Mkt_Return(t)",
                         os.path.join(outdir, "rolling_corr_snet_mkt.png"))

    # ===== Ringkasan angka =====
    ret_stats = (prices.groupby(TICKER_COL)["Return"]
                 .agg(['mean', 'std', 'min', 'max', 'count'])
                 .rename(columns={'mean': 'ret_mean', 'std': 'ret_std', 'count': 'n'}))
    ret_stats.to_csv(os.path.join(outdir, "return_stats_per_ticker.csv"))

    corr_pair = dsc[["Mkt_Return", "S_net_lag1"]].corr().iloc[0, 1] if len(dsc) > 10 else np.nan
    with open(os.path.join(outdir, "key_numbers.txt"), "w", encoding="utf-8") as f:
        f.write(f"Korelasi(Mkt_Return, S_net_lag1): {corr_pair:.4f}\n")
        f.write(f"Jumlah ticker: {prices[TICKER_COL].nunique()}\n")
        f.write(f"Rentang tanggal prices: {prices[DATE_COL].min().date()} .. {prices[DATE_COL].max().date()}\n")
        f.write(f"Rentang tanggal sentimen: {sent[DATE_COL].min().date()} .. {sent[DATE_COL].max().date()}\n")

    print(f"[EDA] Selesai. Output disimpan di: {outdir}")

# =========================
# CLI
# =========================
def parse_args():
    p = argparse.ArgumentParser(description="EDA untuk LQ45 + Sentimen Pasar (harian)")
    p.add_argument("--prices", help="Path CSV gabungan (punya kolom Ticker) ATAU folder berisi banyak CSV per emiten")
    p.add_argument("--prices_folder", help="Alternatif: folder berisi ACES.csv, BBNI.csv, ...")
    p.add_argument("--sentiment", required=True, help="Path CSV sentimen: kolom Date, Positif, Negatif, Netral")
    p.add_argument("--outdir", default="eda_output", help="Folder output untuk grafik & ringkasan")
    p.add_argument("--date_col", default=DATE_COL)
    p.add_argument("--ticker_col", default=TICKER_COL)
    p.add_argument("--target_col", default=TARGET_COL)
    p.add_argument("--sample_tickers", nargs="*", default=None, help="Daftar ticker contoh plot (opsional)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # Tentukan sumber harga: --prices_folder lebih prioritas jika diisi
    prices_input = args.prices_folder if args.prices_folder else (args.prices or "")
    if not prices_input:
        raise SystemExit("Harus isi --prices_folder (folder per emiten) atau --prices (file CSV gabungan).")

    run_eda(
        prices_input=prices_input,
        sentiment_path=args.sentiment,
        outdir=args.outdir,
        date_col=args.date_col,
        ticker_col=args.ticker_col,
        target_col=args.target_col,
        sample_tickers=args.sample_tickers
    )
