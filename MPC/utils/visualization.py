"""
utils/visualization.py
======================
Semua fungsi plotting menggunakan matplotlib:
  - plot_allocation          : bar chart bobot portofolio
  - plot_portfolio_scenarios : perbandingan skenario best/worst/realized
  - plot_benchmark_comparison: perbandingan MPC vs benchmark (baru)
  - _save_fig                : helper simpan dan tutup figure

Setiap fungsi hanya bertanggung jawab untuk membuat dan menyimpan satu plot.
Tidak ada logika kalkulasi di sini — semua angka sudah dihitung sebelum masuk.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional

from config import (
    PREDICTION_BASE_DIR,
    TOTAL_CAPITAL,
)


# ============================================================
# HELPER INTERNAL
# ============================================================

def _save_fig(path: str) -> None:
    """Rapikan layout, simpan figure ke path, lalu tutup."""
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"[SAVED] {path}")


def _ensure_dir(out_dir: str) -> None:
    """Buat direktori output jika belum ada."""
    os.makedirs(out_dir, exist_ok=True)


# ============================================================
# PLOT ALOKASI BOBOT
# ============================================================

def plot_allocation(
    tickers:  List[str],
    weights:  np.ndarray,
    H:        int,
    out_dir:  str,
    filename: Optional[str] = None,
) -> None:
    """
    Bar chart alokasi bobot portofolio per saham.

    Saham dengan bobot < 0.1% disembunyikan agar chart tetap bersih.
    Label persentase ditampilkan di atas setiap bar.

    Parameter
    ---------
    tickers  : list kode saham
    weights  : array bobot (0–1), harus sum = 1
    H        : horizon (untuk judul dan nama file)
    out_dir  : direktori output
    filename : nama file output (default: allocation_H{H}.pdf)
    """
    _ensure_dir(out_dir)

    mask   = weights > 0.001
    t_show = [t for t, m in zip(tickers, mask) if m]
    w_show = weights[mask]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(t_show, w_show * 100, color="steelblue")

    for bar, w in zip(bars, w_show):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{w*100:.1f}%",
            ha="center", va="bottom", fontsize=9,
        )

    plt.ylabel("Bobot (%)")
    plt.title(f"Alokasi Portofolio MPC — H={H}")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)

    fname = filename or f"allocation_H{H}.pdf"
    _save_fig(os.path.join(out_dir, fname))


# ============================================================
# PLOT SKENARIO PORTOFOLIO
# ============================================================

def plot_portfolio_scenarios(
    df_port:  pd.DataFrame,
    tickers:  List[str],
    weights:  np.ndarray,
    df_risk:  pd.DataFrame,
    H:        int,
    capital:  float,
    out_dir:  str,
    filename: Optional[str] = None,
) -> None:
    """
    Plot perbandingan 3 skenario nilai portofolio sepanjang H hari:
      1. Best Case  : dari prediksi Kalman per hari (non-linear)
      2. Worst Case : pred_return - 2*sigma (dari risk analysis, linear)
      3. Realized   : dari data aktual

    Parameter
    ---------
    df_port  : DataFrame output dari evaluate_portfolio (berisi Portfolio_Value)
    tickers  : list kode saham dalam portofolio
    weights  : array bobot portofolio
    df_risk  : DataFrame output dari analyze_risk (berisi Worst_Case_Pct)
    H        : horizon hari
    capital  : modal awal (Rp)
    out_dir  : direktori output
    filename : nama file output (default: scenarios_H{H}.pdf)
    """
    _ensure_dir(out_dir)

    # --- 1. BEST CASE: dari prediksi Kalman per hari ---
    best_case_values = _build_best_case(tickers, weights, H, capital)

    # --- 3. REALIZED: dari data aktual ---
    realized_values = df_port["Portfolio_Value"].values
    realized_days   = np.arange(len(realized_values))

    # --- PLOT ---
    plt.figure(figsize=(12, 7))

    if best_case_values is not None:
        days_best = np.arange(len(best_case_values))
        best_ret  = (best_case_values[-1] / capital) - 1
        plt.plot(
            days_best, best_case_values,
            color="green", linewidth=2, linestyle="--",
            label=f"Best Case (Prediksi LSTM): +{best_ret*100:.2f}%",
            marker="o", markersize=4,
            markevery=max(1, H // 10), alpha=0.8,
        )

    realized_ret = (realized_values[-1] / capital) - 1
    plt.plot(
        realized_days, realized_values,
        color="blue", linewidth=2.5,
        label=f"Realized (Aktual): {realized_ret*100:+.2f}%",
        marker="^", markersize=5,
        markevery=max(1, len(realized_days) // 10),
    )

    plt.axhline(y=capital, color="gray", linestyle=":", alpha=0.5, label="Modal Awal")

    plt.xlabel("Hari ke-", fontsize=11)
    plt.ylabel("Nilai Portofolio (Rp)", fontsize=11)
    plt.title(
        f"Perbandingan Skenario Portofolio — H={H} hari\n"
        f"Modal: Rp{capital:,.0f}",
        fontsize=13, fontweight="bold",
    )
    plt.legend(loc="best", fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)

    ax = plt.gca()
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"Rp{x:,.0f}")
    )

    fname = filename or f"scenarios_H{H}.pdf"
    _save_fig(os.path.join(out_dir, fname))


# ============================================================
# PLOT PERBANDINGAN BENCHMARK (fungsi baru)
# ============================================================

def plot_benchmark_comparison(
    results:  dict,
    H:        int,
    capital:  float,
    out_dir:  str,
    filename: Optional[str] = None,
) -> None:
    """
    Plot perbandingan kumulatif return antara MPC dan benchmark.

    Parameter
    ---------
    results : dict dengan struktur:
              {
                "MPC":          {"df_port": pd.DataFrame, "color": "blue"},
                "Equal Weight": {"df_port": pd.DataFrame, "color": "orange"},
                "Mean-Variance":{"df_port": pd.DataFrame, "color": "green"},
              }
    H       : horizon hari
    capital : modal awal (Rp)
    out_dir : direktori output
    filename: nama file output (default: benchmark_H{H}.pdf)

    Catatan: df_port harus memiliki kolom Portfolio_Cumulative_Return.
    """
    _ensure_dir(out_dir)
    plt.figure(figsize=(12, 6))

    for method_name, info in results.items():
        df   = info.get("df_port")
        color = info.get("color", None)

        if df is None or df.empty:
            print(f"[WARN] Data {method_name} kosong, dilewati")
            continue

        cum_ret      = df["Portfolio_Cumulative_Return"].values
        final_ret    = cum_ret[-1] * 100
        days         = np.arange(len(cum_ret))

        plt.plot(
            days, cum_ret * 100,
            label=f"{method_name}: {final_ret:+.2f}%",
            color=color, linewidth=2,
        )

    plt.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    plt.xlabel("Hari ke-", fontsize=11)
    plt.ylabel("Cumulative Return (%)", fontsize=11)
    plt.title(
        f"Perbandingan MPC vs Benchmark — H={H} hari\n"
        f"Modal: Rp{capital:,.0f}",
        fontsize=13, fontweight="bold",
    )
    plt.legend(loc="best", fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)

    fname = filename or f"benchmark_H{H}.pdf"
    _save_fig(os.path.join(out_dir, fname))


# ============================================================
# HELPER BANGUN SKENARIO
# ============================================================

def _build_best_case(
    tickers: List[str],
    weights: np.ndarray,
    H:       int,
    capital: float,
) -> Optional[np.ndarray]:
    """
    Bangun array nilai portofolio best case dari prediksi Kalman per hari.

    Return None jika tidak ada data prediksi yang bisa dibaca.
    """
    best_case_daily = []

    for i, ticker in enumerate(tickers):
        pred_file = os.path.join(
            PREDICTION_BASE_DIR, f"H{H}", f"{ticker}_H{H}_forward_pred.csv"
        )
        if not os.path.exists(pred_file):
            continue

        df_pred = pd.read_csv(pred_file)
        if "Pred_Close_Kalman" not in df_pred.columns or len(df_pred) < H:
            continue

        kalman    = pd.to_numeric(df_pred["Pred_Close_Kalman"], errors="coerce").dropna()
        daily_ret = np.log(kalman / kalman.shift(1)).fillna(0).values[:H]
        best_case_daily.append(daily_ret)

    if not best_case_daily:
        return None

    w_used = weights[:len(best_case_daily)]
    weighted_daily = np.average(best_case_daily, axis=0, weights=w_used)
    best_cumret    = np.exp(np.cumsum(weighted_daily)) - 1

    # Tambahkan titik awal (hari 0 = modal penuh)
    return capital * (1 + np.insert(best_cumret, 0, 0))


