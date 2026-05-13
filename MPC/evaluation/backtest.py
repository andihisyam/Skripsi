"""
evaluation/backtest.py
======================
Evaluasi performa portofolio menggunakan data aktual:
  - evaluate_portfolio   : hitung realized return, volatility, Sharpe ratio
  - evaluate_all_methods : evaluasi MPC + semua benchmark sekaligus
  - print_eval_summary   : cetak ringkasan hasil evaluasi ke konsol

Modul ini hanya menghitung metrik — tidak ada plotting atau file I/O di sini.
Untuk plotting gunakan utils/visualization.py.
Untuk save hasil gunakan utils/file_utils.py.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from config import (
    RISK_FREE_RATE,
    TOTAL_CAPITAL,
)
from utils.data_loader import build_portfolio_timeseries


# ============================================================
# EVALUASI SATU PORTOFOLIO
# ============================================================

def evaluate_portfolio(
    weights: np.ndarray,
    tickers: List[str],
    H:       int,
    capital: float = TOTAL_CAPITAL,
) -> dict:
    """
    Hitung metrik performa portofolio dari data aktual.

    Metrik yang dihitung
    --------------------
    realized_return  : cumulative return akhir periode
    realized_profit  : keuntungan/kerugian dalam Rp
    volatility       : standar deviasi return harian
    sharpe           : Sharpe ratio = (mean_daily - rf) / vol_daily
    max_drawdown     : maksimum penurunan dari puncak ke lembah
    realized_returns : array realized return per ticker

    Parameter
    ---------
    weights  : (N,) array bobot portofolio
    tickers  : list kode saham
    H        : horizon hari
    capital  : modal awal (Rp)

    Return
    ------
    dict hasil evaluasi. Kosong jika data aktual tidak tersedia.
    """
    df_port, realized_returns = build_portfolio_timeseries(
        weights, tickers, H, capital
    )

    if df_port is None or df_port.empty:
        print(f"[WARN] Data aktual tidak tersedia untuk evaluasi H={H}")
        return {}

    daily  = df_port["Portfolio_Daily_Return"].values
    # Gunakan sample standard deviation (ddof=1), konsisten dengan
    # rumus varians sampel: sum((R_t - E[R])^2) / (m-1)
    vol    = float(np.std(daily, ddof=1)) if len(daily) > 1 else np.nan
    mean_d = float(np.mean(daily))
    sharpe = (mean_d - RISK_FREE_RATE) / vol if vol > 0 else np.nan
    real_r = float(df_port["Portfolio_Cumulative_Return"].iloc[-1])
    mdd    = _compute_max_drawdown(df_port["Portfolio_Value"].values)

    return {
        "realized_return":  real_r,
        "realized_profit":  real_r * capital,
        "volatility":       vol,
        "sharpe":           sharpe,
        "max_drawdown":     mdd,
        "realized_returns": realized_returns,
        "df_port":          df_port,
    }


# ============================================================
# EVALUASI SEMUA METODE SEKALIGUS
# ============================================================

def evaluate_all_methods(
    method_weights: Dict[str, np.ndarray],
    tickers:        List[str],
    H:              int,
    capital:        float = TOTAL_CAPITAL,
) -> Dict[str, dict]:
    """
    Evaluasi beberapa metode portofolio sekaligus dengan tickers yang sama.

    Parameter
    ---------
    method_weights : dict {nama_metode: weights_array}
                     Contoh: {"MPC": w_mpc, "Equal Weight": w_eq}
    tickers        : list kode saham (sama untuk semua metode)
    H              : horizon hari
    capital        : modal awal (Rp)

    Return
    ------
    dict {nama_metode: hasil_evaluate_portfolio}
    """
    results = {}
    for method_name, weights in method_weights.items():
        print(f"\n[EVAL] Mengevaluasi {method_name}...")
        results[method_name] = evaluate_portfolio(weights, tickers, H, capital)
    return results


# ============================================================
# HELPER METRIK TAMBAHAN
# ============================================================

def _compute_max_drawdown(portfolio_values: np.ndarray) -> float:
    """
    Hitung Maximum Drawdown (MDD) dari time series nilai portofolio.

    MDD = max penurunan dari puncak ke lembah sepanjang periode.
    Nilai negatif menunjukkan kerugian (contoh: -0.15 = drawdown 15%).

    Parameter
    ---------
    portfolio_values : array nilai portofolio per hari

    Return
    ------
    float MDD (0.0 jika tidak ada drawdown)
    """
    if len(portfolio_values) < 2:
        return 0.0

    peak    = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    return float(np.min(drawdown))


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_eval_summary(
    results: Dict[str, dict],
    H:       int,
    capital: float = TOTAL_CAPITAL,
) -> None:
    """
    Cetak tabel perbandingan metrik semua metode ke konsol.

    Parameter
    ---------
    results : output dari evaluate_all_methods
    H       : horizon hari
    capital : modal awal (Rp)
    """
    print(f"\n{'='*70}")
    print(f"[EVALUASI PERBANDINGAN] H={H} | Modal: Rp{capital:,.0f}")
    print(f"{'='*70}")
    print(f"  {'Metode':<20} {'Return':>10} {'Profit (Rp)':>18} "
          f"{'Volatility':>12} {'Sharpe':>8} {'Max DD':>8}")
    print(f"  {'-'*20} {'-'*10} {'-'*18} {'-'*12} {'-'*8} {'-'*8}")

    for method_name, res in results.items():
        if not res:
            print(f"  {method_name:<20} {'N/A':>10}")
            continue

        print(
            f"  {method_name:<20} "
            f"{res['realized_return']*100:>+9.2f}% "
            f"{res['realized_profit']:>+18,.0f} "
            f"{res['volatility']:>12.4f} "
            f"{res['sharpe']:>8.3f} "
            f"{res['max_drawdown']*100:>+7.2f}%"
        )

    print(f"{'='*70}")
