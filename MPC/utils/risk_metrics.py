"""
utils/risk_metrics.py
=====================
Fungsi analisis risiko forward-looking per saham dan portofolio:
  - analyze_risk     : hitung VaR, worst case, volatility per ticker
  - _build_risk_row  : helper bangun satu baris risk per ticker
  - _build_total_row : helper bangun baris total portofolio

Semua metrik dihitung menggunakan:
  - Prediksi LSTM sebagai expected return
  - Volatility historis dari data aktual (LOOKBACK_DAYS hari terakhir)
  - Distribusi normal untuk VaR dan worst case
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import List

from config import (
    CONFIDENCE_LEVEL,
    LOOKBACK_DAYS,
    RISK_FREE_RATE,
    TOTAL_CAPITAL,
)
from utils.data_loader import load_historical_volatility


# ============================================================
# HELPER INTERNAL
# ============================================================

def _build_risk_row(
    ticker:     str,
    weight:     float,
    capital:    float,
    pred_ret:   float,
    confidence: float,
    H:          int,
    z:          float,
) -> dict:
    """
    Bangun satu baris risk analysis untuk satu ticker.

    Parameter
    ---------
    ticker     : kode saham
    weight     : bobot portofolio (0–1)
    capital    : total modal (Rp)
    pred_ret   : predicted return (bukan log — sudah di-exp)
    confidence : confidence score (0–1)
    H          : horizon hari
    z          : z-score dari confidence level (negatif untuk VaR lower tail)
    """
    modal    = weight * capital
    vol_daily = load_historical_volatility(ticker)

    row = {
        "Ticker":            ticker,
        "Weight_Pct":        weight * 100,
        "Modal_Rp":          modal,
        "Confidence_Score":  confidence,
        "Pred_Return_Pct":   pred_ret * 100,
        "Pred_Profit_Rp":    pred_ret * modal,
        "Vol_Harian":        None,
        "VaR_95_Pct":        None,
        "VaR_95_Rp":         None,
        "Worst_Case_Pct":    None,
        "Worst_Case_Rp":     None,
        "Risk_Return_Ratio": None,
        "Sharpe_Pred":       None,
    }

    if vol_daily is not None:
        vol_h       = vol_daily * np.sqrt(H)
        var_pct     = pred_ret + z * vol_h          # VaR = return - z*vol (z negatif)
        worst_pct   = pred_ret - 2 * vol_h          # skenario 2 sigma ke bawah

        row.update({
            "Vol_Harian":         vol_daily,
            "VaR_95_Pct":        var_pct * 100,
            "VaR_95_Rp":         var_pct * modal,
            "Worst_Case_Pct":    worst_pct * 100,
            "Worst_Case_Rp":     worst_pct * modal,
            "Risk_Return_Ratio": pred_ret / vol_h if vol_h > 0 else None,
            "Sharpe_Pred":      (pred_ret - (RISK_FREE_RATE * H)) / vol_h if vol_h > 0 else None,
        })

    return row


def _build_total_row(
    weights:    np.ndarray,
    r_pred_raw: np.ndarray,
    conf_arr:   np.ndarray,
    capital:    float,
    H:          int,
    z:          float,
    avg_vol:    float,
) -> dict:
    """
    Bangun baris TOTAL PORTOFOLIO sebagai ringkasan keseluruhan.

    Expected return total = weighted sum dari individual return.
    Volatility total = rata-rata volatility individual (simplified,
    tidak memperhitungkan korelasi antar saham).
    """
    total_pred_ret = float(np.sum(weights * (np.exp(r_pred_raw.flatten()) - 1)))
    vol_h_total    = avg_vol * np.sqrt(H) if pd.notna(avg_vol) else None

    return {
        "Ticker":            "TOTAL PORTOFOLIO",
        "Weight_Pct":        100.0,
        "Modal_Rp":          capital,
        "Confidence_Score":  float(np.mean(conf_arr)),
        "Pred_Return_Pct":   total_pred_ret * 100,
        "Pred_Profit_Rp":    total_pred_ret * capital,
        "Vol_Harian":        avg_vol,
        "VaR_95_Pct":       (total_pred_ret + z * vol_h_total) * 100
                             if vol_h_total else None,
        "VaR_95_Rp":        (total_pred_ret + z * vol_h_total) * capital
                             if vol_h_total else None,
        "Worst_Case_Pct":   (total_pred_ret - 2 * vol_h_total) * 100
                             if vol_h_total else None,
        "Worst_Case_Rp":    (total_pred_ret - 2 * vol_h_total) * capital
                             if vol_h_total else None,
        "Risk_Return_Ratio": total_pred_ret / vol_h_total
                             if vol_h_total else None,
        "Sharpe_Pred":      (total_pred_ret - (RISK_FREE_RATE * H)) / vol_h_total
                             if vol_h_total else None,
    }


# ============================================================
# FUNGSI UTAMA
# ============================================================

def analyze_risk(
    weights:    np.ndarray,
    tickers:    List[str],
    r_pred_raw: np.ndarray,
    conf_arr:   np.ndarray,
    H:          int,
    capital:    float = TOTAL_CAPITAL,
) -> pd.DataFrame:
    """
    Analisis risiko forward-looking per saham dan total portofolio.

    Menggunakan prediksi LSTM sebagai expected return dan volatility
    historis (LOOKBACK_DAYS hari) untuk menghitung distribusi return.

    Metrik yang dihitung
    --------------------
    VaR 95%     : batas kerugian dengan 95% keyakinan selama H hari
                  VaR = pred_return + z * vol_H  (z = -1.645 untuk 95%)
    Worst Case  : skenario 2 standar deviasi di bawah prediksi
                  worst = pred_return - 2 * vol_H
    Volatility  : volatility harian historis dari LOOKBACK_DAYS hari terakhir

    Return
    ------
    DataFrame dengan satu baris per ticker + baris TOTAL PORTOFOLIO di akhir.
    """
    # z-score untuk lower tail (negatif karena kerugian)
    z    = norm.ppf(1 - CONFIDENCE_LEVEL)   # contoh: -1.645 untuk CL=0.95
    rows = []

    for i, ticker in enumerate(tickers):
        pred_ret   = float(np.exp(float(r_pred_raw.flatten()[i])) - 1)
        confidence = float(conf_arr.flatten()[i])

        row = _build_risk_row(
            ticker=ticker,
            weight=weights[i],
            capital=capital,
            pred_ret=pred_ret,
            confidence=confidence,
            H=H,
            z=z,
        )
        rows.append(row)

    df = pd.DataFrame(rows)

    # Baris total portofolio
    avg_vol  = df["Vol_Harian"].mean()
    total_row = _build_total_row(
        weights=weights,
        r_pred_raw=r_pred_raw,
        conf_arr=conf_arr,
        capital=capital,
        H=H,
        z=z,
        avg_vol=avg_vol,
    )
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

    _print_risk_table(df, H, capital)
    return df


# ============================================================
# PRINT KONSOL
# ============================================================

def _print_risk_table(df: pd.DataFrame, H: int, capital: float) -> None:
    """Cetak tabel risk analysis ke konsol dalam format yang rapi."""
    print(f"\n{'='*70}")
    print(f"[FORWARD RISK ANALYSIS] H={H} | Modal: Rp{capital:,.0f}")
    print(f"Volatility dari data historis {LOOKBACK_DAYS} hari terakhir")
    print(f"{'='*70}")

    for _, row in df.iterrows():
        print(f"\n  Saham              : {row['Ticker']}")
        print(f"  Modal              : Rp{row['Modal_Rp']:>15,.0f}  ({row['Weight_Pct']:.1f}%)")
        print(f"  Prediksi Return    : {row['Pred_Return_Pct']:>+10.2f}%  ->  "
              f"Rp{row['Pred_Profit_Rp']:>+15,.0f}")

        if row["Vol_Harian"] is not None:
            print(f"  Volatility Harian  : {row['Vol_Harian']:>10.4f}")
            print(f"  VaR 95% (H={H:2d})   : {row['VaR_95_Pct']:>+10.2f}%  ->  "
                  f"Rp{row['VaR_95_Rp']:>+15,.0f}")
            print(f"  Worst Case         : {row['Worst_Case_Pct']:>+10.2f}%  ->  "
                  f"Rp{row['Worst_Case_Rp']:>+15,.0f}")
            print(f"  Risk/Return Ratio  : {row['Risk_Return_Ratio']:>10.4f}")
            print(f"  Sharpe Pred (H)    : {row['Sharpe_Pred']:>10.4f}")

    print(f"{'='*70}")
