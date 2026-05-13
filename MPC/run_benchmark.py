"""
main.py
=======
Script utama untuk menjalankan MPC Portfolio Optimization.

Alur eksekusi:
    1. Load & seleksi prediksi LSTM (top-N saham)
    2. Jalankan MPC optimization
    3. Evaluasi dengan data aktual (realized return, Sharpe, MDD)
    4. Forward risk analysis (VaR, worst case)
    5. Plot alokasi + skenario
    6. Simpan hasil ke Excel

Untuk perbandingan dengan benchmark, jalankan run_benchmarks.py.

Contoh penggunaan:
    python main.py
    python main.py --horizons 5 10 22 --capital 200000000
"""

import argparse
import os
import numpy as np
import pandas as pd

from config import (
    HORIZONS,
    TOTAL_CAPITAL,
    N_SELECTED_ASSETS,
    MAX_WEIGHT,
    OUTPUT_DIR,
    PREDICTION_BASE_DIR,
)
from utils.data_loader import (
    load_predicted_returns,
    select_top_n_assets,
)
from utils.risk_metrics import analyze_risk
from utils.visualization import (
    plot_allocation,
    plot_portfolio_scenarios,
)
from utils.file_utils import (
    ensure_output_dirs,
    save_excel,
)
from optimizer.mpc import run_mpc, compute_dynamic_max_weight
from evaluation.backtest import evaluate_portfolio, print_eval_summary


# ============================================================
# SATU HORIZON
# ============================================================

def run_single_horizon(
    H:       int,
    capital: float,
    dirs:    dict,
) -> dict:
    """
    Jalankan full pipeline MPC untuk satu horizon H.

    Return
    ------
    dict ringkasan hasil untuk horizon ini.
    Kosong jika ada error fatal (prediksi tidak tersedia, solver gagal).
    """
    print(f"\n{'#'*70}")
    print(f"[MPC] Horizon H={H}")
    print(f"{'#'*70}")

    # --- 1. Load & seleksi prediksi ---
    try:
        r_raw, r_adj, conf, tickers = load_predicted_returns(H)
        r_raw, r_adj, conf, tickers = select_top_n_assets(
            r_raw, r_adj, conf, tickers
        )
    except Exception as e:
        print(f"[SKIP] H={H}: {e}")
        return {}

    print(f"\n[INFO] {len(tickers)} emiten digunakan untuk optimasi H={H}")

    # --- 2. MPC optimization ---
    dynamic_max_w = compute_dynamic_max_weight(len(tickers))

    try:
        weights = run_mpc(
            r_pred=r_adj,
            tickers=tickers,
            confidence=conf,
            max_weight=dynamic_max_w,
        )
    except Exception as e:
        print(f"[ERROR] MPC H={H} gagal: {e}")
        return {}

    # --- 3. Expected return ---
    exp_ret_raw = float(np.sum(weights * r_raw.flatten()))
    exp_ret_adj = float(np.sum(weights * r_adj.flatten()))
    _print_expected_return(exp_ret_raw, exp_ret_adj, capital)

    # --- 4. Tabel alokasi ---
    dynamic_max_w_vec = dynamic_max_w * (0.5 + 0.5 * conf.flatten())
    df_alloc = _build_allocation_table(
        tickers, weights, conf, r_raw, r_adj, dynamic_max_w_vec, capital
    )
    print(f"\n[ALOKASI] H={H}:")
    print(df_alloc[["Ticker", "Confidence_Score", "Max_Weight_Pct",
                     "Weight", "Allocation_Rp",
                     "Pred_Return_Raw_Pct", "Pred_Return_Adj_Pct"]].to_string(index=False))

    # --- 5. Evaluasi realized ---
    mpc_result = evaluate_portfolio(weights, tickers, H, capital)
    df_port    = None

    if mpc_result:
        realized_returns = mpc_result["realized_returns"]
        df_port          = mpc_result["df_port"]

        df_alloc["Realized_Return_Pct"] = realized_returns * 100
        df_alloc["Realized_Profit_Rp"]  = weights * capital * realized_returns

        print_eval_summary({"MPC": mpc_result}, H, capital)

    # --- 6. Forward risk analysis ---
    df_risk = analyze_risk(weights, tickers, r_raw, conf, H, capital)

    # --- 7. Plot ---
    plot_allocation(tickers, weights, H, out_dir=dirs["plots"])

    if df_port is not None:
        plot_portfolio_scenarios(
            df_port=df_port,
            tickers=tickers,
            weights=weights,
            df_risk=df_risk,
            H=H,
            capital=capital,
            out_dir=dirs["plots"],
        )

    # --- 8. Gabungkan tabel alokasi + risk ---
    df_combined = _merge_alloc_risk(df_alloc, df_risk)

    # --- 9. Summary row ---
    return {
        "sheet":          df_combined,
        "H_FWD":          H,
        "N_Emiten":       len(tickers),
        "Expected_Profit_Raw_Rp": exp_ret_raw * capital,
        "Expected_Profit_Adj_Rp": exp_ret_adj * capital,
        "Realized_Profit_Rp":     mpc_result.get("realized_profit") if mpc_result else None,
        "Volatility":             mpc_result.get("volatility")       if mpc_result else None,
        "Sharpe_Ratio":           mpc_result.get("sharpe")           if mpc_result else None,
        "Max_Drawdown":           mpc_result.get("max_drawdown")     if mpc_result else None,
    }


# ============================================================
# MAIN
# ============================================================

def main(
    horizons: list  = HORIZONS,
    capital:  float = TOTAL_CAPITAL,
    out_dir:  str   = OUTPUT_DIR,
) -> None:
    """
    Entry point utama. Jalankan MPC untuk semua horizon yang dikonfigurasi.
    """
    dirs          = ensure_output_dirs(out_dir)
    all_summaries = []
    all_sheets    = {}

    for H in horizons:
        result = run_single_horizon(H, capital, dirs)

        if not result:
            continue

        all_sheets[f"H{H}"] = result.pop("sheet")
        all_summaries.append(result)

    # --- Summary gabungan ---
    if all_summaries:
        df_summary              = pd.DataFrame(all_summaries)
        all_sheets["Summary"]   = df_summary

        save_excel(
            sheets=all_sheets,
            out_dir=dirs["excel"],
            filename="mpc_results.xlsx",
        )

        print(f"\n{'='*70}")
        print("[SUMMARY] Hasil semua horizon:")
        print(df_summary.to_string(index=False))

    print("\n[DONE] MPC selesai!")


# ============================================================
# HELPER INTERNAL
# ============================================================

def _print_expected_return(
    exp_ret_raw: float,
    exp_ret_adj: float,
    capital:     float,
) -> None:
    """Cetak expected return RAW dan ADJ ke konsol."""
    print(f"\n[RESULT] Expected Return RAW : {exp_ret_raw:+.4f} "
          f"({(np.exp(exp_ret_raw)-1)*100:+.2f}%)")
    print(f"[RESULT] Expected Return ADJ : {exp_ret_adj:+.4f} "
          f"({(np.exp(exp_ret_adj)-1)*100:+.2f}%)")
    print(f"[RESULT] Expected Profit RAW : Rp{exp_ret_raw * capital:,.0f}")
    print(f"[RESULT] Expected Profit ADJ : Rp{exp_ret_adj * capital:,.0f}")


def _build_allocation_table(
    tickers:       list,
    weights:       np.ndarray,
    conf:          np.ndarray,
    r_raw:         np.ndarray,
    r_adj:         np.ndarray,
    max_w_vec:     np.ndarray,
    capital:       float,
) -> pd.DataFrame:
    """Buat DataFrame tabel alokasi portofolio."""
    return (
        pd.DataFrame({
            "Ticker":              tickers,
            "Confidence_Score":    conf.flatten(),
            "Max_Weight_Pct":      max_w_vec * 100,
            "Weight":              weights,
            "Allocation_Rp":       weights * capital,
            "Pred_Return_Raw_Pct": (np.exp(r_raw.flatten()) - 1) * 100,
            "Pred_Return_Adj_Pct": (np.exp(r_adj.flatten()) - 1) * 100,
            "Expected_Profit_Rp":  weights * capital * r_raw.flatten(),
        })
        .sort_values("Weight", ascending=False)
        .reset_index(drop=True)
    )


def _merge_alloc_risk(
    df_alloc: pd.DataFrame,
    df_risk:  pd.DataFrame,
) -> pd.DataFrame:
    """Gabungkan tabel alokasi dengan tabel risk analysis."""
    df_combined = pd.merge(
        df_alloc,
        df_risk.rename(columns={"Modal_Rp": "Modal_Rp_Risk"}),
        on="Ticker",
        how="left",
        suffixes=("", "_risk"),
    )
    cols_to_drop = [c for c in df_combined.columns if c.endswith("_risk")]
    return df_combined.drop(columns=cols_to_drop)


# ============================================================
# CLI
# ============================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MPC Portfolio Optimization dengan prediksi LSTM"
    )
    parser.add_argument(
        "--horizons", nargs="+", type=int,
        default=HORIZONS,
        help=f"Horizon prediksi (default: {HORIZONS})",
    )
    parser.add_argument(
        "--capital", type=float,
        default=TOTAL_CAPITAL,
        help=f"Modal awal dalam Rp (default: {TOTAL_CAPITAL:,.0f})",
    )
    parser.add_argument(
        "--out-dir", type=str,
        default=OUTPUT_DIR,
        help=f"Direktori output (default: {OUTPUT_DIR})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        horizons=args.horizons,
        capital=args.capital,
        out_dir=args.out_dir,
    )