"""
main.py
=======
Script utama untuk menjalankan MPC Portfolio Optimization secara interaktif.

Mode interaktif:
  1) otomatis : pilih top-N emiten terbaik berdasarkan DA_IC/Confidence dari all_predictions.csv
  2) emiten   : user pilih ticker, fitur diambil otomatis dari all_predictions.csv

Catatan:
- Horizon dikunci H=22.
- Sumber prediksi utama adalah predictions_forward/all_predictions.csv.
"""

import argparse
import numpy as np
import pandas as pd

from config import (
    TOTAL_CAPITAL,
    OUTPUT_DIR,
)
from utils.data_loader import (
    load_prediction_summary,
    load_predicted_returns_from_selected_pairs,
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

FIXED_HORIZON = 22


# ============================================================
# INTERACTIVE INPUT
# ============================================================

def _ask_mode() -> str:
    while True:
        print("\n[PILIH MODE]")
        print("  1. otomatis")
        print("  2. emiten")
        ans = input("Masukkan pilihan (1/2): ").strip()
        mapping = {"1": "otomatis", "2": "emiten"}
        if ans in mapping:
            return mapping[ans]
        print("[WARN] Pilihan tidak valid.")


def _ask_positive_int(prompt: str) -> int:
    while True:
        ans = input(prompt).strip()
        try:
            n = int(ans)
            if n > 0:
                return n
        except Exception:
            pass
        print("[WARN] Masukkan angka bulat > 0.")


def _ask_ticker_list_with_retry(available_tickers: set[str]) -> list[str]:
    for attempt in range(1, 4):
        raw = input("Masukkan daftar emiten dipisah spasi (contoh: MEDC AKRA ADMR): ").strip()
        tickers = [t.upper().strip() for t in raw.split() if t.strip()]
        tickers = list(dict.fromkeys(tickers))

        invalid = [t for t in tickers if t not in available_tickers]
        if not tickers:
            print("[WARN] Tidak ada emiten yang diinput.")
            continue

        if not invalid:
            return tickers

        print(f"[WARN] Emiten tidak tersedia: {', '.join(invalid)}")
        if attempt < 3:
            print(f"[INFO] Ulangi input ({attempt}/3).")
        else:
            valid = [t for t in tickers if t in available_tickers]
            print("[INFO] Batas retry tercapai. Ticker invalid akan di-skip.")
            return valid

    return []


def collect_user_selection(H: int) -> dict:
    """Kumpulkan keputusan mode + universe dari user via terminal."""
    mode = _ask_mode()
    n_assets = _ask_positive_int("Jumlah emiten target (N): ")

    pred_summary = load_prediction_summary(H)
    available_tickers = set(pred_summary["Ticker"].unique().tolist())

    if mode == "otomatis":
        # Gunakan semua ticker yang tersedia di all_predictions,
        # nanti dipilih top-N berdasarkan DA_IC/Confidence.
        all_tickers = sorted(available_tickers)
        pairs = [(t, None) for t in all_tickers]
        return {
            "mode": mode,
            "top_n": n_assets,
            "pairs": pairs,
        }

    # mode == emiten
    tickers = _ask_ticker_list_with_retry(available_tickers)
    if not tickers:
        raise ValueError("Tidak ada emiten valid dari input user")

    pairs = [(t, None) for t in tickers]
    return {
        "mode": mode,
        "top_n": n_assets,
        "pairs": pairs,
    }


# ============================================================
# HELPER SELEKSI TOP-N BERDASARKAN DA_IC/CONFIDENCE
# ============================================================

def _select_top_n_by_score(
    r_raw: np.ndarray,
    r_adj: np.ndarray,
    conf: np.ndarray,
    tickers: list[str],
    fiturs: list[str],
    top_n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    if top_n is None or top_n >= len(tickers):
        return r_raw, r_adj, conf, tickers, fiturs

    df = pd.DataFrame(
        {
            "Ticker": tickers,
            "Fitur": fiturs,
            "Raw": r_raw.flatten(),
            "Adj": r_adj.flatten(),
            "Conf": conf.flatten(),
        }
    )

    # Ranking utama: confidence (DA_IC), tie-break: adjusted return.
    df = (
        df.sort_values(["Conf", "Adj"], ascending=[False, False])
          .iloc[:top_n]
          .reset_index(drop=True)
    )

    print(f"\n[SELECT TOP-{top_n} by DA_IC/Confidence]")
    print(df[["Ticker", "Fitur", "Conf", "Adj", "Raw"]].to_string(index=False))

    return (
        df["Raw"].values.reshape(-1, 1),
        df["Adj"].values.reshape(-1, 1),
        df["Conf"].values.reshape(-1, 1),
        df["Ticker"].tolist(),
        df["Fitur"].tolist(),
    )


# ============================================================
# SATU HORIZON
# ============================================================

def run_single_horizon(
    H: int,
    capital: float,
    dirs: dict,
    selection: dict,
) -> dict:
    print(f"\n{'#'*70}")
    print(f"[MPC] Horizon H={H}")
    print(f"{'#'*70}")

    mode = selection["mode"]

    try:
        r_raw, r_adj, conf, tickers, fiturs = load_predicted_returns_from_selected_pairs(
            H=H,
            ticker_feature_pairs=selection["pairs"],
            auto_feature_from_training=False,
        )

        r_raw, r_adj, conf, tickers, fiturs = _select_top_n_by_score(
            r_raw=r_raw,
            r_adj=r_adj,
            conf=conf,
            tickers=tickers,
            fiturs=fiturs,
            top_n=selection["top_n"],
        )
    except Exception as e:
        print(f"[SKIP] H={H}: {e}")
        return {}

    used_features = {t: f for t, f in zip(tickers, fiturs)}

    print(f"\n[INFO] {len(tickers)} emiten digunakan untuk optimasi H={H}")

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

    exp_ret_raw = float(np.sum(weights * r_raw.flatten()))
    exp_ret_adj = float(np.sum(weights * r_adj.flatten()))
    _print_expected_return(exp_ret_raw, exp_ret_adj, capital)

    dynamic_max_w_vec = dynamic_max_w * (0.5 + 0.5 * conf.flatten())
    df_alloc = _build_allocation_table(
        tickers=tickers,
        weights=weights,
        conf=conf,
        r_raw=r_raw,
        r_adj=r_adj,
        max_w_vec=dynamic_max_w_vec,
        capital=capital,
        used_features=used_features,
    )
    print(f"\n[ALOKASI] H={H}:")
    print(df_alloc[
        [
            "Ticker", "Fitur", "Confidence_Score", "Max_Weight_Pct",
            "Weight", "Allocation_Rp", "Pred_Return_Raw_Pct", "Pred_Return_Adj_Pct",
        ]
    ].to_string(index=False))

    mpc_result = evaluate_portfolio(weights, tickers, H, capital)
    df_port = None

    if mpc_result:
        realized_returns = mpc_result["realized_returns"]
        df_port = mpc_result["df_port"]

        df_alloc["Realized_Return_Pct"] = realized_returns * 100
        df_alloc["Realized_Profit_Rp"] = weights * capital * realized_returns

        print_eval_summary({"MPC": mpc_result}, H, capital)

    df_risk = analyze_risk(weights, tickers, r_raw, conf, H, capital)
    pred_sharpe_total = None
    if not df_risk.empty:
        total_row = df_risk[df_risk["Ticker"] == "TOTAL PORTOFOLIO"]
        if not total_row.empty and "Sharpe_Pred" in total_row.columns:
            pred_sharpe_total = total_row["Sharpe_Pred"].iloc[0]

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

    df_combined = _merge_alloc_risk(df_alloc, df_risk)

    return {
        "sheet": df_combined,
        "H_FWD": H,
        "Mode": mode,
        "N_Emiten": len(tickers),
        "Expected_Profit_Raw_Rp": exp_ret_raw * capital,
        "Expected_Profit_Adj_Rp": exp_ret_adj * capital,
        "Expected_Sharpe_Pred":    pred_sharpe_total,
        "Realized_Profit_Rp": mpc_result.get("realized_profit") if mpc_result else None,
        "Volatility": mpc_result.get("volatility") if mpc_result else None,
        "Sharpe_Ratio": mpc_result.get("sharpe") if mpc_result else None,
        "Max_Drawdown": mpc_result.get("max_drawdown") if mpc_result else None,
    }


# ============================================================
# MAIN
# ============================================================

def main(
    capital: float = TOTAL_CAPITAL,
    out_dir: str = OUTPUT_DIR,
) -> None:
    H = FIXED_HORIZON
    dirs = ensure_output_dirs(out_dir)

    try:
        selection = collect_user_selection(H)
    except Exception as e:
        print(f"[ERROR] Gagal membaca input interaktif: {e}")
        return

    result = run_single_horizon(H, capital, dirs, selection)
    if not result:
        print("\n[DONE] MPC selesai tanpa hasil tersimpan.")
        return

    df_summary = pd.DataFrame([{k: v for k, v in result.items() if k != "sheet"}])
    all_sheets = {
        f"H{H}": result["sheet"],
        "Summary": df_summary,
    }

    save_excel(
        sheets=all_sheets,
        out_dir=dirs["excel"],
        filename="mpc_results.xlsx",
    )

    print(f"\n{'='*70}")
    print("[SUMMARY] Hasil:")
    print(df_summary.to_string(index=False))
    print("\n[DONE] MPC selesai!")


# ============================================================
# HELPER INTERNAL
# ============================================================

def _print_expected_return(
    exp_ret_raw: float,
    exp_ret_adj: float,
    capital: float,
) -> None:
    print(f"\n[RESULT] Expected Return RAW : {exp_ret_raw:+.4f} "
          f"({(np.exp(exp_ret_raw)-1)*100:+.2f}%)")
    print(f"[RESULT] Expected Return ADJ : {exp_ret_adj:+.4f} "
          f"({(np.exp(exp_ret_adj)-1)*100:+.2f}%)")
    print(f"[RESULT] Expected Profit RAW : Rp{exp_ret_raw * capital:,.0f}")
    print(f"[RESULT] Expected Profit ADJ : Rp{exp_ret_adj * capital:,.0f}")


def _build_allocation_table(
    tickers: list,
    weights: np.ndarray,
    conf: np.ndarray,
    r_raw: np.ndarray,
    r_adj: np.ndarray,
    max_w_vec: np.ndarray,
    capital: float,
    used_features: dict[str, str] | None = None,
) -> pd.DataFrame:
    fitur_col = [used_features.get(t, "AUTO") if used_features else "AUTO" for t in tickers]

    return (
        pd.DataFrame({
            "Ticker": tickers,
            "Fitur": fitur_col,
            "Confidence_Score": conf.flatten(),
            "Max_Weight_Pct": max_w_vec * 100,
            "Weight": weights,
            "Allocation_Rp": weights * capital,
            "Pred_Return_Raw_Pct": (np.exp(r_raw.flatten()) - 1) * 100,
            "Pred_Return_Adj_Pct": (np.exp(r_adj.flatten()) - 1) * 100,
            "Expected_Profit_Rp": weights * capital * r_raw.flatten(),
        })
        .sort_values("Weight", ascending=False)
        .reset_index(drop=True)
    )


def _merge_alloc_risk(
    df_alloc: pd.DataFrame,
    df_risk: pd.DataFrame,
) -> pd.DataFrame:
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
        description="MPC Portfolio Optimization interaktif (Horizon 22)"
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
        capital=args.capital,
        out_dir=args.out_dir,
    )
