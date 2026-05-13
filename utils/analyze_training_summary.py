"""
Analyze ALL_training_results.xlsx / ALL_training_summary.csv and print
compact tables for thesis reporting.

Default output:
- horizon summary table
- top-N models per horizon by selected metric or combined score

Examples:
  python utils/analyze_training_summary.py
  python utils/analyze_training_summary.py --input seq2seq_model/ALL_training_results.xlsx --top-n 5
  python utils/analyze_training_summary.py --metric Test_RMSE --horizons 22

The script prints Markdown tables to stdout so the output can be copied
directly into a report or sent back for further analysis.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


LOWER_IS_BETTER = {
    "Test_RMSE",
    "Test_MAE",
    "Avg_CV_Loss",
    "Final_BestValLoss_RMSE",
    "Val_RMSE",
    "Val_MAE",
    "TestSize",
}


def _find_input_path(raw: Optional[str]) -> Path:
    if not raw:
        raise FileNotFoundError(
            "Wajib isi --input dengan file ALL_training_summary.csv atau ALL_training_results.xlsx yang kamu upload."
        )

    p = Path(raw)
    if p.exists():
        return p

    raise FileNotFoundError(f"Input file tidak ditemukan: {raw}")


def load_summary(path: Path, sheet_name: str = "ALL_Summary") -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        xls = pd.ExcelFile(path)
        sheet = sheet_name if sheet_name in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(path, sheet_name=sheet)
    else:
        df = pd.read_csv(path)

    required = {"H_FWD", "Ticker", "Fitur"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {sorted(missing)}")

    return df.copy()


def normalize_metric_value(metric: str, value):
    try:
        if pd.isna(value):
            return value
    except Exception:
        pass

    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except Exception:
            return value
    return value


def is_higher_better(metric: str) -> bool:
    return metric not in LOWER_IS_BETTER


def fmt_float(x, digits: int = 6):
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def compute_da_ic_score(df: pd.DataFrame, da_col: str = "Test_DA", ic_col: str = "Test_IC") -> pd.Series:
    """
    Build a combined score from DA and IC.
    Higher is better.

    Default weights:
      - DA: 0.7
      - IC: 0.3

    Missing IC is handled by rescaling weights onto the available metrics.
    """
    da = pd.to_numeric(df.get(da_col, pd.Series([pd.NA] * len(df), index=df.index)), errors="coerce")
    ic = pd.to_numeric(df.get(ic_col, pd.Series([pd.NA] * len(df), index=df.index)), errors="coerce")

    def minmax(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        mn, mx = s.min(), s.max()
        if pd.isna(mn) or pd.isna(mx) or np.isclose(mx, mn):
            return pd.Series([0.5] * len(s), index=s.index)
        return (s - mn) / (mx - mn)

    da_norm = minmax(da)
    ic_norm = minmax(ic)
    metric_map = {"DA": da_norm, "IC": ic_norm}
    weights = {"DA": 0.7, "IC": 0.3}

    scores = []
    for idx in df.index:
        vals = []
        for name, series in metric_map.items():
            val = series.loc[idx]
            if pd.notna(val):
                vals.append((name, float(val)))

        if not vals:
            scores.append(np.nan)
            continue

        wsum = sum(weights[name] for name, _ in vals)
        if np.isclose(wsum, 0.0):
            scores.append(np.nan)
            continue

        score = sum((weights[name] / wsum) * val for name, val in vals)
        scores.append(score)

    return pd.Series(scores, index=df.index, name="DA_IC_Score")


def _ranking_metric_name(metric: str) -> str:
    return "DA_IC_Score" if metric.upper() == "DA_IC" else metric


def _filter_da_ic_valid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combined DA+IC ranking only makes sense when IC exists and is positive.
    """
    if "Test_IC" not in df.columns:
        return df.iloc[0:0].copy()
    return df[df["Test_IC"].notna() & (df["Test_IC"] > 0)].copy()


def pick_top_rows(df: pd.DataFrame, metric: str, top_n: int, horizons: Optional[List[int]] = None) -> pd.DataFrame:
    dfx = df.copy()
    rank_metric = _ranking_metric_name(metric)
    if rank_metric not in dfx.columns:
        if metric.upper() == "DA_IC":
            dfx = _filter_da_ic_valid(dfx)
            dfx[rank_metric] = compute_da_ic_score(dfx)
        else:
            dfx[rank_metric] = dfx[metric].apply(lambda v: normalize_metric_value(metric, v))
    dfx = dfx.dropna(subset=[rank_metric, "Ticker", "Fitur"]).copy()

    if horizons:
        dfx = dfx[dfx["H_FWD"].isin(horizons)].copy()

    if dfx.empty:
        return dfx

    asc = not is_higher_better(metric) if metric.upper() != "DA_IC" else False
    dfx = dfx.sort_values(["H_FWD", rank_metric, "Ticker"], ascending=[True, asc, True])
    dfx["Rank"] = dfx.groupby("H_FWD")[rank_metric].rank(method="first", ascending=asc).astype(int)
    dfx = dfx[dfx["Rank"] <= top_n].copy()

    cols = ["H_FWD", "Rank", "Ticker", "Fitur", rank_metric]
    for extra in ["Final_BestEpoch", "Val_RMSE", "Val_DA", "Val_IC", "Test_RMSE", "Test_DA", "Test_IC"]:
        if extra in dfx.columns and extra not in cols:
            cols.append(extra)

    return dfx[cols].sort_values(["H_FWD", "Rank"]).reset_index(drop=True)


def _pick_best_row_in_group(group: pd.DataFrame, metric: str) -> pd.Series:
    """
    Pick one representative row from a group.
    Priority:
    1) best combined metric value
    2) if ties exist, the row with the most non-null reporting metrics
    3) finally, the first row in the sorted order
    """
    g = group.copy()
    rank_metric = _ranking_metric_name(metric)
    if metric.upper() == "DA_IC":
        g = _filter_da_ic_valid(g)
        if g.empty:
            return pd.Series(dtype=object)
        if rank_metric not in g.columns:
            g[rank_metric] = compute_da_ic_score(g)
    else:
        g[rank_metric] = pd.to_numeric(g[metric], errors="coerce")
    g = g.dropna(subset=[rank_metric]).copy()
    if g.empty:
        return pd.Series(dtype=object)

    asc = not is_higher_better(metric) if metric.upper() != "DA_IC" else False
    best_value = g[rank_metric].min() if asc else g[rank_metric].max()
    candidates = g[g[rank_metric] == best_value].copy()

    score_cols = [
        "Final_BestEpoch",
        "Val_RMSE",
        "Val_DA",
        "Val_IC",
        "Test_RMSE",
        "Test_DA",
        "Test_IC",
    ]
    available_score_cols = [c for c in score_cols if c in candidates.columns]
    if available_score_cols:
        candidates["_complete_score"] = candidates[available_score_cols].notna().sum(axis=1)
        candidates = candidates.sort_values(
            ["_complete_score", "Ticker", "Fitur"],
            ascending=[False, True, True],
        )
    else:
        candidates = candidates.sort_values(["Ticker", "Fitur"], ascending=[True, True])

    return candidates.iloc[0].drop(labels=["_complete_score"], errors="ignore")


def pick_top_unique_tickers(
    df: pd.DataFrame,
    metric: str,
    top_n: int,
    horizons: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Select the best row per ticker first, then rank those ticker-level
    winners so the final table contains different emiten only.
    """
    dfx = df.copy()
    rank_metric = _ranking_metric_name(metric)
    if rank_metric not in dfx.columns:
        if metric.upper() == "DA_IC":
            dfx = _filter_da_ic_valid(dfx)
            dfx[rank_metric] = compute_da_ic_score(dfx)
        else:
            dfx[rank_metric] = dfx[metric].apply(lambda v: normalize_metric_value(metric, v))
    dfx = dfx.dropna(subset=[rank_metric, "Ticker", "Fitur"]).copy()

    if horizons:
        dfx = dfx[dfx["H_FWD"].isin(horizons)].copy()

    if dfx.empty:
        return dfx

    best_per_ticker = []
    for (h, ticker), g in dfx.groupby(["H_FWD", "Ticker"], as_index=False):
        best_per_ticker.append(_pick_best_row_in_group(g, metric))

    best_df = pd.DataFrame(best_per_ticker)
    asc = not is_higher_better(metric) if metric.upper() != "DA_IC" else False
    best_df = best_df.sort_values(["H_FWD", rank_metric, "Ticker"], ascending=[True, asc, True]).copy()
    best_df["Rank"] = best_df.groupby("H_FWD")[rank_metric].rank(method="first", ascending=asc).astype(int)
    best_df = best_df[best_df["Rank"] <= top_n].copy()

    cols = ["H_FWD", "Rank", "Ticker", "Fitur", rank_metric]
    for extra in ["Final_BestEpoch", "Val_RMSE", "Val_DA", "Val_IC", "Test_RMSE", "Test_DA", "Test_IC"]:
        if extra in best_df.columns and extra not in cols:
            cols.append(extra)

    return best_df[cols].sort_values(["H_FWD", "Rank"]).reset_index(drop=True)


def summary_by_horizon(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h, g in df.groupby("H_FWD"):
        g = g.copy()
        g_ic = _filter_da_ic_valid(g)
        if not g_ic.empty:
            g_ic["DA_IC_Score"] = compute_da_ic_score(g_ic)
        row = {
            "H_FWD": h,
            "Count": len(g),
            "Avg_Test_RMSE": g["Test_RMSE"].mean() if "Test_RMSE" in g.columns else None,
            "Avg_Test_DA": g["Test_DA"].mean() if "Test_DA" in g.columns else None,
            "Avg_Test_IC": g["Test_IC"].mean() if "Test_IC" in g.columns else None,
            "Best_DA_IC_Ticker": None,
            "Best_DA_IC_Value": None,
            "Best_Test_DA_Ticker": None,
            "Best_Test_DA_Value": None,
            "Best_Test_IC_Ticker": None,
            "Best_Test_IC_Value": None,
            "Best_Test_RMSE_Ticker": None,
            "Best_Test_RMSE_Value": None,
        }

        if not g_ic.empty and "DA_IC_Score" in g_ic.columns:
            idx_combo = g_ic["DA_IC_Score"].astype(float).idxmax()
            row["Best_DA_IC_Ticker"] = g_ic.loc[idx_combo, "Ticker"]
            row["Best_DA_IC_Value"] = g_ic.loc[idx_combo, "DA_IC_Score"]

        if "Test_DA" in g.columns:
            idx_da = g["Test_DA"].astype(float).idxmax()
            row["Best_Test_DA_Ticker"] = g.loc[idx_da, "Ticker"]
            row["Best_Test_DA_Value"] = g.loc[idx_da, "Test_DA"]

        if "Test_IC" in g.columns:
            idx_ic = g["Test_IC"].astype(float).idxmax()
            row["Best_Test_IC_Ticker"] = g.loc[idx_ic, "Ticker"]
            row["Best_Test_IC_Value"] = g.loc[idx_ic, "Test_IC"]

        if "Test_RMSE" in g.columns:
            idx_rmse = g["Test_RMSE"].astype(float).idxmin()
            row["Best_Test_RMSE_Ticker"] = g.loc[idx_rmse, "Ticker"]
            row["Best_Test_RMSE_Value"] = g.loc[idx_rmse, "Test_RMSE"]

        rows.append(row)

    out = pd.DataFrame(rows).sort_values("H_FWD").reset_index(drop=True)
    return out


def to_markdown(df: pd.DataFrame, float_digits: int = 6) -> str:
    if df.empty:
        return "(no rows)"

    display_df = df.copy()
    for c in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[c]) or pd.api.types.is_numeric_dtype(display_df[c]):
            display_df[c] = display_df[c].apply(lambda x: fmt_float(x, float_digits))
    cols = list(display_df.columns)
    rows = display_df.astype(str).values.tolist()

    def cell(v: str) -> str:
        return str(v).replace("|", "\\|")

    header = "| " + " | ".join(cell(c) for c in cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def parse_horizons(raw: Optional[str]) -> Optional[List[int]]:
    if not raw:
        return None
    vals = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            vals.append(int(part))
    return vals or None


def main():
    parser = argparse.ArgumentParser(description="Analyze training summary and print thesis-ready tables.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path ke file ALL_training_summary.csv atau ALL_training_results.xlsx yang dipilih/upload secara eksplisit",
    )
    parser.add_argument("--sheet", type=str, default="ALL_Summary", help="Excel sheet name to read when input is xlsx")
    parser.add_argument(
        "--metric",
        type=str,
        default="DA_IC",
        help="Ranking metric. Use DA_IC to rank by combined Test_DA and Test_IC.",
    )
    parser.add_argument("--top-n", type=int, default=5, help="How many rows to show per horizon")
    parser.add_argument("--horizons", type=str, default=None, help="Comma-separated horizons, e.g. 22 or 22,44")
    parser.add_argument("--out-csv", type=str, default=None, help="Optional path to save the top-N table as CSV")
    parser.add_argument("--summary-csv", type=str, default=None, help="Optional path to save the horizon summary as CSV")
    parser.add_argument(
        "--unique-tickers",
        action="store_true",
        help="Rank best rows per ticker first so the output contains unique emiten only",
    )
    args = parser.parse_args()

    input_path = _find_input_path(args.input)
    horizons = parse_horizons(args.horizons)

    df = load_summary(input_path, sheet_name=args.sheet)
    needed_metrics = ["Test_RMSE", "Test_MAE", "Test_R2", "Test_DA", "Test_IC", "Val_RMSE", "Val_DA", "Val_IC", "Final_BestValLoss_RMSE"]
    for c in needed_metrics:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if horizons:
        df = df[df["H_FWD"].isin(horizons)].copy()

    if df.empty:
        raise ValueError("Tidak ada data setelah filter horizon.")

    print(f"# Input: {input_path}")
    print(f"# Rows: {len(df)}")
    print(f"# Metric ranking: {args.metric}")
    print()

    sum_df = summary_by_horizon(df)
    print("## Horizon Summary")
    print(to_markdown(sum_df))
    print()

    if args.unique_tickers:
        top_df = pick_top_unique_tickers(df, metric=args.metric, top_n=args.top_n, horizons=horizons)
    else:
        top_df = pick_top_rows(df, metric=args.metric, top_n=args.top_n, horizons=horizons)
    pretty_metric = "DA+IC" if args.metric.upper() == "DA_IC" else args.metric
    print(f"## Top {args.top_n} by {pretty_metric}")
    print(to_markdown(top_df))
    print()

    # Optional extra: show the best row per horizon by the chosen metric only.
    if not top_df.empty:
        best_rows = []
        for h, g in df.groupby("H_FWD"):
            row = _pick_best_row_in_group(g, args.metric)
            if row.empty:
                continue
            rank_metric = _ranking_metric_name(args.metric)
            keep_cols = ["H_FWD", "Ticker", "Fitur", "Final_BestEpoch", "Val_RMSE", "Val_DA", "Val_IC", "Test_RMSE", "Test_DA", "Test_IC"]
            if rank_metric not in keep_cols:
                keep_cols.append(rank_metric)
            row = row[keep_cols].to_dict()
            best_rows.append(row)
        if best_rows:
            best_df = pd.DataFrame(best_rows).sort_values("H_FWD").reset_index(drop=True)
            print(f"## Best row per horizon by {pretty_metric}")
            print(to_markdown(best_df))
            print()

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        top_df.to_csv(out_path, index=False)
        print(f"[saved] {out_path}")

    if args.summary_csv:
        out_path = Path(args.summary_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sum_df.to_csv(out_path, index=False)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
