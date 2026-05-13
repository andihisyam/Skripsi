"""
Compare training results from two experiment folders.

Intended use:
  python utils/compare_training_results.py

Default folders:
  - d:/Kuliah/Skripsi/Code/seq2seq_model
  - d:/Kuliah/Skripsi/Code/eksperimen15/hasil

The script reads ALL_training_results.xlsx -> sheet ALL_Summary,
then compares the models on the same ticker and horizon.

Outputs:
- horizon coverage
- model summary on common tickers
- side-by-side ticker comparison
- winner counts per horizon
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


DEFAULT_FOLDERS = {
    "Seq2Seq+Sentimen": r"d:/Kuliah/Skripsi/Code/seq2seq_model",
    "Sklearn+Sentimen": r"d:/Kuliah/Skripsi/Code/models2",
}

LOWER_IS_BETTER = {
    "Test_RMSE",
    "Test_MAE",
    "Avg_CV_Loss",
    "Final_BestValLoss_RMSE",
    "Val_RMSE",
    "Val_MAE",
    "TestSize",
}


def _pick_first_existing(df, candidates, default=np.nan):
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series([default] * len(df), index=df.index)


def is_higher_better(metric: str) -> bool:
    return metric not in LOWER_IS_BETTER


def load_training_results(folder: str, model_name: str) -> pd.DataFrame:
    folder_path = Path(folder)
    file_path = folder_path / "ALL_training_results.xlsx"
    if not file_path.exists():
        raise FileNotFoundError(f"Tidak menemukan {file_path}")

    df = pd.read_excel(file_path, sheet_name="ALL_Summary").copy()

    required = {"H_FWD", "Ticker", "Fitur"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan pada {file_path}: {sorted(missing)}")

    out = pd.DataFrame(index=df.index)
    out["Model"] = model_name
    out["SourceFile"] = str(file_path)
    out["H_FWD"] = pd.to_numeric(df["H_FWD"], errors="coerce")
    out["Ticker"] = df["Ticker"].astype(str).str.upper()
    out["Fitur"] = df["Fitur"].astype(str)

    metric_cols = [
        "Final_BestEpoch",
        "Val_RMSE",
        "Val_MAE",
        "Val_R2",
        "Val_DA",
        "Val_IC",
        "Test_RMSE",
        "Test_MAE",
        "Test_R2",
        "Test_DA",
        "Test_IC",
        "Avg_CV_Loss",
        "Final_BestValLoss_RMSE",
        "TestSize",
    ]
    for c in metric_cols:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce")

    return out


def _pick_best_row_in_group(group: pd.DataFrame, metric: str) -> pd.Series:
    g = group.copy()
    g[metric] = pd.to_numeric(g[metric], errors="coerce")
    g = g.dropna(subset=[metric]).copy()
    if g.empty:
        return pd.Series(dtype=object)

    asc = not is_higher_better(metric)
    best_value = g[metric].min() if asc else g[metric].max()
    candidates = g[g[metric] == best_value].copy()

    score_cols = [
        "Final_BestEpoch",
        "Val_RMSE",
        "Val_DA",
        "Val_IC",
        "Test_RMSE",
        "Test_DA",
        "Test_IC",
    ]
    available = [c for c in score_cols if c in candidates.columns]
    if available:
        candidates["_complete_score"] = candidates[available].notna().sum(axis=1)
        candidates = candidates.sort_values(
            ["_complete_score", "Ticker", "Fitur"],
            ascending=[False, True, True],
        )
    else:
        candidates = candidates.sort_values(["Ticker", "Fitur"], ascending=[True, True])

    return candidates.iloc[0].drop(labels=["_complete_score"], errors="ignore")


def select_best_per_ticker(df: pd.DataFrame, metric: str, horizons: Optional[List[int]] = None) -> pd.DataFrame:
    dfx = df.copy()
    if horizons:
        dfx = dfx[dfx["H_FWD"].isin(horizons)].copy()
    if dfx.empty:
        return dfx

    best_rows = []
    for (h, ticker), g in dfx.groupby(["H_FWD", "Ticker"], as_index=False):
        row = _pick_best_row_in_group(g, metric)
        if row.empty:
            continue
        best_rows.append(row)

    if not best_rows:
        return pd.DataFrame()
    return pd.DataFrame(best_rows).reset_index(drop=True)


def to_markdown(df: pd.DataFrame, float_digits: int = 6) -> str:
    if df.empty:
        return "(no rows)"

    display_df = df.copy()
    for c in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[c]) or pd.api.types.is_numeric_dtype(display_df[c]):
            display_df[c] = display_df[c].apply(lambda x: "" if pd.isna(x) else f"{float(x):.{float_digits}f}")
    cols = list(display_df.columns)
    rows = display_df.astype(str).values.tolist()

    def cell(v: str) -> str:
        return str(v).replace("|", "\\|")

    header = "| " + " | ".join(cell(c) for c in cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def summarize_common(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, h), g in df.groupby(["Model", "H_FWD"], dropna=False):
        rows.append(
            {
                "Model": model,
                "H_FWD": h,
                "N": len(g),
                "Val_RMSE": g["Val_RMSE"].mean() if "Val_RMSE" in g.columns else np.nan,
                "Val_DA": g["Val_DA"].mean() if "Val_DA" in g.columns else np.nan,
                "Test_RMSE": g["Test_RMSE"].mean() if "Test_RMSE" in g.columns else np.nan,
                "Test_DA": g["Test_DA"].mean() if "Test_DA" in g.columns else np.nan,
                "Test_IC": g["Test_IC"].mean() if "Test_IC" in g.columns else np.nan,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["H_FWD", "Model"]).reset_index(drop=True)


def build_side_by_side(seq_df: pd.DataFrame, skl_df: pd.DataFrame, horizons: Optional[List[int]] = None) -> pd.DataFrame:
    seq = seq_df.copy()
    skl = skl_df.copy()
    if horizons:
        seq = seq[seq["H_FWD"].isin(horizons)].copy()
        skl = skl[skl["H_FWD"].isin(horizons)].copy()

    common_keys = sorted(set(zip(seq["H_FWD"], seq["Ticker"])).intersection(set(zip(skl["H_FWD"], skl["Ticker"]))))
    rows = []
    for h, ticker in common_keys:
        a = seq[(seq["H_FWD"] == h) & (seq["Ticker"] == ticker)].iloc[0]
        b = skl[(skl["H_FWD"] == h) & (skl["Ticker"] == ticker)].iloc[0]
        rows.append(
            {
                "H_FWD": h,
                "Ticker": ticker,
                "Seq2Seq_Fitur": a["Fitur"],
                "Sklearn_Fitur": b["Fitur"],
                "Seq2Seq_Test_DA": a.get("Test_DA", np.nan),
                "Sklearn_Test_DA": b.get("Test_DA", np.nan),
                "Seq2Seq_Test_RMSE": a.get("Test_RMSE", np.nan),
                "Sklearn_Test_RMSE": b.get("Test_RMSE", np.nan),
                "Seq2Seq_Test_IC": a.get("Test_IC", np.nan),
                "Sklearn_Test_IC": b.get("Test_IC", np.nan),
                "Seq2Seq_Val_DA": a.get("Val_DA", np.nan),
                "Sklearn_Val_DA": b.get("Val_DA", np.nan),
                "Seq2Seq_Val_RMSE": a.get("Val_RMSE", np.nan),
                "Sklearn_Val_RMSE": b.get("Val_RMSE", np.nan),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["H_FWD", "Ticker"]).reset_index(drop=True)


def winner_counts(side_df: pd.DataFrame, metric_left: str = "Seq2Seq_Test_DA", metric_right: str = "Sklearn_Test_DA") -> pd.DataFrame:
    if side_df.empty:
        return pd.DataFrame()

    rows = []
    for (h, ticker), g in side_df.groupby(["H_FWD", "Ticker"], dropna=False):
        l = g.iloc[0][metric_left]
        r = g.iloc[0][metric_right]
        if pd.isna(l) or pd.isna(r):
            winner = "NA"
        elif np.isclose(float(l), float(r)):
            winner = "Tie"
        elif float(l) > float(r):
            winner = "Seq2Seq+Sentimen"
        else:
            winner = "Sklearn+Sentimen"
        rows.append({"H_FWD": h, "Ticker": ticker, "Winner": winner})

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return (
        out.groupby(["H_FWD", "Winner"])["Ticker"]
        .count()
        .reset_index(name="WinCount")
        .sort_values(["H_FWD", "WinCount"], ascending=[True, False])
        .reset_index(drop=True)
    )


def parse_horizons(raw: Optional[str]) -> Optional[List[int]]:
    if not raw:
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Fair compare training results from two experiment folders.")
    parser.add_argument("--seq2seq", type=str, default=DEFAULT_FOLDERS["Seq2Seq+Sentimen"], help="Folder path for seq2seq results")
    parser.add_argument("--sklearn", type=str, default=DEFAULT_FOLDERS["Sklearn+Sentimen"], help="Folder path for sklearn results")
    parser.add_argument("--metric", type=str, default="Test_DA", help="Metric for best-row selection per ticker")
    parser.add_argument("--horizons", type=str, default=None, help="Comma-separated horizons, e.g. 22 or 22,44")
    args = parser.parse_args()

    horizons = parse_horizons(args.horizons)

    seq_raw = load_training_results(args.seq2seq, "Seq2Seq+Sentimen")
    skl_raw = load_training_results(args.sklearn, "Sklearn+Sentimen")

    if horizons:
        seq_raw = seq_raw[seq_raw["H_FWD"].isin(horizons)].copy()
        skl_raw = skl_raw[skl_raw["H_FWD"].isin(horizons)].copy()

    seq_best = select_best_per_ticker(seq_raw, args.metric, horizons=horizons)
    skl_best = select_best_per_ticker(skl_raw, args.metric, horizons=horizons)

    common_horizons = sorted(set(seq_best["H_FWD"].dropna().unique()).intersection(set(skl_best["H_FWD"].dropna().unique())))
    side = build_side_by_side(seq_best, skl_best, horizons=common_horizons if common_horizons else horizons)

    print(f"# Seq2Seq folder: {args.seq2seq}")
    print(f"# Sklearn folder: {args.sklearn}")
    print(f"# Metric selection: {args.metric}")
    print(f"# Horizons: {common_horizons if common_horizons else horizons}")
    print()

    print("## Best per ticker summary")
    common_counts = (
        side.groupby("H_FWD")["Ticker"]
        .count()
        .reset_index(name="CommonTickers")
        if not side.empty
        else pd.DataFrame(columns=["H_FWD", "CommonTickers"])
    )
    print(to_markdown(common_counts))
    print()

    print("## Model summary on common ticker set")
    if side.empty:
        print("(no rows)")
        return

    seq_common = seq_best[seq_best.set_index(["H_FWD", "Ticker"]).index.isin(side.set_index(["H_FWD", "Ticker"]).index)]
    skl_common = skl_best[skl_best.set_index(["H_FWD", "Ticker"]).index.isin(side.set_index(["H_FWD", "Ticker"]).index)]
    summary = summarize_common(pd.concat([seq_common, skl_common], ignore_index=True))
    print(to_markdown(summary))
    print()

    print("## Side-by-side per ticker")
    print(to_markdown(side))
    print()

    print("## Winner counts")
    print(to_markdown(winner_counts(side)))


if __name__ == "__main__":
    main()
