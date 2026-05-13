"""
utils/file_utils.py
===================
Helper untuk menyimpan output ke file:
  - save_excel       : simpan dict of DataFrames ke satu file .xlsx
  - save_csv         : simpan satu DataFrame ke .csv
  - ensure_output_dirs: buat semua subdirektori output sekaligus

Tidak ada logika kalkulasi di sini — hanya I/O.
"""

import os
import pandas as pd
from typing import Dict

from config import OUTPUT_DIR


# ============================================================
# DIREKTORI
# ============================================================

def ensure_output_dirs(base_dir: str = OUTPUT_DIR) -> Dict[str, str]:
    """
    Buat semua subdirektori output yang dibutuhkan sekaligus.

    Return dict berisi path masing-masing subdirektori:
        {
            "base":  "mpc_results/",
            "excel": "mpc_results/excel/",
            "plots": "mpc_results/plots/",
            "logs":  "mpc_results/logs/",
        }
    """
    subdirs = {
        "base":  base_dir,
        "excel": os.path.join(base_dir, "excel"),
        "plots": os.path.join(base_dir, "plots"),
        "logs":  os.path.join(base_dir, "logs"),
    }
    for path in subdirs.values():
        os.makedirs(path, exist_ok=True)

    return subdirs


# ============================================================
# SAVE EXCEL
# ============================================================

def save_excel(
    sheets:   Dict[str, pd.DataFrame],
    out_dir:  str,
    filename: str = "mpc_results.xlsx",
) -> str:
    """
    Simpan semua DataFrame ke satu file Excel.

    Setiap key di `sheets` menjadi nama sheet.
    Sheet name dipotong otomatis jika > 31 karakter (batas Excel).

    Parameter
    ---------
    sheets   : dict {nama_sheet: DataFrame}
    out_dir  : direktori tujuan
    filename : nama file output

    Return
    ------
    Path lengkap file yang disimpan.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            # Excel membatasi nama sheet maksimal 31 karakter
            safe_name = str(sheet_name)[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)

    print(f"[SAVED] Excel: {path}")
    return path


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(
    df:       pd.DataFrame,
    out_dir:  str,
    filename: str,
) -> str:
    """
    Simpan satu DataFrame ke file CSV.

    Parameter
    ---------
    df       : DataFrame yang akan disimpan
    out_dir  : direktori tujuan
    filename : nama file output (termasuk ekstensi .csv)

    Return
    ------
    Path lengkap file yang disimpan.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    df.to_csv(path, index=False)
    print(f"[SAVED] CSV: {path}")
    return path