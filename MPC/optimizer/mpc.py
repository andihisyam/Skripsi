"""
optimizers/mpc.py
=================
Model Predictive Control (MPC) untuk optimasi bobot portofolio.

Formulasi masalah:
    min  u^T (Q + R) u  +  2 f^T u  -  lambda * r^T u
    s.t. sum(u) = 1
         u >= MIN_WEIGHT
         u_i <= MAX_WEIGHT * (0.5 + 0.5 * conf_i)   ← adaptif per saham

Dimana:
    u      = vektor bobot portofolio (variabel keputusan)
    Q      = matriks state-cost (Q_SCALE * I)
    R      = matriks control-cost (R_SCALE * I)
    f      = -Q @ r_pred  (arah minimisasi)
    r_pred = adjusted predicted return per saham
    conf_i = confidence score saham ke-i

Batas bobot atas bersifat adaptif: saham dengan confidence tinggi
boleh mendapat alokasi lebih besar, dan sebaliknya.
"""

import numpy as np
import cvxpy as cp
from typing import List, Optional

from config import (
    Q_SCALE,
    R_SCALE,
    MIN_WEIGHT,
    MAX_WEIGHT,
    LAMBDA_RETURN,
)


# ============================================================
# MPC OPTIMIZER
# ============================================================

def run_mpc(
    r_pred:        np.ndarray,
    tickers:       List[str],
    confidence:    Optional[np.ndarray] = None,
    max_weight:    float = MAX_WEIGHT,
    lambda_return: float = LAMBDA_RETURN,
) -> np.ndarray:
    """
    Optimasi bobot portofolio dengan MPC menggunakan CVXPY.

    Parameter
    ---------
    r_pred        : (N,1) adjusted predicted log return per saham
    tickers       : list nama ticker (untuk pesan error)
    confidence    : (N,1) confidence score per saham (opsional)
                    Jika None, semua saham mendapat batas atas = max_weight
    max_weight    : batas atas bobot maksimum (sebelum penyesuaian confidence)
    lambda_return : skalar penalti return dalam objective

    Return
    ------
    weights : (N,) array bobot portofolio yang sudah dinormalisasi (sum = 1)

    Raises
    ------
    ValueError jika solver tidak menemukan solusi optimal.
    """
    n     = len(tickers)
    Q     = np.eye(n) * Q_SCALE
    R     = np.eye(n) * R_SCALE
    H_mat = Q + R          # karena A=B=I dan state awal = 0, disederhanakan
    f     = -Q @ r_pred    # gradien arah return

    u = cp.Variable((n, 1))

    objective = cp.Minimize(
        cp.quad_form(u, H_mat)
        + 2 * f.T @ u
        - lambda_return * (r_pred.T @ u)
    )

    max_weight_vec = _compute_adaptive_weights(n, max_weight, confidence)

    constraints = [
        cp.sum(u) == 1,
        u >= MIN_WEIGHT,
        u <= max_weight_vec.reshape(-1, 1),
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(
            f"MPC gagal untuk {len(tickers)} saham: status = {prob.status}\n"
            f"Coba turunkan MIN_CONFIDENCE atau naikkan MAX_WEIGHT di config.py"
        )

    # Clip nilai negatif kecil akibat floating point, lalu normalisasi
    w = np.maximum(u.value.flatten(), 0)
    return w / w.sum()


# ============================================================
# HELPER INTERNAL
# ============================================================

def _compute_adaptive_weights(
    n:          int,
    max_weight: float,
    confidence: Optional[np.ndarray],
) -> np.ndarray:
    """
    Hitung batas atas bobot per saham secara adaptif.

    Formula: max_w_i = max_weight * (0.5 + 0.5 * confidence_i)
      - confidence = 1.0 → batas atas = max_weight          (penuh)
      - confidence = 0.5 → batas atas = max_weight * 0.75   (dikurangi)
      - confidence = 0.0 → batas atas = max_weight * 0.5    (setengah)

    Jika total cap < 1.0 (tidak feasible karena terlalu ketat),
    semua batas dinaikkan proporsional agar sum(max_w) >= 1.

    Return
    ------
    max_weight_vec : (N,) array batas atas per saham
    """
    if confidence is None:
        return np.full(n, max_weight)

    conf_vec       = np.clip(confidence.reshape(-1), 0.0, 1.0)
    max_weight_vec = max_weight * (0.5 + 0.5 * conf_vec)

    # Pastikan problem feasible: total cap harus >= 1
    cap_sum = float(np.sum(max_weight_vec))
    if cap_sum < 1.0:
        print(
            f"[WARN] Total cap bobot = {cap_sum:.3f} < 1.0 — "
            f"dinaikkan proporsional agar feasible"
        )
        max_weight_vec *= 1.0 / cap_sum

    return max_weight_vec


def compute_dynamic_max_weight(n_assets: int, base_max: float = MAX_WEIGHT) -> float:
    """
    Hitung MAX_WEIGHT dinamis berdasarkan jumlah aset yang diseleksi.

    Memastikan batas atas tidak terlalu ketat saat jumlah aset sedikit.
    Contoh: 4 saham → minimal 25% per saham agar bisa sum = 1.

    Parameter
    ---------
    n_assets : jumlah saham dalam portofolio
    base_max : nilai MAX_WEIGHT dari config

    Return
    ------
    max_weight yang dipakai (maks dari base_max dan 1/n_assets)
    """
    if n_assets <= 0:
        return base_max
    return max(base_max, 1.0 / n_assets)