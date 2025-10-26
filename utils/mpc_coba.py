import numpy as np, pandas as pd, cvxpy as cp
from config import DATE_COL

def estimate_cov(returns_df: pd.DataFrame, lookback=60):
    # returns_df: index Date, columns = tickers, values = daily returns
    # ambil window terakhir
    window = returns_df.tail(lookback)
    return np.cov(window.dropna().T)

def mpc_optimize(expected_ret: np.ndarray, cov: np.ndarray, lam=5.0, wmax=0.15):
    """
    expected_ret: (N_assets,) ekspektasi return (mis. rata-rata prediksi horizon / atau return 1-bulan)
    cov: (N,N) matriks kovarians risiko
    Minim: 0.5 * w^T cov w - lam * expected_ret^T w
    s.t. sum w = 1, w >= 0, w <= wmax
    """
    n = expected_ret.shape[0]
    w = cp.Variable(n)
    obj = 0.5 * cp.quad_form(w, cov) - lam * expected_ret @ w
    constraints = [cp.sum(w) == 1, w >= 0, w <= wmax]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.OSQP)
    return np.clip(w.value, 0, 1)

# Contoh pemakaian (pseudo):
# 1) Ambil prediksi 1-bulan ke depan untuk semua 45 emiten → bentuk vektor expected_ret (bisa rata-rata y_pred horizon)
# 2) Hitung kovarians dari return historis terbaru (mis. 60 hari)
# 3) Panggil mpc_optimize → dapat bobot portofolio
