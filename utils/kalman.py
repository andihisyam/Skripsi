# utils/kalman_utils.py
import pandas as pd
from pykalman import KalmanFilter

def apply_kalman_filter(series: pd.Series, transition_cov=0.01, observation_cov=1.0):
    """
    Terapkan Kalman Filter 1D untuk smoothing data time series.

    Args:
        series (pd.Series): Data deret waktu (misal kolom 'Close').
        transition_cov (float): Variansi proses (Q) -> semakin besar, filter lebih responsif.
        observation_cov (float): Variansi observasi (R) -> semakin besar, hasil lebih halus.

    Returns:
        pd.Series: Deret hasil smoothing dengan indeks sama.
    """
    if not isinstance(series, pd.Series):
        raise ValueError("Input harus berupa pd.Series")

    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=series.iloc[0],
        initial_state_covariance=1,
        observation_covariance=observation_cov,
        transition_covariance=transition_cov
    )

    state_means, _ = kf.filter(series.values)
    return pd.Series(state_means.flatten(), index=series.index)
