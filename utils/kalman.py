# utils/kalman_utils.py
import numpy as np
import pandas as pd
from pykalman import KalmanFilter

def apply_kalman_filter_level_trend(
    series: pd.Series,
    level_var: float = 1.0,
    trend_var: float = 0.1,
    observation_var: float = 1.0,
) -> pd.Series:
    """
    Kalman Filter 2D: Level + Trend (local linear trend model).

    State:
      x_t = [level_t, trend_t]^T

    Transition:
      level_t = level_{t-1} + trend_{t-1} + noise_level
      trend_t = trend_{t-1} + noise_trend

    Observation:
      y_t = level_t + noise_obs

    Parameter:
      level_var:    variansi noise untuk level (semakin besar => lebih responsif)
      trend_var:    variansi noise untuk trend (semakin besar => trend lebih mudah berubah)
      observation_var: variansi noise observasi (semakin besar => smoothing lebih kuat)
    """
    if not isinstance(series, pd.Series):
        raise ValueError("Input harus berupa pd.Series")
    if len(series) < 2:
        return series.copy()

    y = series.astype(float).values

    # Transition matrix (A) and observation matrix (H)
    A = np.array([[1.0, 1.0],
                  [0.0, 1.0]])

    H = np.array([[1.0, 0.0]])

    # Initial state: level = first value, trend = first difference (atau 0 jika tidak ada)
    init_level = float(y[0])
    init_trend = float(y[1] - y[0]) if len(y) >= 2 else 0.0
    initial_state_mean = np.array([init_level, init_trend])

    # Initial covariance: cukup “longgar” agar cepat adaptasi di awal
    initial_state_covariance = np.array([[10.0, 0.0],
                                         [0.0, 10.0]])

    # Process noise covariance (Q)
    Q = np.array([[float(level_var), 0.0],
                  [0.0, float(trend_var)]])

    # Observation noise covariance (R)
    R = np.array([[float(observation_var)]])

    kf = KalmanFilter(
        transition_matrices=A,
        observation_matrices=H,
        initial_state_mean=initial_state_mean,
        initial_state_covariance=initial_state_covariance,
        transition_covariance=Q,
        observation_covariance=R,
    )

    state_means, _ = kf.filter(y)
    smooth_level = state_means[:, 0]
    return pd.Series(smooth_level, index=series.index)
