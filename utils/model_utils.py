"""
Compatibility shim.

Gunakan modul baru:
- utils.model_utils_seq2seq untuk pipeline seq2seq (default training sekarang)
- utils.model_utils_sklearn untuk wrapper LSTM style sklearn
"""

from utils.model_utils_seq2seq import (
    StableLoss,
    RMSELoss,
    LSTMRegressorSeq2Seq,
    build_lstm_model,
    build_optimizer,
)
from utils.model_utils_sklearn import (
    LSTMSklearnNet,
    LSTMRegressorSklearn,
)


# Backward-compat alias:
LSTMRegressor = LSTMRegressorSeq2Seq

__all__ = [
    "StableLoss",
    "RMSELoss",
    "LSTMRegressorSeq2Seq",
    "LSTMRegressor",
    "LSTMSklearnNet",
    "LSTMRegressorSklearn",
    "build_lstm_model",
    "build_optimizer",
]

