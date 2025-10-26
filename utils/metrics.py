import numpy as np
from sklearn.metrics import r2_score

def evaluate(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = r2_score(y_true.flatten(), y_pred.flatten())
    return rmse, r2
