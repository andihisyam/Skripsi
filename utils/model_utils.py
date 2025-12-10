import torch
import torch.nn as nn
import torch.optim as optim

# ======================
#  RMSE Loss
# ======================
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


# ======================
#  LSTM Model
# ======================
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, dense_units, output_dim, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, dense_units)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(dense_units, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc_out(out)
        return out


# ======================
# Build LSTM from params
# ======================
def build_lstm_model(input_shape, dense_units, hidden_units, dropout, output_dim):
    num_features = input_shape[-1]
    return LSTMRegressor(
        input_dim=num_features,
        hidden_dim=hidden_units,
        dense_units=dense_units,
        output_dim=output_dim,
        dropout=dropout
    )


# ======================
#  Optimizer builder
# ======================
def build_optimizer(name: str, params, lr: float):
    name = name.lower()
    if name == "adam":
        return optim.Adam(params, lr=lr)
    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=1e-2)
    if name == "rmsprop":
        return optim.RMSprop(params, lr=lr)
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9)

    raise ValueError(f"Optimizer '{name}' tidak didukung.")
