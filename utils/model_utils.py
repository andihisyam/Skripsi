from sklearn.base import BaseEstimator, RegressorMixin
from torch.utils.data import DataLoader, TensorDataset
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


# LSTM Model definition
class LSTMRegressor(nn.Module):
    def __init__(self, input_size ,dense_units=64, hidden_units=128, dropout=0.3, output_dim=22):
        super(LSTMRegressor, self).__init__()

        self.hidden_units = hidden_units
        self.dense_units = dense_units
        self.dropout = dropout

        # LSTM Layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_units, batch_first=True)  # Pastikan input_size disesuaikan dengan jumlah fitur

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_units, dense_units)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(dense_units, output_dim)

        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        # LSTM forward pass
        x, _ = self.lstm(x)

        # Select the last output of LSTM (many-to-one)
        x = x[:, -1, :]

        # Fully connected layers with dropout
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout_layer(x)
        x = self.fc_out(x)
        return x

# Fungsi untuk membangun model LSTM
def build_lstm_model(Xtr, dense_units=64, hidden_units=128, dropout=0.3, output_dim=1):
    # Dapatkan jumlah fitur dari Xtr
    input_size = Xtr.shape[2]  # Menyediakan input_size berdasarkan data (jumlah fitur)
    
    # Membangun dan mengembalikan model LSTM
    model = LSTMRegressor(
        input_size=input_size,
        dense_units=dense_units,
        hidden_units=hidden_units,
        dropout=dropout,
        output_dim=output_dim
    )
    return model

# Pembungkus untuk LSTM yang kompatibel dengan GridSearchCV
class LSTMRegressorSklearn(BaseEstimator, RegressorMixin):
    def __init__(self, dense_units=64, hidden_units=128, dropout=0.3, lr=1e-3, optimizer='Adam', batch_size=32,epochs=100,verbose=False):
        # Store necessary parameters
        self.dense_units = dense_units
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.lr = lr
        self.optimizer_name = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None  # LSTM model
        self.verbose=verbose
        self.optimizer = optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y):
        # Ubah data menjadi tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        # Inisialisasi model LSTM
        input_size = X_tensor.shape[2]  # Menentukan input_size berdasarkan data 
        
        # Berikan input_size pada inisialisasi model
        self.model = LSTMRegressor(input_size=input_size, 
                                dense_units=self.dense_units, 
                                hidden_units=self.hidden_units, 
                                dropout=self.dropout, 
                                output_dim=22
                                ).to(self.device)

        # Tentukan optimizer dan loss function
        optimizer = getattr(optim, self.optimizer_name)(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor.to(self.device))
            target = y_tensor.to(self.device)
            
            if outputs.shape != target.shape:
                target = target.view(outputs.shape)

            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            # Print loss
            if self.verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.6f}")

        return self


    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self.model(X_tensor)
        return predictions.cpu().numpy()


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
