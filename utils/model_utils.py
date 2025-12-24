from sklearn.base import BaseEstimator, RegressorMixin
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
    def __init__(self, dense_units=64, hidden_units=128, dropout=0.3, lr=1e-3, optimizer='Adam', batch_size=32,epochs=100):
        # Store necessary parameters
        self.dense_units = dense_units
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.lr = lr
        self.optimizer_name = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None  # LSTM model
        self.optimizer = optimizer

    def fit(self, X, y):
        # Ubah data menjadi tensor
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        # Print dimensi data sebelum masuk ke model
        print(f"Dimensi data X: {X.shape}, Dimensi target y: {y.shape}")

        # Inisialisasi model LSTM
        input_size = X.shape[2]  # Menentukan input_size berdasarkan data (dimensi terakhir dari data X)
        
        # Berikan input_size pada inisialisasi model
        self.model = LSTMRegressor(input_size=input_size, 
                                dense_units=self.dense_units, 
                                hidden_units=self.hidden_units, 
                                dropout=self.dropout, 
                                output_dim=22
                                )

        # Tentukan optimizer dan loss function
        optimizer = getattr(optim, self.optimizer_name)(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        # Training loop
        for epoch in range(100):  # Misalnya train selama 100 epoch
            self.model.train()
            optimizer.zero_grad()

            # Lakukan prediksi
            outputs = self.model(X)

            # Print dimensi output model dan target sebelum perhitungan loss
            print(f"Dimensi output model: {outputs.shape}, Dimensi target y: {y.shape}")

            # Jika output dan target tidak sesuai, perbaiki dengan reshape
            if outputs.shape != y.shape:
                print(f"Dimensi mismatch: Output {outputs.shape} vs Target {y.shape}")
                y = y.view(-1, 22)  # reshape y menjadi (batch_size, 1) jika perlu

            # Perhitungan loss
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            # Print setiap epoch untuk memantau progress
            print(f"Epoch {epoch+1}/{100}, Loss: {loss.item()}")

        return self


    def predict(self, X):
        # Prediction with the trained model
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            predictions = self.model(X)
        return predictions.detach().cpu().numpy()


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
