from sklearn.base import BaseEstimator, RegressorMixin
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class LSTMSklearnNet(nn.Module):
    """
    LSTM direct multi-output:
    - membaca sequence historis sekali
    - output langsung vector H-step (tanpa decoder autoregressive)
    """

    def __init__(
        self,
        input_size,
        dense_units=32,
        hidden_units=64,
        dropout=0.2,
        output_dim=1,
        num_layers=2,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_units,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.fc1 = nn.Linear(hidden_units, dense_units)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(dense_units, self.output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]
        x = self.fc1(h_last)
        x = self.relu(x)
        x = self.drop(x)
        return self.out(x)


class LSTMRegressorSklearn(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        dense_units=32,
        hidden_units=64,
        dropout=0.2,
        lr=1e-3,
        optimizer="AdamW",
        batch_size=32,
        epochs=100,
        verbose=False,
        output_dim=1,
        num_layers=2,
    ):
        self.dense_units = dense_units
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.lr = lr
        self.optimizer_name = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.output_dim = int(output_dim)
        self.verbose = verbose
        self.num_layers = num_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_optimizer(self):
        opt = str(self.optimizer_name).lower()
        if opt == "rmsprop":
            return optim.RMSprop(self.model.parameters(), lr=self.lr)
        if opt == "adam":
            return optim.Adam(self.model.parameters(), lr=self.lr)
        return optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-2)

    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        if y_tensor.dim() == 1:
            y_tensor = y_tensor.unsqueeze(1)

        input_size = X_tensor.shape[2]
        self.model = LSTMSklearnNet(
            input_size=input_size,
            dense_units=self.dense_units,
            hidden_units=self.hidden_units,
            dropout=self.dropout,
            output_dim=self.output_dim,
            num_layers=self.num_layers,
        ).to(self.device)

        optimizer = self._build_optimizer()
        criterion = nn.HuberLoss(delta=1.0)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch)

                if outputs.shape != y_batch.shape:
                    y_batch = y_batch.view(outputs.shape)

                loss = criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            if self.verbose and (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / max(n_batches, 1)
                print(f"Epoch {epoch + 1}/{self.epochs}, Avg Loss: {avg_loss:.6f}")

        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model belum di-fit!")
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self.model(X_tensor)
        return predictions.cpu().numpy()

