import torch
import torch.nn as nn
import torch.optim as optim


class StableLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.loss = nn.HuberLoss(delta=delta)

    def forward(self, yhat, y):
        return self.loss(yhat, y)


class RMSELoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y) + self.eps)


def build_lstm_model(Xtr, dense_units=64, hidden_units=128, dropout=0.3, output_dim=1, num_layers=1):
    input_size = Xtr.shape[2]
    return LSTMRegressorSeq2Seq(
        input_size=input_size,
        dense_units=dense_units,
        hidden_units=hidden_units,
        dropout=dropout,
        output_dim=output_dim,
        num_layers=num_layers,
    )


class LSTMRegressorSeq2Seq(nn.Module):
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

        self.is_seq2seq = True
        self.output_dim = int(output_dim)
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_units,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )

        self.decoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_units,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )

        self.decoder_dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_units, dense_units)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)
        self.fc_out = nn.Linear(dense_units, 1)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                if ("encoder" in name or "decoder" in name) and param.shape[0] % 4 == 0:
                    n = param.shape[0] // 4
                    param.data[n:2 * n].fill_(1.0)
            elif "weight" in name:
                nn.init.xavier_uniform_(param)

    def _decode_step(self, dec_state):
        x = self.decoder_dropout(dec_state)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout_layer(x)
        return self.fc_out(x)

    def forward(self, x, teacher_forcing_targets=None, teacher_forcing_ratio: float = 0.0):
        _, (h, c) = self.encoder(x)

        batch_size = x.size(0)
        dtype = x.dtype
        device = x.device

        dec_input = torch.zeros(batch_size, 1, 1, dtype=dtype, device=device)

        tf_targets = None
        if teacher_forcing_targets is not None:
            tf_targets = teacher_forcing_targets
            if tf_targets.dim() == 1:
                tf_targets = tf_targets.unsqueeze(1)

        outputs = []
        for t in range(self.output_dim):
            dec_out, (h, c) = self.decoder(dec_input, (h, c))
            step_hidden = dec_out[:, -1, :]
            y_step = self._decode_step(step_hidden)
            outputs.append(y_step)

            if (
                self.training
                and tf_targets is not None
                and teacher_forcing_ratio > 0.0
                and t < tf_targets.size(1)
            ):
                mask = (torch.rand(batch_size, 1, device=device) < teacher_forcing_ratio)
                tf_val = tf_targets[:, t:t + 1].to(device=device, dtype=dtype)
                next_step = torch.where(mask, tf_val, y_step)
            else:
                next_step = y_step

            dec_input = next_step.unsqueeze(1)

        return torch.cat(outputs, dim=1)


def build_optimizer(name: str, params, lr: float):
    name = name.lower()
    if name in ["adam", "adamw"]:
        return optim.AdamW(params, lr=lr, weight_decay=1e-3)
    if name == "rmsprop":
        return optim.RMSprop(params, lr=lr)
    return optim.Adam(params, lr=lr)

