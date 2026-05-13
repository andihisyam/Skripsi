import optuna
from optuna.pruners import MedianPruner
import torch


from sklearn.metrics import (
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)

from utils.model_utils_seq2seq import build_lstm_model
from utils.model_utils_seq2seq import build_optimizer
from utils.model_utils_seq2seq import RMSELoss

def train_one_fold_optuna(
    trial,
    Xtr,
    ytr,
    Xva,
    yva,
    device,
    input_shape,
    H_output=1,
    MAX_EPOCHS=80
):
    dense_units = trial.suggest_int("dense_units", 32, 256, step=32)
    hidden_units = trial.suggest_int("hidden_units", 32, 256, step=32)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "RMSprop", "SGD"])

    model = build_lstm_model(
        Xtr,
        dense_units=dense_units,
        hidden_units=hidden_units,
        dropout=dropout,
        output_dim=H_output
    ).to(device)

    optimizer = build_optimizer(optimizer_name, model.parameters(), lr)
    criterion = RMSELoss()

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32).to(device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32).to(device)
    Xva_t = torch.tensor(Xva, dtype=torch.float32).to(device)
    yva_t = torch.tensor(yva, dtype=torch.float32).to(device)

    n_train = Xtr_t.size(0)

    best_val_loss = float("inf")
    best_pred = None
    best_epoch = 0
    best_state = None

    # ✅ Tambahkan history
    train_losses = []
    val_losses = []

    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n_train)

        running_loss = 0.0
        n_seen = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            batch_X = Xtr_t[idx]
            batch_y = ytr_t[idx]

            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

            bs = batch_X.size(0)
            running_loss += loss.item() * bs
            n_seen += bs

        # rata-rata train loss per epoch
        train_epoch_loss = running_loss / max(n_seen, 1)

        model.eval()
        with torch.no_grad():
            pred_va = model(Xva_t)
            vloss = criterion(pred_va, yva_t).item()

        train_losses.append(train_epoch_loss)
        val_losses.append(vloss)

        if vloss < best_val_loss:
            best_val_loss = vloss
            best_pred = pred_va.detach().cpu().numpy()
            best_epoch = epoch
            best_state = model.state_dict()

        trial.report(best_val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    if best_pred is not None and len(best_pred.shape) != len(yva.shape):
        best_pred = best_pred.reshape(-1, 1)

    return {
        "loss": best_val_loss,
        "pred": best_pred,
        "epoch": best_epoch,
        "params": {
            "dense_units": dense_units,
            "hidden_units": hidden_units,
            "dropout": dropout,
            "lr": lr,
            "batch_size": batch_size,
            "optimizer": optimizer_name,
        },
        "state_dict": best_state,
        # ✅ return history
        "history": {
            "train_loss": train_losses,
            "val_loss": val_losses,
        }
    }


# ================================
# Wrapper untuk Optuna per fold
# ================================
def run_optuna_for_fold(
    input_shape,
    Xtr,
    ytr,
    Xva,
    yva,
    device,
    H_output,
    trials=30,
    MAX_EPOCHS=80
):
    best_result = {
        "loss": float("inf"),
        "params": None,
        "pred": None,
        "epoch": None,
        "state_dict": None,
        "history": None,
    }

    def objective(trial):
        result = train_one_fold_optuna(
            trial=trial,
            input_shape=input_shape,
            Xtr=Xtr,
            ytr=ytr,
            Xva=Xva,
            yva=yva,
            device=device,
            H_output=H_output,
            MAX_EPOCHS=MAX_EPOCHS
        )

        nonlocal best_result
        if result["loss"] < best_result["loss"]:
            best_result = result

        return result["loss"]

    study = optuna.create_study(direction="minimize", pruner=MedianPruner())
    study.optimize(objective, n_trials=trials)

    return best_result
