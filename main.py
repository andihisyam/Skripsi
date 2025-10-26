import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.data_loader import load_data
from utils.preprocessing import prepare_sequences
from utils.model_bulider import build_lstm
from utils.trainer import train_model
from utils.metrics import evaluate

# ------------------------
# KONFIGURASI
# ------------------------
DATA_PATH = r"D:\Kuliah\Skripsi\Data\Saham\aces.csv"
N_STEPS = 10
EPOCHS = 30
BATCH_SIZE = 32

# Kombinasi fitur
input_combinations = {
    "Open": ["Open"],
    "Open_High": ["Open", "High"],
    "Open_High_Low": ["Open", "High", "Low"],
    "Open_High_Low_Volume": ["Open", "High", "Low", "Volume"],
}

# ------------------------
# 1. LOAD DATA
# ------------------------
df = load_data(DATA_PATH)
print("Data loaded:", df.shape)

# ------------------------
# 2. LOOP TIAP KOMBINASI FITUR
# ------------------------
for comb_name, features in input_combinations.items():
    print(f"\n=== Kombinasi: {comb_name} ({features}) ===")

    # ambil kolom sesuai kombinasi
    data = df[features].values
    if len(data) < 2 * N_STEPS:
        print("⚠️ Data terlalu pendek untuk kombinasi ini, dilewati.")
        continue

    # preprocessing
    X, y, scaler = prepare_sequences(data, N_STEPS)

    # build model
    model = build_lstm(N_STEPS, len(features))

    # training
    model, history, (X_test, y_test) = train_model(
        model, X, y, epochs=EPOCHS, batch_size=BATCH_SIZE
    )

    # prediksi
    y_pred = model.predict(X_test)

    # balikkan ke skala asli (pakai kolom pertama dari kombinasi sebagai target)
    min_val, max_val = scaler.data_min_[0], scaler.data_max_[0]
    y_test_orig = y_test * (max_val - min_val) + min_val
    y_pred_orig = y_pred * (max_val - min_val) + min_val

    # evaluasi
    rmse, r2 = evaluate(y_test_orig, y_pred_orig)
    print(f"RMSE: {rmse:.4f}, R2: {r2:.4f}")

    # plot hasil
    plt.figure(figsize=(12,6))
    plt.plot(y_test_orig.flatten(), label="Actual")
    plt.plot(y_pred_orig.flatten(), label="Predicted")
    plt.title(f"Prediksi Harga Saham ACES ({comb_name})")
    plt.xlabel("Time Step")
    plt.ylabel("Harga")
    plt.legend()
    plt.show()
