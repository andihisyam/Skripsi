import numpy as np, pandas as pd, tensorflow as tf
from tensorflow.keras import layers, models
from utils.preprocess_coba import load_csv, merge_sentiment, add_simple_returns, prepare_sequences, time_split_by_date
from config import DATE_COL, TARGET_COL, PRICE_FEATURES, SENT_FEATURES, N_STEPS, H_1M, TRAIN_END, VAL_END

# 1) Load
prices = load_csv("data/prices_lq45.csv")      # harus ada kolom Ticker
sent   = load_csv("data/sentiment_daily.csv")

# 2) Merge per Ticker, lalu kumpulkan X,y
X_tr_list, y_tr_list, X_va_list, y_va_list = [], [], [], []
scaler = None  # biarkan fit ulang per ticker; atau pakai return dan 1 scaler global (lebih aman)
tickers = prices["Ticker"].unique()

for t in tickers:
    p_i = prices[prices["Ticker"] == t].copy()
    df_i = merge_sentiment(p_i, sent, method="inner")
    df_i = df_i.dropna(subset=[TARGET_COL] + SENT_FEATURES)

    # (opsional) gunakan return sbg target agar lintas saham comparable
    df_i = df_i.assign(Ticker=t)
    df_i = add_simple_returns(df_i)

    # PILIH TARGET: Close atau Return (untuk skripsi simple: Close ok)
    target_name = TARGET_COL  # atau "Return"

    # split waktu
    tr, va, _ = time_split_by_date(df_i, train_end=TRAIN_END, val_end=VAL_END)

    # buat sequence untuk train & val (pakai scaler None agar fit per set → aman)
    try:
        Xtr, ytr, _, _ = prepare_sequences(tr, n_steps=N_STEPS, target_col_name=target_name, horizon=H_1M, scaler=None)
        Xva, yva, _, _ = prepare_sequences(va, n_steps=N_STEPS, target_col_name=target_name, horizon=H_1M, scaler=None)
        X_tr_list.append(Xtr); y_tr_list.append(ytr)
        X_va_list.append(Xva); y_va_list.append(yva)
    except Exception:
        # emiten dengan data terlalu pendek, lewati
        continue

X_tr = np.vstack(X_tr_list)
y_tr = np.vstack(y_tr_list)
X_va = np.vstack(X_va_list)
y_va = np.vstack(y_va_list)

print("Train shape:", X_tr.shape, y_tr.shape, "| Val:", X_va.shape, y_va.shape)

# 3) Bangun model LSTM multi-output (output = horizon)
model = models.Sequential([
    layers.Input(shape=(N_STEPS, X_tr.shape[-1])),
    layers.LSTM(128, return_sequences=False),
    layers.Dropout(0.2),
    layers.Dense(64, activation="relu"),
    layers.Dense(H_1M)  # multi-step
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])

# 4) Train
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
]
model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=30, batch_size=256, callbacks=callbacks)

# 5) Simpan
model.save("forecast_lq45_h1m.keras")
