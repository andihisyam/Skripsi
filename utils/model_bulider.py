from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def build_lstm(n_steps, n_features):
    model = Sequential()
    model.add(LSTM(50, activation="relu", input_shape=(n_steps, n_features)))
    model.add(Dense(n_steps))
    model.compile(optimizer="adam", loss="mse")
    return model
