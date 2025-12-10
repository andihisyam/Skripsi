# config.py
DATE_COL = "Date"
TEXT_SENT_COL = "Isi_sentimen" 
TARGET_COL = "Close"
PRICE_FEATURES = ["High", "Low", "Open", "Volume"]
SENT_FEATURES  = ["Positif", "Negatif", "Netral"]
TICKER_COL = "Ticker"

# forecasting
N_STEPS = 60            # panjang input window (±3 bulan bursa)
H_1M    = 22            # horizon 1 bulan
H_3M    = 66            # horizon 3 bulan

# split (pakai batas tanggal biar fair time-series)
TRAIN_END = "2022-12-31"
VAL_END   = "2024-12-31"

# split ratio fallback (untuk saham baru)
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.2


# technical indicators
MA_PERIODS  = [5, 10, 20, 30, 50]
EMA_PERIODS = [5, 12, 26, 30, 50]
RSI_PERIODS = [7, 14, 21]
MACD_CONFIGS = [(12, 26), (5, 20), (10, 30)]

FEATURE_COMBINATIONS = {
    # Single (selalu + lag dari Close)
    "Open_LAG": ["Open", "Close_lag1", "Close_lag2", "Close_lag3"],
    "High_LAG": ["High", "Close_lag1", "Close_lag2", "Close_lag3"],
    "Low_LAG": ["Low", "Close_lag1", "Close_lag2", "Close_lag3"],
    "Volume_LAG": ["Volume", "Close_lag1", "Close_lag2", "Close_lag3"],
    "Close_LAG": ["Close_lag1", "Close_lag2", "Close_lag3"],

    # Pairwise
    "Open_High_LAG": ["Open", "High", "Close_lag1", "Close_lag2", "Close_lag3"],
    "Open_Low_LAG": ["Open", "Low", "Close_lag1", "Close_lag2", "Close_lag3"],
    "Open_Volume_LAG": ["Open", "Volume", "Close_lag1", "Close_lag2", "Close_lag3"],
    "High_Low_LAG": ["High", "Low", "Close_lag1", "Close_lag2", "Close_lag3"],
    "High_Volume_LAG": ["High", "Volume", "Close_lag1", "Close_lag2", "Close_lag3"],
    "Low_Volume_LAG": ["Low", "Volume", "Close_lag1", "Close_lag2", "Close_lag3"],

    # Triplet
    "Open_High_Low_LAG": ["Open", "High", "Low", "Close_lag1", "Close_lag2", "Close_lag3"],
    "Open_High_Volume_LAG": ["Open", "High", "Volume", "Close_lag1", "Close_lag2", "Close_lag3"],
    "High_Low_Volume_LAG": ["High", "Low", "Volume", "Close_lag1", "Close_lag2", "Close_lag3"],

    # Full Base
    "OHLVC_LAG": ["Open", "High", "Low", "Volume", "Close_lag1", "Close_lag2", "Close_lag3"],

    # Full Base + MA
    "OHLVC_LAG_MA": [
        "Open", "High", "Low", "Volume",
        "Close_lag1", "Close_lag2", "Close_lag3",
        "MA5", "MA10", "MA20", "MA30", "MA50"
    ],

    # Full Base + EMA
    "OHLVC_LAG_EMA": [
        "Open", "High", "Low", "Volume",
        "Close_lag1", "Close_lag2", "Close_lag3",
        "EMA5", "EMA12", "EMA26", "EMA30", "EMA50"
    ],

    # Full Base + RSI
    "OHLVC_LAG_RSI": [
        "Open", "High", "Low", "Volume",
        "Close_lag1", "Close_lag2", "Close_lag3",
        "RSI7", "RSI14", "RSI21"
    ],

    # Full Base + MACD
    "OHLVC_LAG_MACD": [
        "Open", "High", "Low", "Volume",
        "Close_lag1", "Close_lag2", "Close_lag3",
        "MACD_5_20", "MACD_10_30", "MACD_12_26"
    ],

    # Campuran indikator
    "OHLVC_LAG_MA_EMA": [
        "Open", "High", "Low", "Volume",
        "Close_lag1", "Close_lag2", "Close_lag3",
        "MA5", "MA10", "MA20", "MA30", "MA50",
        "EMA5", "EMA12", "EMA26", "EMA30", "EMA50"
    ],
    "OHLVC_LAG_MA_RSI": [
        "Open", "High", "Low", "Volume",
        "Close_lag1", "Close_lag2", "Close_lag3",
        "MA5", "MA10", "MA20", "MA30", "MA50",
        "RSI7", "RSI14", "RSI21"
    ],
    "OHLVC_LAG_MA_MACD": [
        "Open", "High", "Low", "Volume",
        "Close_lag1", "Close_lag2", "Close_lag3",
        "MA5", "MA10", "MA20", "MA30", "MA50",
        "MACD_5_20", "MACD_10_30", "MACD_12_26"
    ],
    "OHLVC_LAG_EMA_RSI": [
        "Open", "High", "Low", "Volume",
        "Close_lag1", "Close_lag2", "Close_lag3",
        "EMA5", "EMA12", "EMA26", "EMA30", "EMA50",
        "RSI7", "RSI14", "RSI21"
    ],
    "OHLVC_LAG_EMA_MACD": [
        "Open", "High", "Low", "Volume",
        "Close_lag1", "Close_lag2", "Close_lag3",
        "EMA5", "EMA12", "EMA26", "EMA30", "EMA50",
        "MACD_5_20", "MACD_10_30", "MACD_12_26"
    ],
    "OHLVC_LAG_RSI_MACD": [
        "Open", "High", "Low", "Volume",
        "Close_lag1", "Close_lag2", "Close_lag3",
        "RSI7", "RSI14", "RSI21",
        "MACD_5_20", "MACD_10_30", "MACD_12_26"
    ],

    # Kombinasi 3 indikator
    "OHLVC_LAG_MA_EMA_RSI": [
        "Open", "High", "Low", "Volume",
        "Close_lag1", "Close_lag2", "Close_lag3",
        "MA5", "MA10", "MA20", "MA30", "MA50",
        "EMA5", "EMA12", "EMA26", "EMA30", "EMA50",
        "RSI7", "RSI14", "RSI21"
    ],
    "OHLVC_LAG_MA_RSI_MACD": [
        "Open", "High", "Low", "Volume",
        "Close_lag1", "Close_lag2", "Close_lag3",
        "MA5", "MA10", "MA20", "MA30", "MA50",
        "RSI7", "RSI14", "RSI21",
        "MACD_5_20", "MACD_10_30", "MACD_12_26"
    ],
    "OHLVC_LAG_EMA_RSI_MACD": [
        "Open", "High", "Low", "Volume",
        "Close_lag1", "Close_lag2", "Close_lag3",
        "EMA5", "EMA12", "EMA26", "EMA30", "EMA50",
        "RSI7", "RSI14", "RSI21",
        "MACD_5_20", "MACD_10_30", "MACD_12_26"
    ],

    # Kombinasi semua indikator
    "OHLVC_LAG_MA_EMA_RSI_MACD": [
        "Open", "High", "Low", "Volume",
        "Close_lag1", "Close_lag2", "Close_lag3",
        "MA5", "MA10", "MA20", "MA30", "MA50",
        "EMA5", "EMA12", "EMA26", "EMA30", "EMA50",
        "RSI7", "RSI14", "RSI21",
        "MACD_5_20", "MACD_10_30", "MACD_12_26"
    ]
}