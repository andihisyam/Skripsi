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
