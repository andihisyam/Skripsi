import argparse
import os
import logging
import logging.handlers

from utils import train_optuna_sentimen
from utils.predict_sentimen_torch import predict_forward_sentiment


# ======================================================
# 1. WARNA LOGGING UNTUK TERMINAL
# ======================================================
class Color:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.INFO: Color.BLUE,
        logging.WARNING: Color.YELLOW,
        logging.ERROR: Color.RED,
        logging.CRITICAL: Color.RED + Color.BOLD,
        logging.DEBUG: Color.GREEN,
    }

    def format(self, record):
        color = self.COLORS.get(record.levelno, Color.RESET)
        message = super().format(record)
        return f"{color}{message}{Color.RESET}"


# ======================================================
# 2. SETUP LOGGING
# ======================================================
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColorFormatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        "%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(console_handler)

    file_handler = logging.handlers.RotatingFileHandler(
        "app.log",
        maxBytes=5_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Logging system initialized")


setup_logging()
logger = logging.getLogger(__name__)


# ======================================================
# 3. MAIN PROGRAM (SENTIMENT ONLY)
# ======================================================
def main():
    parser = argparse.ArgumentParser(
        description="Main runner untuk training dan prediksi LSTM PyTorch berbasis sentimen"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_train = sub.add_parser("train-each", help="Latih model LSTM sentimen per emiten")
    p_train.add_argument("--data_dir", default="../Data/Saham")
    p_train.add_argument("--out_dir", default="models")
    p_train.add_argument("--tickers", nargs="+")
    p_train.add_argument("--subset", nargs="+", help="Subset fitur untuk training")
    p_train.add_argument(
        "--model_backend",
        choices=["seq2seq", "lstm_sklearn"],
        default="seq2seq",
        help="Backend model: seq2seq (encoder-decoder) atau lstm_sklearn (direct multi-output).",
    )
    p_train.add_argument(
        "--sentiment_path",
        default="../Data/Sentimen/daily_sentiment_final.csv",
        help="Path file sentimen (CSV/XLSX)",
    )

    p_predict = sub.add_parser(
        "predict-forward",
        help="Prediksi forward berbasis ranking DA+IC dan hasilkan Confidence_Score untuk MPC",
    )
    p_predict.add_argument(
        "--summary_file",
        default="FIX_MODELS/ALL_training_summary.xlsx",
        help="Path ke ALL_training_results.xlsx atau ALL_training_summary.csv",
    )
    p_predict.add_argument("--data_dir", default="../Data/Saham")
    p_predict.add_argument(
        "--sentiment_path",
        default="../Data/Sentimen/daily_sentiment_final.csv",
        help="Path file sentimen (CSV/XLSX)",
    )
    p_predict.add_argument(
        "--horizon_base",
        default="../Data",
        help="Folder base data aktual, berisi Saham_Horizon_22, Saham_Horizon_44, dst.",
    )
    p_predict.add_argument("--out_dir", default="predictions_forward")
    p_predict.add_argument("--horizons", nargs="+", type=int, default=[22, 44])
    p_predict.add_argument("--tickers", nargs="+")
    p_predict.add_argument("--subset", nargs="+", help="Subset fitur untuk prediksi (nama Fitur di summary)")
    p_predict.add_argument(
        "--model_backend",
        choices=["auto", "seq2seq", "lstm_sklearn"],
        default="auto",
        help="Override backend saat prediksi. auto = pakai kolom ModelBackend dari summary.",
    )

    args = parser.parse_args()
    logger.info(f"Mode: {args.mode}")

    if args.mode == "train-each":
        logger.info(f"Sentiment mode aktif | file: {args.sentiment_path} | backend: {args.model_backend}")
        train_optuna_sentimen.train_each(
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            tickers_filter=args.tickers,
            subset_filter=args.subset,
            sentimen_path=args.sentiment_path,
            model_backend=args.model_backend,
        )
        logger.info("Training selesai dan model berhasil disimpan!")
    elif args.mode == "predict-forward":
        logger.info("Menjalankan prediksi forward sentimen (metric tetap: DA_IC)...")
        predict_forward_sentiment(
            summary_file=args.summary_file,
            data_dir=args.data_dir,
            sentimen_path=args.sentiment_path,
            out_dir=args.out_dir,
            horizons=args.horizons,
            horizon_data_base_dir=args.horizon_base,
            tickers_filter=args.tickers,
            subset_filter=args.subset,
            metric="DA_IC",
            model_backend_override=args.model_backend,
        )
        logger.info("Prediksi selesai.")


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    logger.info("Folder 'models' dipastikan tersedia.")
    main()
