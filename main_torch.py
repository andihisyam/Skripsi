# main.py
import argparse
import os
import logging
import logging.handlers
from utils import train_no_sentimen, eda_tes_no_sentimen, predict_no_sentimen_torch,train_multi_no_sentime_torch
from utils.predict_multi_no_sentimen import predict_next_multi
from utils.predict_no_sentimen_torch import predict_next
from config import DATE_COL, TARGET_COL, TICKER_COL, FEATURE_COMBINATIONS

# ======================================================
# 1️⃣ WARNA LOGGING UNTUK TERMINAL
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
        logging.DEBUG: Color.GREEN
    }

    def format(self, record):
        color = self.COLORS.get(record.levelno, Color.RESET)
        message = super().format(record)
        return f"{color}{message}{Color.RESET}"


# ======================================================
# 2️⃣ SETUP LOGGING (CONSOLE + ROTATING FILE)
# ======================================================
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Formatter standar (tanpa warna)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 🎨 Console handler (dengan warna)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColorFormatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        "%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(console_handler)

    # 📝 Rotating file handler (max 5 MB, keep 5 files)
    file_handler = logging.handlers.RotatingFileHandler(
        "app.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Logging system initialized 🌐")


setup_logging()
logger = logging.getLogger(__name__)


# ======================================================
# 3️⃣ MAIN PROGRAM
# ======================================================
def main():
    parser = argparse.ArgumentParser(
        description="Main runner untuk EDA, Training LSTM PyTorch, dan Prediction"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # ======================================================
    # EDA
    # ======================================================
    p_eda = sub.add_parser("eda", help="Jalankan Exploratory Data Analysis")
    p_eda.add_argument("--prices_folder", default="../Data/Saham")
    p_eda.add_argument("--outdir", default="eda_output")

    # ======================================================
    # TRAIN
    # ======================================================
    p_te = sub.add_parser("train-each", help="Latih model LSTM PyTorch per emiten")
    p_te.add_argument("--data_dir", default="../Data/Saham")
    p_te.add_argument("--out_dir", default="models")
    p_te.add_argument("--tickers", nargs="+")
    p_te.add_argument("--subset", nargs="+", help="Subset fitur untuk training")

    # ======================================================
    # MODE: TRAIN MULTI STOCK
    # ======================================================
    p_tm = sub.add_parser("train-multi", help="Latih model LSTM untuk multi-emiten")
    p_tm.add_argument("--data_dir", default="../Data/Saham", help="Folder CSV")
    p_tm.add_argument("--out_dir", default="models_multi", help="Folder output model")
    p_tm.add_argument("--tickers", nargs="+", required=True, help="Daftar emiten (min 2)")
    p_tm.add_argument("--subset", nargs="+", help="Subset fitur yang digunakan")

    # ======================================================
    # PREDICT MULTI STOCK
    # ======================================================
    p_pm = sub.add_parser("predict-multi", help="Prediksi harga untuk model multi-emiten")
    p_pm.add_argument("--tickers", nargs="+", required=True)
    p_pm.add_argument("--data_dir", default="../Data/Saham")
    p_pm.add_argument("--model", required=True)
    p_pm.add_argument("--scaler", required=True)
    p_pm.add_argument("--features", nargs="+", required=True)
    p_pm.add_argument("--out_dir", default="predictions_multi")

    # ======================================================
    # PREDICT
    # ======================================================
    p_pr = sub.add_parser("predict", help="Prediksi harga 1 emiten")
    p_pr.add_argument("--ticker")
    p_pr.add_argument("--csv")
    p_pr.add_argument("--data_dir", default="../Data/Saham")
    p_pr.add_argument("--model", required=True)
    p_pr.add_argument("--features", nargs="+", required=True)
    p_pr.add_argument("--out_dir", default="predictions")

    args = parser.parse_args()

    logger.info(f"Mode: {args.mode}")

    # ======================================================
    # 4️⃣ EKSEKUSI
    # ======================================================
    if args.mode == "eda":
        logger.info("🚀 Menjalankan EDA...")
        eda_tes_no_sentimen.run_eda(
            prices_input=args.prices_folder,
            outdir=args.outdir,
            date_col=DATE_COL,
            ticker_col=TICKER_COL,
            target_col=TARGET_COL,
        )
        logger.info("✨ EDA selesai!")

    elif args.mode == "train-each":
        logger.info("🚀 Mulai training model LSTM PyTorch...")
        train_no_sentimen.train_each(
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            tickers_filter=args.tickers,
            subset_filter=args.subset,
        )
        logger.info("✨ Training selesai dan model berhasil disimpan!")
    
    elif args.mode == "train-multi":
        logger.info("🚀 Mulai training multi-emiten ...")
        train_multi_no_sentime_torch.train_multi(
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            tickers=args.tickers,
            subset_features=args.subset
        )
        logger.info("✨ Training multi-emiten selesai!")
    
    elif args.mode == "predict-multi":
        logger.info("🚀 Mode: Prediksi Multi-Emiten")
        from config import FEATURE_COMBINATIONS
        # validasi subset fitur
        if args.features[0] not in FEATURE_COMBINATIONS:
            raise SystemExit(
                f"❌ Subset fitur '{args.features[0]}' tidak valid!\n"
                f"Daftar subset valid: {list(FEATURE_COMBINATIONS.keys())}"
            )
        feature_subset = FEATURE_COMBINATIONS[args.features[0]]
        predict_next_multi(
            tickers=args.tickers,
            data_dir=args.data_dir,
            model_path=args.model,
            scaler_path=args.scaler,
            feature_subset=feature_subset,
            out_dir=args.out_dir
        )

    elif args.mode == "predict":
        logger.info("🚀 Mode: Prediksi harga saham (PyTorch, No-Sentiment)")

    # -------------------------
    # 1️⃣ Tentukan CSV input
    # -------------------------
    if args.tickers:
        csv_path = os.path.join(args.data_dir, f"{args.tickers}.csv")
        logger.info(f"📂 Menggunakan CSV otomatis: {csv_path}")
    else:
        csv_path = args.csv
        logger.info(f"📂 Menggunakan CSV manual: {csv_path}")

    if not csv_path or not os.path.exists(csv_path):
        raise SystemExit(f"❌ CSV tidak ditemukan: {csv_path}")


    from config import FEATURE_COMBINATIONS

    if not args.features:
        raise SystemExit(
            "❌ Kamu harus memilih subset fitur!\n"
            "Gunakan: --features <nama_subset>\n"
            f"Daftar subset tersedia: {list(FEATURE_COMBINATIONS.keys())}"
        )

    if isinstance(args.features, list):
        # CLI biasanya menerima list, contoh: ["Low_LAG"]
        feature_name = args.features[0]
    else:
        feature_name = args.features

    if feature_name not in FEATURE_COMBINATIONS:
        raise SystemExit(
            f"❌ Subset fitur '{feature_name}' tidak valid!\n"
            f"Daftar subset valid: {list(FEATURE_COMBINATIONS.keys())}"
        )

    feature_subset = FEATURE_COMBINATIONS[feature_name]
    logger.info(f"🔎 Subset fitur dipilih: {feature_name} → {feature_subset}")

    model_path = args.model
    if not os.path.exists(model_path):
        raise SystemExit(f"❌ File model tidak ditemukan: {model_path}")

    logger.info(f"📦 Model yang digunakan: {os.path.basename(model_path)}")

    try:
        logger.info("🔮 Menjalankan prediksi...")
        result_df = predict_next(
            prices_path=csv_path,
            model_path=model_path,
            feature_subset=feature_subset,
            out_dir=args.out_dir,
        )
    except Exception as e:
        logger.error(f"❌ Terjadi error saat prediksi: {e}")
        raise SystemExit(str(e))

    logger.info("✨ Prediksi selesai! CSV & grafik berhasil disimpan.")


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    logger.info("Folder 'models' dipastikan tersedia.")
    main()
