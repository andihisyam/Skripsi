# main.py
import argparse
import os
from utils import train_no_sentimen, eda_tes_no_sentimen,predict_no_sentimen
from config import DATE_COL,TARGET_COL,TICKER_COL

def main():
    parser = argparse.ArgumentParser(description="Main runner untuk EDA dan training per emiten")
    sub = parser.add_subparsers(dest="mode", required=True)

    # --- EDA ---
    p_eda = sub.add_parser("eda", help="Jalankan EDA")
    p_eda.add_argument("--prices_folder", default="../Data/Saham", help="Folder *.csv harga per emiten")
    p_eda.add_argument("--outdir", default="eda_output", help="Folder output EDA")

    # --- Train per emiten ---
    p_te = sub.add_parser("train-each", help="Latih model per emiten (satu file per ticker)")
    p_te.add_argument("--data_dir", default="../Data/Saham")
    p_te.add_argument("--out_dir", default="models", help="Folder simpan model")
    p_te.add_argument("--tickers", nargs="+", help="Daftar ticker spesifik untuk dilatih (pisahkan dengan spasi)")
    p_te.add_argument("--subset", nargs="+", help="Nama subset fitur spesifik yang ingin dilatih (misal: OHLVC_MA_EMA OHLVC_RSI)")

    # --- Predict ---
    p_pr = sub.add_parser("predict", help="Prediksi harga untuk 1 emiten")
    p_pr.add_argument("--ticker", help="Ticker emiten, contoh: ACES")
    p_pr.add_argument("--csv", help="Path CSV data emiten (jika tidak pakai --ticker)")
    p_pr.add_argument("--data_dir", default="../Data/Saham", help="Folder data saham (untuk --ticker)")
    p_pr.add_argument("--model", required=True, help="Path model .keras yang dilatih")
    p_pr.add_argument("--features", nargs="+", required=True, help="Daftar fitur yang dipakai model")
    p_pr.add_argument("--out_dir", default="predictions", help="Folder simpan hasil prediksi")


    args = parser.parse_args()

    if args.mode == "eda":
        eda_tes_no_sentimen.run_eda(
            prices_input=args.prices_folder,   # ✅ pake nama yg sesuai
            outdir=args.outdir,
            date_col=DATE_COL,       # atau DATE_COL dari config.py
            ticker_col=TICKER_COL,   # atau TICKER_COL dari config.py
            target_col=TARGET_COL     # atau TARGET_COL dari config.py
        )
    elif args.mode == "train-each":
        train_no_sentimen.train_each(
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            tickers_filter=args.tickers,
            subset_filter=args.subset
        )
    elif args.mode == "predict":
        # kalau --ticker dipakai, cari file csv otomatis
        if args.ticker:
            csv_path = os.path.join(args.data_dir, f"{args.ticker}.csv")
        else:
            csv_path = args.csv
        
        predict_no_sentimen.predict_next(
            prices_path=csv_path,
            model_path=args.model,
            feature_subset=args.features,
            out_dir=args.out_dir
        )
    else:
        raise SystemExit("Mode tidak dikenali")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    main()
