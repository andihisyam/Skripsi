# mpc.py (versi final lengkap)
import os
import glob
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt

# ===============================
# KONFIGURASI
# ===============================
PREDICTION_DIR = "predictions"
ACTUAL_DIR = "D:/Kuliah/Skripsi/Data/Saham_Horizon_22"
TOTAL_CAPITAL = 100_000_000
MIN_WEIGHT = 0.0
MAX_WEIGHT = 1.0
Q_SCALE = 0.5
R_SCALE = 0.1
RISK_FREE_RATE = 0.0001  # 0.01% per hari

# ===============================
# UTILITAS
# ===============================
def load_latest_returns(pred_dir):
    """Ambil return prediksi dari hasil Kalman (Smoothed_Close)."""
    files = glob.glob(os.path.join(pred_dir, "*_prediction.csv"))
    tickers, returns = [], []

    for f in files:
        ticker = os.path.basename(f).split("_")[0]
        df = pd.read_csv(f)

        if "Smoothed_Close" not in df.columns:
            print(f"⚠️  Lewati {ticker} (kolom 'Smoothed_Close' tidak ditemukan)")
            continue

        df["Return"] = df["Smoothed_Close"].pct_change()
        ret = df["Return"].iloc[-1]
        if pd.isna(ret):
            print(f"⚠️  Return tidak valid untuk {ticker}")
            continue

        tickers.append(ticker)
        returns.append(ret)

    if len(returns) == 0:
        raise ValueError("❌ Tidak ada data return valid ditemukan.")
    return np.array(returns).reshape(-1, 1), tickers


def run_mpc(r_pred, tickers, lambda_return=1.0):
    """Optimasi bobot portofolio dengan preferensi ke return tinggi."""
    n = len(tickers)
    A = B = np.eye(n)
    Q = np.eye(n) * Q_SCALE
    R = np.eye(n) * R_SCALE
    xk = np.zeros((n, 1))

    H = B.T @ Q @ B + R
    f = B.T @ Q @ (A @ xk - r_pred)
    u = cp.Variable((n, 1))

    # 🔥 Fungsi objektif baru: penalti stabilitas + preferensi return
    objective = cp.Minimize(cp.quad_form(u, H) + 2 * f.T @ u - lambda_return * (r_pred.T @ u))

    constraints = [
        cp.sum(u) == 1,
        u >= MIN_WEIGHT,
        u <= MAX_WEIGHT
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"❌ MPC gagal menemukan solusi optimal: {prob.status}")

    weights = np.maximum(u.value.flatten(), 0)
    weights /= np.sum(weights)
    return weights



def load_actual_prices(ticker):
    """Ambil harga aktual dan hitung return harian & kumulatif."""
    path = os.path.join(ACTUAL_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        print(f"⚠️ Data aktual {ticker} tidak ditemukan.")
        return None

    df = pd.read_csv(path)
    price_col = "Price" if "Price" in df.columns else df.columns[1]
    df[price_col] = df[price_col].astype(str).str.replace(",", "").astype(float)
    df=df.sort_values("Date")
    df["Return_Daily"] = df[price_col].pct_change()
    df["Cumulative_Return"] = (df[price_col] / df[price_col].iloc[0]) - 1
    return df[["Date", price_col, "Return_Daily", "Cumulative_Return"]].dropna()


# ===============================
# MAIN PROGRAM
# ===============================
def main():
    print("📊 Memuat return hasil prediksi Kalman dari semua saham...")
    r_pred, tickers = load_latest_returns(PREDICTION_DIR)
    print(f"✅ Ditemukan {len(tickers)} saham dengan return valid.")

    print("\n🚀 Menjalankan MPC Optimization...")
    weights = run_mpc(r_pred, tickers)

    # EXPECTED PORTFOLIO RETURN
    expected_port_return = np.sum(weights * r_pred.flatten())
    expected_profit = expected_port_return * TOTAL_CAPITAL

    print(f"\n🎯 Expected Portfolio Return (Prediksi): {expected_port_return:.4%}")
    print(f"💰 Expected Profit (Prediksi): Rp{expected_profit:,.0f}")

    # ==================== LOAD DATA AKTUAL ====================
    all_prices, realized_returns = [], []
    for t in tickers:
        df = load_actual_prices(t)
        if df is None or df.empty:
            realized_returns.append(np.nan)
            continue
        df = df.rename(columns={"Cumulative_Return": f"{t}_Cum", "Return_Daily": f"{t}_Daily"})
        all_prices.append(df.set_index("Date")[[f"{t}_Cum", f"{t}_Daily"]])
        realized_returns.append(df[f"{t}_Cum"].iloc[-1])

    realized_returns = np.array(realized_returns)

    # ==================== REALIZED PORTFOLIO ====================
    df_merge = pd.concat(all_prices, axis=1).fillna(method="ffill").dropna()
    daily_returns = df_merge[[c for c in df_merge.columns if "_Daily" in c]].values
    cumulative_returns = df_merge[[c for c in df_merge.columns if "_Cum" in c]].values

    weighted_daily = np.sum(daily_returns * weights, axis=1)
    weighted_cumulative = np.sum(cumulative_returns * weights, axis=1)

    df_port = pd.DataFrame({
        "Date": df_merge.index,
        "Portfolio_Daily_Return": weighted_daily,
        "Portfolio_Cumulative_Return": weighted_cumulative,
        "Portfolio_Value": TOTAL_CAPITAL * (1 + weighted_cumulative)
    })

    # ==================== METRIK RISIKO ====================
    vol = np.std(weighted_daily)
    mean_daily_return = np.mean(weighted_daily)
    sharpe = (mean_daily_return - RISK_FREE_RATE) / vol if vol > 0 else np.nan
    realized_return = weighted_cumulative[-1]
    realized_profit = realized_return * TOTAL_CAPITAL

    # ==================== RINGKASAN ====================
    print("\n💹 Realized Portfolio Return (Aktual): {:.4%}".format(realized_return))
    print("💸 Realized Profit (Aktual): Rp{:,.0f}".format(realized_profit))
    print("⚙️ Portfolio Volatility: {:.4%}".format(vol))
    print("📊 Average Daily Return: {:.4%}".format(mean_daily_return))
    print("💎 Sharpe Ratio: {:.3f}".format(sharpe))

    # ==================== PER SAHAM: VISUALISASI HARGA PREDIKSI VS AKTUAL ====================
    print("\n📈 Membuat line plot harga prediksi vs aktual per saham (22 hari)...")

    for t in tickers:
        pred_path = os.path.join(PREDICTION_DIR, f"{t}_prediction.csv")
        actual_path = os.path.join(ACTUAL_DIR, f"{t}.csv")

        if not os.path.exists(pred_path):
            print(f"⚠️  Lewati {t} (file prediksi tidak ditemukan)")
            continue
        if not os.path.exists(actual_path):
            print(f"⚠️  Lewati {t} (data aktual tidak ditemukan)")
            continue

        # --- Baca data ---
        df_pred = pd.read_csv(pred_path)
        df_actual = pd.read_csv(actual_path)

        # --- Normalisasi format tanggal ---
        # Prediksi biasanya format MM/DD/YYYY
        df_pred["Date"] = pd.to_datetime(
            df_pred["Date"], errors="coerce", dayfirst=False
        )
        # Aktual biasanya format DD/MM/YYYY (data Indonesia)
        df_actual["Date"] = pd.to_datetime(
            df_actual["Date"], format="%d/%m/%Y", errors="coerce"
        )

        # --- Bersihkan NaT dan urutkan naik (3 Maret → 10 April) ---
        df_pred = df_pred.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        df_actual = df_actual.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

        # --- Ambil 22 hari terakhir dari keduanya ---
        df_pred_tail = df_pred.tail(22)
        df_actual_tail = df_actual.tail(22)

        # --- Cek range tanggal untuk debugging ---
        print(f"\n{t} - Prediksi range:", df_pred_tail["Date"].min().date(), "→", df_pred_tail["Date"].max().date())
        print(f"{t} - Aktual range  :", df_actual_tail["Date"].min().date(), "→", df_actual_tail["Date"].max().date())

        # --- Pastikan tidak kosong ---
        if df_pred_tail.empty or df_actual_tail.empty:
            print(f"⚠️  Lewati {t} (data kosong setelah preprocessing)")
            continue

        # --- Gabung hanya tanggal yang sama persis ---
        df_merge = pd.merge(df_pred_tail, df_actual_tail, on="Date", how="inner")

        if df_merge.empty:
            print(f"⚠️  Lewati {t} (tidak ada tanggal yang cocok setelah merge)")
            continue

        # --- Tentukan kolom harga aktual (bisa 'Price' atau kolom ke-2) ---
        price_col = "Price" if "Price" in df_actual_tail.columns else df_actual_tail.columns[1]

        # --- Plot garis ---
        plt.figure(figsize=(8, 5))
        plt.plot(df_merge["Date"], df_merge["Smoothed_Close"], label="Prediksi (Smoothed)", color="blue", linestyle="--")
        plt.plot(df_merge["Date"], df_merge[price_col], label="Aktual", color="green", linewidth=2)
        plt.title(f"📈 {t} - Harga Prediksi vs Aktual (22 Hari)")
        plt.xlabel("Tanggal")
        plt.ylabel("Harga (Rp)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()

        # --- Simpan plot ---
        os.makedirs(os.path.join(PREDICTION_DIR, "plots"), exist_ok=True)
        out_plot = os.path.join(PREDICTION_DIR, "plots", f"{t}_pred_vs_actual.png")
        plt.savefig(out_plot)
        plt.close()

        print(f"✅ Plot {t} disimpan ke {out_plot}")


    # ==================== PER SAHAM ====================
    df_allocation = pd.DataFrame({
        "Ticker": tickers,
        "Weight": weights,
        "Allocation_Rp": weights * TOTAL_CAPITAL,
        "Predicted_Return": r_pred.flatten(),
        "Realized_Return": realized_returns,
        "Expected_Profit_Rp": weights * TOTAL_CAPITAL * r_pred.flatten(),
        "Realized_Profit_Rp": weights * TOTAL_CAPITAL * realized_returns
    }).sort_values("Weight", ascending=False)

    print("\n📈 Alokasi Dana per Saham:")
    print(df_allocation.head(len(tickers)))

    # ==================== VISUALISASI ====================

    # Perbandingan expected vs realized return
    plt.figure(figsize=(8, 5))
    plt.bar(["Expected", "Realized"], [expected_port_return * 100, realized_return * 100], color=["blue", "green"])
    plt.ylabel("Return (%)")
    plt.title("📊 Perbandingan Expected vs Realized Portfolio Return")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    # ==================== PERBANDINGAN NILAI PORTOFOLIO EXPECTED vs REALIZED ====================
    days = np.arange(len(df_port))
    expected_curve = TOTAL_CAPITAL * (1 + expected_port_return) ** (days / len(days))
    realized_curve = df_port["Portfolio_Value"].values

    plt.figure(figsize=(10, 5))
    plt.plot(df_port["Date"], expected_curve, label="Expected Portfolio Value (Prediksi)", linestyle="--", color="blue")
    plt.plot(df_port["Date"], realized_curve, label="Realized Portfolio Value (Aktual)", linewidth=2, color="green")
    plt.title("📊 Perbandingan Nilai Portofolio: Expected vs Realized")
    plt.xlabel("Tanggal")
    plt.ylabel("Nilai Portofolio (Rp)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ==================== SIMPAN HASIL ====================
    out_summary = os.path.join(PREDICTION_DIR, "mpc_portfolio_summary.csv")
    out_allocation = os.path.join(PREDICTION_DIR, "mpc_portfolio_allocation.csv")

    summary = {
        "Expected_Return": expected_port_return,
        "Expected_Profit": expected_profit,
        "Realized_Return": realized_return,
        "Realized_Profit": realized_profit,
        "Volatility": vol,
        "Sharpe_Ratio": sharpe
    }

    pd.DataFrame([summary]).to_csv(out_summary, index=False)
    df_allocation.to_csv(out_allocation, index=False)

    print(f"\n📝 Ringkasan metrik disimpan di: {out_summary}")
    print(f"💾 Detail alokasi & profit per saham disimpan di: {out_allocation}")


if __name__ == "__main__":
    main()
