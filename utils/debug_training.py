import numpy as np


def debug_trial(
    trial_num,
    fold,
    ytr,
    yva,
    best_pred_va,
    epoch_train_losses,
    epoch_val_losses,
    prefix="",
):
    """
    Cetak ringkasan diagnostik per trial per fold.
    Panggil setelah train_model_cv selesai.

    Checks:
      1. Model collapse   → std prediksi vs std target
      2. Loss movement    → apakah loss benar-benar turun
      3. Directional acc  → arah prediksi vs aktual
      4. Gap train/val    → seberapa besar overfitting
    """
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"{prefix}[DEBUG] Trial {trial_num} | Fold {fold}")
    print(sep)

    yva_flat   = np.asarray(yva).reshape(-1)
    pred_flat  = np.asarray(best_pred_va).reshape(-1)
    ytr_flat   = np.asarray(ytr).reshape(-1)

    # -------------------------------------------------------
    # 1. MODEL COLLAPSE CHECK
    # -------------------------------------------------------
    std_pred   = float(np.std(pred_flat))
    std_target = float(np.std(yva_flat))
    collapse_ratio = std_pred / (std_target + 1e-12)

    collapse_status = (
        "COLLAPSE - model hanya predict mean"  if collapse_ratio < 0.05 else
        "LEMAH - variasi prediksi sangat kecil" if collapse_ratio < 0.20 else
        "OK"
    )
    print(f"  [1] Model collapse check")
    print(f"      Std prediksi : {std_pred:.6f}")
    print(f"      Std target   : {std_target:.6f}")
    print(f"      Ratio        : {collapse_ratio:.4f}  →  {collapse_status}")

    # -------------------------------------------------------
    # 2. LOSS MOVEMENT CHECK
    # -------------------------------------------------------
    if epoch_train_losses and epoch_val_losses:
        tr_first  = float(epoch_train_losses[0])
        tr_last   = float(epoch_train_losses[-1])
        val_first = float(epoch_val_losses[0])
        val_best  = float(min(epoch_val_losses))

        tr_drop   = tr_first - tr_last
        val_drop  = val_first - val_best

        tr_status  = "OK - turun" if tr_drop  > 0.01 else "STUCK - tidak belajar"
        val_status = "OK - turun" if val_drop > 0.01 else "STUCK - tidak belajar"

        print(f"  [2] Loss movement check")
        print(f"      Train : {tr_first:.4f} → {tr_last:.4f}  (drop {tr_drop:+.4f})  →  {tr_status}")
        print(f"      Val   : {val_first:.4f} → {val_best:.4f}  (drop {val_drop:+.4f})  →  {val_status}")

    # -------------------------------------------------------
    # 3. DIRECTIONAL ACCURACY
    # -------------------------------------------------------
    da = float(np.mean(np.sign(yva_flat) == np.sign(pred_flat)))
    da_status = (
        "BAGUS"      if da >= 0.57 else
        "LUMAYAN"    if da >= 0.53 else
        "COIN FLIP"  if da >= 0.50 else
        "BURUK - lebih jelek dari random"
    )
    print(f"  [3] Directional accuracy : {da:.4f}  →  {da_status}")

    # -------------------------------------------------------
    # 4. TRAIN VS VAL GAP
    # -------------------------------------------------------
    if epoch_train_losses and epoch_val_losses:
        tr_min  = float(min(epoch_train_losses))
        val_min = float(min(epoch_val_losses))
        gap     = val_min - tr_min
        gap_pct = gap / (tr_min + 1e-12) * 100

        gap_status = (
            "OK - gap wajar untuk saham"  if gap_pct < 40  else
            "PERHATIAN - gap cukup besar" if gap_pct < 80  else
            "OVERFITTING - gap terlalu besar"
        )
        print(f"  [4] Train/Val gap")
        print(f"      Best train loss : {tr_min:.6f}")
        print(f"      Best val loss   : {val_min:.6f}")
        print(f"      Gap             : {gap:+.6f} ({gap_pct:+.1f}%)  →  {gap_status}")

    print(sep + "\n")


def debug_sequences(Xtr, ytr, Xva, yva, target_col_name, prefix=""):
    """
    Cetak ringkasan shape dan distribusi sequence sebelum training.
    Panggil setelah prepare_val_with_context.
    """
    sep = "-" * 70
    print(f"\n{sep}")
    print(f"{prefix}[DEBUG SEQUENCES] {target_col_name}")
    print(sep)

    print(f"  Xtr : {Xtr.shape}  |  ytr : {ytr.shape}")
    print(f"  Xva : {Xva.shape}  |  yva : {yva.shape}")

    ytr_flat = np.asarray(ytr).reshape(-1)
    yva_flat = np.asarray(yva).reshape(-1)

    print(f"  ytr — mean={ytr_flat.mean():.4f}  std={ytr_flat.std():.4f}  "
          f"min={ytr_flat.min():.4f}  max={ytr_flat.max():.4f}")
    print(f"  yva — mean={yva_flat.mean():.4f}  std={yva_flat.std():.4f}  "
          f"min={yva_flat.min():.4f}  max={yva_flat.max():.4f}")

    # Cek kalau target hampir semua nol (tanda double shift atau NaN tersisa)
    near_zero = float(np.mean(np.abs(ytr_flat) < 1e-6))
    if near_zero > 0.5:
        print(f"  [WARN] {near_zero:.0%} nilai target mendekati nol — "
              f"kemungkinan masih ada NaN atau double shift")
    else:
        print(f"  [OK] Distribusi target terlihat normal")

    print(sep + "\n")