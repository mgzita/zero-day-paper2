"""
Harmonize the IEC 60870-5-104 Intrusion Detection Dataset into this
project's 18-feature, IQR-scaled pipeline format.
=========================================================================
Maps CICFlowMeter's native column names to the project's snake_case
FEATURE_NAMES, applies the SAME IQR scaler (median/iqr, both shape 18)
used for CIC-IDS2017/CSE-CIC-IDS2018/UNSW-NB15, and saves the result in
the same format as test_unsw_dedup.npz (keys "X", "y"), so it drops
directly into the existing H1-H4/LODO-3 pipeline as a new evaluation
target.

Label convention: "NORMAL" (case-insensitive) -> y=0 (benign); every
other label (the 12 attack types) -> y=1 (attack).

Combines both the train_final and test_final CSVs into a single pool,
since this dataset is intended to be used as a full evaluation TARGET
(like UNSW-NB15), not as a training environment with its own held-out
split.

IMPORTANT: verify the printed scaler-key assumption (median, iqr) and
sanity-check output ranges before trusting this file downstream -- see
the printed diagnostics at the end of this script.

Run from project root:
    python preprocessing\\harmonize_iec104.py
"""

import os
import numpy as np
import pandas as pd

ARTEFACTS_V2_DIR = r"C:\MLProject\zero_day_project\artefacts_v2"
DEDUP_DIR = os.path.join(ARTEFACTS_V2_DIR, "dedup")

# TODO: confirm these paths match where you actually placed the IEC-104 CSVs
IEC104_DIR = r"C:\MLProject\zero_day_project\data\raw\IEC 60870-5-104 IDS\Balanced_IEC104_Train_Test_CSV_Files\iec104_train_test_csvs\tests_cic_120"
TRAIN_CSV = os.path.join(IEC104_DIR, "iec104_CICFLOWMETER_train_final.csv")
TEST_CSV = os.path.join(IEC104_DIR, "iec104_CICFLOWMETER_test_final.csv")

OUTPUT_PATH = os.path.join(DEDUP_DIR, "test_iec104_dedup.npz")

FEATURE_NAMES = [
    "bwd_avg_bytes_bulk", "bwd_iat_mean", "bwd_iat_std",
    "bwd_packet_length_mean", "bwd_packet_length_std", "bwd_packets_s",
    "destination_port", "flow_bytes_s", "flow_duration",
    "fwd_avg_bytes_bulk", "fwd_avg_packets_bulk", "fwd_iat_mean",
    "fwd_iat_min", "fwd_iat_std", "fwd_packet_length_mean",
    "init_win_bytes_backward", "init_win_bytes_forward", "subflow_fwd_bytes",
]

# Confirmed from preprocessing/config.py's LOG_TRANSFORM_COLS: exactly
# these 5 of the 18 harmonized features were log1p-transformed (after
# clipping negatives to 0) BEFORE IQR scaling for CIC17/CIC18/UNSW-NB15.
# The IQR scaler (median/iqr) was fit on ALREADY LOG-TRANSFORMED values
# for these 5 features -- applying it to raw, untransformed IEC-104
# values for these columns would systematically distort the scaling,
# which is very likely why flow_duration, bwd_iat_mean, and
# fwd_iat_mean showed extreme (83-100%) clipping in the initial run.
LOG_TRANSFORM_FEATURES = [
    "flow_bytes_s", "flow_duration", "fwd_iat_mean",
    "bwd_iat_mean", "bwd_packet_length_mean",
]

COLUMN_MAPPING = {
    "Bwd Byts/b Avg": "bwd_avg_bytes_bulk",
    "Bwd IAT Mean": "bwd_iat_mean",
    "Bwd IAT Std": "bwd_iat_std",
    "Bwd Pkt Len Mean": "bwd_packet_length_mean",
    "Bwd Pkt Len Std": "bwd_packet_length_std",
    "Bwd Pkts/s": "bwd_packets_s",
    "Dst Port": "destination_port",
    "Flow Byts/s": "flow_bytes_s",
    "Flow Duration": "flow_duration",
    "Fwd Byts/b Avg": "fwd_avg_bytes_bulk",
    "Fwd Pkts/b Avg": "fwd_avg_packets_bulk",
    "Fwd IAT Mean": "fwd_iat_mean",
    "Fwd IAT Min": "fwd_iat_min",
    "Fwd IAT Std": "fwd_iat_std",
    "Fwd Pkt Len Mean": "fwd_packet_length_mean",
    "Init Bwd Win Byts": "init_win_bytes_backward",
    "Init Fwd Win Byts": "init_win_bytes_forward",
    "Subflow Fwd Byts": "subflow_fwd_bytes",
}


def load_scaler():
    scaler_path = os.path.join(ARTEFACTS_V2_DIR, "iqr_scaler.npz")
    d = np.load(scaler_path, allow_pickle=True)
    keys = list(d.keys())
    print(f"Scaler keys found: {keys}")

    # "median" wasn't found directly; this scaler uses quartile notation
    # (q1, q2, q3, iqr). q2 IS the median by definition (the 2nd quartile).
    median_key = next((k for k in keys if "median" in k.lower()), None)
    if median_key is None:
        median_key = next((k for k in keys if k.lower() == "q2"), None)
    iqr_key = next((k for k in keys if k.lower() == "iqr" or "iqr" in k.lower()), None)

    if median_key is None or iqr_key is None:
        raise ValueError(
            f"Could not confidently identify median/iqr keys in scaler. "
            f"Found keys: {keys}. Please check manually and adjust this script."
        )

    median = d[median_key].astype(np.float64)
    iqr = d[iqr_key].astype(np.float64)
    print(f"Using '{median_key}' as median, '{iqr_key}' as IQR. Both shape: {median.shape}, {iqr.shape}")

    # Consistency check: if q1 and q3 are also present, confirm iqr == q3 - q1
    # (standard definition), as a safeguard that these arrays mean what we
    # think they mean before using them to scale a new dataset.
    if "q1" in keys and "q3" in keys:
        q1 = d["q1"].astype(np.float64)
        q3 = d["q3"].astype(np.float64)
        implied_iqr = q3 - q1
        max_diff = np.max(np.abs(implied_iqr - iqr))
        print(f"Consistency check: max|iqr - (q3-q1)| = {max_diff:.6f} "
              f"({'OK, matches standard definition' if max_diff < 1e-3 else 'MISMATCH -- investigate before trusting this scaler'})")

    if median.shape[0] != len(FEATURE_NAMES) or iqr.shape[0] != len(FEATURE_NAMES):
        raise ValueError(
            f"Scaler arrays have length {median.shape[0]}/{iqr.shape[0]}, "
            f"expected {len(FEATURE_NAMES)} to match FEATURE_NAMES."
        )
    return median, iqr


def load_and_harmonize_csv(path):
    print(f"Loading {path}...")
    df = pd.read_csv(path)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    missing = [col for col in COLUMN_MAPPING if col not in df.columns]
    if missing:
        raise ValueError(f"Expected columns not found in {path}: {missing}")

    df_renamed = df.rename(columns=COLUMN_MAPPING)
    X = df_renamed[FEATURE_NAMES].copy()
    label_col = df["Label"].astype(str).str.strip()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    # Drop rows with any remaining NaN, matching cleaner.py's
    # _drop_nan_rows() exactly (rather than imputing/filling), so
    # IEC-104 is handled identically to CIC17/CIC18/UNSW-NB15.
    n_before = len(X)
    valid_mask = ~X.isna().any(axis=1)
    n_nan_rows = (~valid_mask).sum()
    if n_nan_rows > 0:
        print(f"  Dropping {n_nan_rows} rows with NaN/infinite values "
              f"(infinities are common in Flow Byts/s or Flow Pkts/s for "
              f"zero-duration flows) -- matches cleaner.py's _drop_nan_rows().")
    X = X[valid_mask]
    label_col = label_col[valid_mask]

    y = (label_col.str.upper() != "NORMAL").astype(int).values

    print(f"  {n_before} -> {len(X)} rows after NaN drop")
    print(f"  Label distribution: {int((y == 0).sum())} benign, {int((y == 1).sum())} attack")
    return X.values.astype(np.float64), y



def main():
    median, iqr = load_scaler()

    X_train_raw, y_train = load_and_harmonize_csv(TRAIN_CSV)
    X_test_raw, y_test = load_and_harmonize_csv(TEST_CSV)

    X_raw = np.vstack([X_train_raw, X_test_raw])
    y = np.concatenate([y_train, y_test])
    print(f"\nCombined: {len(X_raw)} rows ({int((y == 0).sum())} benign, {int((y == 1).sum())} attack)")

    # Apply log1p to the same 5 features that were log-transformed for
    # CIC17/CIC18/UNSW-NB15 BEFORE fitting/applying the IQR scaler
    # (matches preprocessing/cleaner.py: negatives clipped to 0 first,
    # since log1p requires non-negative input).
    print(f"\n{'=' * 70}\nLOG1P TRANSFORM (matching cleaner.py, applied before IQR scaling)\n{'=' * 70}")
    print(f"  Features: {LOG_TRANSFORM_FEATURES}")
    X_df = pd.DataFrame(X_raw, columns=FEATURE_NAMES)
    for col in LOG_TRANSFORM_FEATURES:
        n_negative = (X_df[col] < 0).sum()
        if n_negative > 0:
            print(f"    {col}: {n_negative} negative values clipped to 0 before log1p")
        X_df[col] = np.log1p(X_df[col].clip(lower=0))
    X_raw = X_df.values.astype(np.float64)

    iqr_safe = np.where(iqr == 0, 1.0, iqr)
    X_scaled = (X_raw - median) / iqr_safe
    X_clipped = np.clip(X_scaled, -10, 10).astype(np.float32)

    print(f"\n{'=' * 70}\nSANITY CHECK -- per-feature scaled range (should mostly fall within [-10, 10])\n{'=' * 70}")
    for i, name in enumerate(FEATURE_NAMES):
        col = X_clipped[:, i]
        pct_clipped_low = (col == -10).mean() * 100
        pct_clipped_high = (col == 10).mean() * 100
        print(f"  {name:28s} min={col.min():+7.3f} max={col.max():+7.3f} "
              f"mean={col.mean():+7.3f}  clipped_low={pct_clipped_low:.1f}%  clipped_high={pct_clipped_high:.1f}%")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Deduplication: EXACT row match (all 18 feature columns + label
    # combined), keeping the first occurrence -- matches
    # run_dedup_pipeline.py's deduplicate_npz() logic exactly, applied
    # AFTER scaling (matching that script's operating order, confirmed
    # by downstream files being loaded and clipped as already-scaled
    # data).
    print(f"\n{'=' * 70}\nDEDUPLICATION (exact row match, matching run_dedup_pipeline.py)\n{'=' * 70}")
    n_before = len(X_clipped)
    df_dedup = pd.DataFrame(X_clipped)
    df_dedup["__label__"] = y
    df_dedup = df_dedup.drop_duplicates(keep="first").reset_index(drop=True)
    y_final = df_dedup["__label__"].values.astype(np.int64)
    X_final = df_dedup.drop(columns=["__label__"]).values.astype(np.float32)
    n_after = len(X_final)
    n_removed = n_before - n_after
    print(f"  {n_before} -> {n_after} rows ({n_removed} removed, {n_removed / n_before * 100:.2f}%)")
    print(f"  Post-dedup label distribution: {int((y_final == 0).sum())} benign, "
          f"{int((y_final == 1).sum())} attack")

    np.savez(OUTPUT_PATH, X=X_final, y=y_final)
    print(f"\nSaved {OUTPUT_PATH}")
    print(f"Shape: X={X_final.shape}, y={y_final.shape}")
    print("\nIf many features show a large fraction clipped at -10 or +10, that's worth "
          "investigating before treating this as a reliable evaluation target -- it would "
          "mean IEC-104's value ranges differ substantially from the CIC17/CIC18/UNSW "
          "scaler this data is being forced into, which could itself be an interesting "
          "finding (a genuinely large domain gap) or a sign of a mapping/unit mismatch "
          "worth double-checking.")


if __name__ == "__main__":
    main()
