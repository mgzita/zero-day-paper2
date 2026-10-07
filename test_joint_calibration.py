"""
Joint calibration test: does the RELATIONSHIP between stable features
(not just their individual ranges) differ CIC17 vs UNSW?
=========================================================================
Follow-up to test_calibration_hypothesis.py, which found the marginal
(per-feature) ranges of the stable-v2 features were ALREADY well
covered by CIC17-only training -- ruling out simple range-mismatch as
the reason Single+Stable still failed to generalize
(single_stable_only.py).

This tests a more specific hypothesis: an autoencoder's bottleneck
forces it to reconstruct each feature from a compressed JOINT
representation of the others. Even if each stable feature's marginal
distribution matches between CIC17 and UNSW, the RELATIONSHIP between
them could still differ -- causing reconstruction failure on UNSW even
with perfectly calibrated individual ranges.

Two checks:
  1. Correlation structure comparison among the 5 stable-v2 features:
     CIC17 training vs Multi training vs UNSW's actual structure.
  2. Linear reconstruction test: predict each stable feature from the
     other 4, fit once on CIC17-only data and once on Multi data, then
     evaluate prediction error on UNSW's ACTUAL values. A linear proxy
     for what the autoencoder bottleneck does.

No deep model training needed -- pure statistics, fast.
EXPLORATORY analysis (Formal Protocol Section 8).

Run from project root:
    python paper2\\pilot\\test_joint_calibration.py
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

FEATURE_NAMES = [
    "bwd_avg_bytes_bulk", "bwd_iat_mean", "bwd_iat_std",
    "bwd_packet_length_mean", "bwd_packet_length_std", "bwd_packets_s",
    "destination_port", "flow_bytes_s", "flow_duration",
    "fwd_avg_bytes_bulk", "fwd_avg_packets_bulk", "fwd_iat_mean",
    "fwd_iat_min", "fwd_iat_std", "fwd_packet_length_mean",
    "init_win_bytes_backward", "init_win_bytes_forward", "subflow_fwd_bytes",
]

STABLE_V2 = ["bwd_iat_std", "fwd_iat_std", "destination_port",
             "init_win_bytes_forward", "bwd_packet_length_mean"]

DEDUP_DIR = r"C:\MLProject\zero_day_project\artefacts_v2\dedup"  # TODO confirm
PAPER2_ROOT = r"C:\MLProject\zero_day_project\paper2"
PILOT_DIR = os.path.join(PAPER2_ROOT, "pilot")


def load_single_training():
    d = np.load(os.path.join(DEDUP_DIR, "train_dedup.npz"))
    X = np.clip(d["X_train"], -10, 10).astype(np.float32)
    n_per_env = len(X) // 3
    return pd.DataFrame(X[:n_per_env], columns=FEATURE_NAMES)


def load_multi_matched_training():
    d = np.load(os.path.join(DEDUP_DIR, "train_dedup.npz"))
    X_full = np.clip(d["X_train"], -10, 10).astype(np.float32)
    n_per_env = len(X_full) // 3
    n_matched = n_per_env // 3
    X_cic17 = X_full[:n_per_env][:n_matched]
    X_cic18 = X_full[n_per_env:2 * n_per_env][:n_matched]
    X_unsw_train = X_full[2 * n_per_env:][:n_matched]
    X_multi = np.vstack([X_cic17, X_cic18, X_unsw_train])
    return pd.DataFrame(X_multi, columns=FEATURE_NAMES)


def load_unsw_benign():
    d = np.load(os.path.join(DEDUP_DIR, "test_unsw_dedup.npz"))
    X = np.clip(d["X"], -10, 10).astype(np.float32)
    y = d["y"].astype(int)
    return pd.DataFrame(X[y == 0], columns=FEATURE_NAMES)


def correlation_structure_check(single_train, multi_train, unsw_benign):
    print("=" * 90)
    print("CHECK 1: Correlation structure among stable-v2 features")
    print("=" * 90)

    def corr_matrix(df):
        c, _ = spearmanr(df[STABLE_V2].values)
        return pd.DataFrame(c, index=STABLE_V2, columns=STABLE_V2)

    corr_single = corr_matrix(single_train)
    corr_multi = corr_matrix(multi_train)
    corr_unsw = corr_matrix(unsw_benign)

    diff_single_unsw = (corr_single - corr_unsw).abs()
    diff_multi_unsw = (corr_multi - corr_unsw).abs()

    n = len(STABLE_V2)
    mask = ~np.eye(n, dtype=bool)
    mean_diff_single = diff_single_unsw.values[mask].mean()
    mean_diff_multi = diff_multi_unsw.values[mask].mean()

    print("Mean |correlation difference from UNSW's true structure|:")
    print(f"  CIC17-only training vs UNSW:  {mean_diff_single:.4f}")
    print(f"  Multi training vs UNSW:       {mean_diff_multi:.4f}")
    print("  (Lower = training data's correlation structure is closer to UNSW's true structure)")

    if mean_diff_single - mean_diff_multi > 0.05:
        print("  -> Multi's correlation structure is notably closer to UNSW's true "
              "structure than CIC17-only's is -- supports the joint-calibration hypothesis.")
    else:
        print("  -> No large difference in correlation-structure similarity to UNSW.")

    return mean_diff_single, mean_diff_multi


def linear_reconstruction_check(single_train, multi_train, unsw_benign):
    print(f"\n{'=' * 90}")
    print("CHECK 2: Linear reconstruction error on UNSW (proxy for autoencoder bottleneck)")
    print(f"{'=' * 90}")

    results = []
    for target_feat in STABLE_V2:
        predictor_feats = [f for f in STABLE_V2 if f != target_feat]

        lr_single = LinearRegression()
        lr_single.fit(single_train[predictor_feats], single_train[target_feat])
        pred_single = lr_single.predict(unsw_benign[predictor_feats])
        mse_single = mean_squared_error(unsw_benign[target_feat], pred_single)

        lr_multi = LinearRegression()
        lr_multi.fit(multi_train[predictor_feats], multi_train[target_feat])
        pred_multi = lr_multi.predict(unsw_benign[predictor_feats])
        mse_multi = mean_squared_error(unsw_benign[target_feat], pred_multi)

        target_var = unsw_benign[target_feat].var()
        norm_mse_single = mse_single / target_var if target_var > 0 else np.nan
        norm_mse_multi = mse_multi / target_var if target_var > 0 else np.nan

        results.append({
            "target_feature": target_feat,
            "mse_single_fit": mse_single,
            "mse_multi_fit": mse_multi,
            "normalized_mse_single_fit": norm_mse_single,
            "normalized_mse_multi_fit": norm_mse_multi,
            "improvement": norm_mse_single - norm_mse_multi,
        })

    df = pd.DataFrame(results)
    print(df[["target_feature", "normalized_mse_single_fit",
              "normalized_mse_multi_fit", "improvement"]].round(4).to_string(index=False))

    mean_improvement = df["improvement"].mean()
    print(f"\nMean normalized-MSE improvement (Single -> Multi): {mean_improvement:.4f}")
    if mean_improvement > 0.1:
        print("-> STRONG support: predicting one stable feature from the others, "
              "using CIC17-only relationships, generalizes much worse to UNSW than "
              "using Multi-derived relationships -- even though individual feature "
              "ranges already matched. This is direct evidence that diversity "
              "training recalibrates the JOINT relationships between stable "
              "features, not just their marginal ranges.")
    elif mean_improvement > 0.02:
        print("-> MODERATE support for the joint-calibration hypothesis.")
    else:
        print("-> WEAK support -- joint linear relationships among stable features "
              "may also already be well-calibrated under Single training. The "
              "Single+Stable failure may involve nonlinear structure that a linear "
              "proxy cannot detect, or a different mechanism entirely.")

    return df


def main():
    single_train = load_single_training()
    multi_train = load_multi_matched_training()
    unsw_benign = load_unsw_benign()

    print(f"Single-env training: {len(single_train)} rows")
    print(f"Multi-env training:  {len(multi_train)} rows")
    print(f"UNSW-NB15 benign (target): {len(unsw_benign)} rows\n")

    correlation_structure_check(single_train, multi_train, unsw_benign)
    recon_df = linear_reconstruction_check(single_train, multi_train, unsw_benign)

    out_path = os.path.join(PILOT_DIR, "joint_calibration_test.csv")
    recon_df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
