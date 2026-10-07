"""
Calibration hypothesis test: does diversity training widen the
learned "normal" range enough to cover UNSW-NB15?
================================================================
Direct test of the hypothesis that diversity training's benefit comes
from teaching the model a better-CALIBRATED notion of "normal" value
ranges per feature -- not from selecting different features.

No trained models needed -- pure data-distribution comparison:
  - Single-environment training range: p1/p99 of each feature, fit on
    CIC-IDS2017 benign training data only (matches Paper 1 Eq. 1).
  - Multi-environment training range: p1/p99 of each feature, fit on
    the combined (CIC17+CIC18+UNSW) Multi-matched benign training pool.
  - Target: UNSW-NB15 benign evaluation data.

For each feature, reports what percentage of UNSW's actual benign
values fall OUTSIDE the Single-derived range vs the Multi-derived
range. A large drop (Single >> Multi) is direct, quantitative evidence
for the calibration hypothesis.

EXPLORATORY analysis (Formal Protocol Section 8).

Run from project root:
    python paper2\\pilot\\test_calibration_hypothesis.py
"""

import os
import numpy as np
import pandas as pd

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


def pct_outside_range(target_vals, p1, p99):
    outside = (target_vals < p1) | (target_vals > p99)
    return float(outside.mean() * 100)


def main():
    single_train = load_single_training()
    multi_train = load_multi_matched_training()
    unsw_benign = load_unsw_benign()

    print(f"Single-env training: {len(single_train)} rows (CIC-IDS2017 only)")
    print(f"Multi-env training:  {len(multi_train)} rows (CIC17+CIC18+UNSW, matched)")
    print(f"UNSW-NB15 benign (target): {len(unsw_benign)} rows\n")

    rows = []
    for feat in FEATURE_NAMES:
        p1_single, p99_single = np.percentile(single_train[feat], [1, 99])
        p1_multi, p99_multi = np.percentile(multi_train[feat], [1, 99])

        pct_out_single = pct_outside_range(unsw_benign[feat].values, p1_single, p99_single)
        pct_out_multi = pct_outside_range(unsw_benign[feat].values, p1_multi, p99_multi)

        rows.append({
            "feature": feat,
            "in_stable_v2": feat in STABLE_V2,
            "single_p1": p1_single, "single_p99": p99_single,
            "multi_p1": p1_multi, "multi_p99": p99_multi,
            "pct_unsw_outside_single_range": pct_out_single,
            "pct_unsw_outside_multi_range": pct_out_multi,
            "improvement": pct_out_single - pct_out_multi,
        })

    df = pd.DataFrame(rows).sort_values("improvement", ascending=False)

    print("=" * 100)
    print("ALL 18 FEATURES: % of UNSW benign values falling OUTSIDE the training-derived range")
    print("=" * 100)
    print(df[["feature", "in_stable_v2", "pct_unsw_outside_single_range",
              "pct_unsw_outside_multi_range", "improvement"]].round(2).to_string(index=False))

    stable_df = df[df["in_stable_v2"]]
    print(f"\n{'=' * 100}")
    print("STABLE-V2 FEATURES ONLY (the features SHAP/H4 show the model relies on)")
    print(f"{'=' * 100}")
    print(stable_df[["feature", "pct_unsw_outside_single_range",
                     "pct_unsw_outside_multi_range", "improvement"]].round(2).to_string(index=False))

    mean_out_single_stable = stable_df["pct_unsw_outside_single_range"].mean()
    mean_out_multi_stable = stable_df["pct_unsw_outside_multi_range"].mean()
    mean_out_single_all = df["pct_unsw_outside_single_range"].mean()
    mean_out_multi_all = df["pct_unsw_outside_multi_range"].mean()

    print(f"\n{'=' * 100}")
    print("SUMMARY")
    print(f"{'=' * 100}")
    print(f"All 18 features:   mean pct outside range -- Single={mean_out_single_all:.2f}%, "
          f"Multi={mean_out_multi_all:.2f}%  (reduction: {mean_out_single_all - mean_out_multi_all:.2f} pts)")
    print(f"Stable-v2 only:    mean pct outside range -- Single={mean_out_single_stable:.2f}%, "
          f"Multi={mean_out_multi_stable:.2f}%  (reduction: {mean_out_single_stable - mean_out_multi_stable:.2f} pts)")

    if mean_out_single_stable - mean_out_multi_stable > 10:
        print("\n-> STRONG support for the calibration hypothesis: Multi-environment "
              "training's learned range covers substantially more of UNSW's actual "
              "benign distribution for the stable features specifically, directly "
              "explaining why Single+Stable failed despite using the 'correct' features.")
    elif mean_out_single_stable - mean_out_multi_stable > 3:
        print("\n-> MODERATE support for the calibration hypothesis.")
    else:
        print("\n-> WEAK support -- range coverage alone may not fully explain the "
              "Single+Stable failure; other factors (e.g. joint/multivariate "
              "calibration, not just per-feature range) may be involved.")

    out_path = os.path.join(PILOT_DIR, "calibration_hypothesis_test.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
