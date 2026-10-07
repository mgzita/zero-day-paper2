"""
Compute a fresh, SMD-specific Environment Sensitivity Score (ESS),
decorrelated feature set, and Stable/Sensitive/Middle groups -- the
SMD analogue of the IDS pipeline's STABLE_V2/SENSITIVE_V2/MIDDLE_V2.
=========================================================================
SMD's 38 features are anonymized and unrelated to the IDS feature
space, so the existing feature groups cannot be reused. This script
applies the SAME generic recipe used to build the IDS groups:

  1. ESS per feature = mean pairwise Jensen-Shannon divergence of that
     feature's distribution across the 3 training-source environments
     (SMD's 3 groups).
  2. Spearman-correlation-based redundancy removal (|rho| > 0.85),
     matching the IDS pipeline's decorrelation threshold.
  3. Split the remaining decorrelated features into Stable (5 lowest
     ESS), Sensitive (5 highest ESS), and Middle (5 intermediate ESS).

No model training involved -- pure statistical computation.

Run from project root:
    python paper2\\pilot\\smd_compute_ess.py
"""

import json
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

SMD_DIR = r"C:\MLProject\zero_day_project\artefacts_v2\dedup_smd"
PAPER2_ROOT = r"C:\MLProject\zero_day_project\paper2"
PILOT_DIR = os.path.join(PAPER2_ROOT, "pilot")

N_FEATURES = 38
N_BINS = 50
CORRELATION_THRESHOLD = 0.85


def load_group_train(group):
    d_train = np.load(os.path.join(SMD_DIR, "train_smd_dedup.npz"))
    X_train_full = d_train["X_train"]
    n_per_group = len(X_train_full) // 3
    start = (group - 1) * n_per_group
    return X_train_full[start:start + n_per_group]


def js_divergence_feature(x_a, x_b, bins):
    hist_a, _ = np.histogram(x_a, bins=bins, density=True)
    hist_b, _ = np.histogram(x_b, bins=bins, density=True)
    hist_a = hist_a + 1e-10
    hist_b = hist_b + 1e-10
    hist_a = hist_a / hist_a.sum()
    hist_b = hist_b / hist_b.sum()
    return jensenshannon(hist_a, hist_b) ** 2


def compute_ess(group_data):
    ess = np.zeros(N_FEATURES)
    group_pairs = [(1, 2), (1, 3), (2, 3)]
    for feat_idx in range(N_FEATURES):
        all_vals = np.concatenate([group_data[g][:, feat_idx] for g in (1, 2, 3)])
        bin_edges = np.histogram_bin_edges(all_vals, bins=N_BINS)

        divergences = []
        for a, b in group_pairs:
            d = js_divergence_feature(group_data[a][:, feat_idx], group_data[b][:, feat_idx], bin_edges)
            divergences.append(d)
        ess[feat_idx] = np.mean(divergences)

    ess_min, ess_max = ess.min(), ess.max()
    if ess_max > ess_min:
        ess_normalized = (ess - ess_min) / (ess_max - ess_min)
    else:
        ess_normalized = ess
    return ess_normalized


def decorrelate(X_combined, feature_names, threshold=CORRELATION_THRESHOLD):
    df = pd.DataFrame(X_combined, columns=feature_names)
    corr = df.corr(method="spearman").abs()

    to_drop = set()
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            if corr.iloc[i, j] > threshold:
                fi, fj = feature_names[i], feature_names[j]
                if fi not in to_drop and fj not in to_drop:
                    to_drop.add(fj)

    kept = [f for f in feature_names if f not in to_drop]
    return kept, sorted(to_drop)


def main():
    print(f"{'=' * 70}\nSMD-specific ESS computation\n{'=' * 70}")
    feature_names = [f"smd_feat_{i:02d}" for i in range(N_FEATURES)]

    group_data = {g: load_group_train(g) for g in (1, 2, 3)}
    for g, X in group_data.items():
        print(f"  Group {g}: {len(X)} rows")

    # Exclude features already confirmed as "practically constant"
    # (IQR == 0, meaning at least 50% of values are identical) by
    # harmonize_smd.py's scaler-fitting step. Reusing that EXACT same
    # criterion (rather than a stricter std==0 check computed
    # separately) guarantees consistency: a std==0 check misses
    # features that are constant for the bulk of their values but have
    # rare outlier readings (IQR==0 but std>0) -- exactly what caused
    # smd_feat_16/17/28 to incorrectly pass an earlier, stricter
    # version of this check despite being confirmed dead/degenerate
    # metrics.
    print(f"\n{'=' * 70}\nExcluding constant features (IQR == 0, matching harmonize_smd.py exactly)\n{'=' * 70}")
    scaler = np.load(os.path.join(SMD_DIR, "smd_iqr_scaler.npz"))
    iqr_from_scaler = scaler["iqr"]
    constant_mask = (iqr_from_scaler == 0)
    constant_features = [feature_names[i] for i in range(N_FEATURES) if constant_mask[i]]
    active_features = [feature_names[i] for i in range(N_FEATURES) if not constant_mask[i]]
    print(f"  {len(constant_features)} constant feature(s) excluded: {constant_features}")
    print(f"  {len(active_features)} active feature(s) remain for ESS ranking and grouping.")

    print("\nComputing ESS (mean pairwise Jensen-Shannon divergence across groups)...")
    ess = compute_ess(group_data)
    ess_series_all = pd.Series(ess, index=feature_names)
    ess_series = ess_series_all.loc[active_features].sort_values()  # constant features excluded here
    print("\nESS per feature, ACTIVE features only (sorted, lowest = most stable):")
    print(ess_series.to_string())

    print(f"\n{'=' * 70}\nDecorrelation (Spearman |rho| > {CORRELATION_THRESHOLD}), active features only\n{'=' * 70}")
    X_combined = np.vstack([group_data[g] for g in (1, 2, 3)])
    active_idx = [feature_names.index(f) for f in active_features]
    X_combined_active = X_combined[:, active_idx]
    kept_features, dropped_features = decorrelate(X_combined_active, active_features)
    print(f"  Kept {len(kept_features)} features, dropped {len(dropped_features)} redundant: {dropped_features}")

    ess_decorrelated = ess_series.loc[kept_features].sort_values()
    print("\nDecorrelated ESS (sorted):")
    print(ess_decorrelated.to_string())

    n_group = 5
    if len(kept_features) < 3 * n_group:
        print(f"\nWARNING: only {len(kept_features)} decorrelated features remain, "
              f"fewer than {3 * n_group} needed for 3 groups of {n_group}.")
        n_group = len(kept_features) // 3

    stable = ess_decorrelated.index[:n_group].tolist()
    sensitive = ess_decorrelated.index[-n_group:].tolist()
    middle_start = (len(ess_decorrelated) - n_group) // 2
    middle = ess_decorrelated.index[middle_start:middle_start + n_group].tolist()

    print(f"\n{'=' * 70}\nSMD Stable / Sensitive / Middle groups ({n_group} features each)\n{'=' * 70}")
    print(f"  STABLE (lowest ESS):     {stable}")
    print(f"  SENSITIVE (highest ESS): {sensitive}")
    print(f"  MIDDLE (intermediate):   {middle}")

    output = {
        "feature_names": feature_names,
        "constant_features_excluded": constant_features,
        "active_features": active_features,
        "ess_per_feature": {f: float(v) for f, v in ess_series.items()},
        "decorrelated_features": kept_features,
        "dropped_redundant_features": dropped_features,
        "stable_group": stable,
        "sensitive_group": sensitive,
        "middle_group": middle,
    }
    out_path = os.path.join(PILOT_DIR, "smd_ess_feature_groups.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
