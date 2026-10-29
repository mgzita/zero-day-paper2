"""
ESS sensitivity analysis -- SMD: alternate group sizes and decorrelation
thresholds
================================================================================
Reviewer feedback (major comment #8): similar marginal distributions
across training environments (low ESS) don't necessarily mean a feature
will remain useful/stable in an unseen domain. The manuscript's Stable/
Sensitive/Middle groups used one fixed choice (5 features per group,
Spearman |rho| > 0.85 decorrelation threshold) with no sensitivity check.

This is the SMD analogue of smd_compute_ess.py, generalized to sweep:
  - group size n in {3, 4, 5, 6, 7}  (paper used 5)
  - decorrelation threshold in {0.75, 0.80, 0.85, 0.90, 0.95}  (paper used 0.85)

For every (n, threshold) combination, recomputes the Stable/Sensitive
groups and reports:
  - Jaccard overlap of the Stable group against the paper's original
    Stable group (threshold=0.85, n=5)
  - Jaccard overlap of the Sensitive group against the paper's original
    Sensitive group
  - whether the TOP-5 most-stable and TOP-5 most-sensitive features (by
    rank, ignoring group-size choice) stay the same across thresholds

This does not touch any trained model -- pure statistical computation on
the same training-source data smd_compute_ess.py already uses, so there
is no retraining cost here.

NOTE: this only covers SMD. I could not find a standalone script in the
repo that recomputes ESS for the network-traffic features (UNSW-NB15/
IEC-104's STABLE_V2/SENSITIVE_V2/MIDDLE_V2 appear as hardcoded constants
in ablation_h3_lodo3.py / ablation_h4_v2_decorrelated.py / etc., not
derived in a script that's present here) -- if you still have the
original IDS-side ESS computation script (same JS-divergence-across-
training-sources recipe, just over the 18 harmonized features instead of
SMD's 38), send it over and I'll write the equivalent sweep for it.

Run from the repo root (same place as smd_compute_ess.py):
    python smd_ess_sensitivity_sweep.py
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

GROUP_SIZES_TO_SWEEP = [3, 4, 5, 6, 7]
THRESHOLDS_TO_SWEEP = [0.75, 0.80, 0.85, 0.90, 0.95]

ORIGINAL_N_GROUP = 5
ORIGINAL_THRESHOLD = 0.85


def load_group_train(group):
    d_train = np.load(os.path.join(SMD_DIR, "train_smd_dedup.npz"))
    X_train_full = d_train["X_train"]
    n_per_group = len(X_train_full) // 3
    start = (group - 1) * n_per_group
    return X_train_full[start:start + n_per_group]


def js_divergence_feature(x_a, x_b, bins):
    hist_a, _ = np.histogram(x_a, bins=bins, density=True)
    hist_b, _ = np.histogram(x_b, bins=bins, density=True)
    hist_a = (hist_a + 1e-10)
    hist_b = (hist_b + 1e-10)
    hist_a = hist_a / hist_a.sum()
    hist_b = hist_b / hist_b.sum()
    return jensenshannon(hist_a, hist_b) ** 2


def compute_ess(group_data):
    ess = np.zeros(N_FEATURES)
    group_pairs = [(1, 2), (1, 3), (2, 3)]
    for feat_idx in range(N_FEATURES):
        all_vals = np.concatenate([group_data[g][:, feat_idx] for g in (1, 2, 3)])
        bin_edges = np.histogram_bin_edges(all_vals, bins=N_BINS)
        divergences = [js_divergence_feature(group_data[a][:, feat_idx], group_data[b][:, feat_idx], bin_edges)
                       for a, b in group_pairs]
        ess[feat_idx] = np.mean(divergences)
    ess_min, ess_max = ess.min(), ess.max()
    return (ess - ess_min) / (ess_max - ess_min) if ess_max > ess_min else ess


def decorrelate(X_combined, feature_names, threshold):
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


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 1.0


def groups_for(ess_series, kept_features, n_group):
    ess_decorrelated = ess_series.loc[kept_features].sort_values()
    if len(kept_features) < 3 * n_group:
        n_group = len(kept_features) // 3
    stable = ess_decorrelated.index[:n_group].tolist()
    sensitive = ess_decorrelated.index[-n_group:].tolist()
    return stable, sensitive


def main():
    print(f"{'=' * 70}\nSMD ESS sensitivity sweep\n{'=' * 70}")
    feature_names = [f"smd_feat_{i:02d}" for i in range(N_FEATURES)]

    group_data = {g: load_group_train(g) for g in (1, 2, 3)}
    for g, X in group_data.items():
        print(f"  Group {g}: {len(X)} rows")

    scaler = np.load(os.path.join(SMD_DIR, "smd_iqr_scaler.npz"))
    iqr_from_scaler = scaler["iqr"]
    constant_mask = (iqr_from_scaler == 0)
    active_features = [feature_names[i] for i in range(N_FEATURES) if not constant_mask[i]]
    print(f"  {N_FEATURES - len(active_features)} constant feature(s) excluded.")

    print("\nComputing ESS once (threshold-independent)...")
    ess = compute_ess(group_data)
    ess_series_all = pd.Series(ess, index=feature_names)
    ess_series = ess_series_all.loc[active_features]

    X_combined = np.vstack([group_data[g] for g in (1, 2, 3)])
    active_idx = [feature_names.index(f) for f in active_features]
    X_combined_active = X_combined[:, active_idx]

    # Original paper groups, for comparison.
    kept_orig, _ = decorrelate(X_combined_active, active_features, ORIGINAL_THRESHOLD)
    stable_orig, sensitive_orig = groups_for(ess_series, kept_orig, ORIGINAL_N_GROUP)
    print(f"\nOriginal paper groups (threshold={ORIGINAL_THRESHOLD}, n={ORIGINAL_N_GROUP}):")
    print(f"  Stable:    {stable_orig}")
    print(f"  Sensitive: {sensitive_orig}")

    results = []
    print(f"\n{'=' * 70}\nSweep results (Jaccard overlap against original groups)\n{'=' * 70}")
    print(f"{'threshold':>9} {'n_group':>7} {'n_kept':>6} {'stable_jaccard':>14} {'sensitive_jaccard':>18}")
    for threshold in THRESHOLDS_TO_SWEEP:
        kept, dropped = decorrelate(X_combined_active, active_features, threshold)
        for n_group in GROUP_SIZES_TO_SWEEP:
            stable, sensitive = groups_for(ess_series, kept, n_group)
            j_stable = jaccard(stable, stable_orig)
            j_sensitive = jaccard(sensitive, sensitive_orig)
            print(f"{threshold:>9.2f} {n_group:>7} {len(kept):>6} {j_stable:>14.3f} {j_sensitive:>18.3f}")
            results.append({
                "threshold": threshold, "n_group": n_group, "n_kept_features": len(kept),
                "stable_group": stable, "sensitive_group": sensitive,
                "stable_jaccard_vs_original": j_stable,
                "sensitive_jaccard_vs_original": j_sensitive,
            })

    out_path = os.path.join(PILOT_DIR, "smd_ess_sensitivity_sweep.json")
    with open(out_path, "w") as f:
        json.dump({
            "original_threshold": ORIGINAL_THRESHOLD, "original_n_group": ORIGINAL_N_GROUP,
            "original_stable_group": stable_orig, "original_sensitive_group": sensitive_orig,
            "sweep": results,
        }, f, indent=2)
    print(f"\nSaved {out_path}")
    print("\nA Jaccard overlap consistently near 1.0 across the sweep supports the paper's")
    print("groups as robust to this choice; values that swing a lot (especially near the")
    print("original threshold=0.85, n=5 setting) would need flagging as a limitation or")
    print("would need the manuscript's groups re-derived from a more stable setting.")


if __name__ == "__main__":
    main()
