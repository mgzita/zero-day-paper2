"""
Paired statistics on Wasserstein distance for the representation-level
analysis, Normalizing Flow, all three targets -- the corrected companion
to compute_flow_representation_stats.py (which computes the same paired
t-test / Wilcoxon / Cohen's d, but on the MMD^2 values that Section 4.3
shows are unreliable for the IEC-104 cell specifically).
=========================================================================
Round-3 reviewer critique, item 3: Figure 5 (fig:repr) currently plots
only the MMD^2-based d=-4.00 point for Normalizing Flow / IEC-104,
flagged as a diagnostic artifact, with no paired "final" number placed
next to it. The reviewer asks for a real corrected Cohen's d, computed
the same way as every other point in that figure, using the bandwidth-
free Wasserstein measure instead of MMD^2.

This script is compute_flow_representation_stats.py with ONE line
changed: mmd_single/mmd_lodo3 -> wasserstein_single/wasserstein_lodo3.
Same paired t-test, same Wilcoxon signed-rank test, same Cohen's d
formula (mean paired difference / SD of paired differences), same sign
convention (diffs = single - lodo3; positive d = gap narrowed under
Diverse/LODO-3 training, matching the original hypothesis) -- so the
resulting d is directly comparable to, and plottable alongside, every
other point already in Figure 5 / Table 7.

Run from project root, same folder as the flow_representation*.json
files (PILOT_DIR):
    python compute_flow_representation_stats_wasserstein.py
"""

import json
import os

import numpy as np
from scipy import stats

import paper2_followup_tests as pft

PILOT_DIR = pft.PILOT_DIR

FILES = {
    "UNSW-NB15": os.path.join(PILOT_DIR, "flow_representation.json"),
    "IEC-104": os.path.join(PILOT_DIR, "flow_representation_iec104.json"),
    "SMD": os.path.join(PILOT_DIR, "flow_representation_smd.json"),
}


def analyze(name, path):
    print(f"{'=' * 70}\n{name}\n{'=' * 70}")
    if not os.path.exists(path):
        print(f"  NOT FOUND at {path}")
        return None

    with open(path) as f:
        results = json.load(f)

    missing = [r for r in results if "wasserstein_single" not in r or "wasserstein_lodo3" not in r]
    if missing:
        print(f"  WARNING: {len(missing)} record(s) missing wasserstein_single/wasserstein_lodo3 -- dropped")
    results = [r for r in results if "wasserstein_single" in r and "wasserstein_lodo3" in r]
    if not results:
        print("  No usable records.")
        return None

    wass_single = np.array([r["wasserstein_single"] for r in results])
    wass_lodo3 = np.array([r["wasserstein_lodo3"] for r in results])
    n = len(wass_single)

    print(f"  n = {n} seeds")
    print(f"  Wasserstein Single: mean={wass_single.mean():.4f}, std={wass_single.std(ddof=1):.4f}")
    print(f"  Wasserstein LODO-3: mean={wass_lodo3.mean():.4f}, std={wass_lodo3.std(ddof=1):.4f}")

    diffs = wass_single - wass_lodo3
    n_narrowed = int(np.sum(diffs > 0))
    print(f"  Narrowed in {n_narrowed}/{n} seeds")

    t_stat, t_p = stats.ttest_rel(wass_single, wass_lodo3)
    w_p = None
    try:
        w_stat, w_p = stats.wilcoxon(wass_single, wass_lodo3)
    except ValueError as e:
        print(f"  Wilcoxon not computable: {e}")

    d_std = diffs.std(ddof=1)
    cohens_d = float(diffs.mean() / d_std) if d_std > 0 else float("nan")

    outcome = ("Supported (narrowed)" if (t_p < 0.05 and n_narrowed == n)
               else "Reversed (widened, significant)" if (t_p < 0.05 and n_narrowed == 0)
               else "Suggestive" if n_narrowed in (n, 0)
               else "Not supported / mixed")

    print(f"  Paired t-test: t={t_stat:.3f}, p={t_p:.6f}")
    if w_p is not None:
        print(f"  Wilcoxon: p={w_p:.6f}")
    print(f"  Cohen's d (paired, Wasserstein-based): {cohens_d:+.3f}")
    print(f"  Outcome: {outcome}\n")

    return {
        "n": n, "wasserstein_single_mean": float(wass_single.mean()),
        "wasserstein_lodo3_mean": float(wass_lodo3.mean()),
        "n_narrowed": n_narrowed, "t_stat": float(t_stat), "p_value": float(t_p),
        "wilcoxon_p": None if w_p is None else float(w_p), "cohens_d": cohens_d, "outcome": outcome,
    }


def main():
    summary = {}
    for name, path in FILES.items():
        result = analyze(name, path)
        if result:
            summary[name] = result

    print(f"{'=' * 70}\nFor Figure 5 / Table 7 (Normalizing Flow / IEC-104 cell specifically)\n{'=' * 70}")
    if "IEC-104" in summary:
        d = summary["IEC-104"]["cohens_d"]
        print(f"  Original MMD^2 reading:  d=-4.00 (diagnostic artifact, not used for inference)")
        print(f"  Corrected Wasserstein:   d={d:+.3f} (final representation-level result)")

    out_path = os.path.join(PILOT_DIR, "flow_representation_wasserstein_paired_stats.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
