"""
Compute full paired statistics for the SMD (Server Machine Dataset)
AE/DAE/SVDD results -- H1, H3, and representation-level analysis.
=========================================================================
Reads the three saved result files directly and prints a clean,
complete summary: paired t-test, Wilcoxon, and Cohen's d for each
model family and test.

Run from project root:
    python paper2\\pilot\\analyze_smd_results.py
"""

import json
import os
import numpy as np
from scipy import stats

PAPER2_ROOT = r"C:\MLProject\zero_day_project\paper2"
PILOT_DIR = os.path.join(PAPER2_ROOT, "pilot")
MODEL_FAMILIES = ["ae", "dae", "svdd"]


def safe_load(path):
    if not os.path.exists(path):
        print(f"  MISSING: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def paired_test(vals_a, vals_b, label):
    vals_a, vals_b = np.array(vals_a), np.array(vals_b)
    n = len(vals_a)
    diffs = vals_a - vals_b
    n_a_greater = int(np.sum(diffs > 0))
    t, p = stats.ttest_rel(vals_a, vals_b)
    wp = None
    try:
        w, wp = stats.wilcoxon(vals_a, vals_b)
    except ValueError:
        pass
    d_std = diffs.std(ddof=1)
    d = float(diffs.mean() / d_std) if d_std > 0 else float("nan")
    print(f"    {label}: n={n}, mean_a={vals_a.mean():.4f}, mean_b={vals_b.mean():.4f}, "
          f"a>b in {n_a_greater}/{n}, t={t:.3f}, p={p:.5f}"
          f"{f', Wilcoxon p={wp:.5f}' if wp is not None else ''}, d={d:.3f}")
    return {"n": n, "mean_a": float(vals_a.mean()), "mean_b": float(vals_b.mean()),
            "n_a_greater": n_a_greater, "t": float(t), "p": float(p),
            "wilcoxon_p": None if wp is None else float(wp), "cohens_d": d}


def main():
    print(f"{'=' * 70}\nSMD (Server Machine Dataset, Group 2 target) -- full results\n{'=' * 70}")

    h1 = safe_load(os.path.join(PILOT_DIR, "ae_dae_svdd_h1_smd.json"))
    h3 = safe_load(os.path.join(PILOT_DIR, "ae_dae_svdd_h3_smd.json"))
    repr_data = safe_load(os.path.join(PILOT_DIR, "ae_dae_svdd_repr_smd.json"))

    results = {"h1": {}, "h3": {}, "repr": {}}

    print("\nH1 (attribution: ESFRP Single vs Diverse):")
    for fam in MODEL_FAMILIES:
        recs = [r for r in (h1 or []) if r.get("family") == fam and r.get("additivity_ok", True)
                and "esfrp_single" in r]
        n_total = len([r for r in (h1 or []) if r.get("family") == fam])
        n_valid = len(recs)
        print(f"  {fam.upper()}: {n_valid}/{n_total} seeds valid (additivity ok)")
        if n_valid >= 2:
            results["h1"][fam] = paired_test([r["esfrp_lodo3"] for r in recs],
                                              [r["esfrp_single"] for r in recs],
                                              "ESFRP(Diverse) vs ESFRP(Single)")

    print("\nH3 (functional ablation: delta_stable vs delta_sensitive):")
    for fam in MODEL_FAMILIES:
        recs = [r for r in (h3 or []) if r.get("family") == fam]
        n = len(recs)
        print(f"  {fam.upper()}: {n} seeds")
        if n >= 2:
            results["h3"][fam] = paired_test([r["delta_stable"] for r in recs],
                                              [r["delta_sensitive"] for r in recs],
                                              "delta_stable vs delta_sensitive")

    print("\nRepresentation (MMD^2 Single vs Diverse):")
    for fam in MODEL_FAMILIES:
        recs = [r for r in (repr_data or []) if r.get("family") == fam]
        n = len(recs)
        print(f"  {fam.upper()}: {n} seeds")
        if n >= 2:
            results["repr"][fam] = paired_test([r["mmd_single"] for r in recs],
                                                [r["mmd_lodo3"] for r in recs],
                                                "MMD^2(Single) vs MMD^2(Diverse)")

    out_path = os.path.join(PILOT_DIR, "smd_final_stats.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'=' * 70}\nSaved {out_path}\n{'=' * 70}")


if __name__ == "__main__":
    main()
