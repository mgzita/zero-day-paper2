"""
Final combination: merge all UNSW batches (original 5-seed H1/Repr +
new 5-seed extension + existing 10-seed H3) with the complete IEC-104
results, and compute full paired statistics for every
model x dataset x test combination.
=========================================================================
Sources combined:
  UNSW H1:   unsw_lodo3_revalidation_summary.json (5 seeds, original)
           + ae_dae_svdd_h1_unsw_extra5seeds.json (5 seeds, new)
  UNSW H3:   ablation_h3_lodo3_summary.json (5 seeds)
           + ablation_h3_lodo3_summary_extraseeds.json (5 seeds)
  UNSW Repr: ae_dae_svdd_latent_lodo3_comparison.json (5 seeds, original)
           + ae_dae_svdd_repr_unsw_extra5seeds.json (5 seeds, new)
  IEC-104:   ae_dae_svdd_h1_iec104.json / _h3_ / _repr_ (10 seeds each, complete)

This script is defensive about field-name variations across the several
scripts that produced these files over the course of this project --
if a file's structure doesn't match what's expected, it prints a clear
diagnostic rather than silently computing wrong numbers.

Run from project root:
    python paper2\\pilot\\final_combine_all_results.py
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


def combine_unsw_h1():
    print(f"{'=' * 70}\nUNSW H1 (combining original 5 + extra 5 seeds)\n{'=' * 70}")
    original = safe_load(os.path.join(PILOT_DIR, "unsw_lodo3_revalidation_summary.json"))
    extra = safe_load(os.path.join(PILOT_DIR, "ae_dae_svdd_h1_unsw_extra5seeds.json"))
    if original is None or extra is None:
        return {}

    combined = {fam: {"esfrp_single": [], "esfrp_lodo3": []} for fam in MODEL_FAMILIES}
    for r in original:
        fam = r.get("family")
        if fam in combined and r.get("additivity_ok", True) and "esfrp_single" in r:
            combined[fam]["esfrp_single"].append(r["esfrp_single"])
            combined[fam]["esfrp_lodo3"].append(r["esfrp_lodo3"])
    for r in extra:
        fam = r.get("family")
        if fam in combined and r.get("additivity_ok", True) and "esfrp_single" in r:
            combined[fam]["esfrp_single"].append(r["esfrp_single"])
            combined[fam]["esfrp_lodo3"].append(r["esfrp_lodo3"])

    results = {}
    for fam in MODEL_FAMILIES:
        n = len(combined[fam]["esfrp_single"])
        print(f"  {fam.upper()}: {n} seeds combined")
        if n >= 2:
            results[fam] = paired_test(combined[fam]["esfrp_lodo3"], combined[fam]["esfrp_single"],
                                        "ESFRP(LODO-3) vs ESFRP(Single)")
    return results


def combine_unsw_h3():
    print(f"\n{'=' * 70}\nUNSW H3 (combining two existing 5-seed batches -> 10)\n{'=' * 70}")
    batch_a = safe_load(os.path.join(PILOT_DIR, "ablation_h3_lodo3_summary.json"))
    batch_b = safe_load(os.path.join(PILOT_DIR, "ablation_h3_lodo3_summary_extraseeds.json"))
    if batch_a is None or batch_b is None:
        return {}

    all_records = batch_a + batch_b
    results = {}
    for fam in MODEL_FAMILIES:
        stable = {r["seed"]: r["delta_from_full"] for r in all_records
                  if r.get("family") == fam and r.get("condition") == "stable_lodo3"}
        sensitive = {r["seed"]: r["delta_from_full"] for r in all_records
                     if r.get("family") == fam and r.get("condition") == "sensitive_lodo3"}
        seeds_common = sorted(set(stable) & set(sensitive))
        n = len(seeds_common)
        print(f"  {fam.upper()}: {n} seeds combined")
        if n >= 2:
            stable_vals = [stable[s] for s in seeds_common]
            sensitive_vals = [sensitive[s] for s in seeds_common]
            results[fam] = paired_test(stable_vals, sensitive_vals, "delta_stable vs delta_sensitive")
    return results


def combine_unsw_repr():
    print(f"\n{'=' * 70}\nUNSW Representation (combining original 5 + extra 5 seeds)\n{'=' * 70}")
    original = safe_load(os.path.join(PILOT_DIR, "ae_dae_svdd_latent_lodo3_comparison.json"))
    extra = safe_load(os.path.join(PILOT_DIR, "ae_dae_svdd_repr_unsw_extra5seeds.json"))
    if original is None or extra is None:
        return {}

    combined = {fam: {"mmd_single": [], "mmd_lodo3": []} for fam in MODEL_FAMILIES}
    for r in original:
        fam = r.get("family")
        if fam in combined and r.get("condition") == "single":
            combined[fam]["mmd_single"].append(r["mmd_squared"])
        elif fam in combined and r.get("condition") == "lodo3":
            combined[fam]["mmd_lodo3"].append(r["mmd_squared"])
    for r in extra:
        fam = r.get("family")
        if fam in combined and "mmd_single" in r:
            combined[fam]["mmd_single"].append(r["mmd_single"])
            combined[fam]["mmd_lodo3"].append(r["mmd_lodo3"])

    results = {}
    for fam in MODEL_FAMILIES:
        n_s = len(combined[fam]["mmd_single"])
        n_l = len(combined[fam]["mmd_lodo3"])
        print(f"  {fam.upper()}: {n_s} Single values, {n_l} LODO-3 values "
              f"(if counts differ, original file structure may differ from expected -- verify manually)")
        n = min(n_s, n_l)
        if n >= 2:
            results[fam] = paired_test(combined[fam]["mmd_single"][:n], combined[fam]["mmd_lodo3"][:n],
                                        "MMD^2(Single) vs MMD^2(LODO-3)")
    return results


def analyze_iec104():
    print(f"\n{'=' * 70}\nIEC-104 (already complete, 10 seeds each)\n{'=' * 70}")
    h1 = safe_load(os.path.join(PILOT_DIR, "ae_dae_svdd_h1_iec104.json"))
    h3 = safe_load(os.path.join(PILOT_DIR, "ae_dae_svdd_h3_iec104.json"))
    repr_data = safe_load(os.path.join(PILOT_DIR, "ae_dae_svdd_repr_iec104.json"))

    results = {"h1": {}, "h3": {}, "repr": {}}

    print("  H1:")
    for fam in MODEL_FAMILIES:
        recs = [r for r in h1 if r.get("family") == fam and r.get("additivity_ok", True) and "esfrp_single" in r]
        n = len(recs)
        print(f"    {fam.upper()}: {n} valid seeds")
        if n >= 2:
            results["h1"][fam] = paired_test([r["esfrp_lodo3"] for r in recs],
                                              [r["esfrp_single"] for r in recs],
                                              "ESFRP(LODO-3) vs ESFRP(Single)")

    print("  H3:")
    for fam in MODEL_FAMILIES:
        recs = [r for r in h3 if r.get("family") == fam]
        n = len(recs)
        print(f"    {fam.upper()}: {n} seeds")
        if n >= 2:
            results["h3"][fam] = paired_test([r["delta_stable"] for r in recs],
                                              [r["delta_sensitive"] for r in recs],
                                              "delta_stable vs delta_sensitive")

    print("  Representation:")
    for fam in MODEL_FAMILIES:
        recs = [r for r in repr_data if r.get("family") == fam]
        n = len(recs)
        print(f"    {fam.upper()}: {n} seeds")
        if n >= 2:
            results["repr"][fam] = paired_test([r["mmd_single"] for r in recs],
                                                [r["mmd_lodo3"] for r in recs],
                                                "MMD^2(Single) vs MMD^2(LODO-3)")
    return results


def main():
    unsw_h1 = combine_unsw_h1()
    unsw_h3 = combine_unsw_h3()
    unsw_repr = combine_unsw_repr()
    iec104 = analyze_iec104()

    final = {
        "UNSW-NB15": {"h1": unsw_h1, "h3": unsw_h3, "repr": unsw_repr},
        "IEC-104": {"h1": iec104["h1"], "h3": iec104["h3"], "repr": iec104["repr"]},
    }

    out_path = os.path.join(PILOT_DIR, "final_combined_ae_dae_svdd_stats.json")
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\n{'=' * 70}\nSaved {out_path}\n{'=' * 70}")


if __name__ == "__main__":
    main()
