"""
Consolidated verification script: pulls EVERY number that appears in
the Results section draft directly from saved JSON files, so every
figure/table value can be checked against ground truth rather than
trusted from reconstructed memory.
=========================================================================
Covers, for all 4 model families x 3 datasets x 3 tests (H1, H3, Repr):

  AE/DAE/SVDD, UNSW + IEC-104  -> final_combined_ae_dae_svdd_stats.json
  AE/DAE/SVDD, SMD             -> smd_final_stats.json
  Flow, UNSW + IEC-104, Repr   -> flow_representation_paired_stats.json
  Flow, SMD                    -> flow_smd_final_stats.json
  IG-based H1, all 12 combos   -> ig_attribution_paired_stats.json

  Flow, UNSW + IEC-104, H1/H3  -> NOT pre-computed anywhere. Computed
                                   LIVE here from the raw per-seed files,
                                   since no paired-stats file for these
                                   exists. This gap is itself worth
                                   knowing.

Run from project root:
    python paper2\\pilot\\verify_all_manuscript_numbers.py
"""

import json
import os
import numpy as np
from scipy import stats

PAPER2_ROOT = r"C:\MLProject\zero_day_project\paper2"
PILOT_DIR = os.path.join(PAPER2_ROOT, "pilot")


def safe_load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def paired_test(vals_a, vals_b):
    vals_a, vals_b = np.array(vals_a), np.array(vals_b)
    n = len(vals_a)
    diffs = vals_a - vals_b
    n_a_greater = int(np.sum(diffs > 0))
    t, p = stats.ttest_rel(vals_a, vals_b)
    d_std = diffs.std(ddof=1)
    d = float(diffs.mean() / d_std) if d_std > 0 else float("nan")
    return {"n": n, "mean_a": float(vals_a.mean()), "mean_b": float(vals_b.mean()),
            "n_a_greater": n_a_greater, "t": float(t), "p": float(p), "cohens_d": d}


def pct_change(mean_single, mean_lodo3):
    if mean_single == 0:
        return float("nan")
    return 100 * (mean_single - mean_lodo3) / mean_single


def print_row(dataset, family, test, p, d, extra=""):
    sig = "***" if (p is not None and p < 0.05) else "   "
    p_str = f"{p:.5f}" if p is not None else "n/a"
    d_str = f"{d:+.3f}" if d is not None else "n/a"
    print(f"  {sig} {dataset:10s} {family:6s} {test:5s}  p={p_str:>10s}  d={d_str:>8s}  {extra}")


def main():
    print(f"{'=' * 90}\nGROUND-TRUTH VERIFICATION -- every number pulled directly from saved files\n{'=' * 90}\n")

    combined = safe_load(os.path.join(PILOT_DIR, "final_combined_ae_dae_svdd_stats.json"))
    if combined:
        print("--- AE/DAE/SVDD, UNSW + IEC-104 (from final_combined_ae_dae_svdd_stats.json) ---")
        for dataset_key, dataset_label in [("UNSW-NB15", "UNSW"), ("IEC-104", "IEC-104")]:
            for test in ["h1", "h3", "repr"]:
                tests_dict = combined.get(dataset_key, {}).get(test, {})
                for fam, r in tests_dict.items():
                    p = r.get("p")
                    d = r.get("cohens_d")
                    if test == "repr":
                        pct = pct_change(r.get("mean_a", 0), r.get("mean_b", 0))
                        print_row(dataset_label, fam, test, p, d, f"pct_change={pct:+.1f}%")
                    else:
                        print_row(dataset_label, fam, test, p, d)
    else:
        print("MISSING: final_combined_ae_dae_svdd_stats.json")
    print()

    smd = safe_load(os.path.join(PILOT_DIR, "smd_final_stats.json"))
    if smd:
        print("--- AE/DAE/SVDD, SMD (from smd_final_stats.json) ---")
        for test in ["h1", "h3", "repr"]:
            tests_dict = smd.get(test, {})
            for fam, r in tests_dict.items():
                p = r.get("p")
                d = r.get("cohens_d")
                if test == "repr":
                    pct = pct_change(r.get("mean_a", 0), r.get("mean_b", 0))
                    print_row("SMD", fam, test, p, d, f"pct_change={pct:+.1f}%")
                else:
                    print_row("SMD", fam, test, p, d)
    else:
        print("MISSING: smd_final_stats.json")
    print()

    flow_repr = safe_load(os.path.join(PILOT_DIR, "flow_representation_paired_stats.json"))
    if flow_repr:
        print("--- Flow, UNSW + IEC-104, Representation (from flow_representation_paired_stats.json) ---")
        for dataset_key, dataset_label in [("UNSW-NB15", "UNSW"), ("IEC-104", "IEC-104")]:
            r = flow_repr.get(dataset_key, {})
            if r:
                p = r.get("p_value")
                d = r.get("cohens_d")
                pct = pct_change(r.get("mmd_single_mean", 0), r.get("mmd_lodo3_mean", 0))
                print_row(dataset_label, "Flow", "repr", p, d,
                          f"pct_change={pct:+.1f}%  (single={r.get('mmd_single_mean'):.4f}, "
                          f"lodo3={r.get('mmd_lodo3_mean'):.4f})")
    else:
        print("MISSING: flow_representation_paired_stats.json")
    print()

    print("--- Flow, UNSW + IEC-104, H1 + H3 (COMPUTED LIVE -- no pre-saved paired-stats file existed) ---")
    for dataset_key, h1_file, h3_file in [
        ("UNSW", "flow_h1_attribution.json", "flow_h3_ablation.json"),
        ("IEC-104", "flow_h1_attribution_iec104.json", "flow_h3_ablation_iec104.json"),
    ]:
        h1 = safe_load(os.path.join(PILOT_DIR, h1_file))
        if h1:
            recs = [r for r in h1 if r.get("additivity_ok", True) and "esfrp_single" in r]
            if len(recs) >= 2:
                r = paired_test([x["esfrp_lodo3"] for x in recs], [x["esfrp_single"] for x in recs])
                print_row(dataset_key, "Flow", "h1", r["p"], r["cohens_d"], f"n_valid={len(recs)}")
            else:
                print(f"  {dataset_key:10s} Flow   h1     insufficient valid seeds ({len(recs)})")
        else:
            print(f"  MISSING: {h1_file}")

        h3 = safe_load(os.path.join(PILOT_DIR, h3_file))
        if h3:
            r = paired_test([x["delta_stable"] for x in h3], [x["delta_sensitive"] for x in h3])
            print_row(dataset_key, "Flow", "h3", r["p"], r["cohens_d"], f"n={r['n']}")
        else:
            print(f"  MISSING: {h3_file}")
    print()

    flow_smd = safe_load(os.path.join(PILOT_DIR, "flow_smd_final_stats.json"))
    if flow_smd:
        print("--- Flow, SMD (from flow_smd_final_stats.json) ---")
        for test, r in flow_smd.items():
            p = r.get("p")
            d = r.get("cohens_d")
            if test == "repr":
                pct = pct_change(r.get("mean_a", 0), r.get("mean_b", 0))
                print_row("SMD", "Flow", test, p, d, f"pct_change={pct:+.1f}%")
            else:
                print_row("SMD", "Flow", test, p, d, f"n={r.get('n')}")
    else:
        print("MISSING: flow_smd_final_stats.json")
    print()

    ig = safe_load(os.path.join(PILOT_DIR, "ig_attribution_paired_stats.json"))
    if ig:
        print("--- IG-based H1, all 12 combinations (from ig_attribution_paired_stats.json) ---")
        for key, r in ig.items():
            dataset, fam = key.split("_", 1)
            print_row(dataset, fam, "h1(IG)", r.get("p"), r.get("cohens_d"),
                      f"completeness_ok={r.get('n_completeness_ok')}/{r.get('n')}")
    else:
        print("MISSING: ig_attribution_paired_stats.json")
    print()

    print(f"{'=' * 90}\nEnd of verification. Cross-check every row above against the manuscript's "
          f"tables/figures before submission.\n{'=' * 90}")


if __name__ == "__main__":
    main()
