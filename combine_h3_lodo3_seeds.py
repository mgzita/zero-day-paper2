"""
Combine the two independent 5-seed H3 LODO-3 ablation batches into one
true 10-seed paired statistical test.
=========================================================================
Loads the two separately-run result files:
  - ablation_h3_lodo3_summary.json            (seeds 42, 123, 456, 789, 999)
  - ablation_h3_lodo3_summary_extraseeds.json (seeds 111, 222, 333, 444, 555)

and combines them into a single 10-seed paired t-test / Wilcoxon /
Cohen's d for each model family, for both the Stable-vs-Sensitive and
Stable-vs-Middle comparisons.

Run from project root:
    python paper2\\pilot\\combine_h3_lodo3_seeds.py
"""

import json
import os
import numpy as np
from scipy import stats

PAPER2_ROOT = r"C:\MLProject\zero_day_project\paper2"
PILOT_DIR = os.path.join(PAPER2_ROOT, "pilot")

FILE_A = os.path.join(PILOT_DIR, "ablation_h3_lodo3_summary.json")
FILE_B = os.path.join(PILOT_DIR, "ablation_h3_lodo3_summary_extraseeds.json")

MODEL_FAMILIES = ["ae", "dae", "svdd"]


def load_results(path):
    if not os.path.exists(path):
        print(f"WARNING: {path} not found.")
        return []
    with open(path) as f:
        return json.load(f)


def main():
    results_a = load_results(FILE_A)
    results_b = load_results(FILE_B)
    all_results = results_a + results_b

    print(f"Loaded {len(results_a)} records from batch A (original 5 seeds)")
    print(f"Loaded {len(results_b)} records from batch B (extra 5 seeds)")
    print(f"Combined: {len(all_results)} records total\n")

    combined_stats = {}

    for family in MODEL_FAMILIES:
        fam_stable = {r["seed"]: r["delta_from_full"] for r in all_results
                      if r["family"] == family and r["condition"] == "stable_lodo3"}
        fam_sensitive = {r["seed"]: r["delta_from_full"] for r in all_results
                          if r["family"] == family and r["condition"] == "sensitive_lodo3"}
        fam_middle = {r["seed"]: r["delta_from_full"] for r in all_results
                      if r["family"] == family and r["condition"] == "middle_lodo3"}

        seeds_common = sorted(set(fam_stable) & set(fam_sensitive) & set(fam_middle))
        print(f"{'=' * 70}\n{family.upper()} -- {len(seeds_common)} seeds: {seeds_common}\n{'=' * 70}")

        if len(seeds_common) < 2:
            print(f"  Not enough seeds for a paired test.\n")
            continue

        stable_vals = np.array([fam_stable[s] for s in seeds_common])
        sensitive_vals = np.array([fam_sensitive[s] for s in seeds_common])
        middle_vals = np.array([fam_middle[s] for s in seeds_common])

        print(f"  delta_stable:    mean={stable_vals.mean():.4f}, std={stable_vals.std(ddof=1):.4f}")
        print(f"  delta_sensitive: mean={sensitive_vals.mean():.4f}, std={sensitive_vals.std(ddof=1):.4f}")
        print(f"  delta_middle:    mean={middle_vals.mean():.4f}, std={middle_vals.std(ddof=1):.4f}")

        family_stats = {"n_seeds": len(seeds_common), "seeds": seeds_common}
        for label, comparator_vals in [("stable_vs_sensitive", sensitive_vals),
                                        ("stable_vs_middle", middle_vals)]:
            diffs = stable_vals - comparator_vals
            n_stable_better = int(np.sum(diffs < 0))
            t_stat, t_p = stats.ttest_rel(stable_vals, comparator_vals)
            try:
                w_stat, w_p = stats.wilcoxon(stable_vals, comparator_vals)
            except ValueError:
                w_stat, w_p = None, None
            d_std = diffs.std(ddof=1)
            cohens_d = float(diffs.mean() / d_std) if d_std > 0 else float("nan")

            if t_p < 0.05 and n_stable_better == len(seeds_common):
                outcome = "Supported (stable better)"
            elif t_p < 0.05 and n_stable_better == 0:
                outcome = "Significantly REVERSED (comparator better)"
            elif n_stable_better == len(seeds_common):
                outcome = "Suggestive (stable better)"
            elif n_stable_better == 0:
                outcome = "Suggestive reversal (comparator better)"
            else:
                outcome = "Mixed / not supported"

            print(f"  [{label}] stable better in {n_stable_better}/{len(seeds_common)} seeds, "
                  f"paired t={t_stat:.3f}, p={t_p:.4f}, "
                  f"Wilcoxon p={'n/a' if w_p is None else f'{w_p:.4f}'}, "
                  f"Cohen's d={cohens_d:.3f} -> {outcome}")

            family_stats[label] = {
                "n_stable_better": n_stable_better, "n_seeds": len(seeds_common),
                "t_stat": float(t_stat), "p_value": float(t_p),
                "wilcoxon_p": None if w_p is None else float(w_p),
                "cohens_d": cohens_d, "outcome": outcome,
            }
        combined_stats[family] = family_stats
        print()

    out_path = os.path.join(PILOT_DIR, "ablation_h3_lodo3_combined_10seed_stats.json")
    with open(out_path, "w") as f:
        json.dump(combined_stats, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
