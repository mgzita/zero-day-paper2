"""
Compute paired statistics for the Integrated Gradients H1 results
(ig_attribution_all_results.json), across all 4 models x 3 datasets x
10 seeds.

Run from project root:
    python paper2\\pilot\\analyze_ig_results.py
"""

import json
import os
import numpy as np
from scipy import stats

PAPER2_ROOT = r"C:\MLProject\zero_day_project\paper2"
PILOT_DIR = os.path.join(PAPER2_ROOT, "pilot")

DATASETS = ["unsw", "iec104", "smd"]
MODEL_FAMILIES = ["ae", "dae", "svdd", "flow"]


def safe_load(path):
    if not os.path.exists(path):
        print(f"MISSING: {path}")
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
    all_results = safe_load(os.path.join(PILOT_DIR, "ig_attribution_all_results.json"))
    if all_results is None:
        return

    print(f"Total records loaded: {len(all_results)}\n")

    summary = {}
    for dataset in DATASETS:
        print(f"{'=' * 70}\n{dataset.upper()}\n{'=' * 70}")
        for family in MODEL_FAMILIES:
            recs = [r for r in all_results if r.get("dataset") == dataset and r.get("family") == family]
            n_total = len(recs)
            n_completeness_ok = sum(1 for r in recs if r.get("completeness_ok"))
            n_dir_correct = sum(1 for r in recs if r.get("direction_correct"))

            print(f"\n  {family.upper()}: {n_total} seeds, "
                  f"{n_completeness_ok}/{n_total} passed completeness check, "
                  f"{n_dir_correct}/{n_total} direction correct")

            if n_total >= 2:
                result = paired_test([r["esfrp_b"] for r in recs],
                                      [r["esfrp_a"] for r in recs],
                                      "ESFRP(b=diverse/lodo3) vs ESFRP(a=single)")
                result["n_completeness_ok"] = n_completeness_ok
                result["n_direction_correct"] = n_dir_correct
                summary[f"{dataset}_{family}"] = result

    out_path = os.path.join(PILOT_DIR, "ig_attribution_paired_stats.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{'=' * 70}\nSaved {out_path}\n{'=' * 70}")

    print(f"\n{'=' * 70}\nQUICK SIGNIFICANCE SUMMARY (raw p < 0.05, before FDR correction)\n{'=' * 70}")
    for key, r in sorted(summary.items(), key=lambda x: x[1]["p"]):
        sig = "***" if r["p"] < 0.05 else "   "
        print(f"  {sig} {key:20s} p={r['p']:.5f}  d={r['cohens_d']:+.3f}  "
              f"completeness_ok={r['n_completeness_ok']}/{r['n']}")


if __name__ == "__main__":
    main()
