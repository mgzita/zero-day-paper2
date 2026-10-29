"""
Re-analyze H3 functional ablation using raw AUROC, not |AUROC-0.5|
====================================================================
Reviewer feedback (major comment #3): |AUROC-0.5| treats AUROC=0.1 and
AUROC=0.9 as equivalent, despite opposite detection behaviour (one means
"detects attacks very well", the other means "the score is systematically
inverted"). The manuscript's H3 ablation conclusions (Stable vs Sensitive
vs Middle feature-group importance) were built entirely on this symmetric
separation metric.

THIS SCRIPT ONLY COVERS UNSW-NB15 / AE+DAE+SVDD (ablation_h3_lodo3.py +
its extra-seeds batch), because that is the ONLY existing H3 result file
that saved raw "auroc" per condition. Checked directly against your repo:

  - ablation_h3_lodo3.py                    -> saves "auroc" per record. OK.
  - ablation_h3_lodo3_extra_seeds.py         -> saves "auroc" per record. OK.
  - ae_dae_svdd_pipeline_iec104.py (H3 part) -> computes auroc_full/stable/
    sensitive/middle locally but returns ONLY delta_stable/delta_sensitive/
    delta_middle. Raw AUROC is NOT in the saved JSON.
  - ae_dae_svdd_pipeline_smd.py (H3 part)    -> same gap as IEC-104.
  - flow_full_pipeline_iec104.py h3_ablation()  -> same gap.
  - flow_full_pipeline_smd.py h3_ablation()     -> same gap.
  - flow_full_pipeline.py (UNSW Flow H3, if present) -> check the same way;
    not confirmed here.

So this script gets you the UNSW/AE/DAE/SVDD raw-AUROC reanalysis for
free, right now, with NO retraining. For IEC-104, SMD, and the Flow's own
H3 results, the raw AUROC values were never persisted, so getting them
needs each of those four scripts' h3-ablation function to additionally
return (and the caller to additionally save) auroc_full / auroc_stable /
auroc_sensitive / auroc_middle, followed by a rerun of just the H3 part
(ablation training is the cheap part of each pipeline -- a few minutes
per seed/family on CPU, not the slow step). A two-line patch per script:
add those four keys to the returned dict, and the cache/summary write-out
already captures everything else. I did not make that edit myself since
I don't have your live file to patch and verify against -- happy to draft
the exact diff for each of the four scripts if you paste back their current
h3-ablation function/return statement.

AUPRC is not computable from any of the existing saved files (none of them
save per-sample predicted scores or labels, only the summary AUROC) -- that
would need those four scripts to also save y_true/scores, not just the
scalar AUROC.

Run from the repo root (same place as ablation_h3_lodo3.py):
    python reanalyze_h3_raw_auroc_unsw.py
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


def paired_test(vals_a, vals_b, label):
    vals_a, vals_b = np.array(vals_a), np.array(vals_b)
    diffs = vals_a - vals_b
    n = len(vals_a)
    n_a_greater = int(np.sum(diffs > 0))
    t, p = stats.ttest_rel(vals_a, vals_b)
    wp = None
    try:
        _, wp = stats.wilcoxon(vals_a, vals_b)
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
    all_results = load_results(FILE_A) + load_results(FILE_B)
    print(f"Loaded {len(all_results)} total H3 records (UNSW, AE/DAE/SVDD)\n")

    summary = {}
    for family in MODEL_FAMILIES:
        print(f"{'=' * 70}\n{family.upper()}\n{'=' * 70}")

        def by_seed(condition):
            return {r["seed"]: r["auroc"] for r in all_results
                    if r.get("family") == family and r.get("condition") == condition
                    and "auroc" in r}

        auroc_full = by_seed("full_lodo3")
        auroc_stable = by_seed("stable_lodo3")
        auroc_sensitive = by_seed("sensitive_lodo3")
        auroc_middle = by_seed("middle_lodo3")

        seeds = sorted(set(auroc_full) & set(auroc_stable) & set(auroc_sensitive) & set(auroc_middle))
        if len(seeds) < 2:
            print(f"  Only {len(seeds)} seed(s) with raw AUROC on all four conditions -- skipping.")
            continue

        print(f"  {len(seeds)} seeds: {seeds}")
        print(f"  Raw AUROC means -- full={np.mean([auroc_full[s] for s in seeds]):.4f}, "
              f"stable={np.mean([auroc_stable[s] for s in seeds]):.4f}, "
              f"sensitive={np.mean([auroc_sensitive[s] for s in seeds]):.4f}, "
              f"middle={np.mean([auroc_middle[s] for s in seeds]):.4f}")

        fam_summary = {"n_seeds": len(seeds), "seeds": seeds}
        fam_summary["raw_auroc_means"] = {
            "full": float(np.mean([auroc_full[s] for s in seeds])),
            "stable": float(np.mean([auroc_stable[s] for s in seeds])),
            "sensitive": float(np.mean([auroc_sensitive[s] for s in seeds])),
            "middle": float(np.mean([auroc_middle[s] for s in seeds])),
        }

        print("\n  -- Comparison on RAW AUROC (not |AUROC-0.5|) --")
        stable_vals = [auroc_stable[s] for s in seeds]
        sensitive_vals = [auroc_sensitive[s] for s in seeds]
        middle_vals = [auroc_middle[s] for s in seeds]
        fam_summary["raw_auroc_stable_vs_sensitive"] = paired_test(
            stable_vals, sensitive_vals, "raw_auroc_stable vs raw_auroc_sensitive")
        fam_summary["raw_auroc_stable_vs_middle"] = paired_test(
            stable_vals, middle_vals, "raw_auroc_stable vs raw_auroc_middle")

        # Also report the ORIGINAL |AUROC-0.5|-based comparison for the
        # same seeds, side by side, so you can see directly whether the
        # metric choice changes the conclusion.
        def by_seed_delta(condition):
            return {r["seed"]: r["delta_from_full"] for r in all_results
                    if r.get("family") == family and r.get("condition") == condition
                    and "delta_from_full" in r}

        delta_stable = by_seed_delta("stable_lodo3")
        delta_sensitive = by_seed_delta("sensitive_lodo3")
        delta_middle = by_seed_delta("middle_lodo3")
        print("\n  -- Original comparison on delta_from_full (|AUROC-0.5|-based), same seeds --")
        fam_summary["original_delta_stable_vs_sensitive"] = paired_test(
            [delta_stable[s] for s in seeds], [delta_sensitive[s] for s in seeds],
            "delta_stable vs delta_sensitive")
        fam_summary["original_delta_stable_vs_middle"] = paired_test(
            [delta_stable[s] for s in seeds], [delta_middle[s] for s in seeds],
            "delta_stable vs delta_middle")

        summary[family] = fam_summary
        print()

    out_path = os.path.join(PILOT_DIR, "h3_raw_auroc_reanalysis_unsw.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {out_path}")
    print("\nNOTE: this covers UNSW/AE/DAE/SVDD only -- see the module docstring for what's")
    print("still needed for IEC-104, SMD, and the Flow's own H3 results.")


if __name__ == "__main__":
    main()
