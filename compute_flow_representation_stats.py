"""
Compute paired statistics on raw MMD^2 values for the representation-
level analysis, both target domains (UNSW-NB15, IEC-104).
=========================================================================
Reads the raw mmd_single/mmd_lodo3 values already saved by
flow_full_pipeline.py and flow_full_pipeline_iec104.py, and computes a
proper paired t-test, Wilcoxon signed-rank test, and Cohen's d -- rather
than just the binary "narrowed: Yes/No" flag used in the console output.

Run from project root:
    python paper2\\pilot\\compute_flow_representation_stats.py
"""

import json
import os
import numpy as np
from scipy import stats

PAPER2_ROOT = r"C:\MLProject\zero_day_project\paper2"
PILOT_DIR = os.path.join(PAPER2_ROOT, "pilot")

FILES = {
    "UNSW-NB15": os.path.join(PILOT_DIR, "flow_representation.json"),
    "IEC-104": os.path.join(PILOT_DIR, "flow_representation_iec104.json"),
}


def analyze(name, path):
    print(f"{'=' * 70}\n{name}\n{'=' * 70}")
    if not os.path.exists(path):
        print(f"  NOT FOUND at {path}")
        return None

    with open(path) as f:
        results = json.load(f)

    mmd_single = np.array([r["mmd_single"] for r in results])
    mmd_lodo3 = np.array([r["mmd_lodo3"] for r in results])
    n = len(mmd_single)

    print(f"  n = {n} seeds")
    print(f"  MMD^2 Single: mean={mmd_single.mean():.4f}, std={mmd_single.std(ddof=1):.4f}")
    print(f"  MMD^2 LODO-3: mean={mmd_lodo3.mean():.4f}, std={mmd_lodo3.std(ddof=1):.4f}")

    diffs = mmd_single - mmd_lodo3
    n_narrowed = int(np.sum(diffs > 0))
    print(f"  Narrowed in {n_narrowed}/{n} seeds")

    t_stat, t_p = stats.ttest_rel(mmd_single, mmd_lodo3)
    w_p = None
    try:
        w_stat, w_p = stats.wilcoxon(mmd_single, mmd_lodo3)
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
    print(f"  Cohen's d (paired): {cohens_d:.3f}")
    print(f"  Outcome: {outcome}\n")

    return {
        "n": n, "mmd_single_mean": float(mmd_single.mean()), "mmd_lodo3_mean": float(mmd_lodo3.mean()),
        "n_narrowed": n_narrowed, "t_stat": float(t_stat), "p_value": float(t_p),
        "wilcoxon_p": None if w_p is None else float(w_p), "cohens_d": cohens_d, "outcome": outcome,
    }


def main():
    summary = {}
    for name, path in FILES.items():
        result = analyze(name, path)
        if result:
            summary[name] = result

    out_path = os.path.join(PILOT_DIR, "flow_representation_paired_stats.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
