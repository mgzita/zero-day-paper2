"""
Extract the AUROC (Single vs Diverse/LODO-3) values already computed and
saved inside the raw H1 result files, and compute paired statistics.
=========================================================================
Every H1 script in this project computed and saved auroc_single /
auroc_lodo3 (or auroc_diverse) alongside the ESFRP attribution numbers.
This script pulls it out to establish, using this paper's own data
(not Paper 1, which is unpublished), that training-environment
diversity improves detection performance, for all 4 model families
across all 3 target domains.

Attempts to load from the RAW per-seed H1 files. If any expected file
is missing, prints a clear warning rather than silently omitting that row.

Run from project root:
    python paper2\\pilot\\extract_baseline_auroc.py
"""

import json
import os
import numpy as np
from scipy import stats

PAPER2_ROOT = r"C:\MLProject\zero_day_project\paper2"
PILOT_DIR = os.path.join(PAPER2_ROOT, "pilot")

SOURCES = {
    ("UNSW", "AE"):    ["unsw_lodo3_revalidation_summary.json", "ae_dae_svdd_h1_unsw_extra5seeds.json"],
    ("UNSW", "DAE"):   ["unsw_lodo3_revalidation_summary.json", "ae_dae_svdd_h1_unsw_extra5seeds.json"],
    ("UNSW", "SVDD"):  ["unsw_lodo3_revalidation_summary.json", "ae_dae_svdd_h1_unsw_extra5seeds.json"],
    ("UNSW", "Flow"):  ["flow_h1_attribution.json"],
    ("IEC-104", "AE"):    ["ae_dae_svdd_h1_iec104.json"],
    ("IEC-104", "DAE"):   ["ae_dae_svdd_h1_iec104.json"],
    ("IEC-104", "SVDD"):  ["ae_dae_svdd_h1_iec104.json"],
    ("IEC-104", "Flow"):  ["flow_h1_attribution_iec104.json"],
    ("SMD", "AE"):    ["ae_dae_svdd_h1_smd.json"],
    ("SMD", "DAE"):   ["ae_dae_svdd_h1_smd.json"],
    ("SMD", "SVDD"):  ["ae_dae_svdd_h1_smd.json"],
    ("SMD", "Flow"):  ["flow_h1_attribution_smd.json"],
}

ROW_ORDER = [
    ("UNSW", "AE"), ("UNSW", "DAE"), ("UNSW", "SVDD"), ("UNSW", "Flow"),
    ("IEC-104", "AE"), ("IEC-104", "DAE"), ("IEC-104", "SVDD"), ("IEC-104", "Flow"),
    ("SMD", "AE"), ("SMD", "DAE"), ("SMD", "SVDD"), ("SMD", "Flow"),
]

MODEL_KEY_MAP = {"AE": "ae", "DAE": "dae", "SVDD": "svdd", "Flow": "flow"}


def safe_load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def get_auroc_pair(record):
    single = record.get("auroc_single")
    diverse = record.get("auroc_lodo3", record.get("auroc_diverse"))
    return single, diverse


def collect_records(target, model, filenames):
    all_records = []
    family_key = MODEL_KEY_MAP[model]
    for fname in filenames:
        path = os.path.join(PILOT_DIR, fname)
        data = safe_load(path)
        if data is None:
            print(f"    MISSING: {fname}")
            continue
        for r in data:
            if "family" in r and r["family"] != family_key:
                continue
            single, diverse = get_auroc_pair(r)
            if single is not None and diverse is not None:
                all_records.append((single, diverse))
    return all_records


def main():
    print(f"{'=' * 90}\nBASELINE: does diversity training improve detection AUROC? "
          f"(this paper's own data, not Paper 1)\n{'=' * 90}\n")

    results = {}
    for target, model in ROW_ORDER:
        print(f"{target}, {model}:")
        filenames = SOURCES[(target, model)]
        records = collect_records(target, model, filenames)
        n = len(records)
        if n < 2:
            print(f"    Only {n} usable seed(s) found -- cannot compute paired statistics.\n")
            continue

        singles = np.array([s for s, d in records])
        diverses = np.array([d for s, d in records])
        diffs = diverses - singles
        t, p = stats.ttest_rel(diverses, singles)
        d_std = diffs.std(ddof=1)
        cohens_d = float(diffs.mean() / d_std) if d_std > 0 else float("nan")

        print(f"    n={n}, Single AUROC={singles.mean():.4f}, Diverse AUROC={diverses.mean():.4f}, "
              f"mean improvement={diffs.mean():+.4f}, t={t:.3f}, p={p:.5f}, d={cohens_d:+.3f}\n")

        results[(target, model)] = {
            "n": n, "single_mean": float(singles.mean()), "diverse_mean": float(diverses.mean()),
            "improvement": float(diffs.mean()), "t": float(t), "p": float(p), "d": cohens_d,
        }

    out_path = os.path.join(PILOT_DIR, "baseline_auroc_diversity_effect.json")
    with open(out_path, "w") as f:
        json.dump({f"{t}_{m}": v for (t, m), v in results.items()}, f, indent=2)
    print(f"{'=' * 90}\nSaved {out_path} ({len(results)}/12 combinations computed)\n{'=' * 90}")


if __name__ == "__main__":
    main()
