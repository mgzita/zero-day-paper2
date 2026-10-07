"""
Fallback-scorer usage tally, per target domain
=================================================
Round-3 reviewer critique, item 1: Table 3's caption still contains the
placeholder "[fallback-use rate to be inserted per target domain]".

Table 3 documents that the Normalizing Flow's KernelSHAP attribution falls
back to a log(1 + NLL)-transformed scorer whenever the raw-score additivity
check fails under severe distribution shift (see flow_full_pipeline.py /
flow_full_pipeline_iec104.py / flow_full_pipeline_smd.py, ~line 380-410:
each per-seed H1 record is written with a "used_log_transform" boolean).

This script just tallies that boolean, per target, across the three
Normalizing Flow H1-attribution result files:
    flow_h1_attribution.json          (UNSW-NB15)
    flow_h1_attribution_iec104.json   (IEC-104)
    flow_h1_attribution_smd.json      (SMD)

No new computation, no new data -- it only counts a field that is already
being written to disk by the existing pipeline.

Run from the same folder as those three JSON files (PILOT_DIR):
    python fallback_usage_tally.py
"""

import json
import os

import paper2_followup_tests as pft

PILOT_DIR = pft.PILOT_DIR

FILES = {
    "UNSW-NB15": "flow_h1_attribution.json",
    "IEC-104": "flow_h1_attribution_iec104.json",
    "SMD": "flow_h1_attribution_smd.json",
}


def safe_load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def tally():
    print(f"{'=' * 70}\nFallback-scorer (log(1+NLL)) usage, Normalizing Flow, by target\n{'=' * 70}")
    results = {}
    for target, fname in FILES.items():
        path = os.path.join(PILOT_DIR, fname)
        recs = safe_load(path)
        if recs is None:
            print(f"  {target:10s}: FILE NOT FOUND ({fname}) -- skipped")
            results[target] = None
            continue
        n = len(recs)
        used = [r for r in recs if r.get("used_log_transform")]
        n_used = len(used)
        seeds_used = sorted(r.get("seed") for r in used if "seed" in r)
        print(f"  {target:10s}: {n_used}/{n} seeds used the fallback scorer"
              + (f"  (seeds: {seeds_used})" if seeds_used else ""))
        results[target] = {"n_total": n, "n_fallback": n_used, "seeds_fallback": seeds_used}

    print(f"\n{'=' * 70}\nSuggested Table 3 caption sentence\n{'=' * 70}")
    parts = []
    for target in FILES:
        r = results.get(target)
        if r is None:
            parts.append(f"{target}: [file not found]")
        else:
            parts.append(f"{r['n_fallback']}/{r['n_total']} on {target}")
    print("The fallback scorer was required for " + ", ".join(parts) + ".")

    out_path = os.path.join(PILOT_DIR, "fallback_usage_tally.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {out_path}")
    return results


if __name__ == "__main__":
    tally()
