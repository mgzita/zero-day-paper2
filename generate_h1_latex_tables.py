"""
Generate the two H1 LaTeX tables (Table h1_shap, Table h1_ig) directly
from saved result files -- fills every [fill] placeholder automatically,
so no number is ever retyped by hand.
=========================================================================
Pulls from:
  AE/DAE/SVDD, UNSW+IEC-104  -> final_combined_ae_dae_svdd_stats.json
  AE/DAE/SVDD, SMD           -> smd_final_stats.json
  Flow, UNSW+IEC-104         -> computed LIVE from raw per-seed files
  Flow, SMD                  -> flow_smd_final_stats.json
  IG, all 12 combinations    -> ig_attribution_paired_stats.json

Prints two ready-to-paste LaTeX tabular bodies (SHAP, then IG), plus
prose summary numbers (which SHAP rows are significant, largest IG p).

Run from project root:
    python paper2\\pilot\\generate_h1_latex_tables.py
"""

import json
import os
import numpy as np
from scipy import stats

PAPER2_ROOT = r"C:\MLProject\zero_day_project\paper2"
PILOT_DIR = os.path.join(PAPER2_ROOT, "pilot")

ROW_ORDER = [
    ("UNSW", "AE"), ("UNSW", "DAE"), ("UNSW", "SVDD"), ("UNSW", "Flow"),
    ("IEC-104", "AE"), ("IEC-104", "DAE"), ("IEC-104", "SVDD"), ("IEC-104", "Flow"),
    ("SMD", "AE"), ("SMD", "DAE"), ("SMD", "SVDD"), ("SMD", "Flow"),
]

# Single source of truth for family-name -> display-label casing, used
# everywhere a raw family string (e.g. from a JSON key or dict) needs
# to be converted to the exact label used in ROW_ORDER. A bare
# fam.upper() previously mapped "flow" -> "FLOW" instead of "Flow",
# silently breaking the lookup against ROW_ORDER for all three Flow
# rows in the IG table -- this dict is the fix, and the single place
# to update if a new model family is ever added.
MODEL_LABELS = {"ae": "AE", "dae": "DAE", "svdd": "SVDD", "flow": "Flow"}


def safe_load(path):
    if not os.path.exists(path):
        print(f"  WARNING -- missing file: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def live_paired_test(vals_a, vals_b):
    vals_a, vals_b = np.array(vals_a), np.array(vals_b)
    n = len(vals_a)
    diffs = vals_a - vals_b
    t, p = stats.ttest_rel(vals_a, vals_b)
    d_std = diffs.std(ddof=1)
    d = float(diffs.mean() / d_std) if d_std > 0 else float("nan")
    return {"n": n, "single_mean": float(vals_b.mean()), "diverse_mean": float(vals_a.mean()),
            "p": float(p), "d": d}


def fmt_p(p):
    if p is None:
        return "--"
    if p < 0.001:
        return "$<0.001$"
    return f"${p:.3f}$"


def fmt_d(d):
    if d is None:
        return "--"
    return f"${d:+.2f}$"


def build_row(target, model, record):
    if record is None:
        return f"{target} & {model}   & -- & -- & -- & -- \\\\"
    single = record.get("single_mean")
    diverse = record.get("diverse_mean")
    p = record.get("p")
    d = record.get("d")
    single_str = f"${single:.3f}$" if single is not None else "--"
    diverse_str = f"${diverse:.3f}$" if diverse is not None else "--"
    return f"{target:8s} & {model:5s} & {single_str} & {diverse_str} & {fmt_p(p)} & {fmt_d(d)} \\\\"


def collect_shap_records():
    records = {}

    combined = safe_load(os.path.join(PILOT_DIR, "final_combined_ae_dae_svdd_stats.json"))
    if combined:
        for dataset_key, target_label in [("UNSW-NB15", "UNSW"), ("IEC-104", "IEC-104")]:
            h1_dict = combined.get(dataset_key, {}).get("h1", {})
            for fam, r in h1_dict.items():
                model_label = MODEL_LABELS.get(fam, fam.upper())
                records[(target_label, model_label)] = {
                    "single_mean": r.get("mean_b"), "diverse_mean": r.get("mean_a"),
                    "p": r.get("p"), "d": r.get("cohens_d"),
                }

    smd = safe_load(os.path.join(PILOT_DIR, "smd_final_stats.json"))
    if smd:
        h1_dict = smd.get("h1", {})
        for fam, r in h1_dict.items():
            model_label = MODEL_LABELS.get(fam, fam.upper())
            records[("SMD", model_label)] = {
                "single_mean": r.get("mean_b"), "diverse_mean": r.get("mean_a"),
                "p": r.get("p"), "d": r.get("cohens_d"),
            }

    for target_label, h1_file in [("UNSW", "flow_h1_attribution.json"),
                                   ("IEC-104", "flow_h1_attribution_iec104.json")]:
        h1 = safe_load(os.path.join(PILOT_DIR, h1_file))
        if h1:
            recs = [r for r in h1 if r.get("additivity_ok", True) and "esfrp_single" in r]
            if len(recs) >= 2:
                lr = live_paired_test([r["esfrp_lodo3"] for r in recs], [r["esfrp_single"] for r in recs])
                records[(target_label, "Flow")] = lr

    flow_smd = safe_load(os.path.join(PILOT_DIR, "flow_smd_final_stats.json"))
    if flow_smd:
        r = flow_smd.get("h1")
        if r:
            records[("SMD", "Flow")] = {
                "single_mean": r.get("mean_b"), "diverse_mean": r.get("mean_a"),
                "p": r.get("p"), "d": r.get("cohens_d"),
            }

    return records


def collect_ig_records():
    records = {}
    ig = safe_load(os.path.join(PILOT_DIR, "ig_attribution_paired_stats.json"))
    if not ig:
        return records
    key_to_target = {"unsw": "UNSW", "iec104": "IEC-104", "smd": "SMD"}
    for key, r in ig.items():
        dataset_part, fam = key.split("_", 1)
        target_label = key_to_target.get(dataset_part, dataset_part.upper())
        model_label = MODEL_LABELS.get(fam, fam.upper())
        records[(target_label, model_label)] = {
            "single_mean": r.get("mean_b"), "diverse_mean": r.get("mean_a"),
            "p": r.get("p"), "d": r.get("cohens_d"),
        }
    return records


def main():
    print(f"{'=' * 90}\nCollecting SHAP-based H1 records...\n{'=' * 90}")
    shap_records = collect_shap_records()
    print(f"Found {len(shap_records)}/12 SHAP records.\n")

    print(f"{'=' * 90}\nTable: H1, KernelSHAP  (paste directly into tabular body)\n{'=' * 90}")
    for target, model in ROW_ORDER:
        print(build_row(target, model, shap_records.get((target, model))))

    print(f"\n{'=' * 90}\nCollecting Integrated Gradients H1 records...\n{'=' * 90}")
    ig_records = collect_ig_records()
    print(f"Found {len(ig_records)}/12 IG records.\n")

    print(f"{'=' * 90}\nTable: H1, Integrated Gradients\n{'=' * 90}")
    for target, model in ROW_ORDER:
        print(build_row(target, model, ig_records.get((target, model))))

    print(f"\n{'=' * 90}\nPROSE SUMMARY (for the paragraph text)\n{'=' * 90}")
    shap_sig = [(t, m, r) for (t, m), r in shap_records.items() if r.get("p") is not None and r["p"] < 0.05]
    shap_sig.sort(key=lambda x: x[2]["p"])
    print(f"SHAP-significant rows (raw p<0.05), {len(shap_sig)} found:")
    for t, m, r in shap_sig:
        print(f"  {t}, {m}: p={r['p']:.4f}, d={r['d']:+.2f}")

    ig_all_p = [(t, m, r["p"]) for (t, m), r in ig_records.items() if r.get("p") is not None]
    if ig_all_p:
        ig_all_p.sort(key=lambda x: x[2])
        smallest = ig_all_p[0]
        print(f"\nSmallest (most significant) IG p-value across all 12: "
              f"{smallest[0]}, {smallest[1]}, p={smallest[2]:.4f}")
        n_ig_sig = sum(1 for _, _, p in ig_all_p if p < 0.05)
        print(f"IG combinations reaching raw p<0.05: {n_ig_sig}/12")


if __name__ == "__main__":
    main()
