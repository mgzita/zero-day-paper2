"""
Raw-AUROC reanalysis of H3 (Stable vs Sensitive vs Middle) -- IEC-104, SMD
(AE / DAE / SVDD), plus any Flow H3 files that contain raw auroc_* fields.

Addresses reviewer comment #3: |AUROC-0.5| treats AUROC=0.1 and 0.9 as the
same. For every dataset/family this reports, side by side:
  - mean raw AUROC per group and how many seeds fall BELOW 0.5 (inverted)
  - paired tests (t-test + Wilcoxon, Cohen's d) on RAW AUROC
  - the same tests on the old |AUROC-0.5| separation, for comparison
  - whether the conclusion (which group is best) changes between the two

Run:  python reanalyze_h3_raw_auroc_all.py
"""
import json
import os

import numpy as np
from scipy import stats

PILOT_DIR = r"C:\MLProject\zero_day_project\paper2\pilot"
FILES = {
    "IEC-104 (AE/DAE/SVDD)": "ae_dae_svdd_h3_iec104.json",
    "SMD (AE/DAE/SVDD)": "ae_dae_svdd_h3_smd.json",
    "IEC-104 (Flow)": "flow_h3_ablation_iec104.json",
    "SMD (Flow)": "flow_h3_ablation_smd.json",
}
GROUPS = ["stable", "sensitive", "middle"]


def paired(a, b):
    a, b = np.asarray(a), np.asarray(b)
    d = a - b
    t, p = stats.ttest_rel(a, b)
    try:
        wp = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        wp = float("nan")
    sd = d.std(ddof=1)
    return {"mean_diff": float(d.mean()), "t_p": float(p), "wilcoxon_p": wp,
            "cohens_d": float(d.mean() / sd) if sd > 0 else float("nan"),
            "a_greater": f"{int((d > 0).sum())}/{len(d)}"}


def fmt(r):
    return (f"diff={r['mean_diff']:+.3f}  t-p={r['t_p']:.4f}  "
            f"wilcoxon-p={r['wilcoxon_p']:.4f}  d={r['cohens_d']:+.2f}  "
            f"first>second in {r['a_greater']}")


def analyse(name, records):
    print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")
    summary = {}
    families = sorted({r.get("family", "flow") for r in records})
    for fam in families:
        rows = [r for r in records if r.get("family", "flow") == fam]
        need = ["auroc_full"] + [f"auroc_{g}" for g in GROUPS]
        rows = [r for r in rows if all(k in r for k in need)]
        if len(rows) < 3:
            print(f"\n[{fam}] fewer than 3 records with raw AUROC -- skipped.")
            continue
        raw = {g: np.array([r[f"auroc_{g}"] for r in rows]) for g in ["full"] + GROUPS}
        sep = {g: np.abs(v - 0.5) for g, v in raw.items()}
        print(f"\n[{fam}]  n={len(rows)} seeds")
        for g in ["full"] + GROUPS:
            print(f"  {g:>9}: raw AUROC mean={raw[g].mean():.3f} sd={raw[g].std(ddof=1):.3f} "
                  f"| below 0.5 in {(raw[g] < 0.5).sum()}/{len(rows)} seeds "
                  f"| |AUROC-0.5| mean={sep[g].mean():.3f}")
        fam_out = {"n": len(rows)}
        for other in ("sensitive", "middle"):
            r_raw = paired(raw["stable"], raw[other])
            r_sep = paired(sep["stable"], sep[other])
            print(f"  stable vs {other}")
            print(f"    RAW AUROC : {fmt(r_raw)}")
            print(f"    |AUC-0.5| : {fmt(r_sep)}")
            fam_out[f"stable_vs_{other}"] = {"raw": r_raw, "abs_sep": r_sep}
        best_raw = max(GROUPS, key=lambda g: raw[g].mean())
        best_sep = max(GROUPS, key=lambda g: sep[g].mean())
        flag = "" if best_raw == best_sep else "   <-- CONCLUSION CHANGES"
        print(f"  best group: raw={best_raw}, |AUC-0.5|={best_sep}{flag}")
        fam_out["best_raw"], fam_out["best_abs_sep"] = best_raw, best_sep
        summary[fam] = fam_out
    return summary


def main():
    out = {}
    for name, fname in FILES.items():
        path = os.path.join(PILOT_DIR, fname)
        if not os.path.exists(path):
            print(f"\n[{name}] {path} not found -- skipped.")
            continue
        with open(path) as f:
            data = json.load(f)
        out[name] = analyse(name, data)
    out_path = os.path.join(PILOT_DIR, "h3_raw_auroc_reanalysis_all.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
