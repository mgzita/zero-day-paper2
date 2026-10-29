"""
Reviewer comment #7: is the architecture-by-target interaction still there, and how fragile is it to
(a) the definition of the Stable / Sensitive groups and (b) the independence assumption?

Outcome per model x target x seed = the permutation reliance gap from permute_two_source_groups
(reliance on Stable minus reliance on Sensitive; positive = leans on Stable), computed twice:
    paper groups     (ESS from all three environments, including the held-out one)
    two-source groups (ESS from the two training environments only)
Model:  gap ~ C(model) * C(target), cluster-robust Wald F on the 6 interaction terms (4 models x 3 targets).
Three ways of treating dependence between rows are shown side by side:
    A. cluster by model-seed for UNSW and IEC-104 (they share one training pool; SMD rows each their own
       cluster) -- the manuscript's choice
    B. cluster by seed across ALL rows (same seed -> same random stream in every dataset)
    C. no clustering (rows treated as independent)
This does not retrain anything; it reads <PILOT>\\permute_two_source_cache (written by
permute_two_source_groups.py for all 12 cells).

Run from the repo folder:  python -u interaction_two_source.py
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

SEEDS = [42, 123, 456, 789, 999, 111, 222, 333, 444, 555]
DATASETS = ["unsw", "iec104", "smd"]
FAMILIES = ["ae", "dae", "svdd", "flow"]


def load(pilot):
    rows = []
    for ds in DATASETS:
        for fam in FAMILIES:
            for s in SEEDS:
                p = os.path.join(pilot, "permute_two_source_cache", f"{ds}_{fam}_seed{s}.json")
                if not os.path.exists(p):
                    continue
                with open(p) as f:
                    r = json.load(f)
                gp = lambda k, m: r[k]["sensitive"][m] - r[k]["stable"][m]      # = reliance(Stable) - reliance(Sensitive)
                rows.append({"target": ds, "model": fam, "seed": s,
                             "gap_paper": gp("paper", "permute"), "gap_two": gp("two_source", "permute"),
                             "gap_paper_mask": gp("paper", "mask"), "gap_two_mask": gp("two_source", "mask")})
    return pd.DataFrame(rows)


def interaction_test(df, y, cluster):
    fit = smf.ols(f"{y} ~ C(model) * C(target)", data=df)
    res = fit.fit() if cluster is None else fit.fit(cov_type="cluster", cov_kwds={"groups": cluster})
    names = list(res.params.index)
    sel = [i for i, n in enumerate(names) if ":" in n]
    R = np.zeros((len(sel), len(names)))
    for k, i in enumerate(sel):
        R[k, i] = 1.0
    w = res.wald_test(R, use_f=True, scalar=True)
    return float(w.statistic), float(w.pvalue), int(w.df_num), (int(w.df_denom) if w.df_denom is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-dir", default=r"C:\MLProject\zero_day_project\paper2\pilot")
    args = ap.parse_args()
    df = load(args.pilot_dir)
    n_cells = df.groupby(["model", "target"]).ngroups
    print(f"{len(df)} rows, {n_cells} model x target cells (expect 120 rows, 12 cells)")
    if n_cells < 12:
        print("WARNING: not all cells present; run permute_two_source_groups.py for every dataset and family first.")

    print("\nMean reliance gap per cell (positive = leans on Stable):")
    for y, lab in (("gap_paper", "paper groups"), ("gap_two", "two-source groups")):
        t = df.pivot_table(index="model", columns="target", values=y, aggfunc="mean")[[c for c in DATASETS if c in df.target.unique()]]
        print(f"\n  {lab}\n" + t.round(3).to_string())

    df["cl_A"] = np.where(df.target.isin(["unsw", "iec104"]), df.model + "-" + df.seed.astype(str),
                          df.target + "-" + df.model + "-" + df.seed.astype(str))
    df["cl_B"] = df.seed.astype(str)
    out = {}
    print("\n" + "=" * 100 + "\nInteraction  model x target  (Wald F on the interaction terms)\n" + "=" * 100)
    for y, lab in (("gap_paper", "paper groups, permutation"), ("gap_two", "two-source groups, permutation"),
                   ("gap_paper_mask", "paper groups, mean-mask"), ("gap_two_mask", "two-source groups, mean-mask")):
        print(f"\n {lab}")
        for tag, cl, desc in (("A", df.cl_A, "cluster model-seed (UNSW/IEC-104 share a pool)"),
                              ("B", df.cl_B, "cluster by seed across all rows"),
                              ("C", None, "no clustering")):
            F, p, d1, d2 = interaction_test(df, y, None if cl is None else cl.values)
            out[f"{y}_{tag}"] = {"F": F, "p": p, "df_num": d1, "df_den": d2}
            print(f"   {tag}. {desc:<50} F({d1},{d2}) = {F:7.3f}   p = {p:.5f}")
    path = os.path.join(args.pilot_dir, "interaction_two_source.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
