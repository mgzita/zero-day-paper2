"""
Reviewer comment #7 (causation by diversity): does DIVERSITY change which features the models rely on?

permute_two_source_groups.py measured reliance in the Diverse (LODO-3) models only. Here the same
permutation / mean-mask test is run on the SINGLE-source models, with BOTH group definitions
    paper groups        (ESS from all three environments, including the held-out one)
    two-source groups   (ESS from the two training environments only)
and the diversity effect on reliance is tested directly:
    change = reliance gap (Diverse) - reliance gap (Single)       reliance gap = reliance(Stable) - reliance(Sensitive)
Positive change = adding the second source moved the model toward the Stable features.
Per cell: paired t-test across the 10 seeds (BH-corrected over the 12 cells, per group definition).
Across cells: model x target interaction on `change`, cluster-robust as in the manuscript (A), by seed (B), none (C).

Nothing is retrained. Needs, from earlier steps: <PILOT>\\permute_two_source_cache (LODO-3 results, both
definitions) and the saved Single checkpoints. Cached in <PILOT>\\permute_single_cache.

Run from the repo folder, one dataset at a time (Flow is the slow one):
    python -u permute_single_vs_diverse.py --dataset unsw   --families ae dae svdd
    python -u permute_single_vs_diverse.py --dataset iec104 --families ae dae svdd
    python -u permute_single_vs_diverse.py --dataset smd    --families ae dae svdd
    python -u permute_single_vs_diverse.py --dataset unsw   --families flow      (and iec104, smd)
Then:  python -u permute_single_vs_diverse.py --stats-only
"""
import argparse
import importlib
import json
import os

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score

try:
    import permute_two_source_groups_v2 as base
except ImportError:
    import permute_two_source_groups as base

SEEDS, DATASETS, FAMILIES, MODULES = base.SEEDS, base.DATASETS, base.FAMILIES, base.MODULES
DEFS = ["paper", "two_source"]


def score_fn(family, ae_mod, flow_mod, seed, condition):
    if family == "flow":
        scorer = flow_mod.NegativeLogLikelihoodScorer(
            flow_mod.load_flow_model(flow_mod.ensure_flow_checkpoint(seed, condition))).eval()
    else:
        scorer = ae_mod.get_scorer(family, ae_mod.load_model(family, ae_mod.ensure_checkpoint(seed, family, condition)))

    def f(X):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 50000):
                xb = torch.tensor(np.ascontiguousarray(X[i:i + 50000]), dtype=torch.float32)
                out.append(scorer(xb).numpy().flatten())
        return np.nan_to_num(np.concatenate(out), nan=1e10, posinf=1e10, neginf=-1e10)
    return f


def run_one(ds, fam, seed, X, y, ae_mod, flow_mod, base_mod, defs, pilot, n_rep):
    ref_p = os.path.join(pilot, "permute_two_source_cache", f"{ds}_{fam}_seed{seed}.json")
    if not os.path.exists(ref_p):
        return None                                              # LODO-3 result needed for the comparison
    f_model = score_fn(fam, ae_mod, flow_mod, seed, "single")
    base_auroc = float(roc_auc_score(y, f_model(X)))
    base_sep = abs(base_auroc - 0.5)
    idx = {n: i for i, n in enumerate(base_mod.FEATURE_NAMES)}
    X_tr, _ = base_mod.load_single_train_val()
    train_mean = X_tr.mean(axis=0)
    rng = np.random.default_rng(seed + 2000)

    def d_sep(Xm):
        return abs(float(roc_auc_score(y, f_model(Xm))) - 0.5) - base_sep

    rec = {"dataset": ds, "family": fam, "seed": seed, "base_auroc_single": base_auroc}
    for dname in DEFS:
        rec[dname] = {}
        for g in ("stable", "sensitive"):
            cols = [idx[n] for n in defs[dname][g]]
            reps = []
            for _ in range(n_rep):
                Xp = X.copy()
                Xp[:, cols] = X[rng.permutation(len(X))][:, cols]
                reps.append(d_sep(Xp))
            Xm = X.copy()
            Xm[:, cols] = train_mean[cols]
            rec[dname][g] = {"permute": float(np.mean(reps)), "mask": d_sep(Xm)}
    gap = lambda d: rec[d]["sensitive"]["permute"] - rec[d]["stable"]["permute"]
    print(f"  {ds}/{fam}/seed{seed}: Single gap paper={gap('paper'):+.3f} two-source={gap('two_source'):+.3f} "
          f"(base AUROC {base_auroc:.3f})", flush=True)
    return rec


def bh(ps):
    ps = np.array(ps, float)
    order = np.argsort(ps)
    adj, prev = np.empty_like(ps), 1.0
    for rank, i in zip(range(len(ps), 0, -1), order[::-1]):
        prev = min(prev, ps[i] * len(ps) / rank)
        adj[i] = prev
    return adj


def load_pairs(pilot):
    rows = []
    for ds in DATASETS:
        for fam in FAMILIES:
            for s in SEEDS:
                ps = os.path.join(pilot, "permute_single_cache", f"{ds}_{fam}_seed{s}.json")
                pd_ = os.path.join(pilot, "permute_two_source_cache", f"{ds}_{fam}_seed{s}.json")
                if not (os.path.exists(ps) and os.path.exists(pd_)):
                    continue
                with open(ps) as f:
                    a = json.load(f)
                with open(pd_) as f:
                    b = json.load(f)
                row = {"target": ds, "model": fam, "seed": s}
                for dname in DEFS:
                    for kind in ("permute", "mask"):
                        gs = a[dname]["sensitive"][kind] - a[dname]["stable"][kind]
                        gd = b[dname if dname == "two_source" else "paper"]["sensitive"][kind] - \
                            b[dname if dname == "two_source" else "paper"]["stable"][kind]
                        row[f"gs_{dname}_{kind}"], row[f"gd_{dname}_{kind}"] = gs, gd
                        row[f"chg_{dname}_{kind}"] = gd - gs
                    row[f"stab_single_{dname}"] = -a[dname]["stable"]["permute"]
                    row[f"stab_div_{dname}"] = -b[dname if dname == "two_source" else "paper"]["stable"]["permute"]
                rows.append(row)
    return pd.DataFrame(rows)


def interaction(df, y, cluster):
    import statsmodels.formula.api as smf
    fit = smf.ols(f"{y} ~ C(model) * C(target)", data=df)
    res = fit.fit() if cluster is None else fit.fit(cov_type="cluster", cov_kwds={"groups": cluster})
    names = list(res.params.index)
    sel = [i for i, n in enumerate(names) if ":" in n]
    R = np.zeros((len(sel), len(names)))
    for k, i in enumerate(sel):
        R[k, i] = 1.0
    w = res.wald_test(R, use_f=True, scalar=True)
    return float(w.statistic), float(w.pvalue), int(w.df_num), int(w.df_denom) if w.df_denom is not None else None


def stats_report(pilot):
    df = load_pairs(pilot)
    if df.empty:
        print("No paired Single + Diverse results yet.")
        return
    out = {"cells": [], "interaction": {}}
    for dname, lab in (("paper", "PAPER groups"), ("two_source", "TWO-SOURCE groups")):
        print("\n" + "=" * 112 + f"\n{lab}: reliance gap (Stable - Sensitive; positive = leans on Stable), Single vs Diverse (permutation)\n" + "=" * 112)
        rows = []
        for ds in DATASETS:
            for fam in FAMILIES:
                s = df[(df.target == ds) & (df.model == fam)]
                if len(s) < 4:
                    continue
                chg = s[f"chg_{dname}_permute"].values
                sd = chg.std(ddof=1)
                p = float(stats.ttest_1samp(chg, 0.0).pvalue)
                rows.append({"definition": dname, "dataset": ds, "family": fam, "n": len(s),
                             "gap_single": float(s[f"gs_{dname}_permute"].mean()), "gap_diverse": float(s[f"gd_{dname}_permute"].mean()),
                             "change": float(chg.mean()), "p": p, "d": float(chg.mean() / sd) if sd > 0 else float("nan"),
                             "stable_reliance_single": float(s[f"stab_single_{dname}"].mean()),
                             "stable_reliance_diverse": float(s[f"stab_div_{dname}"].mean()),
                             "change_mask": float(s[f"chg_{dname}_mask"].mean())})
        adj = bh([r["p"] for r in rows]) if rows else []
        for r, a in zip(rows, adj):
            r["bh_adj_p"] = float(a)
            print(f" {r['dataset']:>6}/{r['family']:<5} n={r['n']:>2} | gap Single {r['gap_single']:+.3f} -> Diverse {r['gap_diverse']:+.3f} | "
                  f"change {r['change']:+.3f} (d={r['d']:+.2f}, p={r['p']:.4f}, BH {a:.4f}) {'SIG' if a < 0.05 else ''}")
        print(f"  significant after BH: {sum(r['bh_adj_p'] < 0.05 for r in rows)}/{len(rows)}")
        out["cells"] += rows
    n_cells = df.groupby(["model", "target"]).ngroups
    print("\n" + "=" * 112 + "\nModel x target interaction on the CHANGE in reliance gap (Diverse - Single)\n" + "=" * 112)
    if n_cells < 12:
        print(f"  only {n_cells}/12 cells have both Single and Diverse results -- interaction test skipped.")
    else:
        df["cl_A"] = np.where(df.target.isin(["unsw", "iec104"]), df.model + "-" + df.seed.astype(str),
                              df.target + "-" + df.model + "-" + df.seed.astype(str))
        df["cl_B"] = df.seed.astype(str)
        for dname in DEFS:
            for kind in ("permute", "mask"):
                y = f"chg_{dname}_{kind}"
                print(f"\n {dname}, {kind}")
                for tag, cl, desc in (("A", df.cl_A, "cluster model-seed (manuscript choice)"),
                                      ("B", df.cl_B, "cluster by seed (10 clusters: unreliable, sensitivity only)"),
                                      ("C", None, "no clustering")):
                    F, p, d1, d2 = interaction(df, y, None if cl is None else cl.values)
                    out["interaction"][f"{y}_{tag}"] = {"F": F, "p": p, "df": [d1, d2]}
                    print(f"   {tag}. {desc:<62} F({d1},{d2}) = {F:7.3f}  p = {p:.5f}")
    path = os.path.join(pilot, "permute_single_vs_diverse_stats.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {path}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--families", nargs="*", default=FAMILIES)
    ap.add_argument("--n-rep", type=int, default=5)
    ap.add_argument("--stats-only", action="store_true")
    args = ap.parse_args(argv)
    ae_mod = importlib.import_module(MODULES[args.dataset or "iec104"][0])
    pilot = ae_mod.PILOT_DIR
    cache_dir = os.path.join(pilot, "permute_single_cache")
    os.makedirs(cache_dir, exist_ok=True)
    if not args.stats_only:
        if not args.dataset:
            ap.error("--dataset is required unless --stats-only")
        _, flow_name, loader = MODULES[args.dataset]
        flow_mod = importlib.import_module(flow_name)
        X, y = getattr(ae_mod, loader)()
        print(f"{args.dataset}: eval set {len(X)} rows")
        for fam in args.families:
            base_mod = flow_mod if fam == "flow" else ae_mod
            g2, label, _ = base.derive_groups(args.dataset, ae_mod, base_mod)
            defs = {"paper": {"stable": base_mod.STABLE_V2, "sensitive": base_mod.SENSITIVE_V2},
                    "two_source": {"stable": g2["stable"], "sensitive": g2["sensitive"]}}
            for seed in SEEDS:
                cp = os.path.join(cache_dir, f"{args.dataset}_{fam}_seed{seed}.json")
                if os.path.exists(cp):
                    print(f"  [cache] {args.dataset}/{fam}/seed{seed}")
                    continue
                rec = run_one(args.dataset, fam, seed, X, y, ae_mod, flow_mod, base_mod, defs, pilot, args.n_rep)
                if rec is None:
                    print(f"  [skip] {args.dataset}/{fam}/seed{seed}: no LODO-3 result in permute_two_source_cache")
                    continue
                with open(cp, "w") as f:
                    json.dump(rec, f, indent=2)
    stats_report(pilot)


if __name__ == "__main__":
    main()
