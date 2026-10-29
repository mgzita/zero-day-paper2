"""
Do the feature-reliance conclusions survive if the Stable / Sensitive groups are defined WITHOUT the
held-out environment?   (reviewer comment #8 / #7; follow-up to ess_sensitivity_network.py, check 4)

The paper's groups come from ESS computed across all three blocks of the training file, but the
LODO-3 models train on only two of them:
    UNSW-NB15 / IEC-104 : blocks 1 and 2 of train_dedup.npz (the third block is UNSW-NB15, held out)
    SMD                 : Groups 1 and 3 (Group 2 is the target)
So the groups were partly defined from the held-out environment. Here the groups are re-derived from
the TWO training sources only (same JS-divergence ESS recipe, same feature pool as the paper, 5 per group),
and the permutation / mean-mask test is repeated on the SAME saved LODO-3 models.

Nothing is retrained. Per seed this scores the model with the new Stable / Sensitive columns scrambled.
The paper-group results are read from <PILOT>\\eval_extra_cache (written by eval_extra_metrics), so run
that first for the dataset. Results are cached in <PILOT>\\permute_two_source_cache.

Reliance of a group = loss of separation when it is scrambled = -d_sep (d_sep = change in |AUROC-0.5|).
Reliance gap = reliance(Stable) - reliance(Sensitive); positive = the model leans more on Stable.

Run from the repo folder, one dataset at a time:
    python -u permute_two_source_groups.py --dataset unsw
    python -u permute_two_source_groups.py --dataset iec104
    python -u permute_two_source_groups.py --dataset smd
    (add --families ae dae svdd to skip Flow; Flow is the slow one)
Then:  python -u permute_two_source_groups.py --stats-only
"""
import argparse
import importlib
import json
import os

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import roc_auc_score

SEEDS = [42, 123, 456, 789, 999, 111, 222, 333, 444, 555]
DATASETS = ["unsw", "iec104", "smd"]
FAMILIES = ["ae", "dae", "svdd", "flow"]
N_BINS = 50
N_GROUP = 5
MODULES = {
    "unsw": ("extend_ae_dae_svdd_unsw_h1_repr", "flow_full_pipeline", "load_unsw_eval"),
    "iec104": ("ae_dae_svdd_pipeline_iec104", "flow_full_pipeline_iec104", "load_iec104_eval"),
    "smd": ("ae_dae_svdd_pipeline_smd", "flow_full_pipeline_smd", "load_iec104_eval"),
}


# ----------------------------------------------------------------------------- groups
def ess_from_blocks(blocks, n_features):
    out = np.zeros(n_features)
    for k in range(n_features):
        edges = np.histogram_bin_edges(np.concatenate([b[:, k] for b in blocks]), bins=N_BINS)
        hs = []
        for b in blocks:
            h, _ = np.histogram(b[:, k], bins=edges)
            h = h.astype(float)
            hs.append(h / h.sum() if h.sum() > 0 else h)
        v = []
        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                d = jensenshannon(hs[i], hs[j])
                v.append(0.0 if np.isnan(d) else d ** 2)
        out[k] = np.mean(v)
    return out


def two_source_blocks(ds, ae_mod):
    d = os.path.join(ae_mod.DEDUP_DIR)
    if ds == "smd":
        X = np.load(os.path.join(d, "train_smd_dedup.npz"))["X_train"]
        n = len(X) // 3
        groups = tuple(getattr(ae_mod, "DIVERSE_SOURCE_GROUPS", (1, 3)))
        return [X[(g - 1) * n:g * n] for g in groups], f"Groups {groups}"
    X = np.clip(np.load(os.path.join(d, "train_dedup.npz"))["X_train"], -10, 10).astype(np.float32)
    n = len(X) // 3
    return [X[:n], X[n:2 * n]], "blocks 1 and 2 of train_dedup.npz"


def derive_groups(ds, ae_mod, base_mod):
    blocks, label = two_source_blocks(ds, ae_mod)
    names = list(base_mod.FEATURE_NAMES)
    ess = pd.Series(ess_from_blocks(blocks, len(names)), index=names)
    pool = list(base_mod.DECORRELATED_FEATURES)
    order = ess.loc[pool].sort_values(kind="mergesort")
    n = min(N_GROUP, len(pool) // 3)
    st, se = list(order.index[:n]), list(order.index[-n:])
    mid = [f for f in pool if f not in st and f not in se]
    return {"stable": st, "sensitive": se, "middle": mid}, label, ess


def jac(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


# ----------------------------------------------------------------------------- model scoring
def score_fn(family, ae_mod, flow_mod, seed):
    if family == "flow":
        scorer = flow_mod.NegativeLogLikelihoodScorer(
            flow_mod.load_flow_model(flow_mod.ensure_flow_checkpoint(seed, "lodo3"))).eval()
    else:
        scorer = ae_mod.get_scorer(family, ae_mod.load_model(family, ae_mod.ensure_checkpoint(seed, family, "lodo3")))

    def f(X):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 50000):
                xb = torch.tensor(np.ascontiguousarray(X[i:i + 50000]), dtype=torch.float32)
                out.append(scorer(xb).numpy().flatten())
        return np.nan_to_num(np.concatenate(out), nan=1e10, posinf=1e10, neginf=-1e10)
    return f


def run_one(ds, fam, seed, X, y, ae_mod, flow_mod, base_mod, groups2, pilot, n_rep):
    cache_p = os.path.join(pilot, "eval_extra_cache", f"{ds}_{fam}_seed{seed}.json")
    if not os.path.exists(cache_p):
        return None
    with open(cache_p) as f:
        old = json.load(f)
    f_model = score_fn(fam, ae_mod, flow_mod, seed)
    base = float(roc_auc_score(y, f_model(X)))
    base_sep = abs(base - 0.5)
    idx = {n: i for i, n in enumerate(base_mod.FEATURE_NAMES)}
    X_tr, _ = base_mod.load_lodo3_train_val()
    train_mean = X_tr.mean(axis=0)
    rng = np.random.default_rng(seed + 1000)

    def d_sep(Xmod):
        return abs(float(roc_auc_score(y, f_model(Xmod))) - 0.5) - base_sep

    rec = {"dataset": ds, "family": fam, "seed": seed, "base_auroc": base,
           "base_auroc_in_eval_extra": old["lodo3_full"]["auroc"],
           "paper": {g: {"permute": old["perturb"][g]["permute"]["d_sep"], "mask": old["perturb"][g]["mask"]["d_sep"]}
                     for g in ("stable", "sensitive")},
           "random_permute": old["perturb"]["random"]["permute"]["d_sep"], "two_source": {}}
    for g in ("stable", "sensitive"):
        cols = [idx[n] for n in groups2[g]]
        reps = []
        for _ in range(n_rep):
            Xp = X.copy()
            Xp[:, cols] = X[rng.permutation(len(X))][:, cols]
            reps.append(d_sep(Xp))
        Xm = X.copy()
        Xm[:, cols] = train_mean[cols]
        rec["two_source"][g] = {"permute": float(np.mean(reps)), "mask": d_sep(Xm)}
    print(f"  {ds}/{fam}/seed{seed}: reliance gap (Stable-Sensitive) paper="
          f"{rec['paper']['sensitive']['permute'] - rec['paper']['stable']['permute']:+.3f}  two-source="
          f"{rec['two_source']['sensitive']['permute'] - rec['two_source']['stable']['permute']:+.3f}  "
          f"(base AUROC {base:.3f} vs cached {rec['base_auroc_in_eval_extra']:.3f})", flush=True)
    return rec


# ----------------------------------------------------------------------------- statistics
def bh(ps):
    ps = np.array(ps, float)
    order = np.argsort(ps)
    adj, prev = np.empty_like(ps), 1.0
    for rank, i in zip(range(len(ps), 0, -1), order[::-1]):
        prev = min(prev, ps[i] * len(ps) / rank)
        adj[i] = prev
    return adj


def tt(a):
    a = np.asarray(a, float)
    sd = a.std(ddof=1)
    return float(a.mean()), float(stats.ttest_1samp(a, 0.0).pvalue), float(a.mean() / sd) if sd > 0 else float("nan")


def stats_report(pilot, cache_dir):
    rows = []
    print("\n" + "=" * 110 + "\nReliance gap = reliance(Stable) - reliance(Sensitive)   (positive: leans more on Stable)\n"
          "Permutation of the columns; reliance = loss of |AUROC-0.5|. Paper groups vs groups from the two training sources.\n" + "=" * 110)
    for ds in DATASETS:
        for fam in FAMILIES:
            recs = []
            for s in SEEDS:
                p = os.path.join(cache_dir, f"{ds}_{fam}_seed{s}.json")
                if os.path.exists(p):
                    with open(p) as f:
                        recs.append(json.load(f))
            if len(recs) < 4:
                continue
            gap_p = np.array([r["paper"]["sensitive"]["permute"] - r["paper"]["stable"]["permute"] for r in recs])
            gap_n = np.array([r["two_source"]["sensitive"]["permute"] - r["two_source"]["stable"]["permute"] for r in recs])
            mp, pp, dp = tt(gap_p)
            mn, pn, dn = tt(gap_n)
            diff = tt(gap_n - gap_p)
            mism = max(abs(r["base_auroc"] - r["base_auroc_in_eval_extra"]) for r in recs)
            rows.append({"dataset": ds, "family": fam, "n": len(recs), "gap_paper": mp, "p_paper": pp, "d_paper": dp,
                         "gap_two_source": mn, "p_two_source": pn, "d_two_source": dn,
                         "change_in_gap": diff[0], "p_change": diff[1], "max_base_auroc_mismatch": mism,
                         "stable_reliance_paper": float(-np.mean([r["paper"]["stable"]["permute"] for r in recs])),
                         "stable_reliance_two_source": float(-np.mean([r["two_source"]["stable"]["permute"] for r in recs])),
                         "sensitive_reliance_paper": float(-np.mean([r["paper"]["sensitive"]["permute"] for r in recs])),
                         "sensitive_reliance_two_source": float(-np.mean([r["two_source"]["sensitive"]["permute"] for r in recs]))})
    adj = bh([r["p_two_source"] for r in rows]) if rows else []
    for r, a in zip(rows, adj):
        r["bh_adj_p_two_source"] = float(a)
        same = (np.sign(r["gap_paper"]) == np.sign(r["gap_two_source"])) and ((r["p_paper"] < .05) == (r["p_two_source"] < .05))
        r["same_conclusion"] = bool(same)
        print(f" {r['dataset']:>6}/{r['family']:<5} n={r['n']:>2} | paper groups: gap={r['gap_paper']:+.3f} (p={r['p_paper']:.3f}) "
              f"| two-source groups: gap={r['gap_two_source']:+.3f} (p={r['p_two_source']:.3f}, BH {a:.3f}) "
              f"| change {r['change_in_gap']:+.3f} (p={r['p_change']:.3f}) | {'SAME' if same else 'DIFFERENT'} conclusion"
              + ("" if r["max_base_auroc_mismatch"] < 1e-4 else f"  [base AUROC mismatch {r['max_base_auroc_mismatch']:.4f}]"))
    if rows:
        print(f"\n  same conclusion in {sum(r['same_conclusion'] for r in rows)}/{len(rows)} cells")
        out = os.path.join(pilot, "permute_two_source_stats.json")
        with open(out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"  Saved {out}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--families", nargs="*", default=FAMILIES)
    ap.add_argument("--n-rep", type=int, default=5)
    ap.add_argument("--stats-only", action="store_true")
    args = ap.parse_args(argv)
    ae_mod = importlib.import_module(MODULES[args.dataset or "iec104"][0])
    pilot = ae_mod.PILOT_DIR
    cache_dir = os.path.join(pilot, "permute_two_source_cache")
    os.makedirs(cache_dir, exist_ok=True)
    if not args.stats_only:
        if not args.dataset:
            ap.error("--dataset is required unless --stats-only")
        ae_name, flow_name, loader = MODULES[args.dataset]
        flow_mod = importlib.import_module(flow_name)
        X, y = getattr(ae_mod, loader)()
        print(f"{args.dataset}: eval set {len(X)} rows")
        for fam in args.families:
            base_mod = flow_mod if fam == "flow" else ae_mod
            g2, label, ess = derive_groups(args.dataset, ae_mod, base_mod)
            paper = {"stable": base_mod.STABLE_V2, "sensitive": base_mod.SENSITIVE_V2}
            if fam == args.families[0]:
                print(f"\nGroups from the two training sources ({label}):")
                print(f"  Stable    {g2['stable']}   (Jaccard with paper {jac(g2['stable'], paper['stable']):.2f})")
                print(f"  Sensitive {g2['sensitive']}   (Jaccard with paper {jac(g2['sensitive'], paper['sensitive']):.2f})")
                with open(os.path.join(pilot, f"two_source_groups_{args.dataset}.json"), "w") as f:
                    json.dump({"source": label, "groups": g2, "paper": paper, "ess": ess.to_dict()}, f, indent=2)
            for seed in SEEDS:
                cp = os.path.join(cache_dir, f"{args.dataset}_{fam}_seed{seed}.json")
                if os.path.exists(cp):
                    print(f"  [cache] {args.dataset}/{fam}/seed{seed}")
                    continue
                rec = run_one(args.dataset, fam, seed, X, y, ae_mod, flow_mod, base_mod, g2, pilot, args.n_rep)
                if rec is None:
                    print(f"  [skip] {args.dataset}/{fam}/seed{seed}: no eval_extra_cache entry (run eval_extra_metrics first)")
                    continue
                with open(cp, "w") as f:
                    json.dump(rec, f, indent=2)
    stats_report(pilot, cache_dir)


if __name__ == "__main__":
    main()
