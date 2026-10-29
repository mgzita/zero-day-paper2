"""
Reviewer comment #1: separate single-source baselines + a data-volume control.

Existing conditions (checkpoints already on disk, NOT retrained):
    single   = source A alone            (UNSW/IEC-104: first training environment; SMD: Group 1)
    lodo3    = source A + source B       (the "Diverse" condition; 2x the training rows of `single`)
New conditions trained here, with the pipelines' own training functions and hyper-parameters:
    single_b       = source B alone (second environment; SMD: Group 3)
    mixed_matched  = half of A + half of B, TOTAL SIZE = size of `single`
                     (diversity with the data volume held fixed)

All four are scored on the same evaluation set as the rest of the paper. Paired tests (same seeds):
    lodo3 vs single, lodo3 vs single_b, lodo3 vs best-single (per seed), lodo3 vs mixed_matched,
    mixed_matched vs single (diversity at fixed volume), single_b vs single (which source you pick).
BH-FDR over all tests.

Run from the repo folder, one dataset at a time (AE/DAE/SVDD are fast, Flow is slow):
    python -u single_source_baselines.py --dataset iec104 --families ae dae svdd
    python -u single_source_baselines.py --dataset smd    --families ae dae svdd
    python -u single_source_baselines.py --dataset unsw   --families ae dae svdd
    python -u single_source_baselines.py --dataset iec104 --families flow       (and smd, unsw: long)
Per-seed results are cached in <PILOT_DIR>\\single_source_cache, so it can be stopped and restarted.
Then:  python -u single_source_baselines.py --stats-only
"""
import argparse
import importlib
import json
import os
import time

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

SEEDS = [42, 123, 456, 789, 999, 111, 222, 333, 444, 555]
DATASETS = ["unsw", "iec104", "smd"]
FAMS = ["ae", "dae", "svdd", "flow"]
CONDS = ["single", "single_b", "lodo3", "mixed_matched"]
METRICS = ["auroc", "auprc", "recall_at_fpr5"]
MODULES = {
    "unsw": ("extend_ae_dae_svdd_unsw_h1_repr", "flow_full_pipeline", "load_unsw_eval"),
    "iec104": ("ae_dae_svdd_pipeline_iec104", "flow_full_pipeline_iec104", "load_iec104_eval"),
    "smd": ("ae_dae_svdd_pipeline_smd", "flow_full_pipeline_smd", "load_iec104_eval"),
}


def metrics(y, s):
    s = np.nan_to_num(s, nan=1e10, posinf=1e10, neginf=-1e10)
    fpr, tpr, _ = roc_curve(y, s)
    return {"auroc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s)),
            "recall_at_fpr1": float(tpr[fpr <= 0.01].max()), "recall_at_fpr5": float(tpr[fpr <= 0.05].max())}


def score_model(family, ae_mod, flow_mod, model, X):
    model.eval()
    scorer = flow_mod.NegativeLogLikelihoodScorer(model).eval() if family == "flow" \
        else ae_mod.get_scorer(family, model)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 50000):
            out.append(scorer(torch.tensor(np.ascontiguousarray(X[i:i + 50000]), dtype=torch.float32)).numpy().flatten())
    return np.concatenate(out)


def build_sets(mod, family, seed):
    """Return {condition: (X_train, X_val)} for the two NEW conditions plus sizes for reporting."""
    load = mod.load_lodo3_train_val
    X_l, V_l = load()
    X_s, V_s = mod.load_single_train_val()
    ha, hv = len(X_l) // 2, len(V_l) // 2
    A, B, VA, VB = X_l[:ha], X_l[ha:], V_l[:hv], V_l[hv:]
    if len(A) != len(X_s) or not np.array_equal(A[:1000], X_s[:1000]):
        print("  [WARNING] first half of the Diverse set does not match the Single set -- "
              "check the source ordering before trusting single_b")
    n, nv = len(A), len(VA)
    rng = np.random.default_rng(seed)
    ia = rng.permutation(len(A))[: n // 2]
    ib = rng.permutation(len(B))[: n - n // 2]
    ja = rng.permutation(len(VA))[: nv // 2]
    jb = rng.permutation(len(VB))[: nv - nv // 2]
    Xm = np.vstack([A[ia], B[ib]])
    Vm = np.vstack([VA[ja], VB[jb]])
    p = rng.permutation(len(Xm))
    return {"single_b": (B, VB), "mixed_matched": (Xm[p], Vm)}, \
        {"n_single": len(X_s), "n_diverse": len(X_l), "n_single_b": len(B), "n_mixed": len(Xm)}


def train_one(family, ae_mod, flow_mod, X, V, seed):
    d = ae_mod.N_FEATURES
    if family == "ae":
        return ae_mod.train_autoencoder(X, V, seed, d, noise_std=0.0)
    if family == "dae":
        return ae_mod.train_autoencoder(X, V, seed, d, noise_std=ae_mod.NOISE_STD_DAE)
    if family == "svdd":
        return ae_mod.train_svdd(X, V, seed, d)
    if family == "flow":
        return flow_mod.train_flow(X, V, seed, input_dim=d)
    raise ValueError(family)


def load_existing(family, ae_mod, flow_mod, seed, cond):
    if family == "flow":
        return flow_mod.load_flow_model(flow_mod.ensure_flow_checkpoint(seed, cond))
    return ae_mod.load_model(family, ae_mod.ensure_checkpoint(seed, family, cond))


def run(ds, families, pilot, ckpt_dir, cache_dir):
    ae_name, flow_name, loader = MODULES[ds]
    ae_mod, flow_mod = importlib.import_module(ae_name), importlib.import_module(flow_name)
    X_eval, y_eval = getattr(ae_mod, loader)()
    print(f"{ds}: eval set {len(X_eval)} rows ({int((y_eval == 0).sum())} benign, {int((y_eval == 1).sum())} attack)")
    for fam in families:
        mod = flow_mod if fam == "flow" else ae_mod
        for seed in SEEDS:
            cp = os.path.join(cache_dir, f"{ds}_{fam}_seed{seed}.json")
            if os.path.exists(cp):
                print(f"  [cache] {ds}/{fam}/seed{seed}")
                continue
            t0 = time.time()
            rec = {"dataset": ds, "family": fam, "seed": seed, "metrics": {}}
            sets, sizes = build_sets(mod, fam, seed)
            rec["sizes"] = sizes
            for cond in ("single", "lodo3"):
                m = load_existing(fam, ae_mod, flow_mod, seed, cond)
                rec["metrics"][cond] = metrics(y_eval, score_model(fam, ae_mod, flow_mod, m, X_eval))
            for cond, (X, V) in sets.items():
                print(f"  training {ds}/{fam}/seed{seed}/{cond} (n_train={len(X)}) ...", flush=True)
                m = train_one(fam, ae_mod, flow_mod, X, V, seed)
                torch.save(m.state_dict(), os.path.join(ckpt_dir, f"{ds}_{fam}_{cond}_seed{seed}.pt"))
                rec["metrics"][cond] = metrics(y_eval, score_model(fam, ae_mod, flow_mod, m, X_eval))
            with open(cp, "w") as f:
                json.dump(rec, f, indent=2)
            print(f"  {ds}/{fam}/seed{seed}: AUROC " + " ".join(f"{c}={rec['metrics'][c]['auroc']:.3f}" for c in CONDS)
                  + f"  ({time.time() - t0:.0f}s) sizes={sizes}", flush=True)


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    sd = d.std(ddof=1)
    try:
        wp = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        wp = float("nan")
    return {"diff": float(d.mean()), "t_p": float(stats.ttest_rel(a, b).pvalue), "wilcoxon_p": wp,
            "d": float(d.mean() / sd) if sd > 0 else float("nan"), "n_pos": int((d > 0).sum()), "n": len(d)}


def bh(ps):
    ps = np.array(ps, float)
    order = np.argsort(ps)
    adj, prev = np.empty_like(ps), 1.0
    for rank, i in zip(range(len(ps), 0, -1), order[::-1]):
        prev = min(prev, ps[i] * len(ps) / rank)
        adj[i] = prev
    return adj


COMPARISONS = [("lodo3", "single"), ("lodo3", "single_b"), ("lodo3", "best_single"),
               ("lodo3", "mixed_matched"), ("mixed_matched", "single"), ("single_b", "single")]


def stats_report(pilot, cache_dir):
    rows, means = [], []
    for ds in DATASETS:
        for fam in FAMS:
            recs = []
            for seed in SEEDS:
                p = os.path.join(cache_dir, f"{ds}_{fam}_seed{seed}.json")
                if os.path.exists(p):
                    with open(p) as f:
                        recs.append(json.load(f))
            if len(recs) < 4:
                continue
            n = recs[0]["sizes"]
            print(f"\n[{ds}/{fam}] n={len(recs)} seeds | training rows: single={n['n_single']} single_b={n['n_single_b']} "
                  f"diverse={n['n_diverse']} mixed_matched={n['n_mixed']}")
            for m in METRICS:
                v = {c: np.array([r["metrics"][c][m] for r in recs]) for c in CONDS}
                v["best_single"] = np.maximum(v["single"], v["single_b"])
                print(f"   {m:>15}: " + "  ".join(f"{c}={v[c].mean():.3f}" for c in CONDS))
                for a, b in COMPARISONS:
                    t = paired(v[a], v[b])
                    rows.append((ds, fam, m, a, b, t))
    adj = bh([t["t_p"] for *_, t in rows]) if rows else []
    for r, a in zip(rows, adj):
        r[-1]["bh_adj_p"] = float(a)
    print("\n" + "=" * 100 + f"\nTop results (global BH over {len(rows)} tests; positive diff = first condition higher)\n" + "=" * 100)
    for ds, fam, m, a, b, t in sorted(rows, key=lambda x: x[-1]["t_p"])[:25]:
        print(f"  {ds:>6}/{fam:<5}/{m:<15} {a:>13} vs {b:<13} diff={t['diff']:+.3f} d={t['d']:+.2f} "
              f"raw p={t['t_p']:.4f} adj p={t['bh_adj_p']:.4f} {'SIG' if t['bh_adj_p'] < 0.05 else ''}")
    print(f"  significant after BH: {sum(r[-1]['bh_adj_p'] < 0.05 for r in rows)}/{len(rows)}")
    print("\n  AUROC: does Diverse beat each single source and the volume-matched mix? (raw p<0.05, lodo3 higher)")
    for ds, fam, m, a, b, t in rows:
        if m == "auroc" and a == "lodo3" and t["t_p"] < 0.05 and t["diff"] > 0:
            print(f"    {ds}/{fam}: lodo3 > {b} by {t['diff']:+.3f} (p={t['t_p']:.4f})")
    out = os.path.join(pilot, "single_source_baselines_stats.json")
    with open(out, "w") as f:
        json.dump([{"dataset": d, "family": fm, "metric": m, "a": a, "b": b, **t} for d, fm, m, a, b, t in rows], f, indent=2)
    print(f"\nSaved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--families", nargs="*", default=["ae", "dae", "svdd"])
    ap.add_argument("--stats-only", action="store_true")
    args = ap.parse_args()
    ae_mod = importlib.import_module(MODULES[args.dataset or "iec104"][0])
    pilot = ae_mod.PILOT_DIR
    cache_dir = os.path.join(pilot, "single_source_cache")
    ckpt_dir = os.path.join(pilot, "baseline_ckpts")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    if not args.stats_only:
        if not args.dataset:
            ap.error("--dataset is required unless --stats-only")
        run(args.dataset, args.families, pilot, ckpt_dir, cache_dir)
    stats_report(pilot, cache_dir)


if __name__ == "__main__":
    main()
