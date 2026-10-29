"""
Extra evaluation for reviewer comments #2, #3 (AUPRC), #4 and #9.

Uses the EXISTING saved checkpoints (no retraining of anything that already
exists; a missing checkpoint is trained by the pipeline's own ensure_checkpoint).
Nothing already on disk is modified.

For every dataset x family x seed it computes, for the full-feature Single and
LODO-3 ("Diverse") models, on the same evaluation set the pipelines use:
  * AUROC (raw, high score = anomaly, NOT folded around 0.5), AUPRC,
    recall at 1% and 5% false-positive rate                      (#2, #3)
  * feature PERMUTATION and MEAN-MASKING of the Stable / Sensitive / Middle
    groups on the trained LODO-3 full-feature model, plus a random same-size
    group control                                                (#4)
  * the paired Single-vs-LODO-3 statistics                       (#2)
  * (optional --save-scores) per-sample scores/labels as .npz   (#9, bootstrap)

Run from the repo folder, one dataset at a time, e.g.:
    python -u eval_extra_metrics.py --dataset iec104
    python -u eval_extra_metrics.py --dataset smd
    python -u eval_extra_metrics.py --dataset unsw
Per-run results are cached in <PILOT_DIR>\\eval_extra_cache, so it can be
stopped and restarted. Use --n-rep 5 (default) for the permutation repeats.
"""
import argparse
import importlib
import json
import os

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

SEEDS = [42, 123, 456, 789, 999, 111, 222, 333, 444, 555]
FAMILIES = ["ae", "dae", "svdd", "flow"]
CONDITIONS = ["single", "lodo3"]
FPRS = [0.01, 0.05]
GROUP_NAMES = ["stable", "sensitive", "middle"]

# dataset -> (ae/dae/svdd module, flow module, eval-loader name)
MODULES = {
    "unsw": ("extend_ae_dae_svdd_unsw_h1_repr", "flow_full_pipeline", "load_unsw_eval"),
    "iec104": ("ae_dae_svdd_pipeline_iec104", "flow_full_pipeline_iec104", "load_iec104_eval"),
    "smd": ("ae_dae_svdd_pipeline_smd", "flow_full_pipeline_smd", "load_iec104_eval"),
}


def metrics(y, s):
    s = np.nan_to_num(s, nan=1e10, posinf=1e10, neginf=-1e10)
    fpr, tpr, _ = roc_curve(y, s)
    out = {"auroc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s))}
    for t in FPRS:
        out[f"recall_at_fpr{int(t * 100)}"] = float(tpr[fpr <= t].max())
    return out


def score_fn(family, ae_mod, flow_mod, seed, condition):
    """Return a function X -> scores for the full-feature model."""
    if family == "flow":
        ckpt = flow_mod.ensure_flow_checkpoint(seed, condition)
        scorer = flow_mod.NegativeLogLikelihoodScorer(flow_mod.load_flow_model(ckpt)).eval()
    else:
        ckpt = ae_mod.ensure_checkpoint(seed, family, condition)
        scorer = ae_mod.get_scorer(family, ae_mod.load_model(family, ckpt))

    def f(X):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 50000):
                xb = torch.tensor(np.ascontiguousarray(X[i:i + 50000]), dtype=torch.float32)
                out.append(scorer(xb).numpy().flatten())
        return np.concatenate(out)
    return f


def run_one(ds, family, seed, X, y, mod, n_rep, save_dir):
    ae_mod, flow_mod = mod["ae"], mod["flow"]
    base_mod = flow_mod if family == "flow" else ae_mod
    rec = {"dataset": ds, "family": family, "seed": seed, "metrics": {}}
    for cond in CONDITIONS:
        f = score_fn(family, ae_mod, flow_mod, seed, cond)
        s = f(X)
        rec["metrics"][cond] = metrics(y, s)
        if save_dir:
            np.savez_compressed(os.path.join(save_dir, f"{ds}_{family}_{cond}_seed{seed}.npz"),
                                scores=s.astype(np.float32), y=y.astype(np.int8))
        if cond == "lodo3":
            base_scores, f_lodo3 = s, f

    # ---- permutation / masking on the trained LODO-3 full-feature model ----
    names = base_mod.FEATURE_NAMES
    idx = {n: i for i, n in enumerate(names)}
    groups = {"stable": base_mod.STABLE_V2, "sensitive": base_mod.SENSITIVE_V2,
              "middle": base_mod.MIDDLE_V2}
    X_tr, _ = base_mod.load_lodo3_train_val()
    train_mean = X_tr.mean(axis=0)
    rng = np.random.default_rng(seed)
    base = metrics(y, base_scores)
    base_sep = abs(base["auroc"] - 0.5)

    def change(Xmod):
        m = metrics(y, f_lodo3(Xmod))
        return {"d_auroc": m["auroc"] - base["auroc"],
                "d_sep": abs(m["auroc"] - 0.5) - base_sep,
                "d_auprc": m["auprc"] - base["auprc"]}

    def permuted(cols):
        Xp = X.copy()
        perm = rng.permutation(len(X))
        Xp[:, cols] = X[perm][:, cols]      # rows shuffled jointly -> keeps within-group structure
        return Xp

    def masked(cols):
        Xm = X.copy()
        Xm[:, cols] = train_mean[cols]
        return Xm

    perm_out = {}
    for g, feats in groups.items():
        cols = [idx[n] for n in feats]
        reps = [change(permuted(cols)) for _ in range(n_rep)]
        perm_out[g] = {"permute": {k: float(np.mean([r[k] for r in reps])) for k in reps[0]},
                       "mask": change(masked(cols))}
    pool = [idx[n] for n in base_mod.DECORRELATED_FEATURES]
    size = len(groups["stable"])
    reps = []
    for _ in range(n_rep):
        cols = list(rng.choice(pool, size=size, replace=False))
        reps.append(change(permuted(cols)))
    perm_out["random"] = {"permute": {k: float(np.mean([r[k] for r in reps])) for k in reps[0]}}
    rec["lodo3_full"] = {"auroc": base["auroc"], "auprc": base["auprc"]}
    rec["perturb"] = perm_out
    print(f"  {ds}/{family}/seed{seed}: AUROC single={rec['metrics']['single']['auroc']:.3f} "
          f"lodo3={rec['metrics']['lodo3']['auroc']:.3f} | permute d_auroc stable="
          f"{perm_out['stable']['permute']['d_auroc']:+.3f} sensitive="
          f"{perm_out['sensitive']['permute']['d_auroc']:+.3f}", flush=True)
    return rec


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    sd = d.std(ddof=1)
    try:
        wp = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        wp = float("nan")
    return {"n": len(d), "mean_diff": float(d.mean()), "t_p": float(stats.ttest_rel(a, b).pvalue),
            "wilcoxon_p": wp, "cohens_d": float(d.mean() / sd) if sd > 0 else float("nan"),
            "a_greater": int((d > 0).sum())}


def bh(ps):
    ps = np.array(ps, float)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    prev = 1.0
    for rank, i in zip(range(len(ps), 0, -1), order[::-1]):
        prev = min(prev, ps[i] * len(ps) / rank)
        adj[i] = prev
    return adj


def summarize(ds, recs, pilot):
    stats_out, rows = {}, []
    for fam in FAMILIES:
        r = [x for x in recs if x["family"] == fam]
        if len(r) < 3:
            continue
        fo = {"n": len(r)}
        print(f"\n[{ds} / {fam}] n={len(r)}")
        for m in ["auroc", "auprc", "recall_at_fpr1", "recall_at_fpr5"]:
            sg = [x["metrics"]["single"][m] for x in r]
            lo = [x["metrics"]["lodo3"][m] for x in r]
            t = paired(lo, sg)            # positive = Diverse (LODO-3) better
            t.update({"single_mean": float(np.mean(sg)), "lodo3_mean": float(np.mean(lo))})
            fo[f"lodo3_vs_single_{m}"] = t
            rows.append((f"{ds}/{fam}/{m}", t["t_p"]))
            print(f"  {m:>15}: single={t['single_mean']:.3f} lodo3={t['lodo3_mean']:.3f} "
                  f"diff={t['mean_diff']:+.3f} t-p={t['t_p']:.4f} wilcoxon-p={t['wilcoxon_p']:.4f} "
                  f"d={t['cohens_d']:+.2f} (lodo3 higher in {t['a_greater']}/{t['n']})")
        for mode in ["permute", "mask"]:
            for key in ["d_auroc", "d_sep"]:
                means = {g: float(np.mean([x["perturb"][g][mode][key] for x in r]))
                         for g in GROUP_NAMES}
                fo[f"{mode}_{key}_means"] = means
        for key in ["d_auroc", "d_sep"]:
            fo[f"permute_{key}_stable_vs_sensitive"] = paired(
                [x["perturb"]["stable"]["permute"][key] for x in r],
                [x["perturb"]["sensitive"]["permute"][key] for x in r])
        fo["permute_d_auroc_stable_vs_random"] = paired(
            [x["perturb"]["stable"]["permute"]["d_auroc"] for x in r],
            [x["perturb"]["random"]["permute"]["d_auroc"] for x in r])
        fo["permute_d_auroc_sensitive_vs_random"] = paired(
            [x["perturb"]["sensitive"]["permute"]["d_auroc"] for x in r],
            [x["perturb"]["random"]["permute"]["d_auroc"] for x in r])
        pm = fo["permute_d_auroc_means"]
        mm = fo["mask_d_auroc_means"]
        print("  PERMUTATION change in raw AUROC (negative = model relied on the group): "
              + "  ".join(f"{g}={pm[g]:+.3f}" for g in GROUP_NAMES))
        print("  MEAN-MASK   change in raw AUROC: " + "  ".join(f"{g}={mm[g]:+.3f}" for g in GROUP_NAMES))
        s_vs = fo["permute_d_auroc_stable_vs_sensitive"]
        print(f"  permute stable vs sensitive (d_auroc): diff={s_vs['mean_diff']:+.3f} "
              f"t-p={s_vs['t_p']:.4f} wilcoxon-p={s_vs['wilcoxon_p']:.4f} d={s_vs['cohens_d']:+.2f}")
        stats_out[fam] = fo
    adj = bh([p for _, p in rows]) if rows else []
    for (name, _), a in zip(rows, adj):
        stats_out.setdefault("bh_adjusted_p_single_vs_lodo3", {})[name] = float(a)
    path = os.path.join(pilot, f"eval_extra_stats_{ds}.json")
    with open(path, "w") as f:
        json.dump(stats_out, f, indent=2)
    print(f"\nSaved {path}  (BH-adjusted p within this dataset's metric tests included)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(MODULES))
    ap.add_argument("--families", nargs="*", default=FAMILIES)
    ap.add_argument("--n-rep", type=int, default=5)
    ap.add_argument("--save-scores", action="store_true")
    args = ap.parse_args()

    ae_name, flow_name, loader = MODULES[args.dataset]
    ae_mod, flow_mod = importlib.import_module(ae_name), importlib.import_module(flow_name)
    mod = {"ae": ae_mod, "flow": flow_mod}
    pilot = ae_mod.PILOT_DIR
    cache = os.path.join(pilot, "eval_extra_cache")
    os.makedirs(cache, exist_ok=True)
    save_dir = None
    if args.save_scores:
        save_dir = os.path.join(pilot, "score_dumps")
        os.makedirs(save_dir, exist_ok=True)

    X, y = getattr(ae_mod, loader)()
    print(f"{args.dataset}: eval set {len(X)} rows ({int((y == 0).sum())} benign, {int((y == 1).sum())} attack)")
    recs = []
    for fam in args.families:
        for seed in SEEDS:
            cp = os.path.join(cache, f"{args.dataset}_{fam}_seed{seed}.json")
            if os.path.exists(cp):
                with open(cp) as f:
                    recs.append(json.load(f))
                print(f"  [cache] {args.dataset}/{fam}/seed{seed}")
                continue
            rec = run_one(args.dataset, fam, seed, X, y, mod, args.n_rep, save_dir)
            with open(cp, "w") as f:
                json.dump(rec, f, indent=2)
            recs.append(rec)
    with open(os.path.join(pilot, f"eval_extra_all_{args.dataset}.json"), "w") as f:
        json.dump(recs, f, indent=2)
    summarize(args.dataset, recs, pilot)


if __name__ == "__main__":
    main()
