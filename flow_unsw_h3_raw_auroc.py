import json
import os

import numpy as np
from scipy import stats

import flow_full_pipeline as fp

OUT = os.path.join(fp.PILOT_DIR, "flow_h3_ablation_unsw_raw.json")
GROUPS = {"stable": fp.STABLE_V2, "sensitive": fp.SENSITIVE_V2, "middle": fp.MIDDLE_V2}


def run_seed(seed, X_unsw, y_unsw, X_train, X_val):
    rec = {"seed": seed}
    sep_full, rec["auroc_full"] = fp.train_and_score_subset(
        seed, fp.DECORRELATED_FEATURES, X_train, X_val, X_unsw, y_unsw)
    for name, feats in GROUPS.items():
        sep, au = fp.train_and_score_subset(seed, feats, X_train, X_val, X_unsw, y_unsw)
        rec[f"auroc_{name}"] = au
        rec[f"delta_{name}"] = sep_full - sep
    print(f"  seed {seed}: " + "  ".join(
        f"{k}={rec[k]:.3f}" for k in ["auroc_full", "auroc_stable", "auroc_sensitive", "auroc_middle"]),
        flush=True)
    return rec


def paired(a, b):
    a, b = np.asarray(a), np.asarray(b)
    d = a - b
    t, p = stats.ttest_rel(a, b)
    wp = float(stats.wilcoxon(a, b).pvalue)
    sd = d.std(ddof=1)
    return (f"diff={d.mean():+.3f} t-p={p:.4f} wilcoxon-p={wp:.4f} "
            f"d={d.mean() / sd if sd > 0 else float('nan'):+.2f} first>second in {(d > 0).sum()}/{len(d)}")


def main():
    if os.path.exists(OUT):
        with open(OUT) as f:
            recs = json.load(f)
        print(f"Loaded {len(recs)} saved records from {OUT}")
    else:
        X_unsw, y_unsw = fp.load_unsw_eval()
        X_train, X_val = fp.load_lodo3_train_val()
        recs = []
        for seed in fp.SEEDS_TO_RUN:
            recs.append(run_seed(seed, X_unsw, y_unsw, X_train, X_val))
            with open(OUT, "w") as f:
                json.dump(recs, f, indent=2)

    print("\nCheck vs cached flow_pipeline_cache deltas (max abs difference):")
    for g in GROUPS:
        diffs = []
        for r in recs:
            p = os.path.join(fp.FLOW_DIR, f"seed{r['seed']}.json")
            if os.path.exists(p):
                with open(p) as f:
                    diffs.append(abs(json.load(f)["h3"][f"delta_{g}"] - r[f"delta_{g}"]))
        print(f"  delta_{g}: {max(diffs):.6f}" if diffs else f"  delta_{g}: no cache found")

    raw = {g: np.array([r[f"auroc_{g}"] for r in recs]) for g in ["full", *GROUPS]}
    sep = {g: np.abs(v - 0.5) for g, v in raw.items()}
    print("\nUNSW-NB15 / Flow, n=%d seeds" % len(recs))
    for g in raw:
        print(f"  {g:>9}: raw AUROC mean={raw[g].mean():.3f} sd={raw[g].std(ddof=1):.3f} "
              f"| below 0.5 in {(raw[g] < 0.5).sum()}/{len(recs)} | |AUROC-0.5| mean={sep[g].mean():.3f}")
    for other in ("sensitive", "middle"):
        print(f"  stable vs {other}\n    RAW AUROC : {paired(raw['stable'], raw[other])}"
              f"\n    |AUC-0.5| : {paired(sep['stable'], sep[other])}")
    print("  best group: raw=%s, |AUC-0.5|=%s" % (
        max(GROUPS, key=lambda g: raw[g].mean()), max(GROUPS, key=lambda g: sep[g].mean())))


if __name__ == "__main__":
    main()