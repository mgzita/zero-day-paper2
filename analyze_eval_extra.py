"""
Follow-up analysis of eval_extra_metrics output (reads saved JSON only; runs in seconds).

A. Global BH-FDR over ALL Single-vs-LODO-3 detection-metric tests (3 datasets x 4 models x 4 metrics)
   -> reviewer comment #2, with a single multiple-comparison family.
B. Feature permutation / mean-masking, using the change in SEPARATION |AUROC-0.5| (d_sep) as the
   "reliance" measure. d_sep < 0 = the model lost separation when the group was scrambled, whether its
   AUROC was above or below 0.5. (The raw d_auroc printed earlier flips sign for inverted models, so
   do not read it as reliance when the base AUROC is < 0.5.)                       -> comment #4
C. Does the representation change (MMD^2 / Wasserstein narrowing) track the detection change
   (AUROC / AUPRC gain)?  Spearman per cell and pooled (z-scored within cell).     -> comment #9

Run from the repo folder:  python analyze_eval_extra.py
"""
import json
import os

import numpy as np
from scipy import stats

PILOT = r"C:\MLProject\zero_day_project\paper2\pilot"
DATASETS = ["unsw", "iec104", "smd"]
FAMS = ["ae", "dae", "svdd", "flow"]
METRICS = ["auroc", "auprc", "recall_at_fpr1", "recall_at_fpr5"]


def load(name):
    p = os.path.join(PILOT, name)
    if not os.path.exists(p):
        print(f"[missing] {name}")
        return []
    with open(p) as f:
        return json.load(f)


def bh(ps):
    ps = np.array(ps, float)
    order = np.argsort(ps)
    adj, prev = np.empty_like(ps), 1.0
    for rank, i in zip(range(len(ps), 0, -1), order[::-1]):
        prev = min(prev, ps[i] * len(ps) / rank)
        adj[i] = prev
    return adj


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    sd = d.std(ddof=1)
    return {"diff": float(d.mean()), "t_p": float(stats.ttest_rel(a, b).pvalue),
            "d": float(d.mean() / sd) if sd > 0 else float("nan"), "n_pos": int((d > 0).sum()), "n": len(d)}


def main():
    recs = {ds: load(f"eval_extra_all_{ds}.json") for ds in DATASETS}

    # ---------------- A. global FDR ----------------
    print("=" * 100 + "\nA. Single vs LODO-3 detection metrics, global BH-FDR (positive diff = LODO-3 better)\n" + "=" * 100)
    rows = []
    for ds in DATASETS:
        for fam in FAMS:
            r = [x for x in recs[ds] if x["family"] == fam]
            if len(r) < 4:
                continue
            for m in METRICS:
                t = paired([x["metrics"]["lodo3"][m] for x in r], [x["metrics"]["single"][m] for x in r])
                t["single"] = float(np.mean([x["metrics"]["single"][m] for x in r]))
                t["lodo3"] = float(np.mean([x["metrics"]["lodo3"][m] for x in r]))
                rows.append((ds, fam, m, t))
    adj = bh([t["t_p"] for *_, t in rows])
    print(f"{len(rows)} tests")
    for (ds, fam, m, t), a in zip(rows, adj):
        t["bh_adj_p"] = float(a)
    for ds, fam, m, t in sorted(rows, key=lambda x: x[3]["t_p"])[:15]:
        print(f"  {ds:>6}/{fam:<5}/{m:<15} single={t['single']:.3f} lodo3={t['lodo3']:.3f} diff={t['diff']:+.3f} "
              f"d={t['d']:+.2f} raw p={t['t_p']:.4f} adj p={t['bh_adj_p']:.4f} {'SIG' if t['bh_adj_p'] < 0.05 else ''}")
    print(f"  significant after BH: {sum(t['bh_adj_p'] < 0.05 for *_, t in rows)}/{len(rows)}")
    # AUROC vs AUPRC/recall agreement
    print("  Cells where AUROC improves significantly (raw p<0.05, LODO-3 higher) but AUPRC and recall@5% do not:")
    for ds in DATASETS:
        for fam in FAMS:
            sel = {m: t for d_, f_, m, t in rows if d_ == ds and f_ == fam}
            if sel and sel["auroc"]["t_p"] < 0.05 and sel["auroc"]["diff"] > 0 and \
                    sel["auprc"]["t_p"] >= 0.05 and sel["recall_at_fpr5"]["t_p"] >= 0.05:
                print(f"    {ds}/{fam}: AUROC {sel['auroc']['single']:.3f}->{sel['auroc']['lodo3']:.3f}, "
                      f"AUPRC {sel['auprc']['single']:.3f}->{sel['auprc']['lodo3']:.3f}, "
                      f"recall@5%FPR {sel['recall_at_fpr5']['single']:.3f}->{sel['recall_at_fpr5']['lodo3']:.3f}")

    # ---------------- B. permutation on separation ----------------
    print("\n" + "=" * 100 + "\nB. Permutation / mean-mask on the trained LODO-3 full-feature model: change in |AUROC-0.5|\n"
          "   (negative = model lost separation = relied on the group)\n" + "=" * 100)
    perm_out = {}
    for ds in DATASETS:
        for fam in FAMS:
            r = [x for x in recs[ds] if x["family"] == fam]
            if len(r) < 4:
                continue
            g = lambda grp, mode: [x["perturb"][grp][mode]["d_sep"] for x in r]
            inv = sum(x["lodo3_full"]["auroc"] < 0.5 for x in r)
            line = "  ".join(f"{grp}={np.mean(g(grp, 'permute')):+.3f}" for grp in ["stable", "sensitive", "middle", "random"])
            mline = "  ".join(f"{grp}={np.mean(g(grp, 'mask')):+.3f}" for grp in ["stable", "sensitive", "middle"])
            sv = paired(g("stable", "permute"), g("sensitive", "permute"))
            st_r = paired(g("stable", "permute"), g("random", "permute"))
            se_r = paired(g("sensitive", "permute"), g("random", "permute"))
            print(f"\n [{ds}/{fam}] base LODO-3 AUROC<0.5 in {inv}/{len(r)} seeds")
            print(f"   permute d_sep: {line}")
            print(f"   mask    d_sep: {mline}")
            print(f"   stable vs sensitive: diff={sv['diff']:+.3f} p={sv['t_p']:.4f} d={sv['d']:+.2f} | "
                  f"stable vs random: diff={st_r['diff']:+.3f} p={st_r['t_p']:.4f} | "
                  f"sensitive vs random: diff={se_r['diff']:+.3f} p={se_r['t_p']:.4f}")
            perm_out[f"{ds}_{fam}"] = {"stable_vs_sensitive": sv, "stable_vs_random": st_r, "sensitive_vs_random": se_r}

    # ---------------- C. representation vs detection ----------------
    print("\n" + "=" * 100 + "\nC. Does narrowing of source-target distance (Single - LODO-3) track detection gain (LODO-3 - Single)?\n" + "=" * 100)
    repr_sources = {}

    def add(ds, fam, seed, rec):
        repr_sources.setdefault((ds, fam), {})[seed] = rec

    for ds, fn in [("iec104", "ae_dae_svdd_repr_iec104.json"), ("smd", "ae_dae_svdd_repr_smd.json")]:
        for r in load(fn):
            if "mmd_single" in r:
                add(ds, r["family"], r["seed"], r)
    for r in load("ae_dae_svdd_repr_unsw_extra5seeds.json"):
        if "mmd_single" in r:
            add("unsw", r["family"], r["seed"], r)
    orig = load("ae_dae_svdd_latent_lodo3_comparison.json")
    by = {}
    for r in orig:
        if "seed" in r and "condition" in r and "mmd_squared" in r:
            by.setdefault((r["family"], r["seed"]), {})[r["condition"]] = r["mmd_squared"]
    for (fam, seed), d in by.items():
        if "single" in d and "lodo3" in d:
            add("unsw", fam, seed, {"mmd_single": d["single"], "mmd_lodo3": d["lodo3"]})
    for ds, fn in [("unsw", "flow_representation.json"), ("iec104", "flow_representation_iec104.json"),
                   ("smd", "flow_representation_smd.json")]:
        for r in load(fn):
            if "mmd_single" in r:
                add(ds, "flow", r["seed"], r)

    pooled = {m: ([], []) for m in ["mmd_auroc", "mmd_auprc", "wass_auroc", "wass_auprc"]}
    for ds in DATASETS:
        for fam in FAMS:
            src = repr_sources.get((ds, fam), {})
            det = {x["seed"]: x for x in recs[ds] if x["family"] == fam}
            seeds = sorted(set(src) & set(det))
            if len(seeds) < 5:
                print(f" [{ds}/{fam}] only {len(seeds)} matched seeds -- skipped")
                continue
            nar_m = np.array([src[s]["mmd_single"] - src[s]["mmd_lodo3"] for s in seeds])
            has_w = all("wasserstein_single" in src[s] for s in seeds)
            nar_w = np.array([src[s]["wasserstein_single"] - src[s]["wasserstein_lodo3"] for s in seeds]) if has_w else None
            g_auroc = np.array([det[s]["metrics"]["lodo3"]["auroc"] - det[s]["metrics"]["single"]["auroc"] for s in seeds])
            g_auprc = np.array([det[s]["metrics"]["lodo3"]["auprc"] - det[s]["metrics"]["single"]["auprc"] for s in seeds])
            out = f" [{ds}/{fam}] n={len(seeds)}: "
            for tag, nar in [("MMD", nar_m), ("Wass", nar_w)]:
                if nar is None:
                    continue
                for dn, dv in [("AUROC", g_auroc), ("AUPRC", g_auprc)]:
                    rr = stats.spearmanr(nar, dv)
                    out += f"{tag}~{dn} rho={rr.statistic:+.2f} (p={rr.pvalue:.2f})  "
            print(out)
            z = lambda v: (v - v.mean()) / (v.std(ddof=1) or 1.0)
            pooled["mmd_auroc"][0].extend(z(nar_m)); pooled["mmd_auroc"][1].extend(z(g_auroc))
            pooled["mmd_auprc"][0].extend(z(nar_m)); pooled["mmd_auprc"][1].extend(z(g_auprc))
            if nar_w is not None:
                pooled["wass_auroc"][0].extend(z(nar_w)); pooled["wass_auroc"][1].extend(z(g_auroc))
                pooled["wass_auprc"][0].extend(z(nar_w)); pooled["wass_auprc"][1].extend(z(g_auprc))
    print("\n Pooled across cells (z-scored within each cell; seeds treated as exchangeable, so p is indicative only):")
    for k, (x, y) in pooled.items():
        if len(x) > 5:
            rr = stats.spearmanr(x, y)
            print(f"   {k:>11}: n={len(x)} rho={rr.statistic:+.2f} p={rr.pvalue:.4f}")

    with open(os.path.join(PILOT, "eval_extra_followup_stats.json"), "w") as f:
        json.dump({"detection_tests": [{"dataset": d, "family": fm, "metric": m, **t} for d, fm, m, t in rows],
                   "permutation": perm_out}, f, indent=2)
    print("\nSaved eval_extra_followup_stats.json")


if __name__ == "__main__":
    main()
