"""
Reviewer comment #6: direct SHAP-vs-IG comparison (not "significant vs not significant").

For every dataset x family (12 cells) and every seed present in BOTH methods:
    shift_SHAP = ESFRP(LODO-3) - ESFRP(Single)   (KernelSHAP)
    shift_IG   = ESFRP(LODO-3) - ESFRP(Single)   (Integrated Gradients)
ESFRP is the share of attribution on the Stable group for both methods, so the
two shifts are on the same scale. We test, across seeds:
  1. paired difference  shift_SHAP - shift_IG  (t-test, Wilcoxon, Cohen's d, 95% CI)
  2. sign agreement     (seeds where both shifts point the same way)
  3. Spearman rho between the per-seed shifts
For the Flow, IG uses the log(1+NLL) scorer (same function as SHAP). The old
raw-NLL IG rows are shown too, for transparency. BH-FDR over the 12 main tests.
Per-feature attributions were not saved by the pipelines, so feature-ranking
correlations are not possible from the existing files.

Run from the repo folder:  python compare_shap_vs_ig.py
"""
import json
import os

import numpy as np
from scipy import stats

PILOT = r"C:\MLProject\zero_day_project\paper2\pilot"
FAMS = ["ae", "dae", "svdd", "flow"]
DATASETS = ["unsw", "iec104", "smd"]


def load(name):
    p = os.path.join(PILOT, name)
    if not os.path.exists(p):
        print(f"[missing] {name}")
        return []
    with open(p) as f:
        return json.load(f)


def shap_shifts():
    out = {}
    def add(ds, fam, recs):
        for r in recs:
            if r.get("additivity_ok", True) and "esfrp_single" in r:
                out.setdefault((ds, fam), {})[r["seed"]] = r["esfrp_lodo3"] - r["esfrp_single"]
    for r in load("unsw_lodo3_revalidation_summary.json") + load("ae_dae_svdd_h1_unsw_extra5seeds.json"):
        add("unsw", r.get("family"), [r])
    for ds, fn in [("iec104", "ae_dae_svdd_h1_iec104.json"), ("smd", "ae_dae_svdd_h1_smd.json")]:
        for r in load(fn):
            add(ds, r.get("family"), [r])
    for ds, fn in [("unsw", "flow_h1_attribution.json"), ("iec104", "flow_h1_attribution_iec104.json"),
                   ("smd", "flow_h1_attribution_smd.json")]:
        for r in load(fn):
            add(ds, "flow", [r])
    return out


def ig_shifts(records, only_flow=False):
    out, ok = {}, {}
    for r in records:
        if only_flow and r.get("family") != "flow":
            continue
        key = (r["dataset"], r["family"])
        out.setdefault(key, {})[r["seed"]] = r["esfrp_b"] - r["esfrp_a"]   # b = lodo3, a = single
        ok.setdefault(key, {})[r["seed"]] = bool(r.get("completeness_ok", True))
    return out, ok


def compare(shap, ig, ok=None, only_ok=False):
    seeds = sorted(set(shap) & set(ig))
    if only_ok and ok:
        seeds = [s for s in seeds if ok.get(s, True)]
    if len(seeds) < 4:
        return None
    a = np.array([shap[s] for s in seeds])
    b = np.array([ig[s] for s in seeds])
    d = a - b
    sd = d.std(ddof=1)
    se = sd / np.sqrt(len(d))
    tcrit = stats.t.ppf(0.975, len(d) - 1)
    try:
        wp = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        wp = float("nan")
    rho = stats.spearmanr(a, b)
    return {"n": len(seeds), "mean_shift_shap": float(a.mean()), "mean_shift_ig": float(b.mean()),
            "mean_diff": float(d.mean()), "ci95": [float(d.mean() - tcrit * se), float(d.mean() + tcrit * se)],
            "t_p": float(stats.ttest_rel(a, b).pvalue), "wilcoxon_p": wp,
            "cohens_d": float(d.mean() / sd) if sd > 0 else float("nan"),
            "sign_agree": int((np.sign(a) == np.sign(b)).sum()),
            "spearman_rho": float(rho.statistic), "spearman_p": float(rho.pvalue)}


def bh(ps):
    ps = np.array(ps, float)
    order = np.argsort(ps)
    adj, prev = np.empty_like(ps), 1.0
    for rank, i in zip(range(len(ps), 0, -1), order[::-1]):
        prev = min(prev, ps[i] * len(ps) / rank)
        adj[i] = prev
    return adj


def show(tag, r):
    if r is None:
        print(f"  {tag}: too few matched seeds")
        return
    print(f"  {tag}: n={r['n']}  SHAP shift={r['mean_shift_shap']:+.3f}  IG shift={r['mean_shift_ig']:+.3f}  "
          f"diff={r['mean_diff']:+.3f} (95% CI {r['ci95'][0]:+.3f}..{r['ci95'][1]:+.3f})  "
          f"t-p={r['t_p']:.4f}  wilcoxon-p={r['wilcoxon_p']:.4f}  d={r['cohens_d']:+.2f}  "
          f"same sign {r['sign_agree']}/{r['n']}  rho={r['spearman_rho']:+.2f} (p={r['spearman_p']:.2f})")


def main():
    shap = shap_shifts()
    ig_all, ok_all = ig_shifts(load("ig_attribution_all_results.json"))
    ig_flow_log, ok_flow = ig_shifts(load("ig_attribution_flow_logscorer_results.json"))

    results, pvals, names = {}, [], []
    for ds in DATASETS:
        print(f"\n{'=' * 90}\n{ds.upper()}\n{'=' * 90}")
        for fam in FAMS:
            key = (ds, fam)
            ig = ig_flow_log if fam == "flow" else ig_all
            ok = ok_flow if fam == "flow" else ok_all
            if key not in shap or key not in ig:
                print(f"  {fam}: missing data (SHAP={key in shap}, IG={key in ig})")
                continue
            print(f"\n [{fam}]" + ("  (IG with log(1+NLL) scorer = same function as SHAP)" if fam == "flow" else ""))
            r_all = compare(shap[key], ig[key])
            r_ok = compare(shap[key], ig[key], ok[key], only_ok=True)
            show("all seeds       ", r_all)
            if r_ok and r_all and r_ok["n"] != r_all["n"]:
                show("completeness-ok ", r_ok)
            if fam == "flow" and key in ig_all:
                show("old raw-NLL IG  ", compare(shap[key], ig_all[key]))
            results[f"{ds}_{fam}"] = {"all": r_all, "completeness_ok_only": r_ok}
            if r_all:
                pvals.append(r_all["t_p"]); names.append(f"{ds}_{fam}")
    if pvals:
        for n, a in zip(names, bh(pvals)):
            results[n]["bh_adj_p_t"] = float(a)
        print(f"\n{'=' * 90}\nBH-adjusted p (paired SHAP-vs-IG shift difference, 12 cells)\n{'=' * 90}")
        for n, p, a in sorted(zip(names, pvals, bh(pvals)), key=lambda x: x[1]):
            print(f"  {n:>12}: raw p={p:.4f}  adj p={a:.4f}  {'DIFFERENT' if a < 0.05 else ''}")
    out = os.path.join(PILOT, "shap_vs_ig_direct_comparison.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
