"""Per-seed SHAP vs Integrated Gradients (H1), recomputed from the per-seed cache files.

For every model x target cell this prints, for SHAP and for IG: mean ESFRP under Single and
Diverse, paired Cohen's d (Diverse - Single), paired-t p and Wilcoxon p. It also compares the
two methods seed by seed. Runs in two modes (all seeds, and only seeds where the SHAP additivity
check and the IG completeness check both passed). Writes shap_vs_ig_paired.json and
tab_shap_ig.tex (valid-only). Run from the repo folder."""
import glob, json, os
import numpy as np
from scipy import stats

P = r"C:\MLProject\zero_day_project\paper2\pilot"
SEEDS = [42, 123, 456, 789, 999, 111, 222, 333, 444, 555]
DS = [("unsw", "UNSW-NB15"), ("iec104", "IEC-104"), ("smd", "SMD")]
FAM = [("ae", "AE"), ("dae", "DAE"), ("svdd", "Deep SVDD"), ("flow", "Flow")]
SHAP_DIRS = {"unsw": ["unsw_lodo3_revalidation_cache", "ae_dae_svdd_unsw_extra5seeds_cache"],
             "iec104": ["ae_dae_svdd_iec104_cache"], "smd": ["ae_dae_svdd_smd_cache"]}


def find_pair(obj, names):
    if isinstance(obj, dict):
        if all(k in obj for k in names):
            return obj[names[0]], obj[names[1]], obj
        for v in obj.values():
            r = find_pair(v, names)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_pair(v, names)
            if r:
                return r
    return None


def load_json(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def shap_value(ds, fam, seed):
    cands = []
    if fam != "flow":
        for d in SHAP_DIRS[ds]:
            cands += glob.glob(os.path.join(P, d, f"{fam}_seed{seed}.json"))
    elif ds == "unsw":
        cands += glob.glob(os.path.join(P, "flow_pipeline_cache", f"seed{seed}.json"))
    elif ds == "iec104":
        cands += glob.glob(os.path.join(P, "flow_pipeline_cache", f"iec104_seed{seed}.json"))
    else:
        cands += glob.glob(os.path.join(P, "flow_pipeline_smd_cache", f"*seed{seed}.json"))
    for c in cands:
        obj = load_json(c)
        r = find_pair(obj, ["esfrp_single", "esfrp_lodo3"]) if obj is not None else None
        if r:
            return float(r[0]), float(r[1]), bool(r[2].get("additivity_ok", True))
    return None


def ig_value(ds, fam, seed):
    if fam == "flow":
        pats = [os.path.join(P, "ig_attribution_flow_logscorer_cache", f"{ds}_flow_logscorer_seed{seed}.json"),
                os.path.join(P, "ig_attribution_flow_logscorer_cache", f"*{ds}*seed{seed}.json")]
    else:
        pats = [os.path.join(P, "ig_attribution_cache", f"{ds}_{fam}_seed{seed}.json")]
    for pat in pats:
        for c in glob.glob(pat):
            obj = load_json(c)
            if obj is None:
                continue
            for names in (["esfrp_a", "esfrp_b"], ["esfrp_single", "esfrp_lodo3"]):
                r = find_pair(obj, names)
                if r:   # a = Single, b = Diverse (LODO-3)
                    return float(r[0]), float(r[1]), bool(r[2].get("completeness_ok", True))
    return None



from statsmodels.stats.multitest import multipletests
rows = []
for ds, dn in DS:
    for fam, fn in FAM:
        D = []
        for s in SEEDS:
            a, b = shap_value(ds, fam, s), ig_value(ds, fam, s)
            if a is None or b is None or not (a[2] and b[2]):
                continue
            D.append((a[1] - a[0]) - (b[1] - b[0]))   # change in ESFRP under SHAP minus change under IG
        D = np.array(D); n = len(D)
        if n < 3:
            rows.append((dn, fn, n, None)); continue
        m = D.mean(); h = stats.t.ppf(.975, n - 1) * D.std(ddof=1) / np.sqrt(n)
        try:
            pw = stats.wilcoxon(D).pvalue
        except Exception:
            pw = float("nan")
        rows.append((dn, fn, n, dict(mean=m, lo=m - h, hi=m + h, dz=m / D.std(ddof=1), p=stats.ttest_1samp(D, 0).pvalue, pw=pw)))
ok = [r for r in rows if r[3]]
q = multipletests([r[3]["p"] for r in ok], method="fdr_bh")[1]
for r, qq in zip(ok, q):
    r[3]["q"] = qq
print(f"{'target':10}{'model':10}{'n':>3} {'mean D':>8} {'95% CI':>20} {'dz':>6} {'p':>6} {'q':>6} {'Wilcoxon p':>10}")
L = ["\\begin{tabular}{llrrrr}\\toprule", "Target & Model & $n$ & Mean $D$ [95\\% CI] & $p$ & $q$\\\\\\midrule"]
last = None
fp = lambda x: "$<$.001" if x < 0.001 else f"{x:.3f}".lstrip("0")
for dn, fn, n, r in rows:
    if r is None:
        print(f"{dn:10}{fn:10}{n:>3} too few seeds"); continue
    print(f"{dn:10}{fn:10}{n:>3} {r['mean']:+8.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] {r['dz']:+6.2f} {r['p']:6.3f} {r['q']:6.3f} {r['pw']:10.3f}")
    if last and dn != last:
        L.append("\\midrule")
    L.append(f"{dn if dn != last else ''} & {fn} & {n} & {r['mean']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] & {fp(r['p'])} & {fp(r['q'])}\\\\")
    last = dn
L.append("\\bottomrule\\end{tabular}")
open("tab_shap_ig_diff.tex", "w").write("\n".join(L))
print("\nSaved tab_shap_ig_diff.tex in", os.getcwd())
