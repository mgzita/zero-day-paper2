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


def paired(single, diverse):
    s, d = np.array(single), np.array(diverse)
    diff = d - s
    n = len(diff)
    if n < 3 or diff.std(ddof=1) == 0:
        return dict(n=n, mean_single=float(s.mean()) if n else None, mean_diverse=float(d.mean()) if n else None,
                    d=None, p_t=None, p_w=None)
    try:
        pw = float(stats.wilcoxon(d, s).pvalue)
    except Exception:
        pw = None
    return dict(n=n, mean_single=float(s.mean()), mean_diverse=float(d.mean()),
                d=float(diff.mean() / diff.std(ddof=1)), p_t=float(stats.ttest_rel(d, s).pvalue), p_w=pw)


def fp(x):
    return "$<$.001" if x < 0.001 else f"{x:.3f}".lstrip("0")


def run(valid_only):
    res, rows = {}, []
    pool_s, pool_i = [], []
    for ds, dn in DS:
        for fam, fn in FAM:
            S1, S2, I1, I2, ok = [], [], [], [], []
            for s in SEEDS:
                a, b = shap_value(ds, fam, s), ig_value(ds, fam, s)
                if a is None or b is None:
                    continue
                if valid_only and not (a[2] and b[2]):
                    continue
                S1.append(a[0]); S2.append(a[1]); I1.append(b[0]); I2.append(b[1])
            n = len(S1)
            if n < 3:
                rows.append((dn, fn, n, None)); continue
            x = np.array(S2) - np.array(S1)
            y = np.array(I2) - np.array(I1)
            rho, p = stats.spearmanr(x, y)
            same = int(np.sum(np.sign(x) == np.sign(y)))
            pool_s += list((x - x.mean()) / (x.std(ddof=1) or 1)); pool_i += list((y - y.mean()) / (y.std(ddof=1) or 1))
            res[f"{ds}/{fam}"] = dict(n=n, shap=paired(S1, S2), ig=paired(I1, I2), rho=float(rho), rho_p=float(p), same_sign_seeds=same,
                                      same_sign_mean=bool(np.sign(x.mean()) == np.sign(y.mean())))
            rows.append((dn, fn, n, res[f"{ds}/{fam}"]))
    return res, rows, pool_s, pool_i


allres = {}
for mode in (False, True):
    label = "VALID SEEDS ONLY" if mode else "ALL SEEDS"
    res, rows, ps, pi = run(mode)
    allres["valid" if mode else "all"] = res
    print("=" * 118)
    print(label)
    print("=" * 118)
    print(f"{'target':10}{'model':10}{'n':>3} | {'SHAP: Single':>12} {'Diverse':>8} {'d':>6} {'p':>6} | {'IG: Single':>10} {'Diverse':>8} {'d':>6} {'p':>6} | {'rho':>5} {'p':>5} {'same sign':>9}")
    L = ["\\begin{tabular}{llrrrrrrrr}\\toprule",
         " & & & \\multicolumn{2}{c}{SHAP} & \\multicolumn{2}{c}{IG} & \\multicolumn{3}{c}{SHAP vs IG per seed}\\\\",
         "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-10}",
         "Target & Model & $n$ & $d$ & $p$ & $d$ & $p$ & $\\rho$ & $p$ & Same sign\\\\\\midrule"]
    last = None
    for dn, fn, n, r in rows:
        if r is None:
            print(f"{dn:10}{fn:10}{n:>3} | too few paired seeds"); continue
        s, i = r["shap"], r["ig"]
        print(f"{dn:10}{fn:10}{n:>3} | {s['mean_single']:12.4f} {s['mean_diverse']:8.4f} {s['d']:+6.2f} {s['p_t']:6.3f} | "
              f"{i['mean_single']:10.4f} {i['mean_diverse']:8.4f} {i['d']:+6.2f} {i['p_t']:6.3f} | {r['rho']:+5.2f} {r['rho_p']:5.3f} {r['same_sign_seeds']:>5}/{n:<3}")
        if mode:
            if last and dn != last:
                L.append("\\midrule")
            L.append(f"{dn if dn != last else ''} & {fn} & {n} & {s['d']:+.2f} & {fp(s['p_t'])} & {i['d']:+.2f} & {fp(i['p_t'])} & {r['rho']:+.2f} & {fp(r['rho_p'])} & {r['same_sign_seeds']}/{n}\\\\")
            last = dn
    if mode:
        L.append("\\bottomrule\\end{tabular}")
        open("tab_shap_ig.tex", "w").write("\n".join(L))
    if ps:
        rp, pp = stats.spearmanr(ps, pi)
        tot = len(ps)
        same = sum(v["same_sign_seeds"] for v in res.values())
        cells_same_mean = sum(v["same_sign_mean"] for v in res.values())
        print(f"\nPooled seeds n={tot}: Spearman rho={rp:+.3f} (p={pp:.4f}); same sign of change in {same}/{tot} seeds")
        print(f"Cells where the MEAN change has the same sign under SHAP and IG: {cells_same_mean}/{len(res)}")
        print(f"Cells with positive per-seed rho: {sum(v['rho'] > 0 for v in res.values())}/{len(res)}")
        allres[("pooled_valid" if mode else "pooled_all")] = dict(n=tot, rho=float(rp), p=float(pp), same=same, cells_same_mean=int(cells_same_mean))
    print()

json.dump(allres, open("shap_vs_ig_paired.json", "w"), indent=1, default=float)
print("Saved shap_vs_ig_paired.json and tab_shap_ig.tex in", os.getcwd())
