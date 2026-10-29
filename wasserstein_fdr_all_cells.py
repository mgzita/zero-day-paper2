"""
Reviewer comment #10: sliced Wasserstein applied CONSISTENTLY next to MMD^2 in every model x target cell,
with multiple-testing control across ALL cells. From saved results only.

For each of the 12 cells: paired t-test Single vs Diverse(LODO-3) on MMD^2 and on mean sliced Wasserstein.
Effect = % reduction under Diverse = 100*(single - lodo3)/single   (>0: Diverse closer to target-benign data)
BH-FDR over the 12 MMD^2 tests, over the 12 Wasserstein tests, and over all 24 as one family.
Flags: missing Wasserstein seeds, direction differences, significance differences.

Run from the repo folder:   python -u wasserstein_fdr_all_cells.py
"""
import json, os
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import paper2_followup_tests as pft

ALPHA = 0.05


def pct(a, b):
    return 100 * (a - b) / a if a else np.nan


def test_cell(g, s, l):
    g = g.dropna(subset=[s, l])
    if len(g) < 3:
        return None
    d = g[s].values - g[l].values
    t, p = stats.ttest_rel(g[s], g[l])
    sd = d.std(ddof=1)
    return {"n": len(g), "mean_single": float(g[s].mean()), "mean_diverse": float(g[l].mean()),
            "pct_reduction": float(pct(g[s].mean(), g[l].mean())), "seeds_reduced": int((d > 0).sum()),
            "t": float(t), "p": float(p), "dz": float(d.mean() / sd) if sd > 0 else float("nan")}


def main():
    df = pft.repr_cells()
    if df.empty:
        return
    rows = []
    for (model, target), g in df.groupby(["model", "target"]):
        rows.append({"model": model, "target": target, "n_seeds": len(g),
                     "n_wass": int(g[["wasserstein_single", "wasserstein_lodo3"]].dropna().shape[0]),
                     "mmd": test_cell(g, "mmd_single", "mmd_lodo3"),
                     "wass": test_cell(g, "wasserstein_single", "wasserstein_lodo3")})
    for key in ("mmd", "wass"):
        idx = [i for i, r in enumerate(rows) if r[key]]
        adj = multipletests([rows[i][key]["p"] for i in idx], method="fdr_bh")[1]
        for i, a in zip(idx, adj):
            rows[i][key]["q_within"] = float(a)
    allp = [(i, k) for i, r in enumerate(rows) for k in ("mmd", "wass") if r[k]]
    adj = multipletests([rows[i][k]["p"] for i, k in allp], method="fdr_bh")[1]
    for (i, k), a in zip(allp, adj):
        rows[i][k]["q_all"] = float(a)

    print("\n" + "=" * 120)
    print(f"{'cell':<14}{'n':>3} {'nW':>3} | {'MMD2 %red':>10} {'seeds':>6} {'p':>9} {'q(12)':>8} {'q(24)':>8} | "
          f"{'Wass %red':>10} {'seeds':>6} {'p':>9} {'q(12)':>8} {'q(24)':>8} | flag")
    print("=" * 120)
    flags = []
    for r in rows:
        m, w = r["mmd"], r["wass"]
        f = lambda x, k, fmt: format(x[k], fmt) if x else "   n/a"
        flag = []
        if r["n_wass"] < r["n_seeds"]:
            flag.append(f"Wasserstein missing for {r['n_seeds'] - r['n_wass']} seed(s)")
        if m and w:
            if np.sign(m["pct_reduction"]) != np.sign(w["pct_reduction"]):
                flag.append("DIRECTION DIFFERS")
            if (m["q_all"] < ALPHA) != (w["q_all"] < ALPHA):
                flag.append("significance differs")
        if not w:
            flag.append("NO Wasserstein")
        txt = "; ".join(flag)
        if txt:
            flags.append((r["model"], r["target"], txt))
        print(f"{r['model']:<6}{r['target']:<8}{r['n_seeds']:>3} {r['n_wass']:>3} | "
              f"{f(m,'pct_reduction','+10.1f')} {(str(m['seeds_reduced'])+'/'+str(m['n'])) if m else '':>6} {f(m,'p','9.4f')} "
              f"{f(m,'q_within','8.4f')} {f(m,'q_all','8.4f')} | "
              f"{f(w,'pct_reduction','+10.1f')} {(str(w['seeds_reduced'])+'/'+str(w['n'])) if w else '':>6} {f(w,'p','9.4f')} "
              f"{f(w,'q_within','8.4f')} {f(w,'q_all','8.4f')} | {txt}")
    nm = sum(1 for r in rows if r["mmd"] and r["mmd"]["q_within"] < ALPHA)
    nw = sum(1 for r in rows if r["wass"] and r["wass"]["q_within"] < ALPHA)
    na = sum(1 for r in rows for k in ("mmd", "wass") if r[k] and r[k]["q_all"] < ALPHA)
    print(f"\nSignificant at BH q<{ALPHA}: MMD2 {nm}/12 | Wasserstein {nw}/12 | combined family {na}/{len(allp)}")
    print("Cells flagged:" if flags else "No cells flagged.")
    for fl in flags:
        print("  ", *fl)
    out = os.path.join(pft.PILOT_DIR, "wasserstein_fdr_all_cells.json")
    json.dump({"cells": rows, "flags": flags, "n_sig_mmd": nm, "n_sig_wass": nw, "n_sig_all": na},
              open(out, "w"), indent=2, default=float)
    print("Saved", out)


if __name__ == "__main__":
    main()
