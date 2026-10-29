"""
Reviewer comment #3 + #7: rerun the manuscript's architecture x target interaction test for H3 (functional
retraining ablation) on RAW AUROC instead of |AUROC-0.5|.

Uses ONLY saved result files (no retraining) and the manuscript's own statistics from
paper2_followup_tests.py (same folder): OLS  y ~ C(model) * C(target), cluster-robust Wald F on the 6
interaction terms, ART rank robustness check, interaction share of variance, and the within-model
target contrasts with Holm correction.

Outcome, same sign convention as the manuscript:
    manuscript :  y = delta_stable - delta_sensitive,  delta_g = sep(full) - sep(group g only),  sep = |AUROC - 0.5|
    raw AUROC  :  y = delta_stable - delta_sensitive,  delta_g = AUROC(full) - AUROC(group g only)
                    = AUROC(sensitive-only) - AUROC(stable-only)
    y < 0  ->  the Stable-only model keeps MORE of the detection (Stable dominant)
    y > 0  ->  the Sensitive-only model keeps more (reversed)

Sections
  1. Reproduction: the manuscript's own number from the stored deltas (should match F(6,79) = 10.377 unless the
     saved files changed since).
  2. Consistency: |AUROC-0.5| recomputed from the raw fields equals the stored deltas.
  3. RAW AUROC interaction, with three treatments of dependence (A manuscript clustering, B by seed, C none).
  4. Per-cell dominant group on raw vs on |AUROC-0.5|, and the claimed target "reversals" with Holm correction.

Run from the repo folder (needs paper2_followup_tests.py beside it):
    python -u h3_interaction_raw_auroc.py
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

import paper2_followup_tests as pft

PILOT = pft.PILOT_DIR
FAMS = ["ae", "dae", "svdd"]


def load_raw_h3():
    """Per model x target x seed: auroc_full / auroc_stable / auroc_sensitive, plus the stored deltas."""
    rows, missing = [], []

    def add(model, target, seed, rec, full, stab, sens):
        row = {"model": model, "target": target, "seed": seed, "auroc_full": full, "auroc_stable": stab,
               "auroc_sensitive": sens}
        row["stored_delta_stable"] = rec.get("delta_stable")
        row["stored_delta_sensitive"] = rec.get("delta_sensitive")
        rows.append(row)

    # UNSW AE / DAE / SVDD: two 5-seed batches, condition-tagged records
    a = pft.safe_load(os.path.join(PILOT, "ablation_h3_lodo3_summary.json")) or []
    b = pft.safe_load(os.path.join(PILOT, "ablation_h3_lodo3_summary_extraseeds.json")) or []
    recs = a + b
    for fam in FAMS:
        by = {c: {r["seed"]: r for r in recs if r.get("family") == fam and r.get("condition") == c and "auroc" in r}
              for c in ("full_lodo3", "stable_lodo3", "sensitive_lodo3")}
        for s in sorted(set(by["full_lodo3"]) & set(by["stable_lodo3"]) & set(by["sensitive_lodo3"])):
            rec = {"delta_stable": by["stable_lodo3"][s].get("delta_from_full"),
                   "delta_sensitive": by["sensitive_lodo3"][s].get("delta_from_full")}
            add(fam, "UNSW", s, rec, by["full_lodo3"][s]["auroc"], by["stable_lodo3"][s]["auroc"],
                by["sensitive_lodo3"][s]["auroc"])

    def direct(path, target, model=None):
        data = pft.safe_load(path)
        if data is None:
            return
        for r in data:
            m = model if model is not None else r.get("family")
            need = ("auroc_full", "auroc_stable", "auroc_sensitive")
            if m is None or not all(k in r for k in need):
                missing.append(f"{os.path.basename(path)} seed {r.get('seed')}: raw AUROC fields absent")
                continue
            add(m, target, r["seed"], r, r["auroc_full"], r["auroc_stable"], r["auroc_sensitive"])

    # IEC-104 / SMD / Flow: raw AUROC only exists after rerun_h3_save_raw.py
    for key, target, model in (("ae_iec104", "IEC-104", None), ("ae_smd", "SMD", None), ("flow_unsw", "UNSW", "flow"),
                               ("flow_iec104", "IEC-104", "flow"), ("flow_smd", "SMD", "flow")):
        path = os.path.join(PILOT, f"h3_raw_auroc_{key}.json")
        if not os.path.exists(path):
            missing.append(f"h3_raw_auroc_{key}.json not found -> run: python -u rerun_h3_save_raw.py --which {key}")
            continue
        direct(path, target, model)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["y_raw"] = df["auroc_sensitive"] - df["auroc_stable"]
        sep = lambda c: (df[c] - 0.5).abs()
        df["y_sep"] = sep("auroc_sensitive") - sep("auroc_stable")
        df["y_stored"] = df["stored_delta_stable"] - df["stored_delta_sensitive"]
    return df, missing


def wald(df, y, how):
    """A = manuscript clustering, B = by seed, C = none."""
    import statsmodels.formula.api as smf
    if how == "A":
        r = pft.cluster_wald_interaction(df, y_col=y)
        return r["wald_F"], r["wald_p"], int(r["df_num"]), int(r["df_denom"])
    work = df.copy()
    work["model"], work["target"] = work["model"].astype(str), work["target"].astype(str)
    fit = smf.ols(f"{y} ~ C(model) * C(target)", data=work)
    res = fit.fit() if how == "C" else fit.fit(cov_type="cluster", cov_kwds={"groups": work["seed"].astype(str).values})
    names = list(res.params.index)
    sel = [i for i, n in enumerate(names) if ":" in n]
    R = np.zeros((len(sel), len(names)))
    for k, i in enumerate(sel):
        R[k, i] = 1.0
    w = res.wald_test(R, use_f=True, scalar=True)
    return float(w.statistic), float(w.pvalue), int(w.df_num), int(w.df_denom)


def main():
    out = {}
    print("=" * 100 + "\n1. REPRODUCTION of the manuscript's H3 interaction from the stored deltas\n" + "=" * 100)
    h3 = pft.h3_cells()
    if not h3.empty:
        r = pft.cluster_wald_interaction(h3, y_col="y")
        print(f"\n  stored deltas: F({r['df_num']:.0f},{r['df_denom']:.0f}) = {r['wald_F']:.3f}, p = {r['wald_p']:.5f}  "
              f"(n={r['n']}, {r['n_clusters']} clusters)   manuscript: F(6,79) = 10.377")
        out["reproduction"] = {"F": r["wald_F"], "p": r["wald_p"], "n": r["n"]}

    df, missing = load_raw_h3()
    print("\n" + "=" * 100 + "\n2. RAW AUROC data and consistency with the stored deltas\n" + "=" * 100)
    for m in missing:
        print("  MISSING:", m)
    if df.empty:
        print("  No raw-AUROC H3 records found.")
        return
    cells = df.groupby(["model", "target"]).size()
    if df["target"].nunique() < 2:
        print("  Only one target has raw AUROC so far: the model x target interaction cannot be tested yet.")
        return
    print(f"  {len(df)} rows in {len(cells)} model x target cells (expect 120 rows, 12 cells)")
    have = df.dropna(subset=["y_stored"])
    if len(have):
        diff = (have["y_stored"] - have["y_sep"]).abs().max()
        print(f"  max |stored (delta_stable - delta_sensitive) - recomputed from raw AUROC| = {diff:.2e}"
              + ("   [OK]" if diff < 1e-6 else "   [MISMATCH: raw and stored files disagree]"))
        out["consistency_max_abs_diff"] = float(diff)

    print("\n" + "=" * 100 + "\n3. Interaction model x target (Wald F on the 6 interaction terms)\n" + "=" * 100)
    out["interaction"] = {}
    for y, lab in (("y_sep", "|AUROC-0.5| recomputed (manuscript's metric)"), ("y_raw", "RAW AUROC")):
        print(f"\n  {lab}")
        for how, desc in (("A", "cluster model-seed for UNSW/IEC-104 (manuscript)"),
                          ("B", "cluster by seed (10 clusters: sensitivity only)"), ("C", "no clustering")):
            F, p, d1, d2 = wald(df, y, how)
            out["interaction"][f"{y}_{how}"] = {"F": F, "p": p, "df": [d1, d2]}
            print(f"    {how}. {desc:<52} F({d1},{d2}) = {F:7.3f}   p = {p:.5f}")
        art = pft.art_frame(df, y_col=y)
        aw = pft.cluster_wald_interaction(art, y_col="art_rank")
        sh = pft.interaction_share(df, y_col=y)
        print(f"    ART rank check (A-clustering): F({aw['df_num']:.0f},{aw['df_denom']:.0f}) = {aw['wald_F']:.3f}, p = {aw['wald_p']:.5f}"
              f" | interaction share of variance = {sh['interaction_ss_share']:.3f}")
        out["interaction"][f"{y}_ART"] = {"F": aw["wald_F"], "p": aw["wald_p"], "share": sh["interaction_ss_share"]}

    print("\n" + "=" * 100 + "\n4. Dominant group per cell (y < 0 = Stable-only keeps more detection) and claimed reversals\n" + "=" * 100)
    cell_rows, pvals = [], []
    for (model, target), g in df.groupby(["model", "target"]):
        if len(g) < 3:
            continue
        t_raw = stats.ttest_1samp(g["y_raw"], 0.0)
        t_sep = stats.ttest_1samp(g["y_sep"], 0.0)
        cell_rows.append({"model": model, "target": target, "n": len(g), "y_raw": float(g["y_raw"].mean()),
                          "p_raw": float(t_raw.pvalue), "y_sep": float(g["y_sep"].mean()), "p_sep": float(t_sep.pvalue),
                          "inverted_full_seeds": int((g["auroc_full"] < 0.5).sum())})
        pvals.append(t_raw.pvalue)
    adj = multipletests(pvals, method="fdr_bh")[1] if pvals else []
    for r, a in zip(cell_rows, adj):
        r["bh_adj_p_raw"] = float(a)
        dom = lambda v: "Stable" if v < 0 else "Sensitive"
        flag = "" if dom(r["y_raw"]) == dom(r["y_sep"]) else "   <-- dominant group differs"
        print(f"  {r['model']:>5} {r['target']:>8} n={r['n']:>2} | raw y={r['y_raw']:+.3f} (BH p={a:.3f}, {dom(r['y_raw'])}) | "
              f"|.| y={r['y_sep']:+.3f} (p={r['p_sep']:.3f}, {dom(r['y_sep'])}) | full-model AUROC<0.5 in {r['inverted_full_seeds']} seeds{flag}")
    out["cells"] = cell_rows

    for y, lab in (("y_sep", "|AUROC-0.5|"), ("y_raw", "RAW AUROC")):
        cons = []
        for model in df["model"].unique():
            for ta, tb in [("UNSW", "IEC-104"), ("UNSW", "SMD"), ("IEC-104", "SMD")]:
                c = pft.contrast(df, model, ta, tb, y_col=y)
                if c:
                    cons.append(c)
        if cons:
            ph = multipletests([c["p"] for c in cons], method="holm")[1]
            print(f"\n  Within-model target contrasts on {lab} (Holm over {len(cons)}; * = reversal claimed in the manuscript)")
            for c, p in zip(cons, ph):
                star = "*" if ("H3", c["model"], c["target_a"], c["target_b"]) in pft.CLAIMED else " "
                print(f"   {star} {c['model']:>5} {c['target_a']:>8} vs {c['target_b']:<8} diff={c['diff']:+.3f} d={c['cohens_d']:+.2f} "
                      f"p={c['p']:.4f} Holm={p:.4f} {'SIG' if p < 0.05 else ''}")
                c["p_holm"] = float(p)
            out[f"contrasts_{y}"] = cons

    path = os.path.join(PILOT, "h3_interaction_raw_auroc.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
