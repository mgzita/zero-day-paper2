"""
Corrected representation-level interaction test (Wasserstein-based)
=====================================================================
Follow-up to the external reviewer's critique of the manuscript's
formal architecture x target interaction test: the representation-level
version of that test (originally reported as F(6,79)=19.332,
p<0.00001) fed the Normalizing Flow's IEC-104 cell in at its raw
MMD^2 pct_change value, the same value that Section 4.3 of the
manuscript (and flow_iec104_mmd_bandwidth_sweep.py) demonstrates is a
kernel-bandwidth measurement artifact, not a genuine effect. Using a
value in a "this value is unreliable" footnote and then in a headline
confirmatory test in the same paragraph is inconsistent, and a
reviewer is right to flag it.

This script reruns the SAME interaction test (cluster-robust Wald
F-test on the model x target interaction term, ART rank-based
robustness check, interaction SS share), but on mean sliced
Wasserstein distance instead of MMD^2 -- the bandwidth-free distance
measure already used throughout this project as the independent check
on MMD^2. It changes nothing about H3 (functional ablation); that
interaction test (F(6,79)=10.377) already used a measure with no
kernel-bandwidth confound and does not need to be rerun.

It reuses paper2_followup_tests.py's loaders and statistics functions
directly (no duplicated logic): repr_cells() already loads
wasserstein_single/wasserstein_lodo3 per seed for every model-target
cell (see _repr_direct() and _repr_unsw_ae_dae_svdd() in that file),
they just were not previously used as the interaction test's outcome
variable. This script adds a wasserstein_pct_change column (same sign
convention as the existing mmd-based pct_change: positive = gap
narrowed under Diverse/LODO-3 training) and calls the existing
cluster_wald_interaction() / art_frame() / interaction_share() on it
instead.

Cells with no saved Wasserstein value (if any) are dropped and listed
by name, so a partial rerun is visible rather than silently averaged
over.

Run from project root, same folder as paper2_followup_tests.py:
    python paper2\\pilot\\representation_interaction_wasserstein.py
"""

import json
import os

import numpy as np

import paper2_followup_tests as pft

PILOT_DIR = pft.PILOT_DIR


def build_wasserstein_df():
    df = pft.repr_cells()
    if df.empty:
        print("No representation records loaded at all -- nothing to test.")
        return df

    missing = df[df["wasserstein_single"].isna() | df["wasserstein_lodo3"].isna()]
    if not missing.empty:
        print(f"\n{'=' * 70}\nWARNING: {len(missing)} cell(s) have no saved Wasserstein value "
              f"and will be dropped from this test:\n{'=' * 70}")
        for _, r in missing.iterrows():
            print(f"  model={r['model']:6s} target={r['target']:8s} seed={r['seed']}")

    df = df.dropna(subset=["wasserstein_single", "wasserstein_lodo3"]).copy()

    # Same sign convention as the manuscript's MMD^2 pct_change:
    # positive = gap narrowed under Diverse/LODO-3 training (matches
    # the original hypothesis), negative = gap widened.
    df["wasserstein_pct_change"] = np.where(
        df["wasserstein_single"] != 0,
        100 * (df["wasserstein_single"] - df["wasserstein_lodo3"]) / df["wasserstein_single"],
        np.nan,
    )
    df = df.dropna(subset=["wasserstein_pct_change"])

    print(f"\n{'=' * 70}\nWasserstein pct_change by model x target "
          f"({len(df)} rows after dropping missing cells)\n{'=' * 70}")
    for (model, target), g in df.groupby(["model", "target"]):
        print(f"  {model:6s} {target:8s} n={len(g):2d}  "
              f"mean pct_change={g['wasserstein_pct_change'].mean():+6.1f}%")
    return df


def run():
    df = build_wasserstein_df()
    if df.empty or df["model"].nunique() < 2 or df["target"].nunique() < 2:
        print("Insufficient data for the Wasserstein-based interaction test.")
        return {"error": "insufficient data"}

    print(f"\n{'=' * 70}\nWasserstein-based Representation: omnibus model x target "
          f"interaction test\n{'=' * 70}")
    wald = pft.cluster_wald_interaction(df, y_col="wasserstein_pct_change")
    print(f"  Cluster-robust Wald F-test on interaction terms: "
          f"F({wald.get('df_num')},{wald.get('df_denom')})={wald.get('wald_F'):.3f}, "
          f"p={wald.get('wald_p'):.5f}  (n={wald.get('n')}, {wald.get('n_clusters')} clusters)")

    art = pft.art_frame(df, y_col="wasserstein_pct_change")
    art_wald = pft.cluster_wald_interaction(art, y_col="art_rank")
    print(f"  ART robustness check (ranked, main effects removed): "
          f"F({art_wald.get('df_num')},{art_wald.get('df_denom')})={art_wald.get('wald_F'):.3f}, "
          f"p={art_wald.get('wald_p'):.5f}")

    share = pft.interaction_share(df, y_col="wasserstein_pct_change")
    print(f"  Interaction SS share of total variance: {share.get('interaction_ss_share'):.3f}")

    print(f"\n  Within-model pairwise target contrasts (Holm-corrected), Wasserstein-based")
    from statsmodels.stats.multitest import multipletests
    contrasts = []
    for model in df["model"].unique():
        for ta, tb in [("UNSW", "IEC-104"), ("UNSW", "SMD"), ("IEC-104", "SMD")]:
            c = pft.contrast(df, model, ta, tb, y_col="wasserstein_pct_change")
            if c is not None:
                contrasts.append(c)
    if contrasts:
        pvals = [c["p"] for c in contrasts]
        reject, p_adj, _, _ = multipletests(pvals, method="holm")
        for c, p_a, rej in zip(contrasts, p_adj, reject):
            c["p_holm"] = float(p_a)
            c["holm_significant"] = bool(rej)
            print(f"    {c['model']:6s} {c['target_a']:8s} vs {c['target_b']:8s}  "
                  f"diff={c['diff']:+.4f}  d={c['cohens_d']:+.3f}  "
                  f"p={c['p']:.5f}  p_holm={c['p_holm']:.5f}  "
                  f"[{c['test_used']}]" + ("  <-- Holm-significant" if rej else ""))

    result = {"wald": wald, "art_wald": art_wald, "interaction_share": share, "contrasts": contrasts}

    out_path = os.path.join(PILOT_DIR, "representation_interaction_wasserstein.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved {out_path}")
    return result


if __name__ == "__main__":
    run()
