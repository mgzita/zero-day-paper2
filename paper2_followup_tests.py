"""
Follow-up analyses requested by the external reviewer critique:
  (1) A formal model x target interaction test for H3 (functional
      ablation) and Representation, to check whether the "reversal"
      claims already stated in the manuscript (SVDD UNSW-vs-IEC104 H3,
      Flow UNSW-vs-SMD H3, Flow IEC104-vs-SMD Repr) are statistically
      supported, not just eyeballed from the effect-size table.
  (2) An MMD^2-vs-Wasserstein seed-level agreement check for the
      representation-level results, to see whether the two distance
      measures actually agree at the seed level or only on average.

Both use ONLY already-saved JSON files -- no retraining, no new SHAP
runs. Every loader is defensive (uses safe_load / prints MISSING rather
than crashing) so a missing file degrades that one cell rather than
killing the whole run, matching the pattern used throughout this
project's other verification scripts.

Run from project root:
    python paper2\\pilot\\paper2_followup_tests.py interaction
    python paper2\\pilot\\paper2_followup_tests.py wasserstein
    python paper2\\pilot\\paper2_followup_tests.py both      (default)
    python paper2\\pilot\\paper2_followup_tests.py selftest  (type-I error check on synthetic data)
"""

import json
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

PAPER2_ROOT = r"C:\MLProject\zero_day_project\paper2"
PILOT_DIR = os.path.join(PAPER2_ROOT, "pilot")

MODEL_FAMILIES = ["ae", "dae", "svdd"]  # Flow handled separately (no "family" key in its files)
TARGETS = ["UNSW", "IEC-104", "SMD"]

# Claims already stated in the manuscript, so we know exactly which
# contrasts the interaction test needs to speak to directly. Keyed on
# the lowercase internal model tag (matching c["model"]), not the
# title-cased label used in prose -- ae/dae/svdd/flow are literal
# lowercase strings throughout this script's loaders, and title-casing
# "flow" via .upper() previously produced "FLOW" (not "Flow"), which
# silently broke this same kind of lookup earlier in this project.
CLAIMED = {
    ("H3", "svdd", "UNSW", "IEC-104"),
    ("H3", "flow", "UNSW", "SMD"),
    ("Repr", "flow", "IEC-104", "SMD"),
}

NOTES = []


def safe_load(path):
    if not os.path.exists(path):
        NOTES.append(f"MISSING file: {path}")
        print(f"  MISSING: {path}")
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# H3 (functional ablation) loaders
# ---------------------------------------------------------------------------

def _h3_unsw_ae_dae_svdd():
    """UNSW AE/DAE/SVDD H3: two 5-seed batches, condition-tagged records."""
    batch_a = safe_load(os.path.join(PILOT_DIR, "ablation_h3_lodo3_summary.json"))
    batch_b = safe_load(os.path.join(PILOT_DIR, "ablation_h3_lodo3_summary_extraseeds.json"))
    records = []
    if batch_a is None and batch_b is None:
        return records
    all_records = (batch_a or []) + (batch_b or [])
    for fam in MODEL_FAMILIES:
        stable = {r["seed"]: r["delta_from_full"] for r in all_records
                  if r.get("family") == fam and r.get("condition") == "stable_lodo3"}
        sensitive = {r["seed"]: r["delta_from_full"] for r in all_records
                     if r.get("family") == fam and r.get("condition") == "sensitive_lodo3"}
        for seed in sorted(set(stable) & set(sensitive)):
            records.append({"model": fam, "target": "UNSW", "seed": seed,
                             "delta_stable": stable[seed], "delta_sensitive": sensitive[seed]})
    return records


def _h3_direct(path, target, model_from_file=None):
    """IEC-104 / SMD AE-DAE-SVDD files, and all Flow H3 files: flat
    per-seed records with delta_stable / delta_sensitive already present."""
    data = safe_load(path)
    records = []
    if data is None:
        return records
    for r in data:
        model = model_from_file if model_from_file is not None else r.get("family")
        if model is None or "delta_stable" not in r or "delta_sensitive" not in r:
            continue
        records.append({"model": model, "target": target, "seed": r["seed"],
                         "delta_stable": r["delta_stable"], "delta_sensitive": r["delta_sensitive"]})
    return records


def h3_cells():
    print(f"{'=' * 70}\nLoading H3 (functional ablation) cells\n{'=' * 70}")
    records = []
    records += _h3_unsw_ae_dae_svdd()
    records += _h3_direct(os.path.join(PILOT_DIR, "ae_dae_svdd_h3_iec104.json"), "IEC-104")
    records += _h3_direct(os.path.join(PILOT_DIR, "ae_dae_svdd_h3_smd.json"), "SMD")
    records += _h3_direct(os.path.join(PILOT_DIR, "flow_h3_ablation.json"), "UNSW", model_from_file="flow")
    records += _h3_direct(os.path.join(PILOT_DIR, "flow_h3_ablation_iec104.json"), "IEC-104", model_from_file="flow")
    records += _h3_direct(os.path.join(PILOT_DIR, "flow_h3_ablation_smd.json"), "SMD", model_from_file="flow")

    df = pd.DataFrame(records)
    if df.empty:
        print("  No H3 records loaded at all.")
        return df
    df["y"] = df["delta_stable"] - df["delta_sensitive"]
    # y < 0  => Stable dominant (delta_stable smaller, i.e. Stable degrades
    #           performance less than Sensitive does)   -- matches the
    #           manuscript's sign convention in Table "master".
    # y > 0  => Sensitive dominant (reversed).
    for (model, target), g in df.groupby(["model", "target"]):
        print(f"  {model:6s} {target:8s} n={len(g):2d}  mean(delta_stable-delta_sensitive)={g['y'].mean():+.4f}")
    return df


# ---------------------------------------------------------------------------
# Representation (MMD^2 / Wasserstein) loaders
# ---------------------------------------------------------------------------

def _repr_unsw_ae_dae_svdd():
    """UNSW AE/DAE/SVDD repr: original 5-seed condition-tagged file +
    5 new seeds with mmd_single/mmd_lodo3 already paired."""
    original = safe_load(os.path.join(PILOT_DIR, "ae_dae_svdd_latent_lodo3_comparison.json"))
    extra = safe_load(os.path.join(PILOT_DIR, "ae_dae_svdd_repr_unsw_extra5seeds.json"))
    records = []
    if original is not None:
        for fam in MODEL_FAMILIES:
            single = {r["seed"]: r["mmd_squared"] for r in original
                      if r.get("family") == fam and r.get("condition") == "single"}
            lodo3 = {r["seed"]: r["mmd_squared"] for r in original
                     if r.get("family") == fam and r.get("condition") == "lodo3"}
            wass_s = {r["seed"]: r.get("mean_sliced_wasserstein") for r in original
                      if r.get("family") == fam and r.get("condition") == "single"}
            wass_l = {r["seed"]: r.get("mean_sliced_wasserstein") for r in original
                      if r.get("family") == fam and r.get("condition") == "lodo3"}
            for seed in sorted(set(single) & set(lodo3)):
                records.append({"model": fam, "target": "UNSW", "seed": seed,
                                 "mmd_single": single[seed], "mmd_lodo3": lodo3[seed],
                                 "wasserstein_single": wass_s.get(seed), "wasserstein_lodo3": wass_l.get(seed)})
    if extra is not None:
        for r in extra:
            fam = r.get("family")
            if fam in MODEL_FAMILIES and "mmd_single" in r and "mmd_lodo3" in r:
                records.append({"model": fam, "target": "UNSW", "seed": r["seed"],
                                 "mmd_single": r["mmd_single"], "mmd_lodo3": r["mmd_lodo3"],
                                 "wasserstein_single": r.get("wasserstein_single"),
                                 "wasserstein_lodo3": r.get("wasserstein_lodo3")})
    return records


def _repr_direct(path, target, model_from_file=None):
    data = safe_load(path)
    records = []
    if data is None:
        return records
    for r in data:
        model = model_from_file if model_from_file is not None else r.get("family")
        if model is None or "mmd_single" not in r or "mmd_lodo3" not in r:
            continue
        records.append({"model": model, "target": target, "seed": r["seed"],
                         "mmd_single": r["mmd_single"], "mmd_lodo3": r["mmd_lodo3"],
                         "wasserstein_single": r.get("wasserstein_single"),
                         "wasserstein_lodo3": r.get("wasserstein_lodo3")})
    return records


def repr_cells():
    print(f"\n{'=' * 70}\nLoading Representation (MMD^2 / Wasserstein) cells\n{'=' * 70}")
    records = []
    records += _repr_unsw_ae_dae_svdd()
    records += _repr_direct(os.path.join(PILOT_DIR, "ae_dae_svdd_repr_iec104.json"), "IEC-104")
    records += _repr_direct(os.path.join(PILOT_DIR, "ae_dae_svdd_repr_smd.json"), "SMD")
    records += _repr_direct(os.path.join(PILOT_DIR, "flow_representation.json"), "UNSW", model_from_file="flow")
    records += _repr_direct(os.path.join(PILOT_DIR, "flow_representation_iec104.json"), "IEC-104", model_from_file="flow")
    records += _repr_direct(os.path.join(PILOT_DIR, "flow_representation_smd.json"), "SMD", model_from_file="flow")

    df = pd.DataFrame(records)
    if df.empty:
        print("  No representation records loaded at all.")
        return df
    # pct_change > 0  => gap narrowed under LODO-3/Diverse (mmd_lodo3 < mmd_single)
    df["pct_change"] = np.where(
        df["mmd_single"] != 0,
        100 * (df["mmd_single"] - df["mmd_lodo3"]) / df["mmd_single"],
        np.nan,
    )
    df["raw_diff"] = df["mmd_single"] - df["mmd_lodo3"]
    for (model, target), g in df.groupby(["model", "target"]):
        print(f"  {model:6s} {target:8s} n={len(g):2d}  mean pct_change={g['pct_change'].mean():+6.1f}%")
    return df


# ---------------------------------------------------------------------------
# Statistics: cluster-robust interaction test + ART robustness + contrasts
# ---------------------------------------------------------------------------

def cluster_wald_interaction(df, y_col="y"):
    """OLS y ~ C(model)*C(target) with cluster-robust SEs, then a Wald
    F-test on the interaction-term submatrix.

    UNSW and IEC-104 cells for the same (model, seed) share trained
    networks (both come from Single-vs-LODO-3 models trained on the
    same CIC17+CIC18 sources), so they are NOT independent observations;
    they are clustered by "{model}-{seed}". SMD uses entirely separate
    networks and its own feature space, so SMD cells get their own,
    unclustered group per row.
    """
    import statsmodels.formula.api as smf

    work = df.copy()

    def cluster_id(row):
        if row["target"] in ("UNSW", "IEC-104"):
            return f"{row['model']}-{row['seed']}"
        return f"smd-{row['model']}-{row['seed']}-{row.name}"  # effectively unclustered

    work["cluster"] = work.apply(cluster_id, axis=1)
    work["model"] = work["model"].astype(str)
    work["target"] = work["target"].astype(str)

    model = smf.ols(f"{y_col} ~ C(model) * C(target)", data=work)
    fit = model.fit(cov_type="cluster", cov_kwds={"groups": work["cluster"]})

    interaction_terms = [name for name in fit.params.index if ":" in name]
    if not interaction_terms:
        return {"error": "no interaction terms found in design matrix"}

    R = np.zeros((len(interaction_terms), len(fit.params)))
    for i, term in enumerate(interaction_terms):
        R[i, fit.params.index.get_loc(term)] = 1.0
    wald = fit.f_test(R)

    return {
        "n": len(work),
        "n_clusters": work["cluster"].nunique(),
        "interaction_terms": interaction_terms,
        "wald_F": float(np.asarray(wald.fvalue).squeeze()),
        "wald_p": float(np.asarray(wald.pvalue).squeeze()),
        "df_num": float(wald.df_num),
        "df_denom": float(wald.df_denom),
        "params": fit.params.to_dict(),
    }


def art_frame(df, y_col="y"):
    """Aligned Rank Transform: strip the model and target MAIN effects
    (group-mean centering) then rank the residual, as a nonparametric
    robustness check on the interaction test that doesn't depend on the
    OLS normality assumption."""
    work = df.copy()
    grand_mean = work[y_col].mean()
    model_effect = work.groupby("model")[y_col].transform("mean") - grand_mean
    target_effect = work.groupby("target")[y_col].transform("mean") - grand_mean
    work["art_resid"] = work[y_col] - model_effect - target_effect + grand_mean
    work["art_rank"] = stats.rankdata(work["art_resid"])
    return work


def interaction_share(df, y_col="y"):
    """Proportion of total sum-of-squares attributable to the model x
    target interaction: SS(additive model) - SS(full model), over the
    total SS. A quick, assumption-light effect-size companion to the
    Wald test."""
    import statsmodels.formula.api as smf

    work = df.copy()
    work["model"] = work["model"].astype(str)
    work["target"] = work["target"].astype(str)

    total_ss = float(((work[y_col] - work[y_col].mean()) ** 2).sum())
    if total_ss == 0:
        return {"error": "zero total variance"}

    additive = smf.ols(f"{y_col} ~ C(model) + C(target)", data=work).fit()
    full = smf.ols(f"{y_col} ~ C(model) * C(target)", data=work).fit()

    ssr_additive = float((additive.resid ** 2).sum())
    ssr_full = float((full.resid ** 2).sum())
    share = (ssr_additive - ssr_full) / total_ss
    return {"interaction_ss_share": share, "ssr_additive": ssr_additive, "ssr_full": ssr_full, "total_ss": total_ss}


def contrast(df, model, target_a, target_b, y_col="y"):
    """Within-model pairwise target contrast. Paired (matched by seed)
    when both targets are UNSW/IEC-104, since those share trained
    networks per seed; Welch's t-test (unpaired, unequal variance) when
    SMD is one of the two targets, since SMD uses separate networks and
    is not seed-matched to the other target in any meaningful sense."""
    a = df[(df["model"] == model) & (df["target"] == target_a)]
    b = df[(df["model"] == model) & (df["target"] == target_b)]
    if len(a) < 2 or len(b) < 2:
        return None

    shared_networks = {target_a, target_b} <= {"UNSW", "IEC-104"}
    if shared_networks:
        merged = a[["seed", y_col]].merge(b[["seed", y_col]], on="seed", suffixes=("_a", "_b"))
        if len(merged) < 2:
            return None
        va, vb = merged[f"{y_col}_a"].values, merged[f"{y_col}_b"].values
        diffs = va - vb
        t, p = stats.ttest_rel(va, vb)
        n = len(merged)
        d_std = diffs.std(ddof=1)
        d = float(diffs.mean() / d_std) if d_std > 0 else float("nan")
        se = diffs.std(ddof=1) / np.sqrt(n)
        ci = (float(diffs.mean() - 1.96 * se), float(diffs.mean() + 1.96 * se))
        test_used = "paired t-test (shared networks)"
    else:
        va, vb = a[y_col].values, b[y_col].values
        t, p = stats.ttest_ind(va, vb, equal_var=False)
        n = min(len(va), len(vb))
        pooled_sd = np.sqrt((va.var(ddof=1) + vb.var(ddof=1)) / 2)
        d = float((va.mean() - vb.mean()) / pooled_sd) if pooled_sd > 0 else float("nan")
        diff_mean = va.mean() - vb.mean()
        se = np.sqrt(va.var(ddof=1) / len(va) + vb.var(ddof=1) / len(vb))
        ci = (float(diff_mean - 1.96 * se), float(diff_mean + 1.96 * se))
        test_used = "Welch's t-test (independent networks)"

    return {
        "model": model, "target_a": target_a, "target_b": target_b,
        "n": n, "mean_a": float(va.mean()), "mean_b": float(vb.mean()),
        "diff": float(va.mean() - vb.mean()), "ci_95": ci,
        "t": float(t), "p": float(p), "cohens_d": d, "test_used": test_used,
    }


def run_interaction():
    h3 = h3_cells()
    repr_df = repr_cells()

    results = {"H3": {}, "Repr": {}}

    for label, df, y_col in [("H3", h3, "y"), ("Repr", repr_df, "pct_change")]:
        print(f"\n{'=' * 70}\n{label}: omnibus model x target interaction test\n{'=' * 70}")
        if df.empty or df["model"].nunique() < 2 or df["target"].nunique() < 2:
            print(f"  Insufficient data for {label} interaction test.")
            continue

        wald = cluster_wald_interaction(df, y_col=y_col)
        print(f"  Cluster-robust Wald F-test on interaction terms: "
              f"F({wald.get('df_num')},{wald.get('df_denom')})={wald.get('wald_F'):.3f}, "
              f"p={wald.get('wald_p'):.5f}  (n={wald.get('n')}, {wald.get('n_clusters')} clusters)")

        art = art_frame(df, y_col=y_col)
        # ART on ranks: re-run the same Wald test using the ART-ranked outcome
        art_wald = cluster_wald_interaction(art, y_col="art_rank")
        print(f"  ART robustness check (ranked, main effects removed): "
              f"F({art_wald.get('df_num')},{art_wald.get('df_denom')})={art_wald.get('wald_F'):.3f}, "
              f"p={art_wald.get('wald_p'):.5f}")

        share = interaction_share(df, y_col=y_col)
        print(f"  Interaction SS share of total variance: {share.get('interaction_ss_share'):.3f}")

        results[label] = {"wald": wald, "art_wald": art_wald, "interaction_share": share}

        print(f"\n  {label}: within-model pairwise target contrasts (Holm-corrected)")
        contrasts = []
        for model in df["model"].unique():
            for ta, tb in [("UNSW", "IEC-104"), ("UNSW", "SMD"), ("IEC-104", "SMD")]:
                c = contrast(df, model, ta, tb, y_col=y_col)
                if c is not None:
                    contrasts.append(c)

        if contrasts:
            pvals = [c["p"] for c in contrasts]
            reject, p_adj, _, _ = multipletests(pvals, method="holm")
            for c, p_a, rej in zip(contrasts, p_adj, reject):
                c["p_holm"] = float(p_a)
                c["holm_significant"] = bool(rej)
                claimed_flag = "*" if (label, c["model"], c["target_a"], c["target_b"]) in CLAIMED else " "
                print(f"    {claimed_flag} {c['model']:6s} {c['target_a']:8s} vs {c['target_b']:8s}  "
                      f"diff={c['diff']:+.4f}  d={c['cohens_d']:+.3f}  "
                      f"p={c['p']:.5f}  p_holm={c['p_holm']:.5f}  "
                      f"[{c['test_used']}]" + ("  <-- Holm-significant" if rej else ""))
            results[label]["contrasts"] = contrasts

    return results


# ---------------------------------------------------------------------------
# Wasserstein / MMD agreement check
# ---------------------------------------------------------------------------

def run_wasserstein():
    df = repr_cells()
    print(f"\n{'=' * 70}\nMMD^2 vs Wasserstein seed-level agreement check\n{'=' * 70}")
    results = {}
    if df.empty:
        print("  No representation data loaded.")
        return results

    df = df.dropna(subset=["wasserstein_single", "wasserstein_lodo3"])
    if df.empty:
        print("  No rows have both Wasserstein values populated (older UNSW batch may lack them).")
        return results

    df["wass_pct_change"] = np.where(
        df["wasserstein_single"] != 0,
        100 * (df["wasserstein_single"] - df["wasserstein_lodo3"]) / df["wasserstein_single"],
        np.nan,
    )
    df["w_ratio"] = np.where(df["wasserstein_single"] != 0,
                              df["wasserstein_lodo3"] / df["wasserstein_single"], np.nan)
    df["mmd_sign"] = np.sign(df["pct_change"])
    df["wass_sign"] = np.sign(df["wass_pct_change"])

    for (model, target), g in df.groupby(["model", "target"]):
        g = g.dropna(subset=["mmd_sign", "wass_sign", "wass_pct_change"])
        if g.empty:
            continue
        n = len(g)
        agree = int((g["mmd_sign"] == g["wass_sign"]).sum())
        t_p = None
        if n >= 2 and g["wasserstein_single"].nunique() > 1 or g["wasserstein_lodo3"].nunique() > 1:
            try:
                _, t_p = stats.ttest_rel(g["wasserstein_single"], g["wasserstein_lodo3"])
            except Exception:
                t_p = None
        rho, rho_p = (stats.spearmanr(g["pct_change"], g["wass_pct_change"])
                      if n >= 3 else (float("nan"), float("nan")))
        mean_w_ratio = float(g["w_ratio"].mean())

        label = "agree" if agree == n else ("DISAGREE (mean direction)" if
              np.sign(g["pct_change"].mean()) != np.sign(g["wass_pct_change"].mean())
              else "mixed at seed level")

        print(f"  {model:6s} {target:8s} n={n:2d}  seed-level sign agreement={agree}/{n}  "
              f"mean MMD% ={g['pct_change'].mean():+6.1f}  mean Wass%={g['wass_pct_change'].mean():+6.1f}  "
              f"Spearman rho={rho:.3f} (p={rho_p:.3f})  "
              f"paired-t p(Wass)={t_p if t_p is None else round(t_p, 5)}  "
              f"w_ratio(mean Diverse/Single)={mean_w_ratio:.3f}  [{label}]")

        results[f"{model}_{target}"] = {
            "n": n, "sign_agreement": agree, "mean_mmd_pct_change": float(g["pct_change"].mean()),
            "mean_wass_pct_change": float(g["wass_pct_change"].mean()), "spearman_rho": float(rho),
            "spearman_p": float(rho_p), "wass_paired_t_p": t_p, "mean_w_ratio": mean_w_ratio, "label": label,
        }

    return results


# ---------------------------------------------------------------------------
# Monte Carlo self-test: type-I error rate of the interaction test under a
# purely additive null, with heteroskedastic cells and shared-network
# dependence between UNSW/IEC-104 seeds baked in.
# ---------------------------------------------------------------------------

def selftest(n_sim=300, seed=0):
    print(f"\n{'=' * 70}\nSelf-test: type-I error rate under an additive null (n_sim={n_sim})\n{'=' * 70}")
    rng = np.random.default_rng(seed)
    models = ["ae", "dae", "svdd", "flow"]
    targets = ["UNSW", "IEC-104", "SMD"]
    seeds = [42, 123, 456, 789, 999, 111, 222, 333, 444, 555]

    model_effects = {m: rng.normal(0, 0.05) for m in models}
    target_effects = {t: rng.normal(0, 0.05) for t in targets}

    def simulate_once():
        rows = []
        network_noise = {(m, s): rng.normal(0, 0.03) for m in models for s in seeds}
        for m in models:
            for t in targets:
                sd = rng.uniform(0.03, 0.15)
                for s in seeds:
                    shared = network_noise[(m, s)] if t in ("UNSW", "IEC-104") else 0.0
                    y = model_effects[m] + target_effects[t] + shared + rng.normal(0, sd)
                    rows.append({"model": m, "target": t, "seed": s, "y": y})
        return pd.DataFrame(rows)

    raw_rejections = 0
    art_rejections = 0
    for _ in range(n_sim):
        df = simulate_once()
        wald = cluster_wald_interaction(df, y_col="y")
        if wald.get("wald_p", 1.0) < 0.05:
            raw_rejections += 1
        art = art_frame(df, y_col="y")
        art_wald = cluster_wald_interaction(art, y_col="art_rank")
        if art_wald.get("wald_p", 1.0) < 0.05:
            art_rejections += 1

    raw_rate = raw_rejections / n_sim
    art_rate = art_rejections / n_sim
    print(f"  Empirical rejection rate (raw OLS, nominal alpha=0.05): {raw_rate:.3f}")
    print(f"  Empirical rejection rate (ART, nominal alpha=0.05):     {art_rate:.3f}")
    print("  Both should be reasonably close to 0.05 if the test's size is well-calibrated "
          "under this design's dependence structure; a rate far above 0.05 means the test "
          "is anti-conservative here and p-values should be interpreted cautiously.")
    return {"n_sim": n_sim, "raw_rejection_rate": raw_rate, "art_rejection_rate": art_rate}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"

    output = {}
    if mode in ("interaction", "both"):
        output["interaction"] = run_interaction()
    if mode in ("wasserstein", "both"):
        output["wasserstein"] = run_wasserstein()
    if mode == "selftest":
        output["selftest"] = selftest()

    out_path = os.path.join(PILOT_DIR, "followup_tests_results.json")
    try:
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nSaved {out_path}")
    except Exception as e:
        print(f"\nCould not save results JSON: {e}")

    if NOTES:
        print(f"\n{'=' * 70}\nNOTES / assumptions surfaced during this run\n{'=' * 70}")
        for n in NOTES:
            print(f"  - {n}")

    print("\nCaveat: UNSW and IEC-104 cells for the same (model, seed) are treated as "
          "non-independent (clustered by model+seed) because both come from models "
          "trained on the same CIC17+CIC18 source pair; SMD cells use separate "
          "networks and are treated as independent. If that assumption is wrong for "
          "any file combined here, the cluster-robust p-values should be re-checked.")


if __name__ == "__main__":
    main()
