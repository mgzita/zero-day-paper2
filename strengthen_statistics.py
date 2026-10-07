"""
Strengthen the statistical framework: 95% confidence intervals,
multiple-comparison correction (Benjamini-Hochberg FDR), and post-hoc
statistical power, applied across every paired test already run in
this study.
=========================================================================
Loads the already-saved paired-test result files and, for each test:

  1. Computes a 95% confidence interval on the mean paired difference
     (t-distribution based).
  2. Computes post-hoc statistical power given the observed effect
     size and n.
  3. Collects EVERY p-value produced across the whole study and applies
     Benjamini-Hochberg FDR correction across the full family of tests.

Benjamini-Hochberg (not Bonferroni) is used because Bonferroni is
overly conservative for a study with this many related, non-independent
tests, and FDR control is the more standard choice for exploratory
multi-test settings.

Run from project root:
    python paper2\\pilot\\strengthen_statistics.py
"""

import json
import os
import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestPower
from statsmodels.stats.multitest import multipletests

PAPER2_ROOT = r"C:\MLProject\zero_day_project\paper2"
PILOT_DIR = os.path.join(PAPER2_ROOT, "pilot")
ALPHA = 0.05


def safe_load(path):
    if not os.path.exists(path):
        print(f"  [skip] not found: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def ci_and_power(n, mean_diff, t_stat):
    if n < 2 or t_stat == 0:
        return None, None, None
    se = mean_diff / t_stat if t_stat != 0 else float("nan")
    df = n - 1
    t_crit = stats.t.ppf(0.975, df)
    ci_low = mean_diff - t_crit * abs(se)
    ci_high = mean_diff + t_crit * abs(se)

    cohens_d = mean_diff / (se * np.sqrt(n)) if se != 0 else float("nan")
    try:
        power_analysis = TTestPower()
        power = power_analysis.power(effect_size=abs(cohens_d), nobs=n, alpha=ALPHA)
    except Exception:
        power = float("nan")

    return (ci_low, ci_high), cohens_d, power


def process_test_collection(name, tests_dict, all_pvalues, all_labels):
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    results = {}
    for key, t in tests_dict.items():
        n = t.get("n")
        p = t.get("p") if t.get("p") is not None else t.get("p_value")
        t_stat = t.get("t") if t.get("t") is not None else t.get("t_stat")
        mean_a = t.get("mean_a")
        mean_b = t.get("mean_b")
        if n is None or p is None or t_stat is None or mean_a is None or mean_b is None:
            print(f"  [{key}] incomplete record, skipping CI/power computation")
            continue
        mean_diff = mean_a - mean_b
        ci, d, power = ci_and_power(n, mean_diff, t_stat)
        d = d if d is not None else t.get("cohens_d")
        ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "n/a"
        power_str = f"{power:.3f}" if power is not None and not np.isnan(power) else "n/a"
        print(f"  [{key}] n={n}, mean diff={mean_diff:+.4f}, 95% CI={ci_str}, "
              f"p={p:.5f}, Cohen's d={d:.3f}, post-hoc power={power_str}")
        results[key] = {"n": n, "mean_diff": mean_diff, "ci_95": ci, "p": p,
                         "cohens_d": d, "power": power}
        all_pvalues.append(p)
        all_labels.append(f"{name} / {key}")
    return results


def main():
    all_pvalues = []
    all_labels = []
    final_output = {}

    combined = safe_load(os.path.join(PILOT_DIR, "final_combined_ae_dae_svdd_stats.json"))
    if combined:
        for dataset in ["UNSW-NB15", "IEC-104"]:
            for test in ["h1", "h3", "repr"]:
                tests_dict = combined.get(dataset, {}).get(test, {})
                if tests_dict:
                    final_output[f"{dataset}_{test}"] = process_test_collection(
                        f"{dataset} -- {test.upper()}", tests_dict, all_pvalues, all_labels)

    flow_repr = safe_load(os.path.join(PILOT_DIR, "flow_representation_paired_stats.json"))
    if flow_repr:
        tests_dict = {}
        for dataset_name, stats_dict in flow_repr.items():
            n = stats_dict.get("n")
            p = stats_dict.get("p_value")
            t_stat = stats_dict.get("t_stat")
            mean_a = stats_dict.get("mmd_single_mean")
            mean_b = stats_dict.get("mmd_lodo3_mean")
            if None not in (n, p, t_stat, mean_a, mean_b):
                tests_dict[dataset_name] = {"n": n, "p": p, "t": t_stat,
                                             "mean_a": mean_a, "mean_b": mean_b,
                                             "cohens_d": stats_dict.get("cohens_d")}
        if tests_dict:
            final_output["flow_representation"] = process_test_collection(
                "Normalizing Flow -- Representation (both datasets)", tests_dict, all_pvalues, all_labels)

    smd = safe_load(os.path.join(PILOT_DIR, "smd_final_stats.json"))
    if smd:
        for test in ["h1", "h3", "repr"]:
            tests_dict = smd.get(test, {})
            if tests_dict:
                final_output[f"SMD_{test}"] = process_test_collection(
                    f"SMD (Server Machine Dataset) -- {test.upper()}", tests_dict, all_pvalues, all_labels)

    flow_smd = safe_load(os.path.join(PILOT_DIR, "flow_smd_final_stats.json"))
    if flow_smd:
        tests_dict = {test: stats_dict for test, stats_dict in flow_smd.items() if stats_dict}
        if tests_dict:
            final_output["flow_smd"] = process_test_collection(
                "Normalizing Flow -- SMD (H1/H3/Representation)", tests_dict, all_pvalues, all_labels)

    # --- Flow, UNSW + IEC-104, H1 + H3: computed LIVE, no pre-saved
    # paired-stats file exists for these (only representation was
    # pre-saved). Same live-computation logic as
    # verify_all_manuscript_numbers.py, kept in sync with it.
    def live_paired_test(vals_a, vals_b):
        vals_a, vals_b = np.array(vals_a), np.array(vals_b)
        n = len(vals_a)
        diffs = vals_a - vals_b
        t, p = stats.ttest_rel(vals_a, vals_b)
        d_std = diffs.std(ddof=1)
        d = float(diffs.mean() / d_std) if d_std > 0 else float("nan")
        return {"n": n, "mean_a": float(vals_a.mean()), "mean_b": float(vals_b.mean()),
                "t": float(t), "p": float(p), "cohens_d": d}

    for dataset_key, h1_file, h3_file in [
        ("UNSW", "flow_h1_attribution.json", "flow_h3_ablation.json"),
        ("IEC-104", "flow_h1_attribution_iec104.json", "flow_h3_ablation_iec104.json"),
    ]:
        tests_dict = {}
        h1 = safe_load(os.path.join(PILOT_DIR, h1_file))
        if h1:
            recs = [r for r in h1 if r.get("additivity_ok", True) and "esfrp_single" in r]
            if len(recs) >= 2:
                tests_dict["h1"] = live_paired_test([r["esfrp_lodo3"] for r in recs],
                                                     [r["esfrp_single"] for r in recs])
        h3 = safe_load(os.path.join(PILOT_DIR, h3_file))
        if h3:
            tests_dict["h3"] = live_paired_test([r["delta_stable"] for r in h3],
                                                 [r["delta_sensitive"] for r in h3])
        if tests_dict:
            final_output[f"flow_{dataset_key}_live"] = process_test_collection(
                f"Normalizing Flow -- {dataset_key} (H1/H3, live-computed)", tests_dict,
                all_pvalues, all_labels)

    # --- IG-based H1, all 12 combinations ---
    ig = safe_load(os.path.join(PILOT_DIR, "ig_attribution_paired_stats.json"))
    if ig:
        tests_dict = {key: r for key, r in ig.items() if r}
        if tests_dict:
            final_output["ig_h1"] = process_test_collection(
                "Integrated Gradients -- H1, all 12 combinations", tests_dict,
                all_pvalues, all_labels)

    print(f"\n{'=' * 70}\nMULTIPLE-COMPARISON CORRECTION (Benjamini-Hochberg FDR, alpha={ALPHA})\n{'=' * 70}")
    print(f"Total tests collected: {len(all_pvalues)}")
    if len(all_pvalues) > 0:
        reject, p_adjusted, _, _ = multipletests(all_pvalues, alpha=ALPHA, method="fdr_bh")
        n_sig_raw = sum(1 for p in all_pvalues if p < ALPHA)
        n_sig_corrected = sum(reject)
        print(f"Significant at raw p<{ALPHA}: {n_sig_raw}/{len(all_pvalues)}")
        print(f"Significant after FDR correction: {n_sig_corrected}/{len(all_pvalues)}\n")

        print(f"{'Test':<45} {'raw p':>10} {'FDR-adj p':>10} {'Still sig?':>10}")
        print("-" * 78)
        sorted_idx = np.argsort(all_pvalues)
        for i in sorted_idx:
            print(f"{all_labels[i]:<45} {all_pvalues[i]:>10.5f} {p_adjusted[i]:>10.5f} "
                  f"{'YES' if reject[i] else 'no':>10}")

        final_output["fdr_correction"] = {
            "labels": all_labels, "raw_p": all_pvalues,
            "fdr_adjusted_p": p_adjusted.tolist(), "significant_after_fdr": reject.tolist(),
        }

    out_path = os.path.join(PILOT_DIR, "strengthened_statistics_full.json")
    with open(out_path, "w") as f:
        json.dump(final_output, f, indent=2, default=str)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
