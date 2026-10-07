"""
Formal statistical test for the noise dose-response "inverted-U"
pattern, rather than relying on eyeballing the means.
=========================================================================
Loads the per-seed, per-noise-level results already saved by
noise_dose_response_sweep.py and fits a quadratic regression:

    pct_change ~ beta_0 + beta_1 * noise_std + beta_2 * noise_std^2

An inverted-U (rise then fall) implies beta_2 should be significantly
NEGATIVE. Also reports a linear-only comparison and an F-test for
whether the quadratic term significantly improves the fit.

Run from project root:
    python paper2\\pilot\\noise_trend_test.py
"""

import json
import os
import numpy as np
import statsmodels.api as sm

PAPER2_ROOT = r"C:\MLProject\zero_day_project\paper2"
PILOT_DIR = os.path.join(PAPER2_ROOT, "pilot")


def safe_load(path):
    if not os.path.exists(path):
        print(f"NOT FOUND: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def main():
    results = safe_load(os.path.join(PILOT_DIR, "noise_dose_response_summary.json"))
    if results is None:
        return

    noise_std = np.array([r["noise_std"] for r in results], dtype=float)
    pct_change = np.array([r["pct_change"] for r in results], dtype=float)

    print(f"{'=' * 70}\nData summary\n{'=' * 70}")
    unique_noise = sorted(set(noise_std))
    for n in unique_noise:
        vals = pct_change[noise_std == n]
        print(f"  sigma={n}: n_seeds={len(vals)}, mean={vals.mean():+.2f}%, "
              f"std={vals.std(ddof=1) if len(vals) > 1 else 0:.2f}%")

    X_quad = np.column_stack([noise_std, noise_std ** 2])
    X_quad = sm.add_constant(X_quad)
    model_quad = sm.OLS(pct_change, X_quad).fit()

    print(f"\n{'=' * 70}\nQuadratic regression: pct_change ~ b0 + b1*sigma + b2*sigma^2\n{'=' * 70}")
    print(model_quad.summary().tables[1])
    b2 = model_quad.params[2]
    b2_p = model_quad.pvalues[2]
    print(f"\nQuadratic term (b2): {b2:.4f}, p={b2_p:.5f}")
    if b2 < 0 and b2_p < 0.05:
        verdict = "Inverted-U CONFIRMED: quadratic term significantly negative"
    elif b2 < 0:
        verdict = "Inverted-U shape present in the point estimate, but NOT statistically significant"
    else:
        verdict = "No inverted-U pattern supported (quadratic term not negative)"
    print(f"Verdict: {verdict}")

    X_lin = sm.add_constant(noise_std)
    model_lin = sm.OLS(pct_change, X_lin).fit()

    print(f"\n{'=' * 70}\nLinear-only model (for comparison): pct_change ~ b0 + b1*sigma\n{'=' * 70}")
    print(model_lin.summary().tables[1])

    f_stat, f_p, _ = model_quad.compare_f_test(model_lin)
    print(f"\n{'=' * 70}\nF-test: does the quadratic term improve fit over linear-only?\n{'=' * 70}")
    print(f"F={f_stat:.3f}, p={f_p:.5f}")
    if f_p < 0.05:
        print("The quadratic (inverted-U) model fits significantly better than a linear-only model.")
    else:
        print("No significant improvement from adding the quadratic term -- "
              "the data does not statistically distinguish an inverted-U from a simpler "
              "linear or flat relationship at this seed count.")

    output = {
        "quadratic_b2": float(b2), "quadratic_b2_p": float(b2_p),
        "quadratic_r2": float(model_quad.rsquared),
        "linear_r2": float(model_lin.rsquared),
        "f_test_stat": float(f_stat), "f_test_p": float(f_p),
        "verdict": verdict,
        "n_total_points": len(pct_change),
    }
    out_path = os.path.join(PILOT_DIR, "noise_trend_test_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
