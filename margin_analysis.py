"""
Paper 3, RQ3 follow-up: do Multi detectors catch attacks with a thinner margin?
=========================================================================
Feature-sensitivity showed Multi detectors rely LESS on attacker-controllable
features, yet they are easier to evade. A natural explanation is the margin:
if Multi's caught attacks sit just above the alarm threshold, a small push is
enough, however insensitive the detector is per feature.

This reads the per-flow files the evasion grid already wrote
(grid/flows/<family>_<condition>_seed<seed>_both.csv.gz) -- no new runs --
and, for caught CIC-IDS2017 attacks at the calibrated (1% FPR) threshold,
compares Single vs Multi on:

    relative_margin  = (clean score - threshold) / threshold
    share_thin       = fraction of caught attacks within 25% of the threshold
    evaded           = fraction evaded at any budget (min_level >= 0)

It then asks whether margin explains evasion: a logistic model of evasion
on log(margin) per family, and whether Multi still adds evasion once margin
is held fixed.

Run (from the folder with the other Paper 3 scripts):
    python margin_analysis.py
"""

import glob
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RESULTS = r"C:\MLProject\zero_day_paper2_mtsst\paper3\results"
FLOWS = os.path.join(RESULTS, "grid", "flows")
THIN = 0.25


def load():
    rows = []
    for path in glob.glob(os.path.join(FLOWS, "*_both.csv.gz")):
        name = os.path.basename(path).replace("_both.csv.gz", "")
        family, condition, seed = name.split("_")
        if condition not in ("single", "multi"):
            continue
        f = pd.read_csv(path)
        f = f[f.domain == "seen_cic17/calibrated"]
        if f.empty:
            continue
        f["family"], f["condition"], f["seed"] = family, condition, int(seed.replace("seed", ""))
        rows.append(f)
    if not rows:
        raise SystemExit(f"No per-flow files found in {FLOWS}")
    return pd.concat(rows, ignore_index=True)


def main():
    f = load()
    f["evaded"] = (f.min_level >= 0).astype(int)
    f["thin"] = (f.relative_margin < THIN).astype(int)
    f["log_margin"] = np.log(f.relative_margin.clip(lower=1e-4))

    per_model = f.groupby(["family", "condition", "seed"]).agg(
        median_margin=("relative_margin", "median"),
        share_thin=("thin", "mean"),
        evaded=("evaded", "mean")).reset_index()
    per_model.to_csv(os.path.join(RESULTS, "margin_per_model.csv"), index=False)

    pd.set_option("display.width", 200)
    t = per_model.groupby(["family", "condition"])[["median_margin", "share_thin", "evaded"]].mean()
    t["share_thin"] *= 100
    t["evaded"] *= 100
    print("Caught CIC-IDS2017 attacks, mean over 10 seeds")
    print("(median_margin = how far above the threshold; share_thin = % within 25% of it)\n")
    print(t.round(2).to_string())

    import statsmodels.formula.api as smf
    pm = per_model.copy()
    pm["multi"] = (pm.condition == "multi").astype(int)
    pm["sid"] = pm.family + "_" + pm.seed.astype(str)
    print("\nSingle vs Multi (paired mixed model: detector fixed, seed random)")
    for col, lab in [("share_thin", "share of thin-margin attacks (pp)"),
                     ("median_margin", "median relative margin")]:
        y = pm[col] * (100 if col == "share_thin" else 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = smf.mixedlm("y ~ multi + C(family)", pm.assign(y=y), groups=pm.sid).fit(reml=True)
        lo, hi = r.conf_int().loc["multi"]
        print(f"  {lab:38s}: Multi effect {r.params['multi']:+.3f} (95% CI {lo:+.3f} to {hi:+.3f}), "
              f"p = {r.pvalues['multi']:.4f}")

    print("\nDoes margin explain evasion? Flow-level logistic model per family:")
    print("  evaded ~ log(margin) + multi   (multi coefficient = effect left AFTER margin)")
    f["multi"] = (f.condition == "multi").astype(int)
    for fam, g in f.groupby("family"):
        if g.evaded.nunique() < 2:
            print(f"  {fam:5s}: evasion constant, skipped")
            continue
        m0 = smf.logit("evaded ~ multi", g).fit(disp=0)
        m1 = smf.logit("evaded ~ log_margin + multi", g).fit(disp=0)
        print(f"  {fam:5s}: margin coef {m1.params['log_margin']:+.2f} (p={m1.pvalues['log_margin']:.1e}); "
              f"Multi odds ratio {np.exp(m0.params['multi']):.2f} alone -> "
              f"{np.exp(m1.params['multi']):.2f} after margin")
    print(f"\nSaved {os.path.join(RESULTS, 'margin_per_model.csv')}")


if __name__ == "__main__":
    main()
