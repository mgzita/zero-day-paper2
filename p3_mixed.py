"""Mixed-effects check for RQ2 (Single vs Multi, CIC17, mean evasion L1-L5)."""
import sys, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
d = pd.read_csv(sys.argv[1])
d = d[(d["mode"] == "calibrated") & (d.domain == "seen_cic17") & d.condition.isin(["single", "multi"])].copy()
d["evasion"] = d[[f"caught_evade_L{k}" for k in range(1, 6)]].mean(axis=1) * 100
d["multi"] = (d.condition == "multi").astype(int)
d["seed_id"] = d.family + "_" + d.seed.astype(str)

def report(name, res):
    b, se, p = res.params["multi"], res.bse["multi"], res.pvalues["multi"]
    lo, hi = res.conf_int().loc["multi"]
    print(f"{name}\n  Multi effect = {b:+.1f} pp (95% CI {lo:+.1f} to {hi:+.1f}), p = {p:.4f}\n")

# A) detector type as random intercept AND random slope (effect allowed to differ by detector)
m_a = smf.mixedlm("evasion ~ multi", d, groups=d.family, re_formula="~multi").fit(reml=True)
report("A) random intercept + slope by detector", m_a)
# B) detector as fixed effect, seed-within-detector as random intercept (pairing by seed)
m_b = smf.mixedlm("evasion ~ multi + C(family)", d, groups=d.seed_id).fit(reml=True)
report("B) detector fixed, seed random (paired)", m_b)
# C) same as B on detection rate: Multi should NOT change how many attacks are caught
d["detect"] = d.detection_rate_at_1pct_fpr * 100
m_c = smf.mixedlm("detect ~ multi + C(family)", d, groups=d.seed_id).fit(reml=True)
report("C) detection rate (control), detector fixed, seed random", m_c)
