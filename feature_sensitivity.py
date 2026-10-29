"""
Paper 3, RQ3: which features make detectors easy to push, and are they the
environment-sensitive (high-ESS) features from Paper 2?
=========================================================================
For every trained detector (family x condition x seed) this script measures,
on the CIC-IDS2017 attack flows the detector catches at 1% false alarms,
how strongly its anomaly score reacts to each input feature:

    sensitivity_j = mean over caught attacks of |d score / d x_j| / score

(the gradient in the scaled input space, divided by the flow's own score so
that detectors whose scores live on different scales are comparable).

Per-feature sensitivities are then summed within Paper 2's frozen V2 ESS
groups (Stable / Middle / Sensitive, plus the 3 features removed as
redundant), and within the attacker-controllable features used by the
evasion attack. Two views are reported:

    share     : group sensitivity / total sensitivity  (what the detector relies on)
    magnitude : group sensitivity in absolute terms    (how easily it is pushed)

Single vs Multi is then compared with the same paired mixed-effects model
used for RQ2 (detector fixed, seed random).

No training: uses the existing checkpoints. Keep this file next to
pilot_evasion_ae.py and run_evasion_grid.py.

Run:
    python feature_sensitivity.py                 # all families, conditions, seeds
    python feature_sensitivity.py --selftest      # synthetic check, ~30 s
Outputs (in paper3\\results):
    feature_sensitivity_long.csv     one row per model x feature
    feature_sensitivity_groups.csv   one row per model x group
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch

import pilot_evasion_ae as P
import run_evasion_grid as G

warnings.filterwarnings("ignore")

STABLE_V2 = ["bwd_iat_std", "fwd_iat_std", "destination_port",
             "init_win_bytes_forward", "bwd_packet_length_mean"]
SENSITIVE_V2 = ["fwd_iat_mean", "subflow_fwd_bytes", "bwd_packets_s",
                "flow_duration", "bwd_avg_bytes_bulk"]
MIDDLE_V2 = ["bwd_iat_mean", "flow_bytes_s", "fwd_iat_min",
             "fwd_packet_length_mean", "init_win_bytes_backward"]
REDUNDANT = ["bwd_packet_length_std", "fwd_avg_bytes_bulk", "fwd_avg_packets_bulk"]
GROUP = {**{f: "Stable" for f in STABLE_V2}, **{f: "Middle" for f in MIDDLE_V2},
         **{f: "Sensitive" for f in SENSITIVE_V2}, **{f: "Redundant" for f in REDUNDANT}}
assert set(GROUP) == set(P.FEATURE_NAMES)

# Features the default attack (padding + delay) actually changes
ATTACKED = [f for f in P.MUTABLE if f != "init_win_bytes_forward"]
PAD_FEATURES = ["fwd_packet_length_mean", "subflow_fwd_bytes", "flow_bytes_s"]
DELAY_FEATURES = ["fwd_iat_mean", "fwd_iat_min", "flow_duration", "bwd_packets_s"]

MAX_FLOWS = 2000
BATCH = 512


def per_feature_sensitivity(model, X):
    """Mean over flows of |d score / d x| / score, per feature (scaled space)."""
    out = np.zeros(X.shape[1])
    n = 0
    for i in range(0, len(X), BATCH):
        x = torch.tensor(X[i:i + BATCH], dtype=torch.float32, requires_grad=True)
        s = model.score(x)
        g, = torch.autograd.grad(s.sum(), x)
        rel = (g.abs() / s.detach().abs().clamp(min=1e-8).unsqueeze(1)).numpy()
        # features at the +/-10 clip are not differentiable in reality; ignore them
        rel[np.abs(X[i:i + BATCH]) >= P.CLIP - 1e-6] = np.nan
        out += np.nan_to_num(rel).sum(axis=0)
        n += len(x)
    return out / max(n, 1)


def caught_attacks(model, X, y, rng):
    """CIC17 attack flows detected at 1% false alarms on CIC17 benign traffic."""
    with torch.no_grad():
        s = model.score(torch.tensor(X, dtype=torch.float32)).numpy()
    thr = np.percentile(s[y == 0], P.THRESHOLD_PERCENTILE)
    idx = np.where((y == 1) & (s >= thr))[0]
    if len(idx) > MAX_FLOWS:
        idx = np.sort(rng.choice(idx, MAX_FLOWS, replace=False))
    return X[idx]


def analyse(jobs, X, y, out_dir):
    long_rows = []
    for i, (family, condition, seed, model) in enumerate(jobs, 1):
        rng = np.random.default_rng(seed)
        Xc = caught_attacks(model, X, y, rng)
        if len(Xc) == 0:
            print(f"[{i}/{len(jobs)}] {family} {condition} seed{seed}: no caught attacks, skipped")
            continue
        sens = per_feature_sensitivity(model, Xc)
        total = sens.sum()
        for f, v in zip(P.FEATURE_NAMES, sens):
            long_rows.append({"family": family, "condition": condition, "seed": seed,
                              "feature": f, "ess_group": GROUP[f],
                              "attacked": f in ATTACKED,
                              "sensitivity": v, "share": v / total if total > 0 else np.nan,
                              "n_flows": len(Xc)})
        print(f"[{i}/{len(jobs)}] {family} {condition} seed{seed}: {len(Xc)} caught attacks")

    long = pd.DataFrame(long_rows)
    os.makedirs(out_dir, exist_ok=True)
    long.to_csv(os.path.join(out_dir, "feature_sensitivity_long.csv"), index=False)

    keys = ["family", "condition", "seed"]
    parts = []
    for name, mask in [("Stable", long.ess_group == "Stable"),
                       ("Middle", long.ess_group == "Middle"),
                       ("Sensitive", long.ess_group == "Sensitive"),
                       ("Redundant", long.ess_group == "Redundant"),
                       ("Attacked (all)", long.attacked),
                       ("Attacked: padding", long.feature.isin(PAD_FEATURES)),
                       ("Attacked: delay", long.feature.isin(DELAY_FEATURES))]:
        g = long[mask].groupby(keys)[["sensitivity", "share"]].sum().reset_index()
        g["group"] = name
        parts.append(g)
    groups = pd.concat(parts, ignore_index=True)
    groups.to_csv(os.path.join(out_dir, "feature_sensitivity_groups.csv"), index=False)
    return long, groups


def report(long, groups):
    print("\nWhich ESS groups can the attacker touch? (structural, no model involved)")
    for g in ["Stable", "Middle", "Sensitive", "Redundant"]:
        fs = [f for f in P.FEATURE_NAMES if GROUP[f] == g]
        att = [f for f in fs if f in ATTACKED]
        print(f"  {g:9s}: {len(att)}/{len(fs)} attacked  {att}")

    pd.set_option("display.width", 200)
    t = groups.pivot_table(index="group", columns="condition", values="share", aggfunc="mean") * 100
    print("\nShare of score sensitivity (%), mean over families and seeds:")
    print(t.reindex(["Stable", "Middle", "Sensitive", "Redundant", "Attacked (all)",
                     "Attacked: padding", "Attacked: delay"]).round(1).to_string())

    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("\n(statsmodels not installed: skipping mixed-effects tests)")
        return
    print("\nSingle vs Multi (paired mixed model: detector fixed, seed random)")
    print(f"{'group':20s} {'metric':10s} {'Multi effect':>14s} {'95% CI':>20s} {'p':>8s}")
    for metric in ["share", "sensitivity"]:
        for g in ["Stable", "Middle", "Sensitive", "Attacked (all)",
                  "Attacked: padding", "Attacked: delay"]:
            x = groups[(groups.group == g) & groups.condition.isin(["single", "multi"])].copy()
            if x.condition.nunique() < 2:
                continue
            x["multi"] = (x.condition == "multi").astype(int)
            x["sid"] = x.family + "_" + x.seed.astype(str)
            # log scale for magnitudes (multiplicative effect); percentage points for shares
            x["v"] = np.log(x[metric].clip(lower=1e-12)) if metric == "sensitivity" else x[metric] * 100
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = smf.mixedlm("v ~ multi + C(family)", x, groups=x.sid).fit(reml=True)
                b, (lo, hi), p = r.params["multi"], r.conf_int().loc["multi"], r.pvalues["multi"]
            except Exception as e:  # noqa: BLE001
                print(f"{g:20s} {metric:10s} model failed: {e}")
                continue
            if metric == "sensitivity":
                eff, ci = f"x{np.exp(b):.2f}", f"x{np.exp(lo):.2f} to x{np.exp(hi):.2f}"
            else:
                eff, ci = f"{b:+.1f} pp", f"{lo:+.1f} to {hi:+.1f}"
            print(f"{g:20s} {metric:10s} {eff:>14s} {ci:>20s} {p:8.4f}")


def selftest(out_dir):
    _, _, _, syn = P.load_synthetic()
    X, y = syn["seen_cic17"]
    jobs = []
    for family in G.FAMILIES:
        for condition in ["single", "multi"]:
            for seed in (1, 2, 3):
                torch.manual_seed(seed + (0 if condition == "single" else 100))
                m = G.new_model(family)
                m.eval()
                for p in m.parameters():
                    p.requires_grad_(False)
                jobs.append((family, condition, seed, m))
    long, groups = analyse(jobs, X, y, out_dir)
    report(long, groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=G.FAMILIES, choices=G.FAMILIES)
    ap.add_argument("--conditions", nargs="+", default=G.CONDITIONS, choices=G.CONDITIONS)
    ap.add_argument("--seeds", nargs="+", type=int, default=G.SEEDS)
    ap.add_argument("--out-dir", default=P.OUT_DIR)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest(os.path.join(args.out_dir, "sensitivity_selftest"))
        return

    d = np.load(os.path.join(P.DEDUP_DIR, P.TEST_FILES["cic17"]))
    X = np.clip(d["X"], -P.CLIP, P.CLIP).astype(np.float32)
    y = d["y"].astype(int)

    jobs = []
    for family in args.families:
        for condition in args.conditions:
            for seed in args.seeds:
                model, path = G.load_model(family, condition, seed)
                if model is None:
                    print(f"SKIP {family} {condition} seed{seed}: {path} not found")
                    continue
                jobs.append((family, condition, seed, model))
    long, groups = analyse(jobs, X, y, args.out_dir)
    report(long, groups)
    print(f"\nSaved feature_sensitivity_long.csv and feature_sensitivity_groups.csv to {args.out_dir}")


if __name__ == "__main__":
    sys.exit(main())
