"""
Paper 3, RQ3: WHY are Multi detectors easier to evade?
=========================================================================
An attack evades when the clean margin above the threshold is smaller than
how far the attacker's actions can push the score down. Margin was measured
already (it explains little). This script measures the push directly, in the
attacker's own units, and decomposes it feature by feature.

For each caught CIC-IDS2017 attack (calibrated 1% FPR threshold), at the
undisguised flow, using autograd through the same action -> feature
recomputation the attack uses:

  push_pad    = -(d score / d pad_bytes)  / margin     per byte of padding
                (log10 median reported over flows where the action lowers the score;
                 the share of such flows is reported separately as *_pushes_down_pct)
  push_delay  = -(d score / d delay_us)   / margin     per microsecond of delay
                (margin = score - threshold; positive = actions lower the score)

By the chain rule each push is a sum of per-feature contributions:

  d score / d action = sum_j (d score / d x_j) * (d x_j / d action)

and padding / delay each move several features at once. Their contributions
can reinforce or cancel, which per-feature sensitivity cannot see:

  alignment = -(sum_j c_j) / (sum_j |c_j|)
       +1 : every moved feature pushes the score DOWN together
        0 : contributions cancel
       -1 : every moved feature pushes the score UP

Hypothesis: Multi is less sensitive per feature but MORE aligned, so the
total push per byte / microsecond is larger relative to its margin.

Per-feature contributions are also summed by ESS group, to say WHICH
features carry the push.

No training; existing checkpoints only. Keep next to pilot_evasion_ae.py
and run_evasion_grid.py.

Run:
    python action_gradient.py --selftest
    python action_gradient.py
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import torch

import pilot_evasion_ae as P
import run_evasion_grid as G
from feature_sensitivity import GROUP

warnings.filterwarnings("ignore")

MAX_FLOWS = 1000
BATCH = 256


def per_flow_push(model, z_clean, tf, threshold):
    """Push per unit action and per-feature contributions, for a batch of flows."""
    raw = torch.tensor(tf.to_raw(z_clean.astype(np.float64)), dtype=torch.float64)
    n = len(raw)
    out = {}
    for action in ("pad", "delay"):
        a = torch.zeros(n, dtype=torch.float64, requires_grad=True)
        pad = a if action == "pad" else torch.zeros(n, dtype=torch.float64)
        delay = a if action == "delay" else torch.zeros(n, dtype=torch.float64)
        raw_mod = P.apply_actions(raw, pad, delay, None)
        z = P.build_input(z_clean, raw_mod, tf)             # (n, 18), depends on a
        z_leaf = z.detach().clone().requires_grad_(True)
        s = model.score(z_leaf.float()).double()
        g_feat, = torch.autograd.grad(s.sum(), z_leaf)       # d score / d x_j
        dz_da = torch.stack([torch.autograd.grad(z[:, j].sum(), a, retain_graph=True,
                                                 allow_unused=True)[0]
                             if z[:, j].requires_grad else torch.zeros(n, dtype=torch.float64)
                             for j in range(z.shape[1])], dim=1)
        dz_da = torch.nan_to_num(dz_da)
        contrib = (g_feat * dz_da).detach().numpy()          # (n, 18) per-feature contribution
        margin = np.maximum(s.detach().numpy() - threshold, 1e-12)
        total = contrib.sum(axis=1)
        out[action] = {"push": -total / margin,
                       "alignment": -total / np.maximum(np.abs(contrib).sum(axis=1), 1e-30),
                       "moves": np.abs(dz_da.numpy()).sum(axis=1) > 0,
                       "contrib_share": -contrib / np.maximum(np.abs(contrib).sum(axis=1, keepdims=True), 1e-30)}
    return out


def analyse(jobs, X, y, tf):
    rows = []
    for i, (family, condition, seed, model) in enumerate(jobs, 1):
        with torch.no_grad():
            s = model.score(torch.tensor(X, dtype=torch.float32)).numpy()
        thr = np.percentile(s[y == 0], P.THRESHOLD_PERCENTILE)
        idx = np.where((y == 1) & (s >= thr))[0]
        rng = np.random.default_rng(seed)
        if len(idx) > MAX_FLOWS:
            idx = np.sort(rng.choice(idx, MAX_FLOWS, replace=False))
        if len(idx) == 0:
            continue
        acc = {a: {"push": [], "alignment": [], "moves": [], "share": []} for a in ("pad", "delay")}
        for b in range(0, len(idx), BATCH):
            res = per_flow_push(model, X[idx[b:b + BATCH]].astype(np.float64), tf, thr)
            for a in res:
                acc[a]["push"].append(res[a]["push"])
                acc[a]["alignment"].append(res[a]["alignment"])
                acc[a]["moves"].append(res[a]["moves"])
                acc[a]["share"].append(res[a]["contrib_share"])
        row = {"family": family, "condition": condition, "seed": seed, "n": len(idx)}
        for a in ("pad", "delay"):
            push = np.concatenate(acc[a]["push"])
            align = np.concatenate(acc[a]["alignment"])
            moves = np.concatenate(acc[a]["moves"])
            share = np.vstack(acc[a]["share"])
            if moves.any():
                helps = moves & (push > 0)
                # size of the push, only where the action actually lowers the score
                row[f"{a}_log_push"] = (float(np.median(np.log10(push[helps]))) if helps.any() else np.nan)
                row[f"{a}_pushes_down_pct"] = 100 * float((push[moves] > 0).mean())
                row[f"{a}_alignment"] = float(np.median(align[moves]))
                sh = share[moves].mean(axis=0)
                for g in ("Stable", "Middle", "Sensitive", "Redundant"):
                    row[f"{a}_share_{g}"] = float(sum(sh[j] for j, f in enumerate(P.FEATURE_NAMES)
                                                      if GROUP[f] == g))
        rows.append(row)
        print(f"[{i}/{len(jobs)}] {family} {condition} seed{seed}: {len(idx)} flows, "
              f"pad alignment {row.get('pad_alignment', float('nan')):+.2f}, "
              f"delay alignment {row.get('delay_alignment', float('nan')):+.2f}")
    return pd.DataFrame(rows)


def report(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "action_gradient_per_model.csv"), index=False)
    pd.set_option("display.width", 220)
    cols = ["pad_log_push", "pad_pushes_down_pct", "pad_alignment",
            "delay_log_push", "delay_pushes_down_pct", "delay_alignment"]
    print("\nPush per unit action relative to margin (log10, median over flows where the action lowers")
    print("the score), share of flows where it lowers the score, and alignment")
    print("(alignment +1 = moved features all lower the score together; 0 = they cancel)\n")
    print(df.groupby(["family", "condition"])[[c for c in cols if c in df]].mean().round(2).to_string())

    for a in ("pad", "delay"):
        sc = [f"{a}_share_{g}" for g in ("Stable", "Middle", "Sensitive", "Redundant") if f"{a}_share_{g}" in df]
        if sc:
            print(f"\nWhere the {a} push comes from (signed share of contributions by ESS group; "
                  f"positive = lowers the score):")
            print(df.groupby("condition")[sc].mean().round(2).to_string())

    import statsmodels.formula.api as smf
    m = df.copy()
    m["multi"] = (m.condition == "multi").astype(int)
    m["sid"] = m.family + "_" + m.seed.astype(str)
    print("\nSingle vs Multi (paired mixed model: detector fixed, seed random)")
    for col in cols + [c for c in df if "_share_" in c]:
        if col not in m or m[col].isna().all() or (m[col].fillna(0) == 0).all():
            continue   # e.g. Stable/Redundant: padding and delay never move them
        x = m.dropna(subset=[col])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = smf.mixedlm(f"{col} ~ multi + C(family)", x, groups=x.sid).fit(reml=True)
        lo, hi = r.conf_int().loc["multi"]
        note = f"  (= x{10 ** r.params['multi']:.2f} push)" if col.endswith("log_push") else ""
        print(f"  {col:24s}: Multi effect {r.params['multi']:+.3f} (95% CI {lo:+.3f} to {hi:+.3f}), "
              f"p = {r.pvalues['multi']:.4f}{note}")
    print(f"\nSaved {os.path.join(out_dir, 'action_gradient_per_model.csv')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=P.OUT_DIR)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _, tf, _, syn = P.load_synthetic()
        X, y = syn["seen_cic17"]
        jobs = []
        for family in G.FAMILIES:
            for condition in ("single", "multi"):
                for seed in (1, 2):
                    torch.manual_seed(seed + (0 if condition == "single" else 7))
                    m = G.new_model(family)
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad_(False)
                    jobs.append((family, condition, seed, m))
        report(analyse(jobs, X, y, tf), os.path.join(args.out_dir, "action_gradient_selftest"))
        return

    tf, _, tests = G.load_shared_data()
    X, y = tests["cic17"]
    jobs = []
    for family in G.FAMILIES:
        for condition in ("single", "multi"):
            for seed in G.SEEDS:
                model, path = G.load_model(family, condition, seed)
                if model is None:
                    print(f"SKIP {family} {condition} seed{seed}: {path}")
                    continue
                jobs.append((family, condition, seed, model))
    report(analyse(jobs, X, y, tf), args.out_dir)


if __name__ == "__main__":
    main()
