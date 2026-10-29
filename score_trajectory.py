"""
Paper 3, RQ3 figure: what happens to the alarm as the disguise grows?
=========================================================================
For every Single and Multi detector (4 families x 10 seeds), take the
CIC-IDS2017 attacks it catches at 1% false alarms and apply the SAME fixed
disguise to all of them, at increasing strength t in [0, 1]:

    padding = t * 400 bytes per forward packet (never beyond a 1460-byte payload)
    delay   = enough to stretch the flow to (1 + 2t) times its duration

(t = 1 is the largest budget in the evasion grid: 400 B and a 3x slowdown).
Nothing is optimised here: every detector sees exactly the same disguised
flows, so differences come from the detectors, not the attack.

Plotted: % of those attacks the detector still flags, against t, Single vs
Multi, one panel per detector (line = mean over seeds, band = 95% CI of the mean).
If multi-environment training broadens what counts as "normal", Multi's curve
should fall away faster as t grows, while the two start together at t = 0
(both catch 100% of these attacks by construction).

Outputs (paper3\\results):
    score_trajectory.csv            one row per detector x seed x strength
    score_trajectory.png / .pdf     the figure

Run:
    python score_trajectory.py
    python score_trajectory.py --selftest
    python score_trajectory.py --plot-only     # redraw from the saved CSV
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import torch

import pilot_evasion_ae as P
import run_evasion_grid as G

warnings.filterwarnings("ignore")

STRENGTHS = np.linspace(0, 1, 11)
MAX_PAD = 400.0
MAX_RATE = 3.0
MAX_FLOWS = 1000
FAMILY_TITLE = {"ae": "Autoencoder", "dae": "Denoising AE", "svdd": "Deep SVDD", "flow": "Normalizing Flow"}
COLORS = {"single": "#2a78d6", "multi": "#eb6834"}          # validated categorical slots 1-2
LABELS = {"single": "Single (1 network)", "multi": "Multi (3 networks)"}


def disguise(raw, t):
    """Fixed, non-optimised disguise of strength t (same recomputation as the attack)."""
    fpl, _, dur, _, _, n_fwd_time = P.flow_quantities(raw)
    pad = torch.clamp(torch.minimum(torch.full_like(fpl, t * MAX_PAD), P.MAX_PAYLOAD - fpl), min=0.0)
    gaps = n_fwd_time - 1.0
    delay = torch.where(gaps > 0, (t * (MAX_RATE - 1.0)) * dur / torch.clamp(gaps, min=1e-9),
                        torch.zeros_like(dur))
    return P.apply_actions(raw, pad, torch.clamp(delay, min=0.0), None)


def trajectory(model, X, y, tf, seed):
    with torch.no_grad():
        s = model.score(torch.tensor(X, dtype=torch.float32)).numpy()
    thr = np.percentile(s[y == 0], P.THRESHOLD_PERCENTILE)
    idx = np.where((y == 1) & (s >= thr))[0]
    if len(idx) == 0:
        return []
    rng = np.random.default_rng(seed)
    if len(idx) > MAX_FLOWS:
        idx = np.sort(rng.choice(idx, MAX_FLOWS, replace=False))
    z_clean = X[idx].astype(np.float64)
    raw = torch.tensor(tf.to_raw(z_clean), dtype=torch.float64)
    rows = []
    for t in STRENGTHS:
        z = P.build_input(z_clean, disguise(raw, t), tf)
        with torch.no_grad():
            st = model.score(z.float()).numpy()
        rows.append({"strength": float(t), "still_detected_pct": 100 * float((st >= thr).mean()),
                     "n_flows": len(idx)})
    return rows


def plot(df, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#2b2b29", "#6b6a64", "#e6e5e0"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted, "font.family": "DejaVu Sans"})
    fams = [f for f in G.FAMILIES if f in df.family.unique()]
    fig, axes = plt.subplots(1, len(fams), figsize=(2.6 * len(fams) + 0.6, 2.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, fam in zip(axes, fams):
        ends = {}
        for cond, ls in (("single", "-"), ("multi", "--")):
            g = df[(df.family == fam) & (df.condition == cond)]
            if g.empty:
                continue
            agg = g.groupby("strength").still_detected_pct.agg(["mean", "std", "count"])
            from scipy.stats import t as t_dist
            half = t_dist.ppf(0.975, (agg["count"] - 1).clip(lower=1)) * agg["std"] / np.sqrt(agg["count"])
            lo, hi = (agg["mean"] - half).clip(0, 100), (agg["mean"] + half).clip(0, 100)
            ax.fill_between(agg.index, lo, hi, color=COLORS[cond], alpha=0.18, lw=0)
            ax.plot(agg.index, agg["mean"], ls, color=COLORS[cond], lw=2, label=LABELS[cond])
            ends[cond] = (agg.index[-1], agg["mean"].iloc[-1])
        # end labels: one value per line, pushed apart vertically when the lines finish close together
        if len(ends) == 2 and abs(ends["single"][1] - ends["multi"][1]) < 8:
            hi, lo = sorted(ends, key=lambda c: ends[c][1], reverse=True)
            offsets = {hi: 5, lo: -5}
        else:
            offsets = {c: 0 for c in ends}
        for cond, (x_end, y_end) in ends.items():
            ax.plot([x_end], [y_end], "o", ms=3.5, color=COLORS[cond])
            ax.annotate(f"{y_end:.0f}%", (x_end, y_end), xytext=(5, offsets[cond]),
                        textcoords="offset points", va="center", fontsize=8, color=ink)
        ax.set_title(FAMILY_TITLE.get(fam, fam), fontsize=9.5, color=ink, loc="left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 102)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["none", "200 B\n2× time", "400 B\n3× time"])
        ax.grid(axis="y", color=grid, lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Caught attacks still detected (%)")
    fig.supxlabel("Disguise strength (padding per packet, flow slowdown)", fontsize=9, color=ink)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"score_trajectory.{ext}"), dpi=200, bbox_inches="tight")
    print(f"Saved figure: {os.path.join(out_dir, 'score_trajectory.png')} (and .pdf)")


def summarise(df):
    pd.set_option("display.width", 200)
    t = df[df.strength.isin([0.0, 0.2, 0.5, 1.0])].pivot_table(
        index=["family", "strength"], columns="condition", values="still_detected_pct", aggfunc="mean").round(1)
    if {"single", "multi"} <= set(t.columns):
        t["multi_minus_single"] = (t["multi"] - t["single"]).round(1)
    print("\n% of caught attacks still detected under the same fixed disguise (mean over seeds)\n")
    print(t.to_string())

    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return
    print("\nSingle vs Multi at each strength (paired mixed model: detector fixed, seed random)")
    for s in [0.2, 0.5, 1.0]:
        x = df[(df.strength == s) & df.condition.isin(["single", "multi"])].copy()
        if x.condition.nunique() < 2:
            continue
        x["multi"] = (x.condition == "multi").astype(int)
        x["sid"] = x.family + "_" + x.seed.astype(str)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = smf.mixedlm("still_detected_pct ~ multi + C(family)", x, groups=x.sid).fit(reml=True)
        lo, hi = r.conf_int().loc["multi"]
        print(f"  strength {s:.1f}: Multi effect {r.params['multi']:+.1f} pp "
              f"(95% CI {lo:+.1f} to {hi:+.1f}), p = {r.pvalues['multi']:.4f}")


def run(jobs, X, y, tf, out_dir):
    rows = []
    for i, (family, condition, seed, model) in enumerate(jobs, 1):
        r = trajectory(model, X, y, tf, seed)
        for row in r:
            row.update({"family": family, "condition": condition, "seed": seed})
        rows += r
        end = r[-1]["still_detected_pct"] if r else float("nan")
        print(f"[{i}/{len(jobs)}] {family} {condition} seed{seed}: {end:.1f}% still detected at full strength")
    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "score_trajectory.csv"), index=False)
    summarise(df)
    plot(df, out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=P.OUT_DIR)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--plot-only", action="store_true",
                    help="redraw the figure from score_trajectory.csv without rerunning the models")
    args = ap.parse_args()
    if args.plot_only:
        plot(pd.read_csv(os.path.join(args.out_dir, "score_trajectory.csv")), args.out_dir)
        return
    if args.selftest:
        model, tf, _, syn = P.load_synthetic()
        X, y = syn["seen_cic17"]
        jobs = []
        for family in G.FAMILIES:
            for condition in ("single", "multi"):
                for seed in (1, 2, 3):
                    torch.manual_seed(seed + (0 if condition == "single" else 50))
                    m = model if family == "ae" else G.new_model(family)
                    m.eval()
                    jobs.append((family, condition, seed, m))
        run(jobs, X, y, tf, os.path.join(args.out_dir, "trajectory_selftest"))
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
    run(jobs, X, y, tf, args.out_dir)


if __name__ == "__main__":
    main()
