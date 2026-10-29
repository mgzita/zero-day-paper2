"""
Reviewer comment #8 -- ESS threshold and group-size sensitivity for the NETWORK features
(used by UNSW-NB15 and IEC-104; the SMD version is smd_ess_sensitivity_sweep.py).

No model is trained: this recomputes the Stable / Sensitive / Middle groups with the same recipe as
explore_decorrelated_ess.py (Spearman redundancy removal, then per-feature Jensen-Shannon ESS across the
three blocks of train_dedup.npz, 50 histogram bins) and asks how much the groups move when the choices
the paper fixed are changed.

What it reports
  0. REPRODUCTION CHECK: does (threshold 0.85, 5 per group) give exactly STABLE_V2 / SENSITIVE_V2 / MIDDLE_V2
     and exactly the 3 excluded features? If not, the groups in the paper were not produced by this recipe.
  1. GRID: threshold {0.70 ... 0.95, none} x group size {3..7}. Jaccard overlap of Stable and Sensitive with
     the paper's groups, and how often each feature lands in Stable / Sensitive.
  2. BINS: ESS with 20 / 50 / 100 histogram bins (rank correlation with the 50-bin ESS, group overlap).
  3. BOOTSTRAP: resample the rows of each environment; how often is each feature Stable / Sensitive?
  4. TWO-SOURCE ESS: ESS from only the first two blocks (the two environments the UNSW / IEC-104 models are
     trained on) versus all three blocks. The ESS recipe labels the third block "UNSW-NB15"; if that block is
     data from the evaluation domain, groups defined with it are not independent of the target. If the
     two-source groups match the paper's groups, that concern does not change the groups.

Run from the repo folder:
    python -u ess_sensitivity_network.py
Optional:  --dedup-dir PATH  --pilot-dir PATH  --boot 50
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import rankdata, spearmanr

FEATURE_NAMES = [
    "bwd_avg_bytes_bulk", "bwd_iat_mean", "bwd_iat_std",
    "bwd_packet_length_mean", "bwd_packet_length_std", "bwd_packets_s",
    "destination_port", "flow_bytes_s", "flow_duration",
    "fwd_avg_bytes_bulk", "fwd_avg_packets_bulk", "fwd_iat_mean",
    "fwd_iat_min", "fwd_iat_std", "fwd_packet_length_mean",
    "init_win_bytes_backward", "init_win_bytes_forward", "subflow_fwd_bytes",
]
PAPER_STABLE = ["bwd_iat_std", "fwd_iat_std", "destination_port", "init_win_bytes_forward", "bwd_packet_length_mean"]
PAPER_SENSITIVE = ["fwd_iat_mean", "subflow_fwd_bytes", "bwd_packets_s", "flow_duration", "bwd_avg_bytes_bulk"]
PAPER_MIDDLE = ["bwd_iat_mean", "flow_bytes_s", "fwd_iat_min", "fwd_packet_length_mean", "init_win_bytes_backward"]
PAPER_DROPPED = {"bwd_packet_length_std", "fwd_avg_bytes_bulk", "fwd_avg_packets_bulk"}
PAPER_THRESHOLD, PAPER_N = 0.85, 5

THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]   # 1.00 = no redundancy removal
GROUP_SIZES = [3, 4, 5, 6, 7]
N_BINS = 50


def load(dedup_dir, max_rows):
    d = np.load(os.path.join(dedup_dir, "train_dedup.npz"))
    X = np.clip(d["X_train"], -10, 10).astype(np.float32)
    n = len(X) // 3
    envs = [X[:n], X[n:2 * n], X[2 * n:3 * n]]
    if len(X) > max_rows:
        sub = X[np.random.default_rng(0).choice(len(X), max_rows, replace=False)]
    else:
        sub = X
    return sub, envs


def decorrelate(X, threshold):
    """Same greedy rule as explore_decorrelated_ess.fit_correlation_selector (Spearman, keep first of a pair)."""
    corr = np.corrcoef(np.apply_along_axis(rankdata, 0, X).T)
    drop, n = set(), len(FEATURE_NAMES)
    for i in range(n):
        if FEATURE_NAMES[i] in drop:
            continue
        for j in range(i + 1, n):
            if FEATURE_NAMES[j] in drop:
                continue
            if abs(corr[i, j]) > threshold:
                drop.add(FEATURE_NAMES[j])
    return [f for f in FEATURE_NAMES if f not in drop], sorted(drop)


def ess_raw(envs, n_bins=N_BINS):
    """Mean pairwise JS divergence (squared JS distance) per feature across the given environments."""
    out = {}
    for k, f in enumerate(FEATURE_NAMES):
        edges = np.histogram_bin_edges(np.concatenate([e[:, k] for e in envs]), bins=n_bins)
        hists = []
        for e in envs:
            h, _ = np.histogram(e[:, k], bins=edges)
            h = h.astype(float)
            hists.append(h / h.sum() if h.sum() > 0 else h)
        vals = []
        for a in range(len(envs)):
            for b in range(a + 1, len(envs)):
                j = jensenshannon(hists[a], hists[b])
                vals.append(0.0 if np.isnan(j) else j ** 2)
        out[f] = float(np.mean(vals))
    return pd.Series(out)


def groups(ess, kept, n):
    """Stable = n lowest ESS, Sensitive = n highest, Middle = rest (needs at least one left over)."""
    if 2 * n >= len(kept):
        return None
    order = ess.loc[kept].sort_values(kind="mergesort")
    st, se = list(order.index[:n]), list(order.index[-n:])
    return st, se, [f for f in kept if f not in st and f not in se]


def jac(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dedup-dir", default=r"C:\MLProject\zero_day_project\artefacts_v2\dedup")
    ap.add_argument("--pilot-dir", default=r"C:\MLProject\zero_day_project\paper2\pilot")
    ap.add_argument("--boot", type=int, default=50)
    ap.add_argument("--max-rows", type=int, default=200000)
    args = ap.parse_args()

    sub, envs = load(args.dedup_dir, args.max_rows)
    print(f"training rows used for redundancy removal: {len(sub)}; environment blocks: {[len(e) for e in envs]}")
    ess = ess_raw(envs)
    res = {"ess": ess.to_dict()}

    # ---- 0. reproduction ------------------------------------------------------------------------
    kept0, dropped0 = decorrelate(sub, PAPER_THRESHOLD)
    g0 = groups(ess, kept0, PAPER_N)
    ok = {"dropped": set(dropped0) == PAPER_DROPPED,
          "stable": set(g0[0]) == set(PAPER_STABLE),
          "sensitive": set(g0[1]) == set(PAPER_SENSITIVE),
          "middle": set(g0[2]) == set(PAPER_MIDDLE)}
    print("\n" + "=" * 90 + "\n0. REPRODUCTION CHECK (threshold 0.85, 5 per group)\n" + "=" * 90)
    print(f"  dropped    : {dropped0}   matches paper: {ok['dropped']}")
    print(f"  Stable     : {g0[0]}   matches paper: {ok['stable']}")
    print(f"  Sensitive  : {g0[1]}   matches paper: {ok['sensitive']}")
    print(f"  Middle     : {g0[2]}   matches paper: {ok['middle']}")
    if not all(ok.values()):
        print("  >>> NOT reproduced exactly. The paper's groups were not produced by this recipe on this data;"
              " read the grid below as 'how much do groups move', and report this mismatch.")
    res["reproduction"] = {"matches": ok, "dropped": dropped0, "stable": g0[0], "sensitive": g0[1], "middle": g0[2]}

    # ---- 1. grid --------------------------------------------------------------------------------
    print("\n" + "=" * 90 + "\n1. GRID: Jaccard of Stable / Sensitive with the paper's groups\n" + "=" * 90)
    grid, in_stable, in_sens, n_cells = [], {f: 0 for f in FEATURE_NAMES}, {f: 0 for f in FEATURE_NAMES}, 0
    kept_by_thr = {t: decorrelate(sub, t) for t in THRESHOLDS}
    print("  thr  kept | " + "  ".join(f"n={n}  Stable/Sens" for n in GROUP_SIZES))
    for t in THRESHOLDS:
        kept, dropped = kept_by_thr[t]
        line = f"  {t:.2f} {len(kept):>4} | "
        for n in GROUP_SIZES:
            g = groups(ess, kept, n)
            if g is None:
                line += "     --         "
                continue
            js, jse = jac(g[0], PAPER_STABLE), jac(g[1], PAPER_SENSITIVE)
            line += f"     {js:.2f}/{jse:.2f}  "
            n_cells += 1
            for f in g[0]:
                in_stable[f] += 1
            for f in g[1]:
                in_sens[f] += 1
            grid.append({"threshold": t, "n": n, "kept": len(kept), "dropped": dropped, "stable": g[0],
                         "sensitive": g[1], "middle": g[2], "jaccard_stable": js, "jaccard_sensitive": jse})
        print(line)
    print(f"\n  Share of the {n_cells} valid grid cells in which each feature is Stable / Sensitive "
          f"(paper's groups marked S / E):")
    for f in sorted(FEATURE_NAMES, key=lambda x: ess[x]):
        tag = "S" if f in PAPER_STABLE else "E" if f in PAPER_SENSITIVE else "M" if f in PAPER_MIDDLE else "x"
        print(f"    [{tag}] {f:<26} ESS={ess[f]:.4f}  Stable {in_stable[f] / n_cells:>5.0%}   Sensitive {in_sens[f] / n_cells:>5.0%}")
    at_paper_n = [r for r in grid if r["n"] == PAPER_N]
    res["grid"] = grid
    res["grid_summary"] = {
        "n_cells": n_cells,
        "mean_jaccard_stable": float(np.mean([r["jaccard_stable"] for r in grid])),
        "mean_jaccard_sensitive": float(np.mean([r["jaccard_sensitive"] for r in grid])),
        "min_jaccard_stable_at_n5": float(min(r["jaccard_stable"] for r in at_paper_n)),
        "min_jaccard_sensitive_at_n5": float(min(r["jaccard_sensitive"] for r in at_paper_n)),
        "share_stable": {f: in_stable[f] / n_cells for f in FEATURE_NAMES},
        "share_sensitive": {f: in_sens[f] / n_cells for f in FEATURE_NAMES}}
    print(f"\n  mean Jaccard over the grid: Stable {res['grid_summary']['mean_jaccard_stable']:.2f}, "
          f"Sensitive {res['grid_summary']['mean_jaccard_sensitive']:.2f}")

    # ---- 2. bins --------------------------------------------------------------------------------
    print("\n" + "=" * 90 + "\n2. HISTOGRAM BINS (threshold 0.85, 5 per group)\n" + "=" * 90)
    res["bins"] = {}
    for nb in (20, 50, 100):
        e = ess_raw(envs, nb)
        g = groups(e, kept0, PAPER_N)
        rho = spearmanr(e.loc[kept0], ess.loc[kept0]).statistic
        res["bins"][nb] = {"spearman_vs_50bins": float(rho), "jaccard_stable": jac(g[0], PAPER_STABLE),
                           "jaccard_sensitive": jac(g[1], PAPER_SENSITIVE)}
        print(f"  {nb:>3} bins: rank corr with 50-bin ESS = {rho:.3f} | Jaccard Stable {res['bins'][nb]['jaccard_stable']:.2f}"
              f", Sensitive {res['bins'][nb]['jaccard_sensitive']:.2f}")

    # ---- 3. bootstrap ---------------------------------------------------------------------------
    print("\n" + "=" * 90 + f"\n3. BOOTSTRAP over rows ({args.boot} resamples; features kept as in the paper, 5 per group)\n" + "=" * 90)
    rng = np.random.default_rng(123)
    m = min(50000, min(len(e) for e in envs))
    cs, ce, js_l, je_l = {f: 0 for f in kept0}, {f: 0 for f in kept0}, [], []
    for _ in range(args.boot):
        boot = [e[rng.integers(0, len(e), m)] for e in envs]
        g = groups(ess_raw(boot), kept0, PAPER_N)
        for f in g[0]:
            cs[f] += 1
        for f in g[1]:
            ce[f] += 1
        js_l.append(jac(g[0], PAPER_STABLE))
        je_l.append(jac(g[1], PAPER_SENSITIVE))
    print(f"  mean Jaccard with paper groups: Stable {np.mean(js_l):.2f} (min {min(js_l):.2f}), "
          f"Sensitive {np.mean(je_l):.2f} (min {min(je_l):.2f})")
    for f in sorted(kept0, key=lambda x: ess[x]):
        print(f"    {f:<26} Stable {cs[f] / args.boot:>5.0%}   Sensitive {ce[f] / args.boot:>5.0%}")
    res["bootstrap"] = {"n": args.boot, "mean_jaccard_stable": float(np.mean(js_l)),
                        "mean_jaccard_sensitive": float(np.mean(je_l)),
                        "share_stable": {f: cs[f] / args.boot for f in kept0},
                        "share_sensitive": {f: ce[f] / args.boot for f in kept0}}

    # ---- 4. two-source ESS ----------------------------------------------------------------------
    print("\n" + "=" * 90 + "\n4. TWO-SOURCE ESS (first two blocks only = the two training environments) vs all three\n" + "=" * 90)
    e2 = ess_raw(envs[:2])
    g2 = groups(e2, kept0, PAPER_N)
    rho2 = spearmanr(e2.loc[kept0], ess.loc[kept0]).statistic
    res["two_source"] = {"spearman_vs_three": float(rho2), "stable": g2[0], "sensitive": g2[1],
                         "jaccard_stable": jac(g2[0], PAPER_STABLE), "jaccard_sensitive": jac(g2[1], PAPER_SENSITIVE)}
    print(f"  rank correlation of the two ESS rankings: {rho2:.3f}")
    print(f"  Stable    : {g2[0]}  (Jaccard with paper {res['two_source']['jaccard_stable']:.2f})")
    print(f"  Sensitive : {g2[1]}  (Jaccard with paper {res['two_source']['jaccard_sensitive']:.2f})")

    out = os.path.join(args.pilot_dir, "ess_sensitivity_network.json")
    os.makedirs(args.pilot_dir, exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
