"""
Quantitative proof of the calibration hypothesis: does Multi training
reduce reconstruction/distance error on UNSW-NB15's BENIGN flows
specifically, consistently across all 5 seeds?
=========================================================================
Converts the visual finding (reconstruction_error plots: same latent
position, cooler color under Multi) into a hard, statistically tested
claim.

Metric: mean raw anomaly score (reconstruction error for AE/DAE,
distance-from-center for SVDD) computed ONLY on UNSW-NB15's BENIGN
flows -- isolates "how wrong is the model about NORMAL traffic,"
exactly what the calibration hypothesis is about.

For each model family:
  - Computes mean UNSW-benign anomaly score under Single and Multi,
    for each of the 5 seeds.
  - Reports per-seed reduction and whether Multi is lower every seed.
  - Runs a paired t-test AND paired Wilcoxon signed-rank test
    (matching Paper 1's own statistical conventions, Section 6.2).
  - Reports paired Cohen's d.

Requires trained checkpoints (all 5 seeds) to already exist.
EXPLORATORY analysis (Formal Protocol Section 8).

Run from project root:
    python paper2\\pilot\\prove_calibration_hypothesis.py
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

ALL_SEEDS = [42, 123, 456, 789, 999]
MODEL_FAMILIES = ["ae", "dae", "svdd"]

N_FEATURES = 18
DEDUP_DIR = r"C:\MLProject\zero_day_project\artefacts_v2\dedup"  # TODO confirm
PAPER2_ROOT = r"C:\MLProject\zero_day_project\paper2"
PILOT_DIR = os.path.join(PAPER2_ROOT, "pilot")


class AutoencoderModel(nn.Module):
    def __init__(self, input_dim=N_FEATURES):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class SVDDEncoder(nn.Module):
    def __init__(self, input_dim=N_FEATURES, latent_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64, bias=False), nn.ReLU(),
            nn.Linear(64, 32, bias=False), nn.ReLU(),
            nn.Linear(32, latent_dim, bias=False),
        )
        self.register_buffer("center", torch.zeros(latent_dim))

    def forward(self, x):
        return self.net(x)


def checkpoint_path(seed, family, condition):
    ckpt_dir = os.path.join(DEDUP_DIR, f"seed{seed}_{family}")
    fname = "dae_" if family == "dae" else f"{family}_"
    return os.path.join(ckpt_dir, f"{fname}{condition}.pt")


def compute_mean_score(family, ckpt_path, X_unsw_benign):
    if not os.path.exists(ckpt_path):
        return None
    if family in ("ae", "dae"):
        model = AutoencoderModel(N_FEATURES)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.eval()
        with torch.no_grad():
            x_t = torch.tensor(X_unsw_benign, dtype=torch.float32)
            x_hat = model(x_t)
            scores = ((x_t - x_hat) ** 2).mean(dim=1).numpy()
    elif family == "svdd":
        model = SVDDEncoder(N_FEATURES, latent_dim=16)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.eval()
        with torch.no_grad():
            z = model(torch.tensor(X_unsw_benign, dtype=torch.float32))
            scores = ((z - model.center) ** 2).sum(dim=1).numpy()
    else:
        raise ValueError(family)
    return float(np.mean(scores))


def load_unsw_benign():
    d = np.load(os.path.join(DEDUP_DIR, "test_unsw_dedup.npz"))
    X = np.clip(d["X"], -10, 10).astype(np.float32)
    y = d["y"].astype(int)
    return X[y == 0]


def cohens_d_paired(diffs):
    return float(np.mean(diffs) / np.std(diffs, ddof=1)) if np.std(diffs, ddof=1) > 0 else np.nan


def main():
    X_unsw_benign = load_unsw_benign()
    print(f"UNSW-NB15 benign flows: {len(X_unsw_benign)}\n")

    all_results = []
    for family in MODEL_FAMILIES:
        print(f"{'=' * 80}\n{family.upper()}\n{'=' * 80}")
        single_scores, multi_scores = [], []
        for seed in ALL_SEEDS:
            single_ckpt = checkpoint_path(seed, family, "single")
            multi_ckpt = checkpoint_path(seed, family, "multi")
            s_single = compute_mean_score(family, single_ckpt, X_unsw_benign)
            s_multi = compute_mean_score(family, multi_ckpt, X_unsw_benign)
            if s_single is None or s_multi is None:
                print(f"  seed {seed}: SKIPPED (checkpoint missing)")
                continue
            single_scores.append(s_single)
            multi_scores.append(s_multi)
            pct_reduction = 100 * (s_single - s_multi) / s_single if s_single != 0 else np.nan
            print(f"  seed {seed}: Single mean UNSW-benign score={s_single:.6f}, "
                  f"Multi={s_multi:.6f}, reduction={pct_reduction:+.1f}%")
            all_results.append({"family": family, "seed": seed,
                                "single_score": s_single, "multi_score": s_multi,
                                "pct_reduction": pct_reduction})

        single_arr = np.array(single_scores)
        multi_arr = np.array(multi_scores)
        diffs = single_arr - multi_arr
        n_lower = int(np.sum(multi_arr < single_arr))

        print(f"\n  Multi lower than Single in {n_lower}/{len(single_arr)} seeds")
        print(f"  Mean pct reduction: {np.mean(100 * diffs / single_arr):.1f}%")

        if len(diffs) >= 2:
            t_stat, t_p = stats.ttest_rel(single_arr, multi_arr)
            try:
                w_stat, w_p = stats.wilcoxon(single_arr, multi_arr)
            except ValueError:
                w_stat, w_p = np.nan, np.nan
            d = cohens_d_paired(diffs)
            print(f"  Paired t-test:  t={t_stat:.3f}, p={t_p:.4f}")
            if not np.isnan(w_p):
                print(f"  Wilcoxon test:  W={w_stat}, p={w_p:.4f}")
            else:
                print("  Wilcoxon test: not computable (too few / tied samples)")
            print(f"  Cohen's d (paired): {d:.3f}")

            if t_p < 0.05 and n_lower == len(single_arr):
                print(f"\n  -> PROVEN for {family.upper()}: Multi-environment training "
                      f"significantly reduces reconstruction/distance error on UNSW's "
                      f"benign traffic, consistently across all {len(single_arr)} seeds "
                      f"(p={t_p:.4f}, d={d:.2f}).")
            elif n_lower == len(single_arr):
                print(f"\n  -> Consistent direction across all seeds, but not reaching "
                      f"p<0.05 (n={len(single_arr)} is small -- consider this suggestive, "
                      f"not proven, until more seeds are available).")
            else:
                print(f"\n  -> NOT consistent -- {len(single_arr) - n_lower} seed(s) did "
                      f"not show the predicted reduction. Report honestly as mixed evidence.")

    out_path = os.path.join(PILOT_DIR, "calibration_proof_summary.csv")
    pd.DataFrame(all_results).to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
