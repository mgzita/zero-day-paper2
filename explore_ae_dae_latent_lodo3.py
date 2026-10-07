"""
Non-circular latent-representation analysis: does the INTERNAL
representation shift for AE and DAE too, not just Deep SVDD?
=========================================================================
Extends the already-validated Deep SVDD latent-space test (MMD^2,
sliced Wasserstein, non-circular LODO-3 control) to AE and DAE.

Uses the SAME non-circular design already validated for Deep SVDD:
  - Single:  trained on CIC-IDS2017 only        (UNSW unseen)
  - LODO-3:  trained on CIC17 + CSE-CIC-IDS2018  (UNSW STILL unseen)

No new model training required -- reuses existing Single and LODO-3
checkpoints for all three families, all five seeds.

EXPLORATORY analysis, motivated by confirmatory H4/H1-H3 results.

Run from project root:
    python paper2\\pilot\\explore_ae_dae_latent_lodo3.py
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import wasserstein_distance

ALL_SEEDS = [42, 123, 456, 789, 999]
MODEL_FAMILIES = ["ae", "dae", "svdd"]

N_FEATURES = 18
MAX_SAMPLE = 2000

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


def load_embedder(family, ckpt_path):
    if not os.path.exists(ckpt_path):
        return None
    if family in ("ae", "dae"):
        model = AutoencoderModel(N_FEATURES)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.eval()

        def embed(X):
            with torch.no_grad():
                return model.encoder(torch.tensor(X, dtype=torch.float32)).numpy()
        return embed
    elif family == "svdd":
        model = SVDDEncoder(N_FEATURES, latent_dim=16)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.eval()

        def embed(X):
            with torch.no_grad():
                return model(torch.tensor(X, dtype=torch.float32)).numpy()
        return embed
    raise ValueError(family)


def load_unsw_benign():
    d = np.load(os.path.join(DEDUP_DIR, "test_unsw_dedup.npz"))
    X = np.clip(d["X"], -10, 10).astype(np.float32)
    y = d["y"].astype(int)
    return X[y == 0]


def load_cic17_benign_training_block():
    d = np.load(os.path.join(DEDUP_DIR, "train_dedup.npz"))
    X_train_full = np.clip(d["X_train"], -10, 10).astype(np.float32)
    n_per_env = len(X_train_full) // 3
    return X_train_full[:n_per_env]


def mmd_squared_rbf(X, Y, max_sample=MAX_SAMPLE, seed=0):
    rng = np.random.default_rng(seed)
    if len(X) > max_sample:
        X = X[rng.choice(len(X), max_sample, replace=False)]
    if len(Y) > max_sample:
        Y = Y[rng.choice(len(Y), max_sample, replace=False)]
    Z = np.vstack([X, Y])
    n_pairs = min(2000, len(Z))
    idx_a = rng.integers(0, len(Z), n_pairs)
    idx_b = rng.integers(0, len(Z), n_pairs)
    pair_d2 = np.sum((Z[idx_a] - Z[idx_b]) ** 2, axis=1)
    median_d2 = np.median(pair_d2[pair_d2 > 0])
    sigma2 = median_d2 / 2 if median_d2 > 0 else 1.0

    def rbf_kernel_mean(A, B):
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)
        return np.exp(-d2 / (2 * sigma2)).mean()

    kxx = rbf_kernel_mean(X, X)
    kyy = rbf_kernel_mean(Y, Y)
    kxy = rbf_kernel_mean(X, Y)
    return float(kxx + kyy - 2 * kxy)


def mean_sliced_wasserstein(X, Y):
    dists = [wasserstein_distance(X[:, d], Y[:, d]) for d in range(X.shape[1])]
    return float(np.mean(dists))


def main():
    unsw_benign = load_unsw_benign()
    cic17_benign = load_cic17_benign_training_block()
    print(f"UNSW-NB15 benign: {len(unsw_benign)} rows. "
          f"CIC-IDS2017 benign training block: {len(cic17_benign)} rows.\n")
    print("Testing whether internal representation (not input features) shifts "
          "toward the target domain under diverse training, for all three "
          "model families, using the SAME non-circular Single vs LODO-3 design "
          "already validated for Deep SVDD.\n")

    all_results = []
    for family in MODEL_FAMILIES:
        dim_label = "8-dim bottleneck" if family in ("ae", "dae") else "16-dim hypersphere"
        print(f"{'=' * 70}\n{family.upper()} ({dim_label})\n{'=' * 70}")
        for condition in ["single", "lodo3"]:
            mmd_vals, wass_vals = [], []
            for seed in ALL_SEEDS:
                ckpt = checkpoint_path(seed, family, condition)
                embed = load_embedder(family, ckpt)
                if embed is None:
                    print(f"  [skip] {family}/{condition}/seed{seed}: checkpoint not found at {ckpt}")
                    continue
                z_unsw = embed(unsw_benign)
                z_cic17 = embed(cic17_benign)
                mmd = mmd_squared_rbf(z_cic17, z_unsw, seed=seed)
                wass = mean_sliced_wasserstein(z_cic17, z_unsw)
                mmd_vals.append(mmd)
                wass_vals.append(wass)
                all_results.append({
                    "family": family, "condition": condition, "seed": seed,
                    "mmd_squared": mmd, "mean_sliced_wasserstein": wass,
                })
            if mmd_vals:
                std_mmd = np.std(mmd_vals, ddof=1) if len(mmd_vals) > 1 else 0
                std_wass = np.std(wass_vals, ddof=1) if len(wass_vals) > 1 else 0
                print(f"  {condition:6s}: MMD^2 mean={np.mean(mmd_vals):.4f} (std={std_mmd:.4f}), "
                      f"Wasserstein mean={np.mean(wass_vals):.4f} (std={std_wass:.4f})  "
                      f"[n={len(mmd_vals)} seeds]")

        fam_single = [r for r in all_results if r["family"] == family and r["condition"] == "single"]
        fam_lodo3 = [r for r in all_results if r["family"] == family and r["condition"] == "lodo3"]
        if fam_single and fam_lodo3:
            seeds_common = sorted(set(r["seed"] for r in fam_single) & set(r["seed"] for r in fam_lodo3))
            n_narrowed = 0
            for sd in seeds_common:
                s_mmd = next(r["mmd_squared"] for r in fam_single if r["seed"] == sd)
                l_mmd = next(r["mmd_squared"] for r in fam_lodo3 if r["seed"] == sd)
                n_narrowed += int(l_mmd < s_mmd)
            mmd_s = np.mean([r["mmd_squared"] for r in fam_single])
            mmd_l = np.mean([r["mmd_squared"] for r in fam_lodo3])
            pct_change = 100 * (mmd_s - mmd_l) / mmd_s if mmd_s != 0 else float("nan")
            print(f"  Gap narrowed (MMD^2): {mmd_s:.4f} -> {mmd_l:.4f} "
                  f"({pct_change:+.1f}%), consistent in {n_narrowed}/{len(seeds_common)} seeds")
        print()

    out_path = os.path.join(PILOT_DIR, "ae_dae_svdd_latent_lodo3_comparison.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved {out_path}")

    print(f"\n{'=' * 70}\nUNIFIED SUMMARY\n{'=' * 70}")
    for family in MODEL_FAMILIES:
        fam_single = [r for r in all_results if r["family"] == family and r["condition"] == "single"]
        fam_lodo3 = [r for r in all_results if r["family"] == family and r["condition"] == "lodo3"]
        if not (fam_single and fam_lodo3):
            continue
        mmd_s = np.mean([r["mmd_squared"] for r in fam_single])
        mmd_l = np.mean([r["mmd_squared"] for r in fam_lodo3])
        print(f"  {family.upper():6s}: MMD^2 {mmd_s:.4f} -> {mmd_l:.4f} "
              f"({'narrowed' if mmd_l < mmd_s else 'did NOT narrow'})")


if __name__ == "__main__":
    main()
