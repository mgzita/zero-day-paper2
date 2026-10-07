"""
Noise-level dose-response sweep: does the non-circular latent
representational shift scale with denoising noise level?
=========================================================================
Direct test of the hypothesis that DENOISING-specific training pressure,
not autoencoding in general, is what enables training-environment
diversity to reshape a model's internal representation.

AE (noise_std=0.0) showed NO non-circular representational shift under
LODO-3 (MMD^2 did not narrow, 0/5 seeds). DAE (noise_std=0.1) showed a
real, if partial, shift (narrowed in 3/5 seeds). This script tests
whether intermediate and higher noise levels produce a monotonic
"dose-response" pattern: more noise -> larger non-circular
representational shift.

Noise levels tested: 0.0 (= existing AE checkpoints, reused unchanged),
0.05 (new), 0.1 (= existing DAE checkpoints, reused unchanged), 0.2
(new), 0.3 (new).

Trains only the missing noise levels (0.05, 0.2, 0.3), Single and
LODO-3 conditions, seed 42 first.

EXPLORATORY analysis, motivated by the AE-vs-DAE latent-gap finding.

Run from project root:
    python paper2\\pilot\\noise_dose_response_sweep.py
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import wasserstein_distance

SEEDS_TO_RUN = [42]
NOISE_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.3]

N_FEATURES = 18
EPOCHS = 100
BATCH_SIZE = 256
LR = 1e-3
PATIENCE = 10
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


def train_autoencoder(X_train, X_val, seed, noise_std):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = AutoencoderModel(N_FEATURES)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.HuberLoss(delta=1.0)
    X_t, X_v = torch.FloatTensor(X_train), torch.FloatTensor(X_val)
    best_loss, patience_c, best_state = float('inf'), 0, None

    for epoch in range(EPOCHS):
        model.train()
        idx = torch.randperm(len(X_t))
        for i in range(0, len(X_t), BATCH_SIZE):
            batch = X_t[idx[i:i + BATCH_SIZE]]
            noisy = torch.clamp(batch + torch.randn_like(batch) * noise_std, -10, 10) if noise_std > 0 else batch
            optimizer.zero_grad()
            criterion(model(noisy), batch).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            noisy_v = torch.clamp(X_v + torch.randn_like(X_v) * noise_std, -10, 10) if noise_std > 0 else X_v
            val_loss = criterion(model(noisy_v), X_v).item()
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss, best_state, patience_c = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_c += 1
            if patience_c >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


def load_single_train_val():
    d_train = np.load(os.path.join(DEDUP_DIR, "train_dedup.npz"))
    X_train_full = np.clip(d_train["X_train"], -10, 10).astype(np.float32)
    n_per_env = len(X_train_full) // 3
    X_train_single = X_train_full[:n_per_env]

    d_val = np.load(os.path.join(DEDUP_DIR, "val_dedup.npz"))
    X_val_full = np.clip(d_val["X_val"], -10, 10).astype(np.float32)
    n_val_per_env = len(X_val_full) // 3
    X_val_single = X_val_full[:n_val_per_env]
    return X_train_single, X_val_single


def load_lodo3_train_val():
    d_train = np.load(os.path.join(DEDUP_DIR, "train_dedup.npz"))
    X_train_full = np.clip(d_train["X_train"], -10, 10).astype(np.float32)
    n_per_env = len(X_train_full) // 3
    X_train_lodo3 = X_train_full[:2 * n_per_env]

    d_val = np.load(os.path.join(DEDUP_DIR, "val_dedup.npz"))
    X_val_full = np.clip(d_val["X_val"], -10, 10).astype(np.float32)
    n_val_per_env = len(X_val_full) // 3
    X_val_lodo3 = X_val_full[:2 * n_val_per_env]
    return X_train_lodo3, X_val_lodo3


def noise_label(noise_std: float) -> str:
    return f"noise{str(noise_std).replace('.', '')}"


def checkpoint_path(seed: int, noise_std: float, condition: str) -> str:
    if noise_std == 0.0:
        return os.path.join(DEDUP_DIR, f"seed{seed}_ae", f"ae_{condition}.pt")
    if noise_std == 0.1:
        return os.path.join(DEDUP_DIR, f"seed{seed}_dae", f"dae_{condition}.pt")
    ckpt_dir = os.path.join(DEDUP_DIR, f"seed{seed}_{noise_label(noise_std)}")
    return os.path.join(ckpt_dir, f"{noise_label(noise_std)}_{condition}.pt")


def ensure_checkpoint(seed: int, noise_std: float, condition: str) -> str:
    path = checkpoint_path(seed, noise_std, condition)
    if os.path.exists(path):
        print(f"    [checkpoint] Found {path}, reusing.")
        return path

    print(f"    [checkpoint] Training noise_std={noise_std}, condition={condition}, "
          f"seed={seed} (not found at {path})...")
    if condition == "single":
        X_train, X_val = load_single_train_val()
    elif condition == "lodo3":
        X_train, X_val = load_lodo3_train_val()
    else:
        raise ValueError(condition)

    model = train_autoencoder(X_train, X_val, seed, noise_std=noise_std)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"    [checkpoint] Saved -> {path}")
    return path


def load_embedder(ckpt_path: str):
    model = AutoencoderModel(N_FEATURES)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    def embed(X):
        with torch.no_grad():
            return model.encoder(torch.tensor(X, dtype=torch.float32)).numpy()
    return embed


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
    print(f"Noise levels tested: {NOISE_LEVELS}")
    print("(0.0 = existing AE checkpoints, 0.1 = existing DAE checkpoints, both reused unchanged)\n")

    results = []
    for noise_std in NOISE_LEVELS:
        print(f"{'=' * 70}\nnoise_std = {noise_std}\n{'=' * 70}")
        for seed in SEEDS_TO_RUN:
            single_ckpt = ensure_checkpoint(seed, noise_std, "single")
            lodo3_ckpt = ensure_checkpoint(seed, noise_std, "lodo3")

            embed_single = load_embedder(single_ckpt)
            embed_lodo3 = load_embedder(lodo3_ckpt)

            z_unsw_single = embed_single(unsw_benign)
            z_cic17_single = embed_single(cic17_benign)
            mmd_single = mmd_squared_rbf(z_cic17_single, z_unsw_single, seed=seed)
            wass_single = mean_sliced_wasserstein(z_cic17_single, z_unsw_single)

            z_unsw_lodo3 = embed_lodo3(unsw_benign)
            z_cic17_lodo3 = embed_lodo3(cic17_benign)
            mmd_lodo3 = mmd_squared_rbf(z_cic17_lodo3, z_unsw_lodo3, seed=seed)
            wass_lodo3 = mean_sliced_wasserstein(z_cic17_lodo3, z_unsw_lodo3)

            pct_change = 100 * (mmd_single - mmd_lodo3) / mmd_single if mmd_single != 0 else float("nan")
            narrowed = mmd_lodo3 < mmd_single
            print(f"  seed {seed}: MMD^2 Single={mmd_single:.4f} -> LODO-3={mmd_lodo3:.4f} "
                  f"({pct_change:+.1f}%), {'narrowed' if narrowed else 'did NOT narrow'}")

            results.append({
                "noise_std": noise_std, "seed": seed,
                "mmd_single": mmd_single, "mmd_lodo3": mmd_lodo3,
                "wasserstein_single": wass_single, "wasserstein_lodo3": wass_lodo3,
                "pct_change": pct_change, "narrowed": narrowed,
            })
        print()

    out_path = os.path.join(PILOT_DIR, "noise_dose_response_summary.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path}")

    print(f"\n{'=' * 70}\nDOSE-RESPONSE SUMMARY (seed 42)\n{'=' * 70}")
    print(f"{'Noise std':<12} {'MMD^2 Single':>14} {'MMD^2 LODO-3':>14} {'% change':>10} {'Narrowed?':>10}")
    print("-" * 62)
    for r in sorted(results, key=lambda x: x["noise_std"]):
        if r["seed"] == 42:
            print(f"{r['noise_std']:<12} {r['mmd_single']:>14.4f} {r['mmd_lodo3']:>14.4f} "
                  f"{r['pct_change']:>+9.1f}% {'Yes' if r['narrowed'] else 'No':>10}")

    print("\nIf the dose-response hypothesis holds, '% change' should trend from "
          "negative/near-zero at noise=0.0 toward more positive values as noise "
          "increases, at least up to some point.")


if __name__ == "__main__":
    main()
