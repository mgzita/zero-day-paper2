"""
Reconstruction-error-colored latent space: does UNSW's error improve
under Multi, even where the cluster POSITION doesn't move?
========================================================================
Follow-up to visualize_latent_space.py. That script colored points by
SOURCE IDENTITY (CIC17 vs UNSW) and found AE/DAE's clusters stay
visibly separate under both Single and Multi -- unlike SVDD, which
merges dramatically (though that merge is likely inflated by UNSW
being literally inside the Multi training pool -- see
explore_latent_domain_gap_lodo3.py for the non-circular version).

For AE/DAE, cluster POSITION separating isn't actually informative on
its own: the anomaly score is RECONSTRUCTION ERROR, not distance in
latent space. Two points can sit in different latent regions and still
both reconstruct well, if the decoder has learned to handle both
regions. The real question is whether UNSW's reconstruction quality
improves under Multi training, regardless of where it sits.

This script re-uses the same 2D PCA layout, but colors every point by
its ACTUAL anomaly score (reconstruction error for AE/DAE, distance-
from-center for SVDD) instead of by source identity. Same color scale
used for both Single and Multi panels (fit jointly) so brightness is
directly comparable before/after.

If the calibration hypothesis is right: the UNSW-shaped region should
look visibly "hot" (high error) under Single and cool down (lower
error) under Multi -- even if its position in the 2D layout doesn't
change much.

Requires trained checkpoints (seed 42) to already exist.
EXPLORATORY analysis (Formal Protocol Section 8).

Run from project root:
    python paper2\\pilot\\visualize_reconstruction_error.py

Output: PNG figures saved to paper2\\pilot\\figures\\
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

SEED = 42
MODEL_FAMILIES = ["ae", "dae", "svdd"]
N_SAMPLE = 5000

N_FEATURES = 18
DEDUP_DIR = r"C:\MLProject\zero_day_project\artefacts_v2\dedup"  # TODO confirm
PAPER2_ROOT = r"C:\MLProject\zero_day_project\paper2"
PILOT_DIR = os.path.join(PAPER2_ROOT, "pilot")
FIGURES_DIR = os.path.join(PILOT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


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


def load_model_and_functions(family, ckpt_path):
    if not os.path.exists(ckpt_path):
        return None, None

    if family in ("ae", "dae"):
        model = AutoencoderModel(N_FEATURES)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.eval()

        def embed(X):
            with torch.no_grad():
                return model.encoder(torch.tensor(X, dtype=torch.float32)).numpy()

        def score(X):
            with torch.no_grad():
                x_t = torch.tensor(X, dtype=torch.float32)
                x_hat = model(x_t)
                return ((x_t - x_hat) ** 2).mean(dim=1).numpy()

        return embed, score

    elif family == "svdd":
        model = SVDDEncoder(N_FEATURES, latent_dim=16)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.eval()

        def embed(X):
            with torch.no_grad():
                return model(torch.tensor(X, dtype=torch.float32)).numpy()

        def score(X):
            with torch.no_grad():
                z = model(torch.tensor(X, dtype=torch.float32))
                return ((z - model.center) ** 2).sum(dim=1).numpy()

        return embed, score

    raise ValueError(family)


def load_cic17_benign_block(n_sample):
    d = np.load(os.path.join(DEDUP_DIR, "train_dedup.npz"))
    X = np.clip(d["X_train"], -10, 10).astype(np.float32)
    n_per_env = len(X) // 3
    block = X[:n_per_env]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(block), min(n_sample, len(block)), replace=False)
    return block[idx]


def load_unsw_benign(n_sample):
    d = np.load(os.path.join(DEDUP_DIR, "test_unsw_dedup.npz"))
    X = np.clip(d["X"], -10, 10).astype(np.float32)
    y = d["y"].astype(int)
    benign = X[y == 0]
    rng = np.random.default_rng(1)
    idx = rng.choice(len(benign), min(n_sample, len(benign)), replace=False)
    return benign[idx]


def plot_family(family, cic17, unsw):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(f"{family.upper()}: anomaly score (color) at each point's latent position, "
                 f"CIC-IDS2017 + UNSW-NB15 benign flows (seed {SEED})", fontsize=12)

    panel_data = []
    for condition in ["single", "multi"]:
        ckpt = checkpoint_path(SEED, family, condition)
        embed, score = load_model_and_functions(family, ckpt)
        if embed is None:
            panel_data.append(None)
            continue
        X_combined = np.vstack([cic17, unsw])
        z = embed(X_combined)
        s = score(X_combined)
        pca = PCA(n_components=2)
        z2d = pca.fit_transform(z)
        log_s = np.log1p(s)
        panel_data.append({
            "z2d": z2d, "log_s": log_s,
            "explained_var": pca.explained_variance_ratio_,
            "n_cic17": len(cic17),
        })

    valid_logs = np.concatenate([p["log_s"] for p in panel_data if p is not None])
    vmin, vmax = np.percentile(valid_logs, [2, 98])

    sc = None
    for ax, condition, title, data in zip(
            axes, ["single", "multi"],
            ["Single-environment training", "Multi-environment training"],
            panel_data):
        if data is None:
            ax.set_title(f"{title}\n(checkpoint not found)")
            ax.axis("off")
            continue
        sc = ax.scatter(data["z2d"][:, 0], data["z2d"][:, 1], c=data["log_s"],
                        cmap="RdYlBu_r", s=4, alpha=0.6, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel(f"PC1 ({data['explained_var'][0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({data['explained_var'][1]*100:.1f}%)")

    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes, shrink=0.85, pad=0.02)
        cbar.set_label("log(1 + anomaly score)  [same scale, both panels]")

    out_path = os.path.join(FIGURES_DIR, f"reconstruction_error_{family}_seed{SEED}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    cic17 = load_cic17_benign_block(N_SAMPLE)
    unsw = load_unsw_benign(N_SAMPLE)
    print(f"Sampled {len(cic17)} CIC-IDS2017 benign flows, {len(unsw)} UNSW-NB15 benign flows.\n")

    for family in MODEL_FAMILIES:
        print(f"Plotting {family.upper()}...")
        plot_family(family, cic17, unsw)

    print(f"\nAll figures saved to {FIGURES_DIR}")
    print("\nHow to read: compare against visualize_latent_space.py's output "
          "to see which blob is CIC17 vs UNSW. Look at whether the UNSW blob's "
          "COLOR (not position) shifts toward blue/cool (low error) from "
          "Single to Multi.")


if __name__ == "__main__":
    main()
