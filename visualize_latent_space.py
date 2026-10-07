"""
LODO-3 latent-space evaluation.

Compares Single-environment CIC17 training against LODO-3
(CIC17 + CIC18) training while keeping UNSW-NB15 completely
excluded from training.

For each model family:
    AE
    DAE
    Deep SVDD

The script:
    1. Loads Single and LODO-3 checkpoints.
    2. Encodes CIC17 benign traffic.
    3. Encodes UNSW-NB15 benign traffic.
    4. Computes PCA visualizations.
    5. Computes MMD^2.
    6. Computes mean sliced Wasserstein distance.
    7. Reports percentage reduction under LODO-3.

Seed 42 only.

Run from:
    C:\\MLProject\\zero_day_paper2_mtsst

Command:
    python paper2\\pilot\\visualize_latent_space.py
"""

import os
import csv
import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

MODEL_FAMILIES = ["ae", "dae", "svdd"]

N_SAMPLE = 5000
N_FEATURES = 18

DEDUP_DIR = (
    r"C:\MLProject\zero_day_project\artefacts_v2\dedup"
)

PAPER2_ROOT = (
    r"C:\MLProject\zero_day_paper2_mtsst"
)

PILOT_DIR = os.path.join(
    PAPER2_ROOT,
    "paper2",
    "pilot"
)

FIGURES_DIR = os.path.join(
    PILOT_DIR,
    "figures"
)

RESULTS_DIR = os.path.join(
    PILOT_DIR,
    "results"
)

os.makedirs(
    FIGURES_DIR,
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# MODELS
# ============================================================

class AutoencoderModel(nn.Module):

    def __init__(self, input_dim=N_FEATURES):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 8),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),

            nn.Linear(16, 32),
            nn.ReLU(),

            nn.Linear(32, 64),
            nn.ReLU(),

            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        return self.decoder(
            self.encoder(x)
        )


class SVDDEncoder(nn.Module):

    def __init__(
        self,
        input_dim=N_FEATURES,
        latent_dim=16
    ):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                input_dim,
                64,
                bias=False
            ),
            nn.ReLU(),

            nn.Linear(
                64,
                32,
                bias=False
            ),
            nn.ReLU(),

            nn.Linear(
                32,
                latent_dim,
                bias=False
            ),
        )

        self.register_buffer(
            "center",
            torch.zeros(latent_dim)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# CHECKPOINTS
# ============================================================

def checkpoint_path(
    seed,
    family,
    condition
):

    ckpt_dir = os.path.join(
        DEDUP_DIR,
        f"seed{seed}_{family}"
    )

    if family == "dae":
        filename = f"dae_{condition}.pt"
    else:
        filename = f"{family}_{condition}.pt"

    return os.path.join(
        ckpt_dir,
        filename
    )


# ============================================================
# MODEL LOADING
# ============================================================

def load_embedder(
    family,
    ckpt_path
):

    if not os.path.exists(ckpt_path):

        raise FileNotFoundError(
            f"Checkpoint not found:\n{ckpt_path}"
        )

    print(
        f"Loading checkpoint:\n  {ckpt_path}"
    )

    state = torch.load(
        ckpt_path,
        map_location="cpu"
    )

    if family in ("ae", "dae"):

        model = AutoencoderModel(
            N_FEATURES
        )

        model.load_state_dict(state)

        model.eval()

        def embed(X):

            with torch.no_grad():

                tensor = torch.tensor(
                    X,
                    dtype=torch.float32
                )

                return (
                    model.encoder(tensor)
                    .numpy()
                )

        return embed

    if family == "svdd":

        model = SVDDEncoder(
            N_FEATURES,
            latent_dim=16
        )

        model.load_state_dict(state)

        model.eval()

        def embed(X):

            with torch.no_grad():

                tensor = torch.tensor(
                    X,
                    dtype=torch.float32
                )

                return (
                    model(tensor)
                    .numpy()
                )

        return embed

    raise ValueError(
        f"Unknown family: {family}"
    )


# ============================================================
# DATA
# ============================================================

def load_cic17_benign_block(
    n_sample
):

    path = os.path.join(
        DEDUP_DIR,
        "train_dedup.npz"
    )

    d = np.load(path)

    X = np.clip(
        d["X_train"],
        -10,
        10
    ).astype(np.float32)

    n_per_env = len(X) // 3

    # CIC-IDS2017 is the first environment
    block = X[:n_per_env]

    rng = np.random.default_rng(0)

    idx = rng.choice(
        len(block),
        min(
            n_sample,
            len(block)
        ),
        replace=False
    )

    return block[idx]


def load_unsw_benign(
    n_sample
):

    path = os.path.join(
        DEDUP_DIR,
        "test_unsw_dedup.npz"
    )

    d = np.load(path)

    X = np.clip(
        d["X"],
        -10,
        10
    ).astype(np.float32)

    y = d["y"].astype(int)

    benign = X[y == 0]

    rng = np.random.default_rng(1)

    idx = rng.choice(
        len(benign),
        min(
            n_sample,
            len(benign)
        ),
        replace=False
    )

    return benign[idx]


# ============================================================
# MMD^2
# ============================================================

def median_heuristic(
    X,
    Y
):

    combined = np.vstack(
        [X, Y]
    )

    distances = cdist(
        combined,
        combined,
        metric="euclidean"
    )

    distances = distances[
        distances > 0
    ]

    return np.median(
        distances
    )


def rbf_kernel(
    X,
    Y,
    sigma
):

    distances = cdist(
        X,
        Y,
        metric="sqeuclidean"
    )

    return np.exp(
        -distances /
        (2.0 * sigma ** 2)
    )


def compute_mmd2(
    X,
    Y
):

    sigma = median_heuristic(
        X,
        Y
    )

    Kxx = rbf_kernel(
        X,
        X,
        sigma
    )

    Kyy = rbf_kernel(
        Y,
        Y,
        sigma
    )

    Kxy = rbf_kernel(
        X,
        Y,
        sigma
    )

    mmd2 = (
        Kxx.mean()
        + Kyy.mean()
        - 2.0 * Kxy.mean()
    )

    return float(mmd2)


# ============================================================
# MEAN SLICED WASSERSTEIN
# ============================================================

def wasserstein_1d(
    x,
    y
):

    x = np.sort(x)
    y = np.sort(y)

    n = min(
        len(x),
        len(y)
    )

    x = x[
        np.linspace(
            0,
            len(x) - 1,
            n
        ).astype(int)
    ]

    y = y[
        np.linspace(
            0,
            len(y) - 1,
            n
        ).astype(int)
    ]

    return np.mean(
        np.abs(x - y)
    )


def mean_sliced_wasserstein(
    X,
    Y
):

    distances = []

    latent_dim = X.shape[1]

    for d in range(
        latent_dim
    ):

        distances.append(
            wasserstein_1d(
                X[:, d],
                Y[:, d]
            )
        )

    return float(
        np.mean(distances)
    )


# ============================================================
# PCA FIGURE
# ============================================================

def plot_family(
    family,
    cic17,
    unsw
):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5.5)
    )

    fig.suptitle(
        f"{family.upper()}: CIC-IDS2017 vs UNSW-NB15 "
        f"latent-space position "
        f"(LODO-3, seed {SEED})",
        fontsize=13
    )

    conditions = [
        "single",
        "lodo3"
    ]

    titles = [
        "Single-environment\nCIC17 training",
        "LODO-3\nCIC17 + CIC18 training"
    ]

    for ax, condition, title in zip(
        axes,
        conditions,
        titles
    ):

        ckpt = checkpoint_path(
            SEED,
            family,
            condition
        )

        embed = load_embedder(
            family,
            ckpt
        )

        z_cic17 = embed(
            cic17
        )

        z_unsw = embed(
            unsw
        )

        combined = np.vstack(
            [
                z_cic17,
                z_unsw
            ]
        )

        pca = PCA(
            n_components=2
        )

        combined_2d = (
            pca.fit_transform(
                combined
            )
        )

        z_cic17_2d = (
            combined_2d[
                :len(z_cic17)
            ]
        )

        z_unsw_2d = (
            combined_2d[
                len(z_cic17):
            ]
        )

        ax.scatter(
            z_cic17_2d[:, 0],
            z_cic17_2d[:, 1],
            s=4,
            alpha=0.35,
            label="CIC-IDS2017 (source)"
        )

        ax.scatter(
            z_unsw_2d[:, 0],
            z_unsw_2d[:, 1],
            s=4,
            alpha=0.35,
            label="UNSW-NB15 (target)"
        )

        ax.set_title(
            title
        )

        ax.set_xlabel(
            f"PC1 "
            f"({pca.explained_variance_ratio_[0] * 100:.1f}%)"
        )

        ax.set_ylabel(
            f"PC2 "
            f"({pca.explained_variance_ratio_[1] * 100:.1f}%)"
        )

        ax.legend(
            markerscale=3,
            fontsize=9
        )

    plt.tight_layout()

    out_path = os.path.join(
        FIGURES_DIR,
        f"latent_space_lodo3_"
        f"{family}_seed{SEED}.png"
    )

    plt.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Saved figure:\n  {out_path}"
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_family(
    family,
    cic17,
    unsw
):

    print("\n")
    print("=" * 70)
    print(
        f"{family.upper()} — SINGLE vs LODO-3"
    )
    print("=" * 70)

    single_path = checkpoint_path(
        SEED,
        family,
        "single"
    )

    lodo_path = checkpoint_path(
        SEED,
        family,
        "lodo3"
    )

    print(
        f"\nSingle:\n  {single_path}"
    )

    print(
        f"\nLODO-3:\n  {lodo_path}"
    )

    single_embed = load_embedder(
        family,
        single_path
    )

    lodo_embed = load_embedder(
        family,
        lodo_path
    )

    # --------------------------------------------------------
    # Single
    # --------------------------------------------------------

    z_cic_single = (
        single_embed(cic17)
    )

    z_unsw_single = (
        single_embed(unsw)
    )

    mmd_single = compute_mmd2(
        z_cic_single,
        z_unsw_single
    )

    sw_single = (
        mean_sliced_wasserstein(
            z_cic_single,
            z_unsw_single
        )
    )

    # --------------------------------------------------------
    # LODO-3
    # --------------------------------------------------------

    z_cic_lodo = (
        lodo_embed(cic17)
    )

    z_unsw_lodo = (
        lodo_embed(unsw)
    )

    mmd_lodo = compute_mmd2(
        z_cic_lodo,
        z_unsw_lodo
    )

    sw_lodo = (
        mean_sliced_wasserstein(
            z_cic_lodo,
            z_unsw_lodo
        )
    )

    # --------------------------------------------------------
    # REDUCTIONS
    # --------------------------------------------------------

    mmd_reduction = (
        100.0
        * (mmd_single - mmd_lodo)
        / mmd_single
    )

    sw_reduction = (
        100.0
        * (sw_single - sw_lodo)
        / sw_single
    )

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print("\nRESULTS")
    print("-" * 50)

    print(
        f"MMD² Single : {mmd_single:.6f}"
    )

    print(
        f"MMD² LODO-3 : {mmd_lodo:.6f}"
    )

    print(
        f"MMD² reduction : "
        f"{mmd_reduction:.2f}%"
    )

    print()

    print(
        f"SW Single : {sw_single:.6f}"
    )

    print(
        f"SW LODO-3 : {sw_lodo:.6f}"
    )

    print(
        f"SW reduction : "
        f"{sw_reduction:.2f}%"
    )

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    plot_family(
        family,
        cic17,
        unsw
    )

    return {
        "seed": SEED,
        "model": family,
        "mmd_single": mmd_single,
        "mmd_lodo3": mmd_lodo,
        "mmd_reduction_percent": mmd_reduction,
        "sw_single": sw_single,
        "sw_lodo3": sw_lodo,
        "sw_reduction_percent": sw_reduction,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\nLODO-3 LATENT-SPACE EVALUATION"
    )

    print("=" * 70)

    print(
        "Target: UNSW-NB15"
    )

    print(
        "UNSW-NB15 is evaluation-only."
    )

    print(
        "Training comparison:"
    )

    print(
        "  Single = CIC-IDS2017"
    )

    print(
        "  LODO-3 = CIC-IDS2017 + CSE-CIC-IDS2018"
    )

    print(
        f"Seed = {SEED}"
    )

    print(
        f"N_SAMPLE = {N_SAMPLE}"
    )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    cic17 = load_cic17_benign_block(
        N_SAMPLE
    )

    unsw = load_unsw_benign(
        N_SAMPLE
    )

    print(
        f"\nSampled "
        f"{len(cic17)} CIC17 benign flows."
    )

    print(
        f"Sampled "
        f"{len(unsw)} UNSW benign flows."
    )

    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    results = []

    for family in MODEL_FAMILIES:

        result = evaluate_family(
            family,
            cic17,
            unsw
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    csv_path = os.path.join(
        RESULTS_DIR,
        f"lodo3_latent_results_seed{SEED}.csv"
    )

    fieldnames = [
        "seed",
        "model",
        "mmd_single",
        "mmd_lodo3",
        "mmd_reduction_percent",
        "sw_single",
        "sw_lodo3",
        "sw_reduction_percent",
    ]

    with open(
        csv_path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    print("\n")
    print("=" * 70)
    print(
        f"Results saved:\n  {csv_path}"
    )

    print(
        f"Figures saved:\n  {FIGURES_DIR}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()