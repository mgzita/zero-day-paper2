"""
Normalizing Flow anomaly detector -- third, genuinely distinct
unsupervised paradigm (density-based, not reconstruction- or
distance-based).
=========================================================================
Adds a fourth model family to the study: a lightweight RealNVP-style
normalizing flow. Unlike AE/DAE (reconstruction error) and Deep SVDD
(distance from a learned center), this model's anomaly score is the
NEGATIVE LOG-LIKELIHOOD of a flow under a learned probability
distribution over benign traffic -- a density-estimation approach,
genuinely different from both existing categories.

Architecture: a stack of affine coupling layers (Dinh et al., 2017,
"Density estimation using Real NVP"), alternating which half of the
input features are transformed at each layer, each conditioned on the
other half via a small MLP producing scale and translation parameters.
The base distribution is a standard multivariate Gaussian. Because
each coupling layer is exactly invertible with a tractable Jacobian
determinant, the exact log-likelihood of any input is computable in
closed form.

This script first VALIDATES the architecture using the same reproduction
check convention as every other model in this study: Single-environment
vs. size-matched Multi-environment training, evaluated on UNSW-NB15,
seed 42. If Multi shows the same qualitative improvement over Single
seen for AE/DAE/SVDD, the architecture is confirmed to be learning
sensibly and is ready to be integrated into the full H1-H4 / LODO-3
pipeline used elsewhere in this study.

Run from project root:
    python paper2\\pilot\\train_normalizing_flow.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

SEEDS_TO_RUN = [42]

N_FEATURES = 18
N_COUPLING_LAYERS = 6
HIDDEN_DIM = 64
EPOCHS = 100
BATCH_SIZE = 256
LR = 1e-3
PATIENCE = 10

DEDUP_DIR = r"C:\MLProject\zero_day_project\artefacts_v2\dedup"  # TODO confirm


class AffineCouplingLayer(nn.Module):
    """One invertible affine coupling transform. Splits the input into
    two halves via a fixed binary mask; one half passes through
    unchanged and conditions a small MLP that produces a scale (s) and
    translation (t) applied to the other half. Alternating the mask
    across layers ensures every dimension gets transformed eventually."""

    def __init__(self, dim, hidden_dim, mask):
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, dim * 2),
        )

    def forward(self, x):
        x_masked = x * self.mask
        st = self.net(x_masked)
        s, t = st.chunk(2, dim=1)
        s = torch.tanh(s) * (1 - self.mask)
        t = t * (1 - self.mask)
        z = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)
        log_det = s.sum(dim=1)
        return z, log_det

    def inverse(self, z):
        z_masked = z * self.mask
        st = self.net(z_masked)
        s, t = st.chunk(2, dim=1)
        s = torch.tanh(s) * (1 - self.mask)
        t = t * (1 - self.mask)
        x = z_masked + (1 - self.mask) * ((z - t) * torch.exp(-s))
        return x


class NormalizingFlowModel(nn.Module):
    def __init__(self, dim=N_FEATURES, n_layers=N_COUPLING_LAYERS, hidden_dim=HIDDEN_DIM):
        super().__init__()
        masks = []
        for i in range(n_layers):
            mask = torch.zeros(dim)
            mask[i % 2::2] = 1
            masks.append(mask)
        self.layers = nn.ModuleList([
            AffineCouplingLayer(dim, hidden_dim, masks[i]) for i in range(n_layers)
        ])
        self.register_buffer("base_mean", torch.zeros(dim))
        self.register_buffer("base_log_std", torch.zeros(dim))

    def log_prob(self, x):
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        z = x
        for layer in self.layers:
            z, log_det = layer(z)
            log_det_total = log_det_total + log_det
        base_log_prob = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=1)
        return base_log_prob + log_det_total


class NegativeLogLikelihoodScorer(nn.Module):
    def __init__(self, flow_model):
        super().__init__()
        self.flow_model = flow_model

    def forward(self, x):
        return (-self.flow_model.log_prob(x)).unsqueeze(1)


def train_flow(X_train, X_val, seed, input_dim=N_FEATURES):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = NormalizingFlowModel(dim=input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    X_t, X_v = torch.FloatTensor(X_train), torch.FloatTensor(X_val)
    best_loss, patience_c, best_state = float('inf'), 0, None

    for epoch in range(EPOCHS):
        model.train()
        idx = torch.randperm(len(X_t))
        for i in range(0, len(X_t), BATCH_SIZE):
            batch = X_t[idx[i:i + BATCH_SIZE]]
            optimizer.zero_grad()
            loss = -model.log_prob(batch).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = -model.log_prob(X_v).mean().item()
        scheduler.step(val_loss)
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"    epoch {epoch + 1}/{EPOCHS}, val_nll={val_loss:.4f}, patience={patience_c}/{PATIENCE}")
        if np.isfinite(val_loss) and val_loss < best_loss:
            best_loss, best_state, patience_c = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_c += 1
            if patience_c >= PATIENCE:
                print(f"    Early stop at epoch {epoch + 1}")
                break
    if best_state is not None:
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


def load_multi_matched_train_val(seed):
    d_train = np.load(os.path.join(DEDUP_DIR, "train_dedup.npz"))
    X_train_full = np.clip(d_train["X_train"], -10, 10).astype(np.float32)
    n_per_env = len(X_train_full) // 3
    n_matched = n_per_env // 3
    X_cic17 = X_train_full[:n_per_env][:n_matched]
    X_cic18 = X_train_full[n_per_env:2 * n_per_env][:n_matched]
    X_unsw_train = X_train_full[2 * n_per_env:][:n_matched]
    X_multi = np.vstack([X_cic17, X_cic18, X_unsw_train])

    d_val = np.load(os.path.join(DEDUP_DIR, "val_dedup.npz"))
    X_val_full = np.clip(d_val["X_val"], -10, 10).astype(np.float32)
    n_val_per_env = len(X_val_full) // 3
    n_val_matched = n_val_per_env // 3
    X_val_matched = np.vstack([
        X_val_full[:n_val_per_env][:n_val_matched],
        X_val_full[n_val_per_env:2 * n_val_per_env][:n_val_matched],
        X_val_full[2 * n_val_per_env:][:n_val_matched],
    ])

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_multi))
    return X_multi[idx], X_val_matched


def load_unsw_eval():
    d = np.load(os.path.join(DEDUP_DIR, "test_unsw_dedup.npz"))
    X = np.clip(d["X"], -10, 10).astype(np.float32)
    y = d["y"].astype(int)
    return X, y


def evaluate(scorer, X_eval, y_eval):
    with torch.no_grad():
        scores = scorer(torch.tensor(X_eval, dtype=torch.float32)).numpy().flatten()
    auroc = roc_auc_score(y_eval, scores)
    separation = abs(auroc - 0.5)
    return auroc, separation


def main():
    X_unsw, y_unsw = load_unsw_eval()

    for seed in SEEDS_TO_RUN:
        print(f"{'=' * 70}\nNormalizing Flow -- seed {seed}\n{'=' * 70}")

        print("  Training SINGLE (CIC-IDS2017 only)...")
        X_train_single, X_val_single = load_single_train_val()
        model_single = train_flow(X_train_single, X_val_single, seed)
        scorer_single = NegativeLogLikelihoodScorer(model_single).eval()
        auroc_single, sep_single = evaluate(scorer_single, X_unsw, y_unsw)
        print(f"  SINGLE result: AUROC={auroc_single:.4f}, separation={sep_single:.4f}")

        print("  Training MULTI (size-matched, CIC17+CIC18+UNSW)...")
        X_train_multi, X_val_multi = load_multi_matched_train_val(seed)
        model_multi = train_flow(X_train_multi, X_val_multi, seed)
        scorer_multi = NegativeLogLikelihoodScorer(model_multi).eval()
        auroc_multi, sep_multi = evaluate(scorer_multi, X_unsw, y_unsw)
        print(f"  MULTI result:  AUROC={auroc_multi:.4f}, separation={sep_multi:.4f}")

        print(f"\n  SUMMARY (seed {seed}): Single separation={sep_single:.4f}, "
              f"Multi separation={sep_multi:.4f}, "
              f"improvement={'YES' if sep_multi > sep_single else 'NO'}")

        ckpt_dir = os.path.join(DEDUP_DIR, f"seed{seed}_flow")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model_single.state_dict(), os.path.join(ckpt_dir, "flow_single.pt"))
        torch.save(model_multi.state_dict(), os.path.join(ckpt_dir, "flow_multi.pt"))
        print(f"  Saved checkpoints -> {ckpt_dir}")

    print("\nIf Multi's separation clearly exceeds Single's, matching the pattern "
          "already seen for AE/DAE/Deep SVDD, the Normalizing Flow architecture is "
          "validated and ready for integration into the full H1-H4/LODO-3 pipeline.")


if __name__ == "__main__":
    main()
