"""
Single-environment training restricted to stable-v2 features
==================================================================
Tests whether stable-feature restriction ALONE (without diversity
training) is sufficient for generalization -- distinct from the H4
ablation, which only tested feature restriction on top of an already
diversity-trained (Multi) model.

Trains AE, DAE, and Deep SVDD on CIC-IDS2017 ONLY (Single-environment,
same data as Paper 1's Single condition), restricted to the 5
decorrelated Stable-v2 features, and evaluates on UNSW-NB15.

Compares three points for each family:
  (a) Single-environment, FULL features       (Paper 1 / already run)
  (b) Single-environment, STABLE-v2 features  (NEW -- this script)
  (c) Multi-environment,  STABLE-v2 features  (already run,
                                                ablation_h4_v2_summary.json)

Interpretation guide:
  - (b) close to (c), both good  -> stability alone is largely
    sufficient; diversity training's main job may just be revealing
    which features are stable, not adding anything further.
  - (b) close to (a), both poor  -> diversity training does something
    beyond feature selection; restricting features doesn't help a
    Single-environment model on its own.
  - (b) between (a) and (c)      -> partial effect: stable features
    help somewhat even without diversity training, but diversity adds
    something further on top.

Uses the same deduplicated dataset (train_dedup.npz / val_dedup.npz /
test_unsw_dedup.npz under DEDUP_DIR) as every other script in this
pilot. EXPLORATORY analysis (Formal Protocol Section 8).

Run from project root:
    python paper2\\pilot\\single_stable_only.py
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

SEEDS_TO_RUN = [42]
MODEL_FAMILIES = ["ae", "dae", "svdd"]

EPOCHS = 100
BATCH_SIZE = 256
LR = 1e-3
PATIENCE = 10
NOISE_STD_DAE = 0.1

FEATURE_NAMES = [
    "bwd_avg_bytes_bulk", "bwd_iat_mean", "bwd_iat_std",
    "bwd_packet_length_mean", "bwd_packet_length_std", "bwd_packets_s",
    "destination_port", "flow_bytes_s", "flow_duration",
    "fwd_avg_bytes_bulk", "fwd_avg_packets_bulk", "fwd_iat_mean",
    "fwd_iat_min", "fwd_iat_std", "fwd_packet_length_mean",
    "init_win_bytes_backward", "init_win_bytes_forward", "subflow_fwd_bytes",
]
FEATURE_INDEX = {f: i for i, f in enumerate(FEATURE_NAMES)}

STABLE_V2 = ["bwd_iat_std", "fwd_iat_std", "destination_port",
             "init_win_bytes_forward", "bwd_packet_length_mean"]

KNOWN_SINGLE_FULL_SEPARATION = {
    "ae": 0.2785,
    "dae": 0.2029,
    "svdd": 0.1347,
}

DEDUP_DIR = r"C:\MLProject\zero_day_project\artefacts_v2\dedup"  # TODO confirm
PAPER2_ROOT = r"C:\MLProject\zero_day_project\paper2"
PILOT_DIR = os.path.join(PAPER2_ROOT, "pilot")
H4_V2_SUMMARY = os.path.join(PILOT_DIR, "ablation_h4_v2_summary.json")


class AutoencoderModel(nn.Module):
    def __init__(self, input_dim):
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


class ReconstructionErrorScorer(nn.Module):
    def __init__(self, autoencoder):
        super().__init__()
        self.autoencoder = autoencoder

    def forward(self, x):
        x_hat = self.autoencoder(x)
        return ((x - x_hat) ** 2).mean(dim=1, keepdim=True)


class SVDDEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64, bias=False), nn.ReLU(),
            nn.Linear(64, 32, bias=False), nn.ReLU(),
            nn.Linear(32, latent_dim, bias=False),
        )
        self.register_buffer("center", torch.zeros(latent_dim))

    def forward(self, x):
        return self.net(x)


class SVDDScorer(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        z = self.encoder(x)
        return ((z - self.encoder.center) ** 2).sum(dim=1, keepdim=True)


def train_autoencoder(X_train, X_val, seed, input_dim, noise_std):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = AutoencoderModel(input_dim)
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


def train_svdd(X_train, X_val, seed, input_dim):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SVDDEncoder(input_dim, latent_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    X_t, X_v = torch.FloatTensor(X_train), torch.FloatTensor(X_val)

    model.eval()
    with torch.no_grad():
        c = model(X_t[:min(50000, len(X_t))]).mean(dim=0)
        eps = 0.1
        c[(c.abs() < eps) & (c >= 0)] = eps
        c[(c.abs() < eps) & (c < 0)] = -eps
        model.center.copy_(c)

    best_loss, patience_c, best_state = float('inf'), 0, None
    for epoch in range(EPOCHS):
        model.train()
        idx = torch.randperm(len(X_t))
        for i in range(0, len(X_t), BATCH_SIZE):
            batch = X_t[idx[i:i + BATCH_SIZE]]
            optimizer.zero_grad()
            z = model(batch)
            loss = ((z - model.center) ** 2).sum(dim=1).mean()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            z_v = model(X_v)
            val_loss = ((z_v - model.center) ** 2).sum(dim=1).mean().item()
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss, best_state, patience_c = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_c += 1
            if patience_c >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


def train_and_score(family, seed, feature_subset, X_train_full18, X_val_full18,
                     X_unsw_full18, y_unsw):
    cols = [FEATURE_INDEX[f] for f in feature_subset]
    X_train = X_train_full18[:, cols]
    X_val = X_val_full18[:, cols]
    X_unsw = X_unsw_full18[:, cols]
    input_dim = len(cols)

    if family == "ae":
        model = train_autoencoder(X_train, X_val, seed, input_dim, noise_std=0.0)
        scorer = ReconstructionErrorScorer(model).eval()
    elif family == "dae":
        model = train_autoencoder(X_train, X_val, seed, input_dim, noise_std=NOISE_STD_DAE)
        scorer = ReconstructionErrorScorer(model).eval()
    elif family == "svdd":
        model = train_svdd(X_train, X_val, seed, input_dim)
        scorer = SVDDScorer(model).eval()
    else:
        raise ValueError(family)

    with torch.no_grad():
        scores = scorer(torch.tensor(X_unsw, dtype=torch.float32)).numpy().flatten()
    auroc = roc_auc_score(y_unsw, scores)
    separation = abs(auroc - 0.5)
    return separation, auroc


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


def load_unsw_eval():
    d = np.load(os.path.join(DEDUP_DIR, "test_unsw_dedup.npz"))
    X = np.clip(d["X"], -10, 10).astype(np.float32)
    y = d["y"].astype(int)
    return X, y


def load_multi_stable_separation():
    if not os.path.exists(H4_V2_SUMMARY):
        print(f"WARNING: {H4_V2_SUMMARY} not found -- Multi+stable comparison "
              f"column will be blank.")
        return {}
    with open(H4_V2_SUMMARY) as f:
        data = json.load(f)
    result = {}
    for r in data:
        if r["condition"] == "stable_v2" and r["seed"] == 42:
            result[r["family"]] = r["separation"]
    return result


def main():
    X_unsw, y_unsw = load_unsw_eval()
    X_train, X_val = load_single_train_val()
    multi_stable_sep = load_multi_stable_separation()

    results = {}
    for family in MODEL_FAMILIES:
        for seed in SEEDS_TO_RUN:
            print(f"\n[{family.upper()}, seed {seed}] Training SINGLE-environment, "
                  f"STABLE-v2 only (5 features): {STABLE_V2}...")
            sep, auroc = train_and_score(family, seed, STABLE_V2, X_train, X_val, X_unsw, y_unsw)
            print(f"  AUROC={auroc:.4f}, separation={sep:.4f}")
            results[family] = {"auroc": auroc, "separation": sep}

    print(f"\n{'=' * 90}")
    print(f"{'Family':<8} {'(a) Single+Full':>18} {'(b) Single+Stable':>20} "
          f"{'(c) Multi+Stable':>18}")
    print(f"{'=' * 90}")
    for family in MODEL_FAMILIES:
        a = KNOWN_SINGLE_FULL_SEPARATION.get(family)
        b = results[family]["separation"]
        c = multi_stable_sep.get(family)
        a_str = f"{a:.4f}" if a is not None else "N/A"
        c_str = f"{c:.4f}" if c is not None else "N/A"
        print(f"{family.upper():<8} {a_str:>18} {b:>20.4f} {c_str:>18}")

        if a is not None and c is not None:
            if abs(b - c) < abs(b - a):
                verdict = "(b) closer to (c): stability alone looks largely sufficient"
            elif abs(b - a) < abs(b - c):
                verdict = "(b) closer to (a): diversity training adds something beyond feature selection"
            else:
                verdict = "(b) roughly midway: partial effect"
            print(f"         -> {verdict}")

    out_path = os.path.join(PILOT_DIR, "single_stable_only_summary.json")
    with open(out_path, "w") as f:
        json.dump({
            "results": results,
            "known_single_full": KNOWN_SINGLE_FULL_SEPARATION,
            "multi_stable": multi_stable_sep,
        }, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
