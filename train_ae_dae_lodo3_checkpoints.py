"""
Train AE and DAE LODO-3 checkpoints: CIC-IDS2017 + CSE-CIC-IDS2018,
UNSW-NB15 excluded entirely from training.
=========================================================================
Deep SVDD's LODO-3 checkpoint already exists (trained earlier for the
latent-space/MMD non-circularity test, all 5 seeds) and is NOT
retrained here. This script fills the missing piece: AE and DAE never
had a LODO-3 checkpoint trained, since LODO-3 was originally built only
for the SVDD latent-space investigation.

This enables a non-circular re-validation of the CORE H1-H4 attribution
and ablation results (originally established via Single vs Multi, where
UNSW-NB15 is confounded by being inside Multi's training pool) using
Single vs LODO-3 instead -- both leave UNSW-NB15 genuinely unseen.

Uses the FULL available CIC17 and CIC18 blocks (not size-matched),
matching Paper 1's LODO protocol and the existing SVDD LODO-3 script's
convention.

Run from project root:
    python paper2\\pilot\\train_ae_dae_lodo3_checkpoints.py
"""

import os
import numpy as np
import torch
import torch.nn as nn

SEEDS_TO_TRAIN = [42, 123, 456, 789, 999]
MODEL_FAMILIES = ["ae", "dae"]   # svdd's LODO-3 checkpoint already exists

N_FEATURES = 18
EPOCHS = 100
BATCH_SIZE = 256
LR = 1e-3
PATIENCE = 10
NOISE_STD_DAE = 0.1

DEDUP_DIR = r"C:\MLProject\zero_day_project\artefacts_v2\dedup"  # TODO confirm


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
                print(f"    Early stop epoch {epoch + 1}")
                break
    model.load_state_dict(best_state)
    return model


def load_lodo3_train_val():
    d_train = np.load(os.path.join(DEDUP_DIR, "train_dedup.npz"))
    X_train_full = np.clip(d_train["X_train"], -10, 10).astype(np.float32)
    n_per_env = len(X_train_full) // 3
    X_train_lodo3 = X_train_full[:2 * n_per_env]  # CIC17 + CIC18 blocks

    d_val = np.load(os.path.join(DEDUP_DIR, "val_dedup.npz"))
    X_val_full = np.clip(d_val["X_val"], -10, 10).astype(np.float32)
    n_val_per_env = len(X_val_full) // 3
    X_val_lodo3 = X_val_full[:2 * n_val_per_env]

    return X_train_lodo3, X_val_lodo3


def main():
    X_train, X_val = load_lodo3_train_val()
    print(f"LODO-3 (CIC17+CIC18) training set: {len(X_train)} flows, "
          f"validation set: {len(X_val)} flows. UNSW-NB15 excluded entirely.\n")
    print("NOTE: Deep SVDD's LODO-3 checkpoint already exists (trained earlier "
          "for the latent-space test) and is intentionally not retrained here.\n")

    for family in MODEL_FAMILIES:
        for seed in SEEDS_TO_TRAIN:
            ckpt_dir = os.path.join(DEDUP_DIR, f"seed{seed}_{family}")
            fname = "dae_" if family == "dae" else f"{family}_"
            ckpt_path = os.path.join(ckpt_dir, f"{fname}lodo3.pt")
            if os.path.exists(ckpt_path):
                print(f"[{family}/seed{seed}] Checkpoint already exists at {ckpt_path}, skipping.")
                continue

            print(f"[{family}/seed{seed}] Training LODO-3...")
            noise_std = NOISE_STD_DAE if family == "dae" else 0.0
            model = train_autoencoder(X_train, X_val, seed, noise_std=noise_std)

            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            print(f"[{family}/seed{seed}] Saved -> {ckpt_path}\n")


if __name__ == "__main__":
    main()
