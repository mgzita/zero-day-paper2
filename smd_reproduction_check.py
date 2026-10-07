"""
SMD reproduction check: Single vs. Diverse training, all 3 candidate
target groups, seed 42 only.
=========================================================================
Purpose: confirm (a) the diversity effect actually appears in SMD, and
(b) which of the 3 groups shows the largest single-environment deficit
-- the SMD analogue of UNSW-NB15 in the IDS pipeline.

METHODOLOGICAL NOTE: because SMD naturally has exactly 3 groups, this
design is non-circular for BOTH conditions by construction:
  - Single:  train on ONE non-target group only
  - Diverse: train on the OTHER TWO non-target groups combined, size-
             matched to Single's training set size
  - Evaluate BOTH on the held-out target group's test set
The target group never appears in either training condition.

Run from project root:
    python paper2\\pilot\\smd_reproduction_check.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

SEED = 42
N_FEATURES = 38
EPOCHS = 100
BATCH_SIZE = 256
LR = 1e-3
PATIENCE = 10

SMD_DIR = r"C:\MLProject\zero_day_project\artefacts_v2\dedup_smd"


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


def train_autoencoder(X_train, X_val, seed):
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
            optimizer.zero_grad()
            criterion(model(batch), batch).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_v), X_v).item()
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss, best_state, patience_c = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_c += 1
            if patience_c >= PATIENCE:
                print(f"      early stop at epoch {epoch + 1}")
                break
    model.load_state_dict(best_state)
    return model


def load_group_train_val(group):
    d_train = np.load(os.path.join(SMD_DIR, "train_smd_dedup.npz"))
    X_train_full = d_train["X_train"]
    n_per_group = len(X_train_full) // 3
    start = (group - 1) * n_per_group
    X_train_g = X_train_full[start:start + n_per_group]

    d_val = np.load(os.path.join(SMD_DIR, "val_smd_dedup.npz"))
    X_val_full = d_val["X_val"]
    n_val_per_group = len(X_val_full) // 3
    start_v = (group - 1) * n_val_per_group
    X_val_g = X_val_full[start_v:start_v + n_val_per_group]
    return X_train_g, X_val_g


def load_target_test(group):
    d = np.load(os.path.join(SMD_DIR, f"test_smd_group{group}_dedup.npz"))
    return d["X"].astype(np.float32), d["y"].astype(int)


def evaluate(model, X_eval, y_eval):
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_eval)
        recon = model(X_t)
        scores = ((X_t - recon) ** 2).mean(dim=1).numpy()
    auroc = roc_auc_score(y_eval, scores)
    return auroc, abs(auroc - 0.5)


def main():
    all_groups = [1, 2, 3]
    print(f"{'=' * 70}\nSMD reproduction check -- seed {SEED}\n{'=' * 70}")
    print("Design: Single (1 non-target group) vs Diverse (2 non-target groups, "
          "size-matched), both evaluated on the held-out target group's test set. "
          "Non-circular by construction for BOTH conditions.\n")

    results = []
    for target in all_groups:
        source_groups = [g for g in all_groups if g != target]
        single_source = source_groups[0]
        both_sources = source_groups

        print(f"{'=' * 70}\nTarget: Group {target}  (Single source: Group {single_source}, "
              f"Diverse sources: Groups {both_sources})\n{'=' * 70}")

        X_test, y_test = load_target_test(target)
        print(f"  Target test set: {len(X_test)} rows "
              f"({(y_test == 0).sum()} benign, {(y_test == 1).sum()} anomalous)")

        print(f"  Training SINGLE (Group {single_source} only)...")
        X_train_single, X_val_single = load_group_train_val(single_source)
        model_single = train_autoencoder(X_train_single, X_val_single, SEED)
        auroc_single, sep_single = evaluate(model_single, X_test, y_test)
        print(f"    AUROC={auroc_single:.4f}, separation={sep_single:.4f}")

        print(f"  Training DIVERSE (Groups {both_sources} combined, size-matched)...")
        X_train_a, X_val_a = load_group_train_val(both_sources[0])
        X_train_b, X_val_b = load_group_train_val(both_sources[1])
        n_half = len(X_train_single) // 2
        X_train_diverse = np.vstack([X_train_a[:n_half], X_train_b[:n_half]])
        rng = np.random.default_rng(SEED)
        X_train_diverse = X_train_diverse[rng.permutation(len(X_train_diverse))]
        n_half_val = len(X_val_single) // 2
        X_val_diverse = np.vstack([X_val_a[:n_half_val], X_val_b[:n_half_val]])

        model_diverse = train_autoencoder(X_train_diverse, X_val_diverse, SEED)
        auroc_diverse, sep_diverse = evaluate(model_diverse, X_test, y_test)
        print(f"    AUROC={auroc_diverse:.4f}, separation={sep_diverse:.4f}")

        improvement = sep_diverse - sep_single
        print(f"  SUMMARY: Single separation={sep_single:.4f}, Diverse separation={sep_diverse:.4f}, "
              f"improvement={improvement:+.4f} ({'YES' if improvement > 0 else 'NO'})\n")

        results.append({
            "target": target, "auroc_single": auroc_single, "auroc_diverse": auroc_diverse,
            "sep_single": sep_single, "sep_diverse": sep_diverse, "improvement": improvement,
        })

    print(f"\n{'=' * 70}\nFINAL SUMMARY (seed {SEED})\n{'=' * 70}")
    print(f"{'Target':<10} {'Sep(Single)':>12} {'Sep(Diverse)':>13} {'Improvement':>13} {'Deficit size':>13}")
    for r in results:
        print(f"Group {r['target']:<4} {r['sep_single']:>12.4f} {r['sep_diverse']:>13.4f} "
              f"{r['improvement']:>+13.4f} {(1 - r['sep_single']):>13.4f}")

    best_target = max(results, key=lambda r: 1 - r["sep_single"])
    print(f"\nGroup {best_target['target']} shows the largest single-environment deficit "
          f"(Single separation={best_target['sep_single']:.4f}) -- likely the best candidate "
          f"for the full H1/H3/representation pipeline, analogous to UNSW-NB15's role in the "
          f"IDS pipeline.")

    best_improvement = max(results, key=lambda r: r["improvement"])
    print(f"Group {best_improvement['target']} shows the largest diversity IMPROVEMENT "
          f"({best_improvement['improvement']:+.4f}).")


if __name__ == "__main__":
    main()
