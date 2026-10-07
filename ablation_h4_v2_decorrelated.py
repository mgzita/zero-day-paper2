"""
H4 Ablation v2: decorrelated feature groups (5 features each)
==================================================================
Re-runs the H4 functional-importance ablation using the DECORRELATED
groups from explore_decorrelated_ess.py (threshold=0.85), which removed
3 redundant features (bwd_packet_length_std, fwd_avg_bytes_bulk,
fwd_avg_packets_bulk) before re-ranking by ESS. This produces groups of
5 features each (15 remaining / 3), with zero overlap between the new
Stable and Middle groups -- unlike the original 6/6/6 split, where a
near-duplicate pair (bwd_iat_std <-> bwd_packet_length_std, rho=0.894)
straddled the Stable/Middle boundary.

Uses the SAME data source (train_dedup.npz / val_dedup.npz /
test_unsw_dedup.npz under DEDUP_DIR) as every other script in this
pilot -- Paper 1's deduplicated dataset, no change there.

This is EXPLORATORY (Formal Protocol Section 8) and uses REVISED (v2)
feature groups, not the original frozen groups used for the
confirmatory ESFRP/H1-H3 results. Both versions should be reported:
original groups = confirmatory result; this v2 run = robustness check
motivated by the correlation finding.

Starts with seed 42 only across all three families (4 conditions:
full-15, stable-only, sensitive-only, middle-only), matching the
seed-42-first pattern used throughout.

Run from project root:
    python paper2\\pilot\\ablation_h4_v2_decorrelated.py
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from scipy import stats

SEEDS_TO_RUN = [42, 123, 456, 789, 999]
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

DECORRELATED_FEATURES = [f for f in FEATURE_NAMES
                          if f not in ("bwd_packet_length_std", "fwd_avg_bytes_bulk",
                                       "fwd_avg_packets_bulk")]
STABLE_V2 = ["bwd_iat_std", "fwd_iat_std", "destination_port",
             "init_win_bytes_forward", "bwd_packet_length_mean"]
SENSITIVE_V2 = ["fwd_iat_mean", "subflow_fwd_bytes", "bwd_packets_s",
                "flow_duration", "bwd_avg_bytes_bulk"]
MIDDLE_V2 = ["bwd_iat_mean", "flow_bytes_s", "fwd_iat_min",
             "fwd_packet_length_mean", "init_win_bytes_backward"]
assert set(STABLE_V2 + SENSITIVE_V2 + MIDDLE_V2) == set(DECORRELATED_FEATURES)
assert not (set(STABLE_V2) & set(MIDDLE_V2)), "Stable/Middle still overlap!"

DEDUP_DIR = r"C:\MLProject\zero_day_project\artefacts_v2\dedup"  # TODO confirm (Paper 1's dedup dataset)
PAPER2_ROOT = r"C:\MLProject\zero_day_project\paper2"
PILOT_DIR = os.path.join(PAPER2_ROOT, "pilot")
ABLATION_DIR = os.path.join(PILOT_DIR, "ablation_h4_v2_cache")
os.makedirs(ABLATION_DIR, exist_ok=True)


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


def main():
    print(f"Decorrelated groups (5 features each, from {len(DECORRELATED_FEATURES)}-feature "
          f"decorrelated set):")
    print(f"  Stable:    {STABLE_V2}")
    print(f"  Sensitive: {SENSITIVE_V2}")
    print(f"  Middle:    {MIDDLE_V2}\n")

    X_unsw, y_unsw = load_unsw_eval()
    all_results = []

    for family in MODEL_FAMILIES:
        print(f"\n{'=' * 70}\n{family.upper()} -- H4 ablation v2 (decorrelated)\n{'=' * 70}")
        for seed in SEEDS_TO_RUN:
            cache_path = os.path.join(ABLATION_DIR, f"{family}_seed{seed}.json")
            if os.path.exists(cache_path):
                print(f"  [cache] seed {seed} already completed -> loading {cache_path}")
                with open(cache_path) as f:
                    all_results.extend(json.load(f))
                continue

            X_train, X_val = load_multi_matched_train_val(seed)
            seed_results = []

            print(f"  [seed {seed}] Training FULL ({len(DECORRELATED_FEATURES)} decorrelated features)...")
            sep_full, auroc_full = train_and_score(
                family, seed, DECORRELATED_FEATURES, X_train, X_val, X_unsw, y_unsw)
            print(f"    full: AUROC={auroc_full:.4f}, separation={sep_full:.4f}")
            seed_results.append({"family": family, "seed": seed, "condition": "full_decorrelated",
                                  "auroc": auroc_full, "separation": sep_full})

            print(f"  [seed {seed}] Training STABLE-V2 (5 features)...")
            sep_stable, auroc_stable = train_and_score(
                family, seed, STABLE_V2, X_train, X_val, X_unsw, y_unsw)
            delta_stable = sep_full - sep_stable
            print(f"    stable-v2: AUROC={auroc_stable:.4f}, separation={sep_stable:.4f}, "
                  f"delta={delta_stable:.4f}")
            seed_results.append({"family": family, "seed": seed, "condition": "stable_v2",
                                  "auroc": auroc_stable, "separation": sep_stable,
                                  "delta_from_full": delta_stable})

            print(f"  [seed {seed}] Training SENSITIVE-V2 (5 features)...")
            sep_sensitive, auroc_sensitive = train_and_score(
                family, seed, SENSITIVE_V2, X_train, X_val, X_unsw, y_unsw)
            delta_sensitive = sep_full - sep_sensitive
            print(f"    sensitive-v2: AUROC={auroc_sensitive:.4f}, separation={sep_sensitive:.4f}, "
                  f"delta={delta_sensitive:.4f}")
            seed_results.append({"family": family, "seed": seed, "condition": "sensitive_v2",
                                  "auroc": auroc_sensitive, "separation": sep_sensitive,
                                  "delta_from_full": delta_sensitive})

            print(f"  [seed {seed}] Training MIDDLE-V2 (5 features)...")
            sep_mid, auroc_mid = train_and_score(
                family, seed, MIDDLE_V2, X_train, X_val, X_unsw, y_unsw)
            delta_mid = sep_full - sep_mid
            print(f"    middle-v2: AUROC={auroc_mid:.4f}, separation={sep_mid:.4f}, "
                  f"delta={delta_mid:.4f}")
            seed_results.append({"family": family, "seed": seed, "condition": "middle_v2",
                                  "auroc": auroc_mid, "separation": sep_mid,
                                  "delta_from_full": delta_mid})

            h4_supported = delta_stable < delta_sensitive and delta_stable < delta_mid
            print(f"  [seed {seed}] SUMMARY: delta_stable={delta_stable:.4f}, "
                  f"delta_sensitive={delta_sensitive:.4f}, delta_middle={delta_mid:.4f}")
            print(f"    H4 (v2, decorrelated) supported this seed? {h4_supported}")

            with open(cache_path, "w") as f:
                json.dump(seed_results, f, indent=2)
            all_results.extend(seed_results)

    out_path = os.path.join(PILOT_DIR, "ablation_h4_v2_summary.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {out_path}")

    # --- Cross-seed paired statistical summary (Stable vs Sensitive, Stable vs Middle) ---
    print(f"\n{'=' * 70}\nCROSS-SEED PAIRED STATISTICAL SUMMARY (H4, decorrelated groups, n=5 seeds)\n{'=' * 70}")
    stats_summary = {}
    for family in MODEL_FAMILIES:
        fam_stable = {r["seed"]: r["delta_from_full"] for r in all_results
                      if r["family"] == family and r["condition"] == "stable_v2"}
        fam_sensitive = {r["seed"]: r["delta_from_full"] for r in all_results
                          if r["family"] == family and r["condition"] == "sensitive_v2"}
        fam_middle = {r["seed"]: r["delta_from_full"] for r in all_results
                      if r["family"] == family and r["condition"] == "middle_v2"}
        seeds_common = sorted(set(fam_stable) & set(fam_sensitive) & set(fam_middle))
        if len(seeds_common) < 2:
            print(f"\n{family.upper()}: insufficient seeds for paired test ({len(seeds_common)} available)")
            continue

        stable_vals = np.array([fam_stable[s] for s in seeds_common])
        sensitive_vals = np.array([fam_sensitive[s] for s in seeds_common])
        middle_vals = np.array([fam_middle[s] for s in seeds_common])

        print(f"\n{family.upper()} ({len(seeds_common)} seeds)")
        print(f"  delta_stable:    mean={stable_vals.mean():.4f}, std={stable_vals.std(ddof=1):.4f}")
        print(f"  delta_sensitive: mean={sensitive_vals.mean():.4f}, std={sensitive_vals.std(ddof=1):.4f}")
        print(f"  delta_middle:    mean={middle_vals.mean():.4f}, std={middle_vals.std(ddof=1):.4f}")

        family_stats = {"n_seeds": len(seeds_common)}
        for label, comparator_vals in [("stable_vs_sensitive", sensitive_vals),
                                        ("stable_vs_middle", middle_vals)]:
            diffs = stable_vals - comparator_vals  # negative diff = stable is smaller = better
            n_stable_better = int(np.sum(diffs < 0))
            t_stat, t_p = stats.ttest_rel(stable_vals, comparator_vals)
            try:
                w_stat, w_p = stats.wilcoxon(stable_vals, comparator_vals)
            except ValueError:
                w_stat, w_p = None, None
            d_std = diffs.std(ddof=1)
            cohens_d = float(diffs.mean() / d_std) if d_std > 0 else float("nan")

            supported = (t_p < 0.05) and (n_stable_better == len(seeds_common))
            outcome = "Supported" if supported else ("Suggestive" if n_stable_better == len(seeds_common) else "Not supported")

            print(f"  [{label}] stable better in {n_stable_better}/{len(seeds_common)} seeds, "
                  f"paired t={t_stat:.3f}, p={t_p:.4f}, "
                  f"Wilcoxon p={'n/a' if w_p is None else f'{w_p:.4f}'}, "
                  f"Cohen's d={cohens_d:.3f} -> {outcome}")

            family_stats[label] = {
                "n_stable_better": n_stable_better, "n_seeds": len(seeds_common),
                "t_stat": float(t_stat), "p_value": float(t_p),
                "wilcoxon_p": None if w_p is None else float(w_p),
                "cohens_d": cohens_d, "outcome": outcome,
            }
        stats_summary[family] = family_stats

    stats_out_path = os.path.join(PILOT_DIR, "ablation_h4_v2_paired_stats.json")
    with open(stats_out_path, "w") as f:
        json.dump(stats_summary, f, indent=2)
    print(f"\nSaved {stats_out_path}")


if __name__ == "__main__":
    main()
