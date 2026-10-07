"""
AE, DAE, Deep SVDD: full pipeline (H1 attribution, H3 ablation,
representation-level analysis) against IEC 60870-5-104, 10 seeds.
=========================================================================
Extends the same Single-vs-LODO-3 pipeline already validated for the
Normalizing Flow to the three reconstruction/distance-based model
families, evaluated against the harmonized IEC-104 dataset instead of
UNSW-NB15.

Reuses:
  - The exact model class definitions from ablation_h3_lodo3.py /
    revalidate_unsw_attribution_lodo3.py.
  - The additivity gate and log-transform fallback pattern validated
    for the flow model.
  - The NaN/Inf output guard added after IEC-104's degenerate features
    caused a KernelSHAP crash for the flow model.
  - Per-seed, per-family caching.

Training conditions: Single (existing checkpoints) and LODO-3 (existing
for some seeds from earlier revalidation work; trained on demand for
the rest via ensure_checkpoint, the same pattern used throughout this
project).

Run from project root:
    python paper2\\pilot\\ae_dae_svdd_pipeline_iec104.py
"""

import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from scipy.stats import wasserstein_distance
import shap

SEEDS_TO_RUN = [42, 123, 456, 789, 999, 111, 222, 333, 444, 555]
MODEL_FAMILIES = ["ae", "dae", "svdd"]

N_FEATURES = 18
EPOCHS = 100
BATCH_SIZE = 256
LR = 1e-3
PATIENCE = 10
NOISE_STD_DAE = 0.1
ADDITIVITY_TOL = 1e-3
N_BACKGROUND = 500
N_EVAL = 500
MAX_SAMPLE_REPR = 2000

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

DEDUP_DIR = r"C:\MLProject\zero_day_project\artefacts_v2\dedup"  # TODO confirm
PAPER2_ROOT = r"C:\MLProject\zero_day_project\paper2"
PILOT_DIR = os.path.join(PAPER2_ROOT, "pilot")
CACHE_DIR = os.path.join(PILOT_DIR, "ae_dae_svdd_iec104_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


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


class ReconstructionErrorScorer(nn.Module):
    def __init__(self, autoencoder):
        super().__init__()
        self.autoencoder = autoencoder

    def forward(self, x):
        x_hat = self.autoencoder(x)
        score = ((x - x_hat) ** 2).mean(dim=1, keepdim=True)
        return torch.nan_to_num(score, nan=1e10, posinf=1e10, neginf=-1e10)


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


class SVDDScorer(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        z = self.encoder(x)
        score = ((z - self.encoder.center) ** 2).sum(dim=1, keepdim=True)
        return torch.nan_to_num(score, nan=1e10, posinf=1e10, neginf=-1e10)


class LogCompressedScorer(nn.Module):
    def __init__(self, base_scorer):
        super().__init__()
        self.base_scorer = base_scorer

    def forward(self, x):
        raw = self.base_scorer(x)
        return torch.log1p(torch.clamp(raw, min=0))


def get_embedder(family, model):
    if family in ("ae", "dae"):
        def embed(x):
            return model.encoder(x)
        return embed
    elif family == "svdd":
        def embed(x):
            return model(x)
        return embed
    raise ValueError(family)


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


def load_iec104_eval():
    d = np.load(os.path.join(DEDUP_DIR, "test_iec104_dedup.npz"))
    return d["X"].astype(np.float32), d["y"].astype(int)


def load_cic17_benign_training_block():
    d = np.load(os.path.join(DEDUP_DIR, "train_dedup.npz"))
    X_train_full = np.clip(d["X_train"], -10, 10).astype(np.float32)
    n_per_env = len(X_train_full) // 3
    return X_train_full[:n_per_env]


def checkpoint_path(seed, family, condition, input_dim=N_FEATURES):
    ckpt_dir = os.path.join(DEDUP_DIR, f"seed{seed}_{family}")
    fname = "dae_" if family == "dae" else f"{family}_"
    suffix = "" if input_dim == N_FEATURES else f"_dim{input_dim}"
    return os.path.join(ckpt_dir, f"{fname}{condition}{suffix}.pt")


def ensure_checkpoint(seed, family, condition, feature_subset=None):
    input_dim = len(feature_subset) if feature_subset else N_FEATURES
    path = checkpoint_path(seed, family, condition, input_dim)
    if os.path.exists(path):
        return path

    print(f"    [checkpoint] Training {family}/{condition}/seed{seed} ({input_dim} features)...")
    if condition == "single":
        X_train, X_val = load_single_train_val()
    elif condition == "lodo3":
        X_train, X_val = load_lodo3_train_val()
    else:
        raise ValueError(condition)

    if feature_subset:
        cols = [FEATURE_INDEX[f] for f in feature_subset]
        X_train, X_val = X_train[:, cols], X_val[:, cols]

    if family == "ae":
        model = train_autoencoder(X_train, X_val, seed, input_dim, noise_std=0.0)
    elif family == "dae":
        model = train_autoencoder(X_train, X_val, seed, input_dim, noise_std=NOISE_STD_DAE)
    elif family == "svdd":
        model = train_svdd(X_train, X_val, seed, input_dim)
    else:
        raise ValueError(family)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"    [checkpoint] Saved -> {path}")
    return path


def load_model(family, ckpt_path, input_dim=N_FEATURES):
    if family in ("ae", "dae"):
        model = AutoencoderModel(input_dim)
    elif family == "svdd":
        model = SVDDEncoder(input_dim, latent_dim=16)
    else:
        raise ValueError(family)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
    model.eval()
    return model


def get_scorer(family, model):
    if family in ("ae", "dae"):
        return ReconstructionErrorScorer(model).eval()
    elif family == "svdd":
        return SVDDScorer(model).eval()
    raise ValueError(family)


def normalized_global_shap(shap_values, feature_names):
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    total = mean_abs.sum()
    normalized = mean_abs / total if total > 0 else mean_abs
    return pd.Series(normalized, index=feature_names).sort_values(ascending=False)


def validate_additivity(scorer, evaluation, shap_values, expected_value):
    with torch.no_grad():
        actual = scorer(torch.tensor(evaluation, dtype=torch.float32)).numpy().flatten()
    reconstructed = expected_value + shap_values.sum(axis=1)
    return float(np.max(np.abs(actual - reconstructed)))


def run_shap(scorer, background, evaluation, n_features):
    background_t = torch.tensor(background, dtype=torch.float32)
    evaluation_t = torch.tensor(evaluation, dtype=torch.float32)
    try:
        explainer = shap.DeepExplainer(scorer, background_t)
        shap_values = explainer.shap_values(evaluation_t, check_additivity=False)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = np.array(shap_values).reshape(evaluation.shape[0], n_features)
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = float(np.array(expected_value).flatten()[0])
        max_err = validate_additivity(scorer, evaluation, shap_values, expected_value)
        if max_err < ADDITIVITY_TOL:
            return shap_values, expected_value, "DeepSHAP", evaluation, max_err
    except (AssertionError, RuntimeError):
        pass

    eval_sub = evaluation[:N_EVAL]

    def f(x_np):
        with torch.no_grad():
            return scorer(torch.tensor(x_np, dtype=torch.float32)).numpy()

    background_sub = background[:min(100, len(background))]
    kexplainer = shap.KernelExplainer(f, background_sub)
    shap_values = kexplainer.shap_values(eval_sub, nsamples=2000)
    shap_values = np.array(shap_values).reshape(eval_sub.shape[0], n_features)
    expected_value = float(np.array(kexplainer.expected_value).flatten()[0])
    max_err = validate_additivity(scorer, eval_sub, shap_values, expected_value)
    return shap_values, expected_value, "KernelSHAP", eval_sub, max_err


def h1_attribution(seed, family, X_eval, y_eval):
    print(f"  [H1] {family}/seed{seed}...")
    single_ckpt = ensure_checkpoint(seed, family, "single")
    lodo3_ckpt = ensure_checkpoint(seed, family, "lodo3")
    model_single = load_model(family, single_ckpt)
    model_lodo3 = load_model(family, lodo3_ckpt)
    scorer_single = get_scorer(family, model_single)
    scorer_lodo3 = get_scorer(family, model_lodo3)

    with torch.no_grad():
        x_t = torch.tensor(X_eval, dtype=torch.float32)
        scores_single = scorer_single(x_t).numpy().flatten()
        scores_lodo3 = scorer_lodo3(x_t).numpy().flatten()
    auroc_single = roc_auc_score(y_eval, scores_single)
    auroc_lodo3 = roc_auc_score(y_eval, scores_lodo3)
    print(f"    AUROC: Single={auroc_single:.4f}, LODO-3={auroc_lodo3:.4f}")

    rng = np.random.default_rng(seed)
    background_idx = rng.choice(len(X_eval), size=min(N_BACKGROUND, len(X_eval)), replace=False)
    eval_idx = rng.choice(len(X_eval), size=min(N_EVAL, len(X_eval)), replace=False)
    background = X_eval[background_idx]
    evaluation = X_eval[eval_idx]

    shap_single, ev_s, method_s, eval_used_s, err_s = run_shap(scorer_single, background, evaluation, N_FEATURES)
    shap_lodo3, ev_l, method_l, eval_used_l, err_l = run_shap(scorer_lodo3, background, evaluation, N_FEATURES)
    print(f"    Attribution: Single={method_s} (err={err_s:.2e}), LODO-3={method_l} (err={err_l:.2e})")

    err_s_ok = err_s < ADDITIVITY_TOL
    err_l_ok = err_l < ADDITIVITY_TOL
    used_log_transform = False

    if not (err_s_ok and err_l_ok):
        print(f"    Raw-score attribution failed additivity. Retrying with log(1+score) scorer...")
        used_log_transform = True
        scorer_single_log = LogCompressedScorer(scorer_single).eval()
        scorer_lodo3_log = LogCompressedScorer(scorer_lodo3).eval()
        shap_single, ev_s, method_s, eval_used_s, err_s = run_shap(scorer_single_log, background, evaluation, N_FEATURES)
        shap_lodo3, ev_l, method_l, eval_used_l, err_l = run_shap(scorer_lodo3_log, background, evaluation, N_FEATURES)
        print(f"    [log-transformed retry] Single={method_s} (err={err_s:.2e}), LODO-3={method_l} (err={err_l:.2e})")
        err_s_ok = err_s < ADDITIVITY_TOL
        err_l_ok = err_l < ADDITIVITY_TOL

    if not (err_s_ok and err_l_ok):
        print(f"    [HALT] Additivity check failed even after log-transform retry.")
        return {"seed": seed, "family": family, "additivity_ok": False,
                "used_log_transform": used_log_transform,
                "auroc_single": auroc_single, "auroc_lodo3": auroc_lodo3}

    norm_single = normalized_global_shap(shap_single, FEATURE_NAMES)
    norm_lodo3 = normalized_global_shap(shap_lodo3, FEATURE_NAMES)
    esfrp_single = float(norm_single.loc[STABLE_V2].sum())
    esfrp_lodo3 = float(norm_lodo3.loc[STABLE_V2].sum())
    a, b = norm_single.align(norm_lodo3, fill_value=0.0)
    asi = float(0.5 * np.sum(np.abs(b - a)))
    direction_correct = esfrp_lodo3 > esfrp_single

    print(f"    ESFRP(Single)={esfrp_single:.4f} ESFRP(LODO-3)={esfrp_lodo3:.4f} "
          f"ASI={asi:.4f} Direction correct: {direction_correct}"
          f"{'  [log transform]' if used_log_transform else ''}")

    return {
        "seed": seed, "family": family, "additivity_ok": True,
        "used_log_transform": used_log_transform,
        "auroc_single": auroc_single, "auroc_lodo3": auroc_lodo3,
        "esfrp_single": esfrp_single, "esfrp_lodo3": esfrp_lodo3,
        "asi": asi, "direction_correct": direction_correct,
    }


def train_and_score_subset(seed, family, feature_subset, X_eval_full, y_eval):
    ckpt = ensure_checkpoint(seed, family, "lodo3", feature_subset=feature_subset)
    model = load_model(family, ckpt, input_dim=len(feature_subset))
    scorer = get_scorer(family, model)
    cols = [FEATURE_INDEX[f] for f in feature_subset]
    X_eval_sub = X_eval_full[:, cols]
    with torch.no_grad():
        scores = scorer(torch.tensor(X_eval_sub, dtype=torch.float32)).numpy().flatten()
    auroc = roc_auc_score(y_eval, scores)
    return abs(auroc - 0.5), auroc


def h3_ablation(seed, family, X_eval, y_eval):
    print(f"  [H3] {family}/seed{seed}...")
    sep_full, auroc_full = train_and_score_subset(seed, family, DECORRELATED_FEATURES, X_eval, y_eval)
    sep_stable, auroc_stable = train_and_score_subset(seed, family, STABLE_V2, X_eval, y_eval)
    sep_sensitive, auroc_sensitive = train_and_score_subset(seed, family, SENSITIVE_V2, X_eval, y_eval)
    sep_middle, auroc_middle = train_and_score_subset(seed, family, MIDDLE_V2, X_eval, y_eval)

    delta_stable = sep_full - sep_stable
    delta_sensitive = sep_full - sep_sensitive
    delta_middle = sep_full - sep_middle
    print(f"    delta_stable={delta_stable:.4f} delta_sensitive={delta_sensitive:.4f} "
          f"delta_middle={delta_middle:.4f}")

    return {
        "seed": seed, "family": family,
        "delta_stable": delta_stable, "delta_sensitive": delta_sensitive, "delta_middle": delta_middle,
    }


def mmd_squared_rbf(X, Y, max_sample=MAX_SAMPLE_REPR, seed=0):
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

    return float(rbf_kernel_mean(X, X) + rbf_kernel_mean(Y, Y) - 2 * rbf_kernel_mean(X, Y))


def mean_sliced_wasserstein(X, Y):
    dists = [wasserstein_distance(X[:, d], Y[:, d]) for d in range(X.shape[1])]
    return float(np.mean(dists))


def representation_analysis(seed, family, cic17_benign, iec104_benign):
    print(f"  [Repr] {family}/seed{seed}...")
    single_ckpt = ensure_checkpoint(seed, family, "single")
    lodo3_ckpt = ensure_checkpoint(seed, family, "lodo3")
    model_single = load_model(family, single_ckpt)
    model_lodo3 = load_model(family, lodo3_ckpt)
    embed_single = get_embedder(family, model_single)
    embed_lodo3 = get_embedder(family, model_lodo3)

    with torch.no_grad():
        z_cic17_single = embed_single(torch.tensor(cic17_benign, dtype=torch.float32)).numpy()
        z_iec104_single = embed_single(torch.tensor(iec104_benign, dtype=torch.float32)).numpy()
        z_cic17_lodo3 = embed_lodo3(torch.tensor(cic17_benign, dtype=torch.float32)).numpy()
        z_iec104_lodo3 = embed_lodo3(torch.tensor(iec104_benign, dtype=torch.float32)).numpy()

    mmd_single = mmd_squared_rbf(z_cic17_single, z_iec104_single, seed=seed)
    mmd_lodo3 = mmd_squared_rbf(z_cic17_lodo3, z_iec104_lodo3, seed=seed)
    wass_single = mean_sliced_wasserstein(z_cic17_single, z_iec104_single)
    wass_lodo3 = mean_sliced_wasserstein(z_cic17_lodo3, z_iec104_lodo3)

    pct_change = 100 * (mmd_single - mmd_lodo3) / mmd_single if mmd_single != 0 else float("nan")
    print(f"    MMD^2: Single={mmd_single:.4f}, LODO-3={mmd_lodo3:.4f} ({pct_change:+.1f}%)")

    return {
        "seed": seed, "family": family,
        "mmd_single": mmd_single, "mmd_lodo3": mmd_lodo3,
        "wasserstein_single": wass_single, "wasserstein_lodo3": wass_lodo3,
        "pct_change": pct_change, "narrowed": bool(mmd_lodo3 < mmd_single),
    }


def main():
    X_eval, y_eval = load_iec104_eval()
    eval_benign = X_eval[y_eval == 0]
    cic17_benign = load_cic17_benign_training_block()
    print(f"IEC-104 evaluation set: {len(X_eval)} flows "
          f"({(y_eval == 0).sum()} benign, {(y_eval == 1).sum()} attack)\n")

    all_results = {"h1": [], "h3": [], "repr": []}

    for family in MODEL_FAMILIES:
        for seed in SEEDS_TO_RUN:
            cache_path = os.path.join(CACHE_DIR, f"{family}_seed{seed}.json")
            if os.path.exists(cache_path):
                print(f"[cache] {family}/seed{seed} already completed -> loading")
                with open(cache_path) as f:
                    cached = json.load(f)
                all_results["h1"].append(cached["h1"])
                all_results["h3"].append(cached["h3"])
                all_results["repr"].append(cached["repr"])
                continue

            print(f"\n{'=' * 70}\n{family.upper()} -- seed {seed}\n{'=' * 70}")
            h1_result = h1_attribution(seed, family, X_eval, y_eval)
            h3_result = h3_ablation(seed, family, X_eval, y_eval)
            repr_result = representation_analysis(seed, family, cic17_benign, eval_benign)

            all_results["h1"].append(h1_result)
            all_results["h3"].append(h3_result)
            all_results["repr"].append(repr_result)

            with open(cache_path, "w") as f:
                json.dump({"h1": h1_result, "h3": h3_result, "repr": repr_result}, f, indent=2)
            print(f"[cache] Saved {family}/seed{seed}")

    with open(os.path.join(PILOT_DIR, "ae_dae_svdd_h1_iec104.json"), "w") as f:
        json.dump(all_results["h1"], f, indent=2)
    with open(os.path.join(PILOT_DIR, "ae_dae_svdd_h3_iec104.json"), "w") as f:
        json.dump(all_results["h3"], f, indent=2)
    with open(os.path.join(PILOT_DIR, "ae_dae_svdd_repr_iec104.json"), "w") as f:
        json.dump(all_results["repr"], f, indent=2)
    print("\nSaved: ae_dae_svdd_h1_iec104.json, ae_dae_svdd_h3_iec104.json, ae_dae_svdd_repr_iec104.json")


if __name__ == "__main__":
    main()
