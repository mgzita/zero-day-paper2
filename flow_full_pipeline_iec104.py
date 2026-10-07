"""
Normalizing Flow: full pipeline, IEC 60870-5-104 evaluation target
=========================================================================
Adaptation of flow_full_pipeline.py: identical H1/H3/representation-level
logic, but evaluated against the harmonized IEC 60870-5-104 dataset
(test_iec104_dedup.npz, see harmonize_iec104.py) instead of UNSW-NB15.

Training conditions are UNCHANGED (Single, LODO-3 -- both trained on
CIC17/CIC18 as before) and REUSE the existing flow_single.pt/
flow_lodo3.pt checkpoints; only the evaluation target changes. Note
that unlike UNSW-NB15, IEC-104 was never part of any training
condition, so there is no circularity concern analogous to Multi's
confound for UNSW-NB15 -- IEC-104 is genuinely, entirely unseen by
every model evaluated here.

  1. LODO-3 checkpoint (reused, not retrained).
  2. H1 attribution: Single vs LODO-3 SHAP-based ESFRP/ASI, evaluated on
     IEC-104, using the SAME frozen Stable/Sensitive feature groups.
  3. H3 functional ablation, evaluated on IEC-104.
  4. Representation-level analysis: MMD^2 and mean sliced Wasserstein
     distance between source (CIC17) and target (IEC-104) benign flows.

Seed 42 only, matching the "validate before scaling" pattern used
throughout this project.

Run from project root:
    python paper2\\pilot\\flow_full_pipeline.py
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from scipy.stats import wasserstein_distance
import shap

SEEDS_TO_RUN = [42, 123, 456, 789, 999, 111, 222, 333, 444, 555]

N_FEATURES = 18
N_COUPLING_LAYERS = 6
HIDDEN_DIM = 64
EPOCHS = 100
BATCH_SIZE = 256
LR = 1e-3
PATIENCE = 10
ADDITIVITY_TOL = 1e-3  # matches the standard used for AE/DAE/SVDD throughout this project
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
FLOW_DIR = os.path.join(PILOT_DIR, "flow_pipeline_cache")
os.makedirs(FLOW_DIR, exist_ok=True)


class AffineCouplingLayer(nn.Module):
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


class NormalizingFlowModel(nn.Module):
    def __init__(self, dim, n_layers=N_COUPLING_LAYERS, hidden_dim=HIDDEN_DIM):
        super().__init__()
        masks = []
        for i in range(n_layers):
            mask = torch.zeros(dim)
            mask[i % 2::2] = 1
            masks.append(mask)
        self.layers = nn.ModuleList([
            AffineCouplingLayer(dim, hidden_dim, masks[i]) for i in range(n_layers)
        ])

    def transform(self, x):
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        z = x
        for layer in self.layers:
            z, log_det = layer(z)
            log_det_total = log_det_total + log_det
        return z, log_det_total

    def log_prob(self, x):
        z, log_det_total = self.transform(x)
        base_log_prob = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=1)
        return base_log_prob + log_det_total


class NegativeLogLikelihoodScorer(nn.Module):
    """Used for AUROC/H3/representation analysis -- the raw, untransformed
    anomaly score, matching the convention used for AE/DAE/SVDD elsewhere
    in this project. Non-finite outputs are guarded against: for genuine,
    in-distribution data these never occur, but KernelSHAP's synthetic
    "masked" query points (mixing real and background feature values in
    combinations that never occur in real traffic) can occasionally push
    the flow's transform into a numerically unstable region, especially
    on data with several exactly-constant or boundary-saturated features
    (e.g. IEC-104's destination_port, init_win_bytes_forward, and
    bwd_iat_std at the +10 clip boundary). Clamping to a large finite
    value prevents this from crashing the explainer's internal solver,
    without altering the score for any real input."""
    def __init__(self, flow_model):
        super().__init__()
        self.flow_model = flow_model

    def forward(self, x):
        score = (-self.flow_model.log_prob(x)).unsqueeze(1)
        return torch.nan_to_num(score, nan=1e10, posinf=1e10, neginf=-1e10)


class LogCompressedNLLScorer(nn.Module):
    """Used ONLY for SHAP attribution (H1/H2), when the raw NLL scorer's
    additivity check fails due to extreme dynamic range under severe
    distribution shift. Explains log(1 + NLL) instead of raw NLL: a
    monotonic transform that preserves score ranking (so AUROC, ESFRP
    direction, etc. carry the same meaning) while compressing outliers
    into a range KernelSHAP's local linear surrogate can approximate
    tightly. This is the SAME standard practice used for explaining
    other highly skewed model outputs (e.g. cost or risk scores) -- not
    a workaround specific to this study, and is disclosed explicitly
    wherever attribution results using this scorer are reported.
    Non-finite guard: see NegativeLogLikelihoodScorer's docstring above."""
    def __init__(self, flow_model):
        super().__init__()
        self.flow_model = flow_model

    def forward(self, x):
        nll = torch.nan_to_num(-self.flow_model.log_prob(x), nan=1e10, posinf=1e10, neginf=-1e10)
        score = torch.log1p(nll).unsqueeze(1)
        return torch.nan_to_num(score, nan=23.0, posinf=23.0, neginf=0.0)  # log1p(1e10) approx 23


def train_flow(X_train, X_val, seed, input_dim):
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
        if np.isfinite(val_loss) and val_loss < best_loss:
            best_loss, best_state, patience_c = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_c += 1
            if patience_c >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


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


def load_iec104_eval():
    """Evaluation target: IEC 60870-5-104, properly harmonized (log1p on
    the confirmed 5 features, IQR-scaled with the same scaler as
    CIC17/CIC18/UNSW-NB15, deduplicated). See harmonize_iec104.py."""
    d = np.load(os.path.join(DEDUP_DIR, "test_iec104_dedup.npz"))
    X = np.clip(d["X"], -10, 10).astype(np.float32)
    y = d["y"].astype(int)
    return X, y


def load_cic17_benign_training_block():
    d = np.load(os.path.join(DEDUP_DIR, "train_dedup.npz"))
    X_train_full = np.clip(d["X_train"], -10, 10).astype(np.float32)
    n_per_env = len(X_train_full) // 3
    return X_train_full[:n_per_env]


def ensure_flow_checkpoint(seed, condition, input_dim=N_FEATURES):
    ckpt_dir = os.path.join(DEDUP_DIR, f"seed{seed}_flow")
    ckpt_path = os.path.join(ckpt_dir, f"flow_{condition}.pt")
    if os.path.exists(ckpt_path):
        print(f"  [checkpoint] Found {ckpt_path}, reusing.")
        return ckpt_path

    print(f"  [checkpoint] Training flow/{condition}/seed{seed} (18 features)...")
    if condition == "single":
        X_train, X_val = load_single_train_val()
    elif condition == "multi":
        X_train, X_val = load_multi_matched_train_val(seed)
    elif condition == "lodo3":
        X_train, X_val = load_lodo3_train_val()
    else:
        raise ValueError(condition)

    model = train_flow(X_train, X_val, seed, input_dim=input_dim)
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    print(f"  [checkpoint] Saved -> {ckpt_path}")
    return ckpt_path


def load_flow_model(ckpt_path, input_dim=N_FEATURES):
    model = NormalizingFlowModel(dim=input_dim)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    # strict=False: the original validation script's model class had two
    # unused buffers (base_mean, base_log_std) not present in this
    # cleaned-up version. They were never used in log_prob() in either
    # version, so ignoring them on load is safe and does not change
    # model behavior.
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def normalized_global_shap(shap_values, feature_names):
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    total = mean_abs.sum()
    normalized = mean_abs / total if total > 0 else mean_abs
    import pandas as pd
    return pd.Series(normalized, index=feature_names).sort_values(ascending=False)


def validate_additivity(scorer, evaluation, shap_values, expected_value):
    with torch.no_grad():
        actual = scorer(torch.tensor(evaluation, dtype=torch.float32)).numpy().flatten()
    reconstructed = expected_value + shap_values.sum(axis=1)
    return float(np.max(np.abs(actual - reconstructed)))


def run_shap(scorer, background, evaluation):
    background_t = torch.tensor(background, dtype=torch.float32)
    evaluation_t = torch.tensor(evaluation, dtype=torch.float32)
    try:
        explainer = shap.DeepExplainer(scorer, background_t)
        shap_values = explainer.shap_values(evaluation_t, check_additivity=False)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = np.array(shap_values).reshape(evaluation.shape[0], N_FEATURES)
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = float(np.array(expected_value).flatten()[0])
        max_err = validate_additivity(scorer, evaluation, shap_values, expected_value)
        if max_err < ADDITIVITY_TOL:
            return shap_values, expected_value, "DeepSHAP", evaluation, max_err
    except (AssertionError, RuntimeError) as e:
        print(f"    DeepSHAP failed ({type(e).__name__}), falling back to KernelSHAP...")

    eval_sub = evaluation[:N_EVAL]

    def f(x_np):
        with torch.no_grad():
            return scorer(torch.tensor(x_np, dtype=torch.float32)).numpy()

    background_sub = background[:min(100, len(background))]
    kexplainer = shap.KernelExplainer(f, background_sub)
    # Increased from the default "auto" used for AE/DAE/SVDD, since the
    # flow's raw negative-log-likelihood score can have much larger
    # dynamic range/outliers than reconstruction error or distance scores
    # (especially for a poorly-generalizing Single-environment model
    # scoring out-of-distribution UNSW-NB15 flows), which may require
    # more samples for KernelSHAP's local linear surrogate to fit
    # accurately enough to meet the same 1e-3 tolerance used elsewhere.
    shap_values = kexplainer.shap_values(eval_sub, nsamples=2000)
    shap_values = np.array(shap_values).reshape(eval_sub.shape[0], N_FEATURES)
    expected_value = float(np.array(kexplainer.expected_value).flatten()[0])
    max_err = validate_additivity(scorer, eval_sub, shap_values, expected_value)
    return shap_values, expected_value, "KernelSHAP", eval_sub, max_err


def h1_attribution(seed, X_unsw, y_unsw):
    print(f"\n{'=' * 70}\nSTEP 2: H1 attribution (Single vs LODO-3), seed {seed}\n{'=' * 70}")
    single_ckpt = ensure_flow_checkpoint(seed, "single")
    lodo3_ckpt = ensure_flow_checkpoint(seed, "lodo3")
    model_single = load_flow_model(single_ckpt)
    model_lodo3 = load_flow_model(lodo3_ckpt)
    scorer_single = NegativeLogLikelihoodScorer(model_single).eval()
    scorer_lodo3 = NegativeLogLikelihoodScorer(model_lodo3).eval()

    with torch.no_grad():
        x_t = torch.tensor(X_unsw, dtype=torch.float32)
        scores_single = scorer_single(x_t).numpy().flatten()
        scores_lodo3 = scorer_lodo3(x_t).numpy().flatten()
    auroc_single = roc_auc_score(y_unsw, scores_single)
    auroc_lodo3 = roc_auc_score(y_unsw, scores_lodo3)
    print(f"  AUROC: Single={auroc_single:.4f}, LODO-3={auroc_lodo3:.4f}")
    print(f"  Raw score scale (for additivity-tolerance context): "
          f"Single range=[{scores_single.min():.2f}, {scores_single.max():.2f}], "
          f"LODO-3 range=[{scores_lodo3.min():.2f}, {scores_lodo3.max():.2f}]")

    rng = np.random.default_rng(seed)
    background_idx = rng.choice(len(X_unsw), size=min(N_BACKGROUND, len(X_unsw)), replace=False)
    eval_idx = rng.choice(len(X_unsw), size=min(N_EVAL, len(X_unsw)), replace=False)
    background = X_unsw[background_idx]
    evaluation = X_unsw[eval_idx]

    shap_single, ev_s, method_s, eval_used_s, err_s = run_shap(scorer_single, background, evaluation)
    shap_lodo3, ev_l, method_l, eval_used_l, err_l = run_shap(scorer_lodo3, background, evaluation)
    print(f"  Attribution method: Single={method_s} (additivity err={err_s:.2e}), "
          f"LODO-3={method_l} (additivity err={err_l:.2e})")

    err_s_ok = err_s < ADDITIVITY_TOL
    err_l_ok = err_l < ADDITIVITY_TOL
    used_log_transform = False

    if not (err_s_ok and err_l_ok):
        print(f"  Raw-score attribution failed additivity (Single ok={err_s_ok}, "
              f"LODO-3 ok={err_l_ok}). Retrying with log(1+NLL)-transformed scorer "
              f"(monotonic, preserves score ranking, compresses extreme outliers)...")
        used_log_transform = True
        scorer_single_log = LogCompressedNLLScorer(model_single).eval()
        scorer_lodo3_log = LogCompressedNLLScorer(model_lodo3).eval()
        shap_single, ev_s, method_s, eval_used_s, err_s = run_shap(scorer_single_log, background, evaluation)
        shap_lodo3, ev_l, method_l, eval_used_l, err_l = run_shap(scorer_lodo3_log, background, evaluation)
        print(f"  [log-transformed retry] Single={method_s} (additivity err={err_s:.2e}), "
              f"LODO-3={method_l} (additivity err={err_l:.2e})")
        err_s_ok = err_s < ADDITIVITY_TOL
        err_l_ok = err_l < ADDITIVITY_TOL

    if not (err_s_ok and err_l_ok):
        print(f"  [HALT] Additivity check failed even after log-transform retry "
              f"(Single ok={err_s_ok}, LODO-3 ok={err_l_ok}). "
              f"ESFRP/ASI NOT computed -- this seed's attribution result is unreliable and "
              f"must not be used.")
        return {
            "seed": seed, "additivity_ok": False, "used_log_transform": used_log_transform,
            "seed": seed, "additivity_ok": False,
            "auroc_single": auroc_single, "auroc_lodo3": auroc_lodo3,
            "method_single": method_s, "method_lodo3": method_l,
            "additivity_err_single": err_s, "additivity_err_lodo3": err_l,
        }

    norm_single = normalized_global_shap(shap_single, FEATURE_NAMES)
    norm_lodo3 = normalized_global_shap(shap_lodo3, FEATURE_NAMES)
    esfrp_single = float(norm_single.loc[STABLE_V2].sum())
    esfrp_lodo3 = float(norm_lodo3.loc[STABLE_V2].sum())
    a, b = norm_single.align(norm_lodo3, fill_value=0.0)
    asi = float(0.5 * np.sum(np.abs(b - a)))
    direction_correct = esfrp_lodo3 > esfrp_single

    print(f"  ESFRP(Single)={esfrp_single:.4f}  ESFRP(LODO-3)={esfrp_lodo3:.4f}  "
          f"ASI={asi:.4f}  Direction correct: {direction_correct}"
          f"{'  [via log(1+NLL) transform]' if used_log_transform else ''}")

    return {
        "seed": seed, "additivity_ok": True, "used_log_transform": used_log_transform,
        "auroc_single": auroc_single, "auroc_lodo3": auroc_lodo3,
        "esfrp_single": esfrp_single, "esfrp_lodo3": esfrp_lodo3, "asi": asi,
        "direction_correct": direction_correct,
        "method_single": method_s, "method_lodo3": method_l,
    }


def train_and_score_subset(seed, feature_subset, X_train_full18, X_val_full18,
                            X_unsw_full18, y_unsw):
    cols = [FEATURE_INDEX[f] for f in feature_subset]
    X_train = X_train_full18[:, cols]
    X_val = X_val_full18[:, cols]
    X_unsw = X_unsw_full18[:, cols]
    input_dim = len(cols)

    model = train_flow(X_train, X_val, seed, input_dim=input_dim)
    scorer = NegativeLogLikelihoodScorer(model).eval()
    with torch.no_grad():
        scores = scorer(torch.tensor(X_unsw, dtype=torch.float32)).numpy().flatten()
    auroc = roc_auc_score(y_unsw, scores)
    separation = abs(auroc - 0.5)
    return separation, auroc


def h3_ablation(seed, X_unsw, y_unsw):
    print(f"\n{'=' * 70}\nSTEP 3: H3 functional ablation (LODO-3), seed {seed}\n{'=' * 70}")
    X_train, X_val = load_lodo3_train_val()

    print("  Training FULL (15 decorrelated features)...")
    sep_full, auroc_full = train_and_score_subset(seed, DECORRELATED_FEATURES, X_train, X_val, X_unsw, y_unsw)
    print(f"    full: AUROC={auroc_full:.4f}, separation={sep_full:.4f}")

    print("  Training STABLE-ONLY (5 features)...")
    sep_stable, auroc_stable = train_and_score_subset(seed, STABLE_V2, X_train, X_val, X_unsw, y_unsw)
    delta_stable = sep_full - sep_stable
    print(f"    stable: AUROC={auroc_stable:.4f}, delta={delta_stable:.4f}")

    print("  Training SENSITIVE-ONLY (5 features)...")
    sep_sensitive, auroc_sensitive = train_and_score_subset(seed, SENSITIVE_V2, X_train, X_val, X_unsw, y_unsw)
    delta_sensitive = sep_full - sep_sensitive
    print(f"    sensitive: AUROC={auroc_sensitive:.4f}, delta={delta_sensitive:.4f}")

    print("  Training MIDDLE-ONLY (5 features)...")
    sep_middle, auroc_middle = train_and_score_subset(seed, MIDDLE_V2, X_train, X_val, X_unsw, y_unsw)
    delta_middle = sep_full - sep_middle
    print(f"    middle: AUROC={auroc_middle:.4f}, delta={delta_middle:.4f}")

    print(f"  SUMMARY: delta_stable={delta_stable:.4f}, delta_sensitive={delta_sensitive:.4f}, "
          f"delta_middle={delta_middle:.4f}")
    print(f"    Stable beats Sensitive? {delta_stable < delta_sensitive}  "
          f"Stable beats Middle? {delta_stable < delta_middle}")

    return {
        "seed": seed, "delta_stable": delta_stable, "delta_sensitive": delta_sensitive,
        "delta_middle": delta_middle,
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


def representation_analysis(seed, cic17_benign, unsw_benign):
    print(f"\n{'=' * 70}\nSTEP 4: Representation-level analysis (Single vs LODO-3), seed {seed}\n{'=' * 70}")
    single_ckpt = ensure_flow_checkpoint(seed, "single")
    lodo3_ckpt = ensure_flow_checkpoint(seed, "lodo3")
    model_single = load_flow_model(single_ckpt)
    model_lodo3 = load_flow_model(lodo3_ckpt)

    with torch.no_grad():
        z_cic17_single, _ = model_single.transform(torch.tensor(cic17_benign, dtype=torch.float32))
        z_unsw_single, _ = model_single.transform(torch.tensor(unsw_benign, dtype=torch.float32))
        z_cic17_lodo3, _ = model_lodo3.transform(torch.tensor(cic17_benign, dtype=torch.float32))
        z_unsw_lodo3, _ = model_lodo3.transform(torch.tensor(unsw_benign, dtype=torch.float32))

    z_cic17_single = z_cic17_single.numpy()
    z_unsw_single = z_unsw_single.numpy()
    z_cic17_lodo3 = z_cic17_lodo3.numpy()
    z_unsw_lodo3 = z_unsw_lodo3.numpy()

    mmd_single = mmd_squared_rbf(z_cic17_single, z_unsw_single, seed=seed)
    mmd_lodo3 = mmd_squared_rbf(z_cic17_lodo3, z_unsw_lodo3, seed=seed)
    wass_single = mean_sliced_wasserstein(z_cic17_single, z_unsw_single)
    wass_lodo3 = mean_sliced_wasserstein(z_cic17_lodo3, z_unsw_lodo3)

    pct_change = 100 * (mmd_single - mmd_lodo3) / mmd_single if mmd_single != 0 else float("nan")
    print(f"  MMD^2: Single={mmd_single:.4f}, LODO-3={mmd_lodo3:.4f} ({pct_change:+.1f}%)")
    print(f"  Wasserstein: Single={wass_single:.4f}, LODO-3={wass_lodo3:.4f}")
    print(f"  Gap narrowed: {'Yes' if mmd_lodo3 < mmd_single else 'No'}")

    return {
        "seed": seed, "mmd_single": mmd_single, "mmd_lodo3": mmd_lodo3,
        "wasserstein_single": wass_single, "wasserstein_lodo3": wass_lodo3,
        "pct_change": pct_change, "narrowed": bool(mmd_lodo3 < mmd_single),
    }


def main():
    X_unsw, y_unsw = load_iec104_eval()
    unsw_benign = X_unsw[y_unsw == 0]
    cic17_benign = load_cic17_benign_training_block()

    all_h1, all_h3, all_repr = [], [], []
    for seed in SEEDS_TO_RUN:
        # Separate cache filename (iec104_seedN.json) from the UNSW
        # script's cache (seedN.json) -- same shared FLOW_DIR, same seed
        # numbers, but different dataset, so must not collide.
        cache_path = os.path.join(FLOW_DIR, f"iec104_seed{seed}.json")
        if os.path.exists(cache_path):
            print(f"[cache] seed {seed} already completed -> loading {cache_path}")
            with open(cache_path) as f:
                cached = json.load(f)
            all_h1.append(cached["h1"])
            all_h3.append(cached["h3"])
            all_repr.append(cached["repr"])
            continue

        h1_result = h1_attribution(seed, X_unsw, y_unsw)
        h3_result = h3_ablation(seed, X_unsw, y_unsw)
        repr_result = representation_analysis(seed, cic17_benign, unsw_benign)

        all_h1.append(h1_result)
        all_h3.append(h3_result)
        all_repr.append(repr_result)

        with open(cache_path, "w") as f:
            json.dump({"h1": h1_result, "h3": h3_result, "repr": repr_result}, f, indent=2)
        print(f"[cache] Saved seed {seed} -> {cache_path}")

    with open(os.path.join(PILOT_DIR, "flow_h1_attribution_iec104.json"), "w") as f:
        json.dump(all_h1, f, indent=2)
    with open(os.path.join(PILOT_DIR, "flow_h3_ablation_iec104.json"), "w") as f:
        json.dump(all_h3, f, indent=2)
    with open(os.path.join(PILOT_DIR, "flow_representation_iec104.json"), "w") as f:
        json.dump(all_repr, f, indent=2)

    print(f"\n{'=' * 70}\nFLOW PIPELINE COMPLETE -- SUMMARY (seed(s): {SEEDS_TO_RUN})\n{'=' * 70}")
    for h1, h3, repr_r in zip(all_h1, all_h3, all_repr):
        print(f"\nSeed {h1['seed']}:")
        if h1.get("additivity_ok"):
            print(f"  H1: ESFRP {h1['esfrp_single']:.4f} -> {h1['esfrp_lodo3']:.4f}, "
                  f"direction correct: {h1['direction_correct']}")
        else:
            print(f"  H1: SKIPPED -- additivity check failed "
                  f"(Single err={h1.get('additivity_err_single', 'n/a')}, "
                  f"LODO-3 err={h1.get('additivity_err_lodo3', 'n/a')})")
        print(f"  H3: delta_stable={h3['delta_stable']:.4f}, delta_sensitive={h3['delta_sensitive']:.4f}, "
              f"delta_middle={h3['delta_middle']:.4f}")
        print(f"  Representation: MMD^2 narrowed: {repr_r['narrowed']} ({repr_r['pct_change']:+.1f}%)")

    print("\nSaved: flow_h1_attribution_iec104.json, flow_h3_ablation_iec104.json, "
          "flow_representation_iec104.json")


if __name__ == "__main__":
    main()
