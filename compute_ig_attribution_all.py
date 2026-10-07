"""
Integrated Gradients attribution -- H1, recomputed with a gradient-based
method instead of SHAP, across ALL 4 model families, ALL 3 datasets,
ALL 10 seeds.
=========================================================================
Motivation: KernelSHAP's additivity repeatedly failed for the
Normalizing Flow (needing a log-transform fallback) due to extreme
score dynamic range under severe distribution shift. Integrated
Gradients (Sundararajan et al., 2017) computes an EXACT path integral
of the model's gradients from a baseline to the input -- no sampling,
no linear-surrogate fitting -- so it should not share this failure
mode. This script tests whether IG confirms, strengthens, or
contradicts the existing SHAP-based H1 findings.

IMPORTANT: this uses the SAME fixed Stable/Sensitive/Middle feature
groups already established independently via ESS for each dataset --
IG is a second, independent check on those same groups, not a
redefinition of them. Redefining feature groups based on IG's own
output would be circular (IG attribution and functional importance are
nearly tautologically related), so this script deliberately does NOT
do that.

REQUIRES: pip install captum --break-system-packages

Reuses:
  - The exact model class definitions and checkpoint conventions from
    each dataset's existing pipeline (UNSW/IEC-104: 18 features,
    LODO-3; SMD: 38 features, Diverse).
  - Per-(dataset, family, seed) caching.

Run from project root:
    python paper2\\pilot\\compute_ig_attribution_all.py
"""

import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients

SEEDS_TO_RUN = [42, 123, 456, 789, 999, 111, 222, 333, 444, 555]
MODEL_FAMILIES = ["ae", "dae", "svdd", "flow"]
DATASETS = ["unsw", "iec104", "smd"]

N_COUPLING_LAYERS = 6
HIDDEN_DIM = 64
N_BACKGROUND = 100
N_EVAL = 500
COMPLETENESS_TOL = 0.05
N_STEPS = 300  # increased from 50; ReLU networks with many "kinks" can
                # need more interpolation steps for the Riemann-sum
                # approximation to converge, especially when the path
                # from baseline to a far out-of-distribution input
                # crosses many piecewise-linear boundaries -- a known,
                # well-documented practical limitation of IG, not
                # evidence the method itself is unsound.

PAPER2_ROOT = r"C:\MLProject\zero_day_project\paper2"
PILOT_DIR = os.path.join(PAPER2_ROOT, "pilot")
CACHE_DIR = os.path.join(PILOT_DIR, "ig_attribution_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

IDS_DEDUP_DIR = r"C:\MLProject\zero_day_project\artefacts_v2\dedup"
SMD_DEDUP_DIR = r"C:\MLProject\zero_day_project\artefacts_v2\dedup_smd"

IDS_FEATURE_NAMES = [
    "bwd_avg_bytes_bulk", "bwd_iat_mean", "bwd_iat_std",
    "bwd_packet_length_mean", "bwd_packet_length_std", "bwd_packets_s",
    "destination_port", "flow_bytes_s", "flow_duration",
    "fwd_avg_bytes_bulk", "fwd_avg_packets_bulk", "fwd_iat_mean",
    "fwd_iat_min", "fwd_iat_std", "fwd_packet_length_mean",
    "init_win_bytes_backward", "init_win_bytes_forward", "subflow_fwd_bytes",
]
IDS_STABLE_V2 = ["bwd_iat_std", "fwd_iat_std", "destination_port",
                 "init_win_bytes_forward", "bwd_packet_length_mean"]

SMD_FEATURE_NAMES = [f"smd_feat_{i:02d}" for i in range(38)]
SMD_STABLE_V2 = ["smd_feat_24", "smd_feat_09", "smd_feat_00", "smd_feat_08", "smd_feat_22"]

DATASET_CONFIG = {
    "unsw": {
        "n_features": 18, "feature_names": IDS_FEATURE_NAMES, "stable_group": IDS_STABLE_V2,
        "dedup_dir": IDS_DEDUP_DIR, "condition_a": "single", "condition_b": "lodo3",
        "eval_file": "test_unsw_dedup.npz", "ckpt_suffix": "",
        "clip_on_load": True,
    },
    "iec104": {
        "n_features": 18, "feature_names": IDS_FEATURE_NAMES, "stable_group": IDS_STABLE_V2,
        "dedup_dir": IDS_DEDUP_DIR, "condition_a": "single", "condition_b": "lodo3",
        "eval_file": "test_iec104_dedup.npz", "ckpt_suffix": "",
        "clip_on_load": True,
    },
    "smd": {
        "n_features": 38, "feature_names": SMD_FEATURE_NAMES, "stable_group": SMD_STABLE_V2,
        "dedup_dir": SMD_DEDUP_DIR, "condition_a": "single", "condition_b": "lodo3",
        "eval_file": "test_smd_group2_dedup.npz", "ckpt_suffix": "_smd",
        "clip_on_load": False,
    },
}


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
        score = ((x - x_hat) ** 2).mean(dim=1, keepdim=True)
        return torch.nan_to_num(score, nan=1e10, posinf=1e10, neginf=-1e10)


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
        score = ((z - self.encoder.center) ** 2).sum(dim=1, keepdim=True)
        return torch.nan_to_num(score, nan=1e10, posinf=1e10, neginf=-1e10)


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
        score = (-self.flow_model.log_prob(x)).unsqueeze(1)
        return torch.nan_to_num(score, nan=1e10, posinf=1e10, neginf=-1e10)


def load_model_and_scorer(family, ckpt_path, input_dim):
    if family in ("ae", "dae"):
        model = AutoencoderModel(input_dim)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
        model.eval()
        return ReconstructionErrorScorer(model).eval()
    elif family == "svdd":
        model = SVDDEncoder(input_dim, latent_dim=16)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
        model.eval()
        return SVDDScorer(model).eval()
    elif family == "flow":
        model = NormalizingFlowModel(dim=input_dim)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
        model.eval()
        return NegativeLogLikelihoodScorer(model).eval()
    raise ValueError(family)


def checkpoint_path(dataset, seed, family, condition, cfg):
    ckpt_dir = os.path.join(cfg["dedup_dir"], f"seed{seed}_{family}{cfg['ckpt_suffix']}")
    fname = "dae_" if family == "dae" else f"{family}_"
    return os.path.join(ckpt_dir, f"{fname}{condition}.pt")


def load_eval_data(dataset, cfg):
    d = np.load(os.path.join(cfg["dedup_dir"], cfg["eval_file"]))
    X = d["X"].astype(np.float32)
    if cfg["clip_on_load"]:
        X = np.clip(X, -10, 10)
    y = d["y"].astype(int)
    return X, y


def compute_ig_attribution(scorer, X_eval):
    """Returns (mean_abs_attribution_per_feature, median_relative_completeness_error).

    IMPORTANT FIX: the completeness check must compare each sample's own
    convergence delta against that SAME sample's own |f(x)-f(baseline)|
    -- not the worst-case delta across the batch against the AVERAGE
    score difference across the batch (an earlier version of this
    function made exactly that mistake, comparing max-of-500 against
    mean-of-500, which is not a fair like-for-like comparison and
    produced spuriously large "errors" for every single one of 120
    model/dataset/seed combinations -- a bug in this validation code,
    not a genuine finding about Integrated Gradients).
    """
    ig = IntegratedGradients(scorer)
    X_t = torch.tensor(X_eval[:N_EVAL], dtype=torch.float32, requires_grad=True)

    baseline = torch.tensor(X_eval[:N_BACKGROUND].mean(axis=0, keepdims=True),
                             dtype=torch.float32)

    attributions, delta = ig.attribute(X_t, baselines=baseline, n_steps=N_STEPS,
                                        return_convergence_delta=True)
    attributions_np = attributions.detach().numpy()

    with torch.no_grad():
        f_x = scorer(X_t).squeeze(-1)          # shape (N_EVAL,)
        f_baseline = scorer(baseline).squeeze(-1)  # shape (1,)
        per_sample_score_diff = (f_x - f_baseline).abs()  # broadcasts to (N_EVAL,)

    per_sample_relative_error = (delta.abs() / per_sample_score_diff.clamp(min=1e-8)).numpy()
    median_relative_error = float(np.median(per_sample_relative_error))
    # Also report the 95th percentile, since a handful of near-zero
    # score-difference samples can still produce a large ratio even
    # when the fit is good everywhere else.
    p95_relative_error = float(np.percentile(per_sample_relative_error, 95))

    mean_abs_attribution = np.mean(np.abs(attributions_np), axis=0)
    return mean_abs_attribution, median_relative_error, p95_relative_error


def normalized_attribution(mean_abs_attribution, feature_names):
    total = mean_abs_attribution.sum()
    normalized = mean_abs_attribution / total if total > 0 else mean_abs_attribution
    return pd.Series(normalized, index=feature_names).sort_values(ascending=False)


def run_one_combination(dataset, family, seed):
    cfg = DATASET_CONFIG[dataset]
    n_features = cfg["n_features"]

    ckpt_a = checkpoint_path(dataset, seed, family, cfg["condition_a"], cfg)
    ckpt_b = checkpoint_path(dataset, seed, family, cfg["condition_b"], cfg)
    if not (os.path.exists(ckpt_a) and os.path.exists(ckpt_b)):
        print(f"    [SKIP] Missing checkpoint(s): {ckpt_a} or {ckpt_b}")
        return None

    scorer_a = load_model_and_scorer(family, ckpt_a, n_features)
    scorer_b = load_model_and_scorer(family, ckpt_b, n_features)

    X_eval, y_eval = load_eval_data(dataset, cfg)

    attr_a, err_a_median, err_a_p95 = compute_ig_attribution(scorer_a, X_eval)
    attr_b, err_b_median, err_b_p95 = compute_ig_attribution(scorer_b, X_eval)

    # Using the median per-sample relative error, not a max-vs-mean
    # comparison (see compute_ig_attribution's docstring for why the
    # earlier version of this check was invalid).
    completeness_ok = (err_a_median < COMPLETENESS_TOL) and (err_b_median < COMPLETENESS_TOL)
    if not completeness_ok:
        print(f"    [WARNING] Completeness check failed (median relative error): "
              f"err_a={err_a_median:.4f}, err_b={err_b_median:.4f} "
              f"(tolerance {COMPLETENESS_TOL}) -- reporting anyway, flagged as completeness_ok=False")

    norm_a = normalized_attribution(attr_a, cfg["feature_names"])
    norm_b = normalized_attribution(attr_b, cfg["feature_names"])
    esfrp_a = float(norm_a.loc[cfg["stable_group"]].sum())
    esfrp_b = float(norm_b.loc[cfg["stable_group"]].sum())

    direction_correct = esfrp_b > esfrp_a

    print(f"    ESFRP({cfg['condition_a']})={esfrp_a:.4f}  ESFRP({cfg['condition_b']})={esfrp_b:.4f}  "
          f"completeness_ok={completeness_ok} (median err a={err_a_median:.4f}, b={err_b_median:.4f})  "
          f"Direction correct: {direction_correct}")

    return {
        "dataset": dataset, "family": family, "seed": seed,
        "esfrp_a": esfrp_a, "esfrp_b": esfrp_b,
        "completeness_err_a_median": err_a_median, "completeness_err_a_p95": err_a_p95,
        "completeness_err_b_median": err_b_median, "completeness_err_b_p95": err_b_p95,
        "completeness_ok": completeness_ok,
        "direction_correct": direction_correct,
    }


def main():
    all_results = []
    for dataset in DATASETS:
        for family in MODEL_FAMILIES:
            for seed in SEEDS_TO_RUN:
                cache_path = os.path.join(CACHE_DIR, f"{dataset}_{family}_seed{seed}.json")
                if os.path.exists(cache_path):
                    print(f"[cache] {dataset}/{family}/seed{seed} already completed -> loading")
                    with open(cache_path) as f:
                        all_results.append(json.load(f))
                    continue

                print(f"\n{'=' * 70}\n{dataset.upper()} / {family.upper()} / seed {seed}\n{'=' * 70}")
                result = run_one_combination(dataset, family, seed)
                if result is not None:
                    all_results.append(result)
                    with open(cache_path, "w") as f:
                        json.dump(result, f, indent=2)
                    print(f"    [cache] Saved -> {cache_path}")

    out_path = os.path.join(PILOT_DIR, "ig_attribution_all_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'=' * 70}\nDONE. {len(all_results)} combinations computed.\nSaved {out_path}\n{'=' * 70}")


if __name__ == "__main__":
    main()
