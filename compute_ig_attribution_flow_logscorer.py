"""
Integrated Gradients for the Normalizing Flow, using the SAME scorer SHAP used
================================================================================
Reviewer feedback (major comment #5): the accompanying code shows KernelSHAP
explaining log(1 + NLL) for the Normalizing Flow (LogCompressedNLLScorer,
flow_full_pipeline.py), while Integrated Gradients explained raw NLL
(NegativeLogLikelihoodScorer, compute_ig_attribution_all.py). Per Table 3,
the log-transform fallback was used for 30/30 Flow seed-target combinations
-- i.e. SHAP ALWAYS explained the log-transformed score for the Flow, never
the raw one. So the SHAP-vs-IG "disagreement" for the Flow was partly
comparing attributions of two different functions, not just two different
attribution methods on the same function.

Fix: recompute IG for family=="flow" only, using LogCompressedNLLScorer
(identical definition to flow_full_pipeline.py) instead of
NegativeLogLikelihoodScorer. AE/DAE/SVDD already use the same scorer for
both methods, so they are NOT rerun here.

Reuses the EXACT checkpoint/eval-data conventions from
compute_ig_attribution_all.py -- same checkpoints, same evaluation data, same
completeness check, same N_STEPS/N_BACKGROUND/N_EVAL. Only the Flow scorer
class changes. No retraining.

After recomputing, this also re-runs the paired ESFRP shift test (t-test,
Wilcoxon, Cohen's d) for the three Flow rows only, so you can directly compare
against the existing Table 5 (IG) and Table 4 (SHAP) Flow rows under the new,
consistent scoring, and reassess whether the "does not replicate" conclusion
still holds for the Flow once both methods explain the same function.

BEFORE RUNNING: this assumes compute_ig_attribution_all.py's directory/path
conventions (PAPER2_ROOT, DEDUP_DIR, checkpoint naming) still match your repo
-- adjust the constants below if your layout has moved since that script was
written.

Run from the repo root (same place as compute_ig_attribution_all.py):
    python compute_ig_attribution_flow_logscorer.py
"""

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
from scipy import stats

SEEDS_TO_RUN = [42, 123, 456, 789, 999, 111, 222, 333, 444, 555]
DATASETS = ["unsw", "iec104", "smd"]

N_COUPLING_LAYERS = 6
HIDDEN_DIM = 64
N_BACKGROUND = 100
N_EVAL = 500
COMPLETENESS_TOL = 0.05
N_STEPS = 300

PAPER2_ROOT = r"C:\MLProject\zero_day_project\paper2"
PILOT_DIR = os.path.join(PAPER2_ROOT, "pilot")
CACHE_DIR = os.path.join(PILOT_DIR, "ig_attribution_flow_logscorer_cache")
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


class LogCompressedNLLScorer(nn.Module):
    """Identical definition to flow_full_pipeline.py's LogCompressedNLLScorer
    -- explains log(1 + NLL) instead of raw NLL, matching what KernelSHAP
    actually explained for every one of the 30 Flow seed-target combinations
    (Table 3: fallback used 30/30)."""

    def __init__(self, flow_model):
        super().__init__()
        self.flow_model = flow_model

    def forward(self, x):
        nll = -self.flow_model.log_prob(x)
        nll = torch.nan_to_num(nll, nan=1e10, posinf=1e10, neginf=-1e10)
        return torch.log1p(nll.clamp(min=0)).unsqueeze(1)


def load_flow_scorer(ckpt_path, input_dim):
    model = NormalizingFlowModel(dim=input_dim)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
    model.eval()
    return LogCompressedNLLScorer(model).eval()


def checkpoint_path(dataset, seed, condition, cfg):
    ckpt_dir = os.path.join(cfg["dedup_dir"], f"seed{seed}_flow{cfg['ckpt_suffix']}")
    return os.path.join(ckpt_dir, f"flow_{condition}.pt")


def load_eval_data(cfg):
    d = np.load(os.path.join(cfg["dedup_dir"], cfg["eval_file"]))
    X = d["X"].astype(np.float32)
    if cfg["clip_on_load"]:
        X = np.clip(X, -10, 10)
    y = d["y"].astype(int)
    return X, y


def compute_ig_attribution(scorer, X_eval):
    """Same completeness-check logic as compute_ig_attribution_all.py:
    per-sample relative error against that SAME sample's own
    |f(x)-f(baseline)|, not a max-vs-mean comparison."""
    ig = IntegratedGradients(scorer)
    X_t = torch.tensor(X_eval[:N_EVAL], dtype=torch.float32, requires_grad=True)
    baseline = torch.tensor(X_eval[:N_BACKGROUND].mean(axis=0, keepdims=True), dtype=torch.float32)

    attributions, delta = ig.attribute(X_t, baselines=baseline, n_steps=N_STEPS,
                                        return_convergence_delta=True)
    attributions_np = attributions.detach().numpy()

    with torch.no_grad():
        f_x = scorer(X_t).squeeze(-1)
        f_baseline = scorer(baseline).squeeze(-1)
        per_sample_score_diff = (f_x - f_baseline).abs()

    per_sample_relative_error = (delta.abs() / per_sample_score_diff.clamp(min=1e-8)).numpy()
    median_relative_error = float(np.median(per_sample_relative_error))
    p95_relative_error = float(np.percentile(per_sample_relative_error, 95))

    mean_abs_attribution = np.mean(np.abs(attributions_np), axis=0)
    return mean_abs_attribution, median_relative_error, p95_relative_error


def normalized_attribution(mean_abs_attribution, feature_names):
    total = mean_abs_attribution.sum()
    normalized = mean_abs_attribution / total if total > 0 else mean_abs_attribution
    return pd.Series(normalized, index=feature_names).sort_values(ascending=False)


def run_one_combination(dataset, seed):
    cfg = DATASET_CONFIG[dataset]
    n_features = cfg["n_features"]

    ckpt_a = checkpoint_path(dataset, seed, cfg["condition_a"], cfg)
    ckpt_b = checkpoint_path(dataset, seed, cfg["condition_b"], cfg)
    if not (os.path.exists(ckpt_a) and os.path.exists(ckpt_b)):
        print(f"    [SKIP] Missing checkpoint(s): {ckpt_a} or {ckpt_b}")
        return None

    scorer_a = load_flow_scorer(ckpt_a, n_features)
    scorer_b = load_flow_scorer(ckpt_b, n_features)

    X_eval, y_eval = load_eval_data(cfg)

    attr_a, err_a_median, err_a_p95 = compute_ig_attribution(scorer_a, X_eval)
    attr_b, err_b_median, err_b_p95 = compute_ig_attribution(scorer_b, X_eval)

    completeness_ok = (err_a_median < COMPLETENESS_TOL) and (err_b_median < COMPLETENESS_TOL)
    if not completeness_ok:
        print(f"    [WARNING] Completeness check failed: err_a={err_a_median:.4f}, "
              f"err_b={err_b_median:.4f} (tolerance {COMPLETENESS_TOL})")

    norm_a = normalized_attribution(attr_a, cfg["feature_names"])
    norm_b = normalized_attribution(attr_b, cfg["feature_names"])
    esfrp_a = float(norm_a.loc[cfg["stable_group"]].sum())
    esfrp_b = float(norm_b.loc[cfg["stable_group"]].sum())

    print(f"    [log-scorer] ESFRP(single)={esfrp_a:.4f}  ESFRP(lodo3)={esfrp_b:.4f}  "
          f"completeness_ok={completeness_ok}")

    return {
        "dataset": dataset, "family": "flow", "seed": seed, "scorer": "log1p_nll",
        "esfrp_a": esfrp_a, "esfrp_b": esfrp_b,
        "completeness_err_a_median": err_a_median, "completeness_err_a_p95": err_a_p95,
        "completeness_err_b_median": err_b_median, "completeness_err_b_p95": err_b_p95,
        "completeness_ok": completeness_ok,
    }


def main():
    all_results = []
    for dataset in DATASETS:
        for seed in SEEDS_TO_RUN:
            cache_path = os.path.join(CACHE_DIR, f"{dataset}_flow_logscorer_seed{seed}.json")
            if os.path.exists(cache_path):
                print(f"[cache] {dataset}/flow/seed{seed} (log-scorer) already done -> loading")
                with open(cache_path) as f:
                    all_results.append(json.load(f))
                continue

            print(f"\n{'=' * 70}\n{dataset.upper()} / FLOW (log1p-NLL scorer) / seed {seed}\n{'=' * 70}")
            result = run_one_combination(dataset, seed)
            if result is not None:
                all_results.append(result)
                with open(cache_path, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"    [cache] Saved -> {cache_path}")

    out_path = os.path.join(PILOT_DIR, "ig_attribution_flow_logscorer_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {out_path}")

    # ------------------------------------------------------------------
    # Paired ESFRP shift test, Flow only, consistent-scorer IG -- directly
    # comparable to the existing Table 4 (SHAP) Flow rows, since both now
    # explain log(1 + NLL).
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}\nPaired ESFRP shift test -- Flow, Integrated Gradients, log(1+NLL) scorer\n{'=' * 70}")
    summary = {}
    for dataset in DATASETS:
        rows = [r for r in all_results if r["dataset"] == dataset]
        seeds_common = sorted(r["seed"] for r in rows)
        if len(seeds_common) < 2:
            print(f"  {dataset}: only {len(seeds_common)} seed(s) -- skipping (need >= 2).")
            continue
        esfrp_a = np.array([r["esfrp_a"] for r in rows])
        esfrp_b = np.array([r["esfrp_b"] for r in rows])

        t_stat, t_p = stats.ttest_rel(esfrp_b, esfrp_a)
        try:
            w_stat, w_p = stats.wilcoxon(esfrp_b, esfrp_a)
        except ValueError:
            w_p = None
        diffs = esfrp_b - esfrp_a
        d_std = diffs.std(ddof=1)
        cohens_d = float(diffs.mean() / d_std) if d_std > 0 else float("nan")

        print(f"  {dataset.upper()}: single={esfrp_a.mean():.4f}, lodo3={esfrp_b.mean():.4f}, "
              f"t={t_stat:.3f}, p={t_p:.4f}, "
              f"Wilcoxon p={'n/a' if w_p is None else f'{w_p:.4f}'}, d={cohens_d:+.3f}")

        summary[dataset] = {
            "n_seeds": len(seeds_common), "esfrp_single_mean": float(esfrp_a.mean()),
            "esfrp_lodo3_mean": float(esfrp_b.mean()), "t_stat": float(t_stat),
            "p_value": float(t_p), "wilcoxon_p": None if w_p is None else float(w_p),
            "cohens_d": cohens_d,
        }

    stats_out_path = os.path.join(PILOT_DIR, "ig_attribution_flow_logscorer_paired_stats.json")
    with open(stats_out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {stats_out_path}")
    print("\nCompare these p/d values against Table 4's Flow rows (SHAP, same log1p scorer) "
          "and against the existing Table 5 Flow rows (IG, raw-NLL scorer) to see whether "
          "the 'does not replicate' conclusion for the Flow changes once both methods "
          "explain the same function.")


if __name__ == "__main__":
    main()
