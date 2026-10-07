"""
CSE-CIC-IDS2018 replication: attribution matrix (H1-H3)
=========================================================
Replicates the core attribution/ESFRP analysis with CSE-CIC-IDS2018 as
the evaluation target, instead of UNSW-NB15, to address the single-
target-domain limitation flagged in external review.

KEY EFFICIENCY: no new model training is required. CSE-CIC-IDS2018 is
already one of the three Multi training environments, so the existing
Single and Multi checkpoints (all three families, all five seeds) are
reused unchanged -- only the EVALUATION dataset changes, from UNSW-
NB15's test set to CSE-CIC-IDS2018's test set.

ESS and the frozen Stable/Sensitive/Middle feature groups also do NOT
change: ESS is computed from source-environment TRAINING data only,
independent of which dataset serves as the evaluation target. The same
decorrelated groups used for the UNSW-NB15 analysis apply unchanged.

Requires test_cic18_dedup.npz and all seed42-999 Single/Multi
checkpoints for ae/dae/svdd to already exist.

Run from project root:
    python paper2\\pilot\\replicate_cic18_attribution.py
"""

import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import shap

ALL_SEEDS = [42, 123, 456, 789, 999]
MODEL_FAMILIES = ["ae", "dae", "svdd"]

N_FEATURES = 18
ADDITIVITY_TOL = 1e-3
N_BACKGROUND = 500
N_EVAL = 500

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
SENSITIVE_V2 = ["fwd_iat_mean", "subflow_fwd_bytes", "bwd_packets_s",
                "flow_duration", "bwd_avg_bytes_bulk"]

DEDUP_DIR = r"C:\MLProject\zero_day_project\artefacts_v2\dedup"  # TODO confirm
PAPER2_ROOT = r"C:\MLProject\zero_day_project\paper2"
PILOT_DIR = os.path.join(PAPER2_ROOT, "pilot")
CIC18_DIR = os.path.join(PILOT_DIR, "cic18_replication_cache")
os.makedirs(CIC18_DIR, exist_ok=True)


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
        return ((x - x_hat) ** 2).mean(dim=1, keepdim=True)


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
        return ((z - self.encoder.center) ** 2).sum(dim=1, keepdim=True)


def checkpoint_path(seed, family, condition):
    ckpt_dir = os.path.join(DEDUP_DIR, f"seed{seed}_{family}")
    fname = "dae_" if family == "dae" else f"{family}_"
    return os.path.join(ckpt_dir, f"{fname}{condition}.pt")


def load_scorer(family, ckpt_path):
    if not os.path.exists(ckpt_path):
        return None
    if family in ("ae", "dae"):
        model = AutoencoderModel(N_FEATURES)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.eval()
        return ReconstructionErrorScorer(model).eval()
    elif family == "svdd":
        model = SVDDEncoder(N_FEATURES, latent_dim=16)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.eval()
        return SVDDScorer(model).eval()
    raise ValueError(family)


def load_cic18_evaluation_data():
    path = os.path.join(DEDUP_DIR, "test_cic18_dedup.npz")
    d = np.load(path)
    X = np.clip(d["X"], -10, 10).astype(np.float32)
    y = d["y"].astype(int)
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["label"] = y
    return df


def validate_additivity(scorer, evaluation, shap_values, expected_value):
    with torch.no_grad():
        actual = scorer(torch.tensor(evaluation, dtype=torch.float32)).numpy().flatten()
    reconstructed = expected_value + shap_values.sum(axis=1)
    return float(np.max(np.abs(actual - reconstructed)))


def run_shap(scorer, background, evaluation, n_kernelshap_eval=N_EVAL):
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
            return shap_values, expected_value, "DeepSHAP", evaluation
    except AssertionError:
        pass

    eval_sub = evaluation[:n_kernelshap_eval]

    def f(x_np):
        with torch.no_grad():
            return scorer(torch.tensor(x_np, dtype=torch.float32)).numpy()

    background_sub = background[:min(100, len(background))]
    kexplainer = shap.KernelExplainer(f, background_sub)
    shap_values = kexplainer.shap_values(eval_sub, nsamples="auto")
    shap_values = np.array(shap_values).reshape(eval_sub.shape[0], N_FEATURES)
    expected_value = float(np.array(kexplainer.expected_value).flatten()[0])
    return shap_values, expected_value, "KernelSHAP", eval_sub


def normalized_global_shap(shap_values, feature_names):
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    total = mean_abs.sum()
    normalized = mean_abs / total if total > 0 else mean_abs
    return pd.Series(normalized, index=feature_names, name="normalized_shap").sort_values(ascending=False)


def compute_esfrp(normalized_shap, stable_features):
    return float(normalized_shap.loc[stable_features].sum())


def compute_asi(normalized_single, normalized_multi):
    a, b = normalized_single.align(normalized_multi, fill_value=0.0)
    return float(0.5 * np.sum(np.abs(b - a)))


def check_reproduction_directional(scorer_single, scorer_multi, eval_df, feature_names, margin=0.10):
    with torch.no_grad():
        x = torch.tensor(eval_df[feature_names].values, dtype=torch.float32)
        scores_single = scorer_single(x).numpy().flatten()
        scores_multi = scorer_multi(x).numpy().flatten()
    y = eval_df["label"].values
    auroc_single = roc_auc_score(y, scores_single)
    auroc_multi = roc_auc_score(y, scores_multi)
    sep_single = abs(auroc_single - 0.5)
    sep_multi = abs(auroc_multi - 0.5)
    passed = (sep_multi - sep_single) >= margin
    return auroc_single, auroc_multi, passed


def run_for_seed(family, seed, eval_df):
    cache_path = os.path.join(CIC18_DIR, f"{family}_seed{seed}.json")
    if os.path.exists(cache_path):
        print(f"  [cache] {family}/seed{seed} already completed")
        with open(cache_path) as f:
            return json.load(f)

    single_ckpt = checkpoint_path(seed, family, "single")
    multi_ckpt = checkpoint_path(seed, family, "multi")
    scorer_single = load_scorer(family, single_ckpt)
    scorer_multi = load_scorer(family, multi_ckpt)
    if scorer_single is None or scorer_multi is None:
        print(f"  [skip] {family}/seed{seed}: checkpoint missing")
        return {"family": family, "seed": seed, "skipped": True}

    auroc_single, auroc_multi, repro_passed = check_reproduction_directional(
        scorer_single, scorer_multi, eval_df, FEATURE_NAMES)
    print(f"  {family}/seed{seed}: AUROC Single={auroc_single:.4f}, Multi={auroc_multi:.4f}, "
          f"reproduction check {'PASS' if repro_passed else 'FAIL'}")

    rng = np.random.default_rng(seed)
    background_idx = rng.choice(len(eval_df), size=min(N_BACKGROUND, len(eval_df)), replace=False)
    eval_idx = rng.choice(len(eval_df), size=min(N_EVAL, len(eval_df)), replace=False)
    background = eval_df.iloc[background_idx][FEATURE_NAMES].values
    evaluation = eval_df.iloc[eval_idx][FEATURE_NAMES].values

    shap_single, ev_s, method_s, eval_used_s = run_shap(scorer_single, background, evaluation)
    shap_multi, ev_m, method_m, eval_used_m = run_shap(scorer_multi, background, evaluation)

    err_s = validate_additivity(scorer_single, eval_used_s, shap_single, ev_s)
    err_m = validate_additivity(scorer_multi, eval_used_m, shap_multi, ev_m)
    if not (err_s < ADDITIVITY_TOL and err_m < ADDITIVITY_TOL):
        print(f"  [HALT] {family}/seed{seed}: additivity failed")
        result = {"family": family, "seed": seed, "additivity_ok": False}
        with open(cache_path, "w") as f:
            json.dump(result, f, indent=2)
        return result

    norm_single = normalized_global_shap(shap_single, FEATURE_NAMES)
    norm_multi = normalized_global_shap(shap_multi, FEATURE_NAMES)
    esfrp_single = compute_esfrp(norm_single, STABLE_V2)
    esfrp_multi = compute_esfrp(norm_multi, STABLE_V2)
    asi = compute_asi(norm_single, norm_multi)
    direction_correct = esfrp_multi > esfrp_single

    print(f"    ESFRP(Single)={esfrp_single:.4f}  ESFRP(Multi)={esfrp_multi:.4f}  "
          f"ASI={asi:.4f}  Direction correct: {direction_correct}")

    result = {
        "family": family, "seed": seed, "additivity_ok": True,
        "auroc_single": auroc_single, "auroc_multi": auroc_multi,
        "reproduction_passed": repro_passed,
        "method_single": method_s, "method_multi": method_m,
        "esfrp_single": esfrp_single, "esfrp_multi": esfrp_multi,
        "asi": asi, "direction_correct": direction_correct,
        "normalized_shap_single": norm_single.to_dict(),
        "normalized_shap_multi": norm_multi.to_dict(),
    }
    with open(cache_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    eval_df = load_cic18_evaluation_data()
    print(f"CSE-CIC-IDS2018 evaluation set: {len(eval_df)} flows "
          f"({(eval_df['label'] == 0).sum()} benign, {(eval_df['label'] == 1).sum()} attack)\n")
    print(f"Reused feature groups (target-independent):")
    print(f"  Stable:    {STABLE_V2}")
    print(f"  Sensitive: {SENSITIVE_V2}\n")

    all_results = []
    for family in MODEL_FAMILIES:
        print(f"{'=' * 70}\n{family.upper()}\n{'=' * 70}")
        for seed in ALL_SEEDS:
            result = run_for_seed(family, seed, eval_df)
            all_results.append(result)

    print(f"\n{'=' * 70}\nCROSS-SEED SUMMARY (CSE-CIC-IDS2018 target)\n{'=' * 70}")
    for family in MODEL_FAMILIES:
        fam_results = [r for r in all_results if r.get("family") == family and r.get("additivity_ok")]
        if not fam_results:
            print(f"\n{family.upper()}: no valid results")
            continue
        n_correct = sum(1 for r in fam_results if r.get("direction_correct"))
        esfrp_s = [r["esfrp_single"] for r in fam_results]
        esfrp_m = [r["esfrp_multi"] for r in fam_results]
        print(f"\n{family.upper()}: {n_correct}/{len(fam_results)} seeds correct direction")
        print(f"  ESFRP(Single): mean={np.mean(esfrp_s):.4f}, std={np.std(esfrp_s):.4f}")
        print(f"  ESFRP(Multi):  mean={np.mean(esfrp_m):.4f}, std={np.std(esfrp_m):.4f}")

    out_path = os.path.join(PILOT_DIR, "cic18_replication_summary.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
