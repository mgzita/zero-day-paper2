"""
Non-circular re-validation of the core H1-H4 result: Single vs. LODO-3
attribution on UNSW-NB15
=========================================================================
The original headline result (ESFRP shift toward stable features under
Multi training) was established using Single vs Multi, evaluated on
UNSW-NB15. But UNSW-NB15 is one of the three environments INSIDE
Multi's training pool -- meaning that comparison is not a fully clean
test of generalization to a genuinely unseen domain, the same
circularity issue already identified and corrected for Deep SVDD's
latent-space test.

This script re-runs the attribution/ESFRP comparison using:
  - Single:  trained on CIC-IDS2017 only        (UNSW unseen)
  - LODO-3:  trained on CIC17 + CSE-CIC-IDS2018  (UNSW STILL unseen)

Both conditions leave UNSW-NB15 genuinely unseen, giving a clean,
non-circular re-validation of the original headline finding.

Requires ae_lodo3.pt / dae_lodo3.pt / svdd_lodo3.pt to already exist
for each seed being run.

Run from project root:
    python paper2\\pilot\\revalidate_unsw_attribution_lodo3.py
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
UNSW_LODO3_DIR = os.path.join(PILOT_DIR, "unsw_lodo3_revalidation_cache")
os.makedirs(UNSW_LODO3_DIR, exist_ok=True)


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


def load_unsw_evaluation_data():
    path = os.path.join(DEDUP_DIR, "test_unsw_dedup.npz")
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


def compute_asi(normalized_single, normalized_lodo3):
    a, b = normalized_single.align(normalized_lodo3, fill_value=0.0)
    return float(0.5 * np.sum(np.abs(b - a)))


def check_reproduction_directional(scorer_single, scorer_lodo3, eval_df, feature_names, margin=0.10):
    with torch.no_grad():
        x = torch.tensor(eval_df[feature_names].values, dtype=torch.float32)
        scores_single = scorer_single(x).numpy().flatten()
        scores_lodo3 = scorer_lodo3(x).numpy().flatten()
    y = eval_df["label"].values
    auroc_single = roc_auc_score(y, scores_single)
    auroc_lodo3 = roc_auc_score(y, scores_lodo3)
    sep_single = abs(auroc_single - 0.5)
    sep_lodo3 = abs(auroc_lodo3 - 0.5)
    passed = (sep_lodo3 - sep_single) >= margin
    return auroc_single, auroc_lodo3, passed


def run_for_seed(family, seed, eval_df):
    cache_path = os.path.join(UNSW_LODO3_DIR, f"{family}_seed{seed}.json")
    if os.path.exists(cache_path):
        print(f"  [cache] {family}/seed{seed} already completed")
        with open(cache_path) as f:
            return json.load(f)

    single_ckpt = checkpoint_path(seed, family, "single")
    lodo3_ckpt = checkpoint_path(seed, family, "lodo3")
    scorer_single = load_scorer(family, single_ckpt)
    scorer_lodo3 = load_scorer(family, lodo3_ckpt)
    if scorer_single is None or scorer_lodo3 is None:
        print(f"  [skip] {family}/seed{seed}: checkpoint missing "
              f"(single={scorer_single is not None}, lodo3={scorer_lodo3 is not None})")
        return {"family": family, "seed": seed, "skipped": True}

    auroc_single, auroc_lodo3, repro_passed = check_reproduction_directional(
        scorer_single, scorer_lodo3, eval_df, FEATURE_NAMES)
    print(f"  {family}/seed{seed}: AUROC Single={auroc_single:.4f}, LODO-3={auroc_lodo3:.4f}, "
          f"reproduction check {'PASS' if repro_passed else 'FAIL'}")

    rng = np.random.default_rng(seed)
    background_idx = rng.choice(len(eval_df), size=min(N_BACKGROUND, len(eval_df)), replace=False)
    eval_idx = rng.choice(len(eval_df), size=min(N_EVAL, len(eval_df)), replace=False)
    background = eval_df.iloc[background_idx][FEATURE_NAMES].values
    evaluation = eval_df.iloc[eval_idx][FEATURE_NAMES].values

    shap_single, ev_s, method_s, eval_used_s = run_shap(scorer_single, background, evaluation)
    shap_lodo3, ev_l, method_l, eval_used_l = run_shap(scorer_lodo3, background, evaluation)

    err_s = validate_additivity(scorer_single, eval_used_s, shap_single, ev_s)
    err_l = validate_additivity(scorer_lodo3, eval_used_l, shap_lodo3, ev_l)
    if not (err_s < ADDITIVITY_TOL and err_l < ADDITIVITY_TOL):
        print(f"  [HALT] {family}/seed{seed}: additivity failed")
        result = {"family": family, "seed": seed, "additivity_ok": False}
        with open(cache_path, "w") as f:
            json.dump(result, f, indent=2)
        return result

    norm_single = normalized_global_shap(shap_single, FEATURE_NAMES)
    norm_lodo3 = normalized_global_shap(shap_lodo3, FEATURE_NAMES)
    esfrp_single = compute_esfrp(norm_single, STABLE_V2)
    esfrp_lodo3 = compute_esfrp(norm_lodo3, STABLE_V2)
    asi = compute_asi(norm_single, norm_lodo3)
    direction_correct = esfrp_lodo3 > esfrp_single

    print(f"    ESFRP(Single)={esfrp_single:.4f}  ESFRP(LODO-3)={esfrp_lodo3:.4f}  "
          f"ASI={asi:.4f}  Direction correct: {direction_correct}")

    result = {
        "family": family, "seed": seed, "additivity_ok": True,
        "auroc_single": auroc_single, "auroc_lodo3": auroc_lodo3,
        "reproduction_passed": repro_passed,
        "method_single": method_s, "method_lodo3": method_l,
        "esfrp_single": esfrp_single, "esfrp_lodo3": esfrp_lodo3,
        "asi": asi, "direction_correct": direction_correct,
        "normalized_shap_single": norm_single.to_dict(),
        "normalized_shap_lodo3": norm_lodo3.to_dict(),
    }
    with open(cache_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    eval_df = load_unsw_evaluation_data()
    print(f"UNSW-NB15 evaluation set: {len(eval_df)} flows "
          f"({(eval_df['label'] == 0).sum()} benign, {(eval_df['label'] == 1).sum()} attack)\n")
    print("Reused feature groups (target-independent, same as original UNSW analysis):")
    print(f"  Stable:    {STABLE_V2}")
    print(f"  Sensitive: {SENSITIVE_V2}\n")

    all_results = []
    for family in MODEL_FAMILIES:
        print(f"{'=' * 70}\n{family.upper()}\n{'=' * 70}")
        for seed in ALL_SEEDS:
            result = run_for_seed(family, seed, eval_df)
            all_results.append(result)

    print(f"\n{'=' * 70}\nCROSS-SEED SUMMARY (UNSW-NB15, non-circular LODO-3 re-validation)\n{'=' * 70}")
    for family in MODEL_FAMILIES:
        fam_results = [r for r in all_results if r.get("family") == family and r.get("additivity_ok")]
        if not fam_results:
            print(f"\n{family.upper()}: no valid results")
            continue
        n_correct = sum(1 for r in fam_results if r.get("direction_correct"))
        esfrp_s = [r["esfrp_single"] for r in fam_results]
        esfrp_l = [r["esfrp_lodo3"] for r in fam_results]
        print(f"\n{family.upper()}: {n_correct}/{len(fam_results)} seeds correct direction")
        print(f"  ESFRP(Single): mean={np.mean(esfrp_s):.4f}, std={np.std(esfrp_s):.4f}")
        print(f"  ESFRP(LODO-3): mean={np.mean(esfrp_l):.4f}, std={np.std(esfrp_l):.4f}")

    out_path = os.path.join(PILOT_DIR, "unsw_lodo3_revalidation_summary.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
