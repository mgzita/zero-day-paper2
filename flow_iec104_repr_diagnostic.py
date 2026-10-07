"""
Flow / IEC-104 representation-metric diagnostic
=================================================
Follow-up to paper2_followup_tests.py's Wasserstein-agreement check,
which found 0/10 seed-level sign agreement between MMD^2 and mean
sliced Wasserstein distance for the Flow model on IEC-104 -- the
manuscript's reported "Gap widened (reversed)" claim (d=-4.00) is
based on MMD^2 alone, and Wasserstein disagrees with it on every seed.

This script checks the most likely mechanical explanation: MMD^2's
RBF kernel bandwidth (sigma^2) is set via a per-condition median
heuristic -- see mmd_squared_rbf() in flow_full_pipeline_iec104.py --
estimated SEPARATELY for the Single and LODO-3 embeddings each time.
If the two conditions' embeddings differ in overall scale or have a
few extreme outliers, the two bandwidths differ, and the resulting
MMD^2 values become sensitive to that scale/outlier difference rather
than purely to how much the two distributions overlap. Wasserstein
has no kernel/bandwidth step, so it would not share this artifact.

Three checks, per seed:
  1. Bandwidth comparison: sigma^2 (median heuristic) for Single vs
     LODO-3 embeddings. A large swing supports the bandwidth-artifact
     explanation.
  2. Outlier / tail check: embedding-norm percentiles (median, p95,
     p99, max) for source and target, both conditions. Heavy tails or
     extreme outliers would help explain why a kernel-weighted measure
     (MMD^2) and a measure sensitive to all probability mass
     (Wasserstein) can disagree.
  3. Fixed-bandwidth MMD^2 re-run: recompute MMD^2 for both conditions
     using ONE shared bandwidth (pooled/averaged across Single and
     LODO-3), removing the per-condition bandwidth confound. If the
     "widened vs narrowed" direction flips or disappears under a fixed
     bandwidth, that is direct evidence the original MMD^2 result was
     (at least partly) a bandwidth artifact rather than a genuine,
     bandwidth-independent property of the representations.

This is diagnostic/exploratory. It does not recompute or override the
numbers already reported (those stand as originally computed); it
tries to explain why MMD^2 and Wasserstein disagree for this one cell.
It reuses the checkpoint/model/data-loading functions from
flow_full_pipeline_iec104.py directly (no duplicated logic), so it
must be run from the same folder as that script.

Run from project root:
    python paper2\\pilot\\flow_iec104_repr_diagnostic.py
"""

import json
import os
import sys

import numpy as np
import torch

import flow_full_pipeline_iec104 as ph  # reuse checkpoint/model/data loaders

SEEDS_TO_RUN = ph.SEEDS_TO_RUN
PILOT_DIR = ph.PILOT_DIR

NOTES = []


def safe_load(path):
    if not os.path.exists(path):
        msg = f"MISSING file: {path}"
        NOTES.append(msg)
        print(msg)
        return None
    with open(path) as f:
        return json.load(f)


def get_models(seed):
    single_ckpt = ph.ensure_flow_checkpoint(seed, "single")
    lodo3_ckpt = ph.ensure_flow_checkpoint(seed, "lodo3")
    model_single = ph.load_flow_model(single_ckpt)
    model_lodo3 = ph.load_flow_model(lodo3_ckpt)
    return model_single, model_lodo3


def embed(model, X):
    with torch.no_grad():
        z, _ = model.transform(torch.tensor(X, dtype=torch.float32))
    return z.numpy()


def median_heuristic_sigma2(X, Y, max_sample, seed):
    """Replicates the sigma^2 computation inside mmd_squared_rbf(), but
    returns sigma^2 itself (rather than the final MMD^2 value) so it can
    be compared across conditions."""
    rng = np.random.default_rng(seed)
    Xs, Ys = X, Y
    if len(Xs) > max_sample:
        Xs = Xs[rng.choice(len(Xs), max_sample, replace=False)]
    if len(Ys) > max_sample:
        Ys = Ys[rng.choice(len(Ys), max_sample, replace=False)]
    Z = np.vstack([Xs, Ys])
    n_pairs = min(2000, len(Z))
    idx_a = rng.integers(0, len(Z), n_pairs)
    idx_b = rng.integers(0, len(Z), n_pairs)
    pair_d2 = np.sum((Z[idx_a] - Z[idx_b]) ** 2, axis=1)
    median_d2 = np.median(pair_d2[pair_d2 > 0])
    return float(median_d2 / 2) if median_d2 > 0 else 1.0


def mmd_squared_fixed_bandwidth(X, Y, sigma2, max_sample, seed):
    rng = np.random.default_rng(seed)
    Xs, Ys = X, Y
    if len(Xs) > max_sample:
        Xs = Xs[rng.choice(len(Xs), max_sample, replace=False)]
    if len(Ys) > max_sample:
        Ys = Ys[rng.choice(len(Ys), max_sample, replace=False)]

    def rbf_kernel_mean(A, B):
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)
        return np.exp(-d2 / (2 * sigma2)).mean()

    return float(rbf_kernel_mean(Xs, Xs) + rbf_kernel_mean(Ys, Ys) - 2 * rbf_kernel_mean(Xs, Ys))


def norm_percentiles(Z):
    norms = np.linalg.norm(Z, axis=1)
    return {
        "median": float(np.median(norms)),
        "p95": float(np.percentile(norms, 95)),
        "p99": float(np.percentile(norms, 99)),
        "max": float(np.max(norms)),
    }


def run_seed(seed, cic17_benign, iec104_benign):
    print(f"\n{'=' * 70}\nSeed {seed}\n{'=' * 70}")
    model_single, model_lodo3 = get_models(seed)

    z_cic_single = embed(model_single, cic17_benign)
    z_iec_single = embed(model_single, iec104_benign)
    z_cic_lodo3 = embed(model_lodo3, cic17_benign)
    z_iec_lodo3 = embed(model_lodo3, iec104_benign)

    # --- Check 1: bandwidth comparison ---
    sigma2_single = median_heuristic_sigma2(z_cic_single, z_iec_single, ph.MAX_SAMPLE_REPR, seed)
    sigma2_lodo3 = median_heuristic_sigma2(z_cic_lodo3, z_iec_lodo3, ph.MAX_SAMPLE_REPR, seed)
    bandwidth_ratio = sigma2_lodo3 / sigma2_single if sigma2_single != 0 else float("nan")
    print(f"  sigma^2: single={sigma2_single:.4f}  lodo3={sigma2_lodo3:.4f}  "
          f"ratio(lodo3/single)={bandwidth_ratio:.3f}")

    # --- Check 2: outlier / tail check ---
    tails = {
        "cic_single": norm_percentiles(z_cic_single),
        "iec_single": norm_percentiles(z_iec_single),
        "cic_lodo3": norm_percentiles(z_cic_lodo3),
        "iec_lodo3": norm_percentiles(z_iec_lodo3),
    }
    for name, t in tails.items():
        tail_ratio = t["max"] / t["median"] if t["median"] > 0 else float("nan")
        print(f"  {name:12s} norm: median={t['median']:.3f}  p95={t['p95']:.3f}  "
              f"p99={t['p99']:.3f}  max={t['max']:.3f}  (max/median={tail_ratio:.2f})")

    # --- Check 3: fixed shared-bandwidth MMD^2 recompute ---
    sigma2_fixed = (sigma2_single + sigma2_lodo3) / 2
    mmd_single_fixed = mmd_squared_fixed_bandwidth(z_cic_single, z_iec_single, sigma2_fixed, ph.MAX_SAMPLE_REPR, seed)
    mmd_lodo3_fixed = mmd_squared_fixed_bandwidth(z_cic_lodo3, z_iec_lodo3, sigma2_fixed, ph.MAX_SAMPLE_REPR, seed)
    pct_change_fixed = (100 * (mmd_single_fixed - mmd_lodo3_fixed) / mmd_single_fixed
                         if mmd_single_fixed != 0 else float("nan"))
    narrowed_fixed = mmd_lodo3_fixed < mmd_single_fixed

    # original (per-condition bandwidth) MMD^2, recomputed here for direct comparison
    mmd_single_orig = ph.mmd_squared_rbf(z_cic_single, z_iec_single, seed=seed)
    mmd_lodo3_orig = ph.mmd_squared_rbf(z_cic_lodo3, z_iec_lodo3, seed=seed)
    pct_change_orig = (100 * (mmd_single_orig - mmd_lodo3_orig) / mmd_single_orig
                        if mmd_single_orig != 0 else float("nan"))
    narrowed_orig = mmd_lodo3_orig < mmd_single_orig

    print(f"  MMD^2 (original, per-condition bandwidth): single={mmd_single_orig:.4f}  "
          f"lodo3={mmd_lodo3_orig:.4f}  ({pct_change_orig:+.1f}%)  narrowed={narrowed_orig}")
    print(f"  MMD^2 (fixed, shared bandwidth):            single={mmd_single_fixed:.4f}  "
          f"lodo3={mmd_lodo3_fixed:.4f}  ({pct_change_fixed:+.1f}%)  narrowed={narrowed_fixed}")
    if narrowed_orig != narrowed_fixed:
        print("  >>> DIRECTION FLIPS under fixed bandwidth -- supports the bandwidth-artifact explanation.")
    else:
        print("  >>> Direction unchanged under fixed bandwidth -- bandwidth choice does NOT explain the "
              "MMD^2/Wasserstein disagreement for this seed.")

    return {
        "seed": seed,
        "sigma2_single": sigma2_single, "sigma2_lodo3": sigma2_lodo3, "bandwidth_ratio": bandwidth_ratio,
        "tails": tails,
        "mmd_single_orig": mmd_single_orig, "mmd_lodo3_orig": mmd_lodo3_orig,
        "pct_change_orig": pct_change_orig, "narrowed_orig": narrowed_orig,
        "mmd_single_fixed": mmd_single_fixed, "mmd_lodo3_fixed": mmd_lodo3_fixed,
        "pct_change_fixed": pct_change_fixed, "narrowed_fixed": narrowed_fixed,
        "direction_flips": bool(narrowed_orig != narrowed_fixed),
    }


def main():
    X_iec, y_iec = ph.load_iec104_eval()
    iec104_benign = X_iec[y_iec == 0]
    cic17_benign = ph.load_cic17_benign_training_block()

    results = []
    for seed in SEEDS_TO_RUN:
        results.append(run_seed(seed, cic17_benign, iec104_benign))

    n_flip = sum(r["direction_flips"] for r in results)
    mean_bandwidth_ratio = float(np.mean([r["bandwidth_ratio"] for r in results]))
    print(f"\n{'=' * 70}\nSUMMARY across {len(results)} seeds\n{'=' * 70}")
    print(f"  Seeds where fixed-bandwidth MMD^2 flips direction vs the original: {n_flip}/{len(results)}")
    print(f"  Mean bandwidth ratio (sigma^2 lodo3/single): {mean_bandwidth_ratio:.3f}")
    if n_flip >= len(results) // 2:
        print("  -> Bandwidth sensitivity plausibly explains most of the MMD^2/Wasserstein disagreement.")
    elif n_flip > 0:
        print("  -> Bandwidth sensitivity explains SOME but not most seeds; other factors likely also contribute.")
    else:
        print("  -> Fixing the bandwidth does not change direction for any seed -- the bandwidth-artifact "
              "explanation is NOT supported by this check. The MMD^2/Wasserstein disagreement more likely "
              "reflects a genuine difference in what the two metrics are sensitive to, not a fixable artifact.")

    if NOTES:
        print(f"\n{'=' * 70}\nNOTES (missing files etc.)\n{'=' * 70}")
        for n in NOTES:
            print(f"  - {n}")

    out_path = os.path.join(PILOT_DIR, "flow_iec104_repr_diagnostic.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
