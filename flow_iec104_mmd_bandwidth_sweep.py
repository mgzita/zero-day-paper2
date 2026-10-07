"""
Flow / IEC-104 MMD^2 bandwidth-sweep + kernel-term decomposition
====================================================================
This is the rigorous follow-up to flow_iec104_repr_diagnostic.py's
single "average bandwidth" check, which only flipped direction for
1/10 seeds -- inconclusive, because that fixed bandwidth was still
tiny (~3-14) relative to the actual squared-distance scale implied by
the Single-condition IEC-104 latent-norm blow-up (norms in the
hundreds, so squared distances on the order of 1e5).

This script proves or disproves the "kernel saturation" explanation
directly, two ways, per seed:

  1. DECOMPOSITION at the ORIGINAL (per-condition, median-heuristic)
     bandwidth: MMD^2 = mean(k(X,X)) + mean(k(Y,Y)) - 2*mean(k(X,Y)).
     If the cross term mean(k(X,Y)) has collapsed to ~0 relative to
     the within-cluster terms, that IS the saturation signature,
     directly, not inferred.

  2. BANDWIDTH SWEEP: recompute MMD^2 (single vs lodo3, SAME shared
     bandwidth applied to both conditions at each sweep point) across
     ~20 log-spaced bandwidth values, from near the original estimate
     up to well beyond the actual squared-distance scale observed.
     Reports the "narrowed" direction at every sweep point and the
     crossover bandwidth (if any) where the direction flips relative
     to the original per-condition result.

Pairwise squared-distance matrices are computed ONCE per condition
per seed (the expensive O(n^2 x dim) step); every sweep bandwidth then
just re-applies exp(-d2/(2*sigma2)).mean() to the already-computed
matrix, which is cheap. No retraining -- reuses cached checkpoints via
flow_full_pipeline_iec104.py, exactly like flow_iec104_repr_diagnostic.py.

Run from project root (same folder as flow_full_pipeline_iec104.py and
flow_iec104_repr_diagnostic.py):
    python paper2\\pilot\\flow_iec104_mmd_bandwidth_sweep.py
"""

import json
import os

import numpy as np
import torch

import flow_full_pipeline_iec104 as ph  # reuse checkpoint/model/data loaders

SEEDS_TO_RUN = ph.SEEDS_TO_RUN
PILOT_DIR = ph.PILOT_DIR
N_SWEEP = 20

NOTES = []


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


def subsample(X, max_sample, rng):
    if len(X) > max_sample:
        return X[rng.choice(len(X), max_sample, replace=False)]
    return X


def sq_dist_matrix(A, B):
    return ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)


def median_heuristic_sigma2(d2_pool_sample):
    d2_pool_sample = d2_pool_sample[d2_pool_sample > 0]
    med = np.median(d2_pool_sample)
    return float(med / 2) if med > 0 else 1.0


def kernel_mean(d2, sigma2):
    return float(np.exp(-d2 / (2 * sigma2)).mean())


def mmd_from_d2(d2_xx, d2_yy, d2_xy, sigma2):
    kxx = kernel_mean(d2_xx, sigma2)
    kyy = kernel_mean(d2_yy, sigma2)
    kxy = kernel_mean(d2_xy, sigma2)
    return kxx + kyy - 2 * kxy, kxx, kyy, kxy


def run_seed(seed, cic17_benign, iec104_benign):
    print(f"\n{'=' * 70}\nSeed {seed}\n{'=' * 70}")
    model_single, model_lodo3 = get_models(seed)

    z_cic_single = embed(model_single, cic17_benign)
    z_iec_single = embed(model_single, iec104_benign)
    z_cic_lodo3 = embed(model_lodo3, cic17_benign)
    z_iec_lodo3 = embed(model_lodo3, iec104_benign)

    max_sample = ph.MAX_SAMPLE_REPR
    rng = np.random.default_rng(seed)
    Xs = subsample(z_cic_single, max_sample, rng)
    Ys = subsample(z_iec_single, max_sample, rng)
    Xl = subsample(z_cic_lodo3, max_sample, rng)
    Yl = subsample(z_iec_lodo3, max_sample, rng)

    # --- pairwise squared-distance matrices, computed ONCE per condition ---
    d2_xx_s = sq_dist_matrix(Xs, Xs)
    d2_yy_s = sq_dist_matrix(Ys, Ys)
    d2_xy_s = sq_dist_matrix(Xs, Ys)
    d2_xx_l = sq_dist_matrix(Xl, Xl)
    d2_yy_l = sq_dist_matrix(Yl, Yl)
    d2_xy_l = sq_dist_matrix(Xl, Yl)

    # --- original per-condition median-heuristic bandwidth (matches
    #     mmd_squared_rbf's own estimate, recomputed here for the decomposition) ---
    Z_s = np.vstack([Xs, Ys])
    n_pairs = min(2000, len(Z_s))
    idx_a = rng.integers(0, len(Z_s), n_pairs)
    idx_b = rng.integers(0, len(Z_s), n_pairs)
    sigma2_orig_single = median_heuristic_sigma2(np.sum((Z_s[idx_a] - Z_s[idx_b]) ** 2, axis=1))

    Z_l = np.vstack([Xl, Yl])
    n_pairs = min(2000, len(Z_l))
    idx_a = rng.integers(0, len(Z_l), n_pairs)
    idx_b = rng.integers(0, len(Z_l), n_pairs)
    sigma2_orig_lodo3 = median_heuristic_sigma2(np.sum((Z_l[idx_a] - Z_l[idx_b]) ** 2, axis=1))

    # --- Check 1: decomposition at the ORIGINAL bandwidths ---
    mmd_s_orig, kxx_s, kyy_s, kxy_s = mmd_from_d2(d2_xx_s, d2_yy_s, d2_xy_s, sigma2_orig_single)
    mmd_l_orig, kxx_l, kyy_l, kxy_l = mmd_from_d2(d2_xx_l, d2_yy_l, d2_xy_l, sigma2_orig_lodo3)
    print(f"  Original bandwidth: sigma2_single={sigma2_orig_single:.3f}  sigma2_lodo3={sigma2_orig_lodo3:.3f}")
    print(f"  SINGLE decomposition: k(X,X)={kxx_s:.4f}  k(Y,Y)={kyy_s:.4f}  k(X,Y)={kxy_s:.6f}  "
          f"-> MMD^2={mmd_s_orig:.4f}")
    print(f"  LODO3  decomposition: k(X,X)={kxx_l:.4f}  k(Y,Y)={kyy_l:.4f}  k(X,Y)={kxy_l:.6f}  "
          f"-> MMD^2={mmd_l_orig:.4f}")
    cross_term_collapsed_single = kxy_s < 0.01 * max(kxx_s, kyy_s, 1e-12)
    print(f"  Cross term k(X,Y) collapsed relative to within-cluster terms (Single)? "
          f"{cross_term_collapsed_single}  (k(X,Y)={kxy_s:.6f} vs max(k(X,X),k(Y,Y))={max(kxx_s, kyy_s):.4f})")

    # --- Check 2: bandwidth sweep, SAME shared sigma2 applied to both conditions ---
    all_d2 = np.concatenate([d2_xx_s.ravel(), d2_yy_s.ravel(), d2_xy_s.ravel(),
                              d2_xx_l.ravel(), d2_yy_l.ravel(), d2_xy_l.ravel()])
    all_d2 = all_d2[all_d2 > 0]
    sweep_lo = min(sigma2_orig_single, sigma2_orig_lodo3) * 0.5
    sweep_hi = float(np.max(all_d2)) * 2.0
    sweep_values = np.geomspace(sweep_lo, sweep_hi, N_SWEEP)

    narrowed_orig_direction = mmd_l_orig < mmd_s_orig  # direction at the ORIGINAL (unshared) bandwidths
    sweep_rows = []
    crossover_sigma2 = None
    for sigma2 in sweep_values:
        mmd_s, _, _, _ = mmd_from_d2(d2_xx_s, d2_yy_s, d2_xy_s, sigma2)
        mmd_l, _, _, _ = mmd_from_d2(d2_xx_l, d2_yy_l, d2_xy_l, sigma2)
        pct = 100 * (mmd_s - mmd_l) / mmd_s if mmd_s != 0 else float("nan")
        narrowed = mmd_l < mmd_s
        sweep_rows.append({"sigma2": float(sigma2), "mmd_single": mmd_s, "mmd_lodo3": mmd_l,
                            "pct_change": pct, "narrowed": bool(narrowed)})
        if crossover_sigma2 is None and narrowed != narrowed_orig_direction:
            crossover_sigma2 = float(sigma2)

    print(f"\n  Bandwidth sweep ({N_SWEEP} points, sigma2 from {sweep_lo:.2f} to {sweep_hi:.1f}):")
    for r in sweep_rows:
        flag = "  <-- direction differs from original" if r["narrowed"] != narrowed_orig_direction else ""
        print(f"    sigma2={r['sigma2']:>12.2f}  MMD^2: single={r['mmd_single']:.4f}  "
              f"lodo3={r['mmd_lodo3']:.4f}  ({r['pct_change']:+7.1f}%)  narrowed={r['narrowed']}{flag}")

    if crossover_sigma2 is not None:
        print(f"  >>> CROSSOVER found at sigma2~={crossover_sigma2:.2f}: direction flips once bandwidth "
              f"is large enough. This directly confirms the kernel-saturation explanation.")
    else:
        print(f"  >>> NO crossover across the full sweep range (up to {sweep_hi:.1f}, well beyond the "
              f"observed distance scale) -- the direction is stable regardless of bandwidth. The "
              f"kernel-saturation explanation is NOT supported for this seed; MMD^2's original reading "
              f"appears to reflect a genuine property of the representations, not a bandwidth artifact.")

    return {
        "seed": seed,
        "sigma2_orig_single": sigma2_orig_single, "sigma2_orig_lodo3": sigma2_orig_lodo3,
        "decomposition_single": {"kxx": kxx_s, "kyy": kyy_s, "kxy": kxy_s, "mmd2": mmd_s_orig},
        "decomposition_lodo3": {"kxx": kxx_l, "kyy": kyy_l, "kxy": kxy_l, "mmd2": mmd_l_orig},
        "cross_term_collapsed_single": bool(cross_term_collapsed_single),
        "narrowed_orig_direction": bool(narrowed_orig_direction),
        "sweep": sweep_rows,
        "crossover_sigma2": crossover_sigma2,
    }


def main():
    X_iec, y_iec = ph.load_iec104_eval()
    iec104_benign = X_iec[y_iec == 0]
    cic17_benign = ph.load_cic17_benign_training_block()

    results = []
    for seed in SEEDS_TO_RUN:
        results.append(run_seed(seed, cic17_benign, iec104_benign))

    n_collapsed = sum(r["cross_term_collapsed_single"] for r in results)
    n_crossover = sum(r["crossover_sigma2"] is not None for r in results)
    print(f"\n{'=' * 70}\nSUMMARY across {len(results)} seeds\n{'=' * 70}")
    print(f"  Seeds where the cross term k(X,Y) collapsed under the original bandwidth (Single): "
          f"{n_collapsed}/{len(results)}")
    print(f"  Seeds where a crossover bandwidth exists (direction flips somewhere in the sweep): "
          f"{n_crossover}/{len(results)}")
    if n_collapsed >= len(results) // 2 and n_crossover >= len(results) // 2:
        print("  -> Kernel saturation is CONFIRMED as the mechanism for most seeds: the cross term "
              "collapses under the original bandwidth, and the direction reliably flips once bandwidth "
              "is large enough. MMD^2's original 'gap widened' reading is a measurement artifact.")
    elif n_collapsed >= len(results) // 2:
        print("  -> The cross term collapses (consistent with saturation), but the direction does not "
              "reliably flip even at much larger bandwidths -- saturation alone does not fully explain "
              "the discrepancy; treat the mechanism as partially, not fully, confirmed.")
    else:
        print("  -> Kernel saturation is NOT confirmed as the primary mechanism. The MMD^2 vs Wasserstein "
              "disagreement likely has a different or additional cause.")

    if NOTES:
        for n in NOTES:
            print(f"  - {n}")

    out_path = os.path.join(PILOT_DIR, "flow_iec104_mmd_bandwidth_sweep.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
