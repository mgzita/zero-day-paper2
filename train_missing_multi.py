"""
Train the missing size-matched Multi checkpoints for Paper 3
=========================================================================
The evasion grid found Multi checkpoints for only 5 seeds (ae, dae, svdd)
and 1 seed (flow), while Single and LODO-3 have all 10. This script trains
the missing ones so the Single-vs-Multi comparison (RQ2) is balanced.

To guarantee the new models are trained EXACTLY like the existing ones, it
does not copy any training code. It imports Paper 2's own functions:

  ae / dae / svdd : train_autoencoder, train_svdd
                    from ae_dae_svdd_pipeline_iec104.py
  flow            : train_flow
                    from flow_full_pipeline.py
  Multi data      : load_multi_matched_train_val(seed)
                    from flow_full_pipeline.py (CIC17 + CIC18 + UNSW,
                    each cut to one third, so the total equals Single's size)

Existing checkpoints are never overwritten. Each finished model is saved
immediately, so if the run stops, starting it again carries on.

Run from C:\\MLProject\\zero_day_paper2_mtsst (where the Paper 2 scripts are):
    python train_missing_multi.py --dry-run     # list what would be trained
    python train_missing_multi.py               # train everything missing
    python train_missing_multi.py --families flow
"""

import argparse
import os
import sys
import time
import types

# Import order matters on Windows/Anaconda: loading torch before numpy can load
# two copies of the OpenMP runtime (OMP Error #15). Import numpy first, exactly
# as the Paper 2 scripts do.
import numpy  # noqa: F401

# The Paper 2 modules import shap at the top, but training never uses it.
# A placeholder avoids loading shap's compiled libraries (another OpenMP source).
if "shap" not in sys.modules:
    sys.modules["shap"] = types.ModuleType("shap")

import ae_dae_svdd_pipeline_iec104 as AE   # AE/DAE/SVDD training recipe (Paper 2)
import flow_full_pipeline as FL            # flow training recipe + Multi data loader
import torch

FAMILIES = ["ae", "dae", "svdd", "flow"]
SEEDS = [42, 123, 456, 789, 999, 111, 222, 333, 444, 555]


def multi_path(family, seed):
    return os.path.join(AE.DEDUP_DIR, f"seed{seed}_{family}", f"{family}_multi.pt")


def train_one(family, seed):
    X_train, X_val = FL.load_multi_matched_train_val(seed)
    print(f"  data: {len(X_train)} training flows, {len(X_val)} validation flows")
    if family == "ae":
        return AE.train_autoencoder(X_train, X_val, seed, AE.N_FEATURES, noise_std=0.0)
    if family == "dae":
        return AE.train_autoencoder(X_train, X_val, seed, AE.N_FEATURES, noise_std=AE.NOISE_STD_DAE)
    if family == "svdd":
        return AE.train_svdd(X_train, X_val, seed, AE.N_FEATURES)
    if family == "flow":
        return FL.train_flow(X_train, X_val, seed, input_dim=FL.N_FEATURES)
    raise ValueError(family)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=FAMILIES, choices=FAMILIES)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    todo = [(f, s) for f in args.families for s in args.seeds if not os.path.exists(multi_path(f, s))]
    print(f"{len(todo)} Multi checkpoints missing:")
    for f, s in todo:
        print(f"  {multi_path(f, s)}")
    if args.dry_run or not todo:
        return

    times = []
    for i, (family, seed) in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] Training {family} Multi, seed {seed}")
        t0 = time.time()
        model = train_one(family, seed)
        path = multi_path(family, seed)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        torch.save(model.state_dict(), tmp)
        os.replace(tmp, path)            # never leaves a half-written checkpoint
        times.append(time.time() - t0)
        left = (len(todo) - i) * sum(times) / len(times) / 60
        print(f"  saved {path} ({times[-1] / 60:.1f} min; about {left:.0f} min left)")

    print("\nAll missing Multi checkpoints trained. Now rerun the grid:")
    print("    python run_evasion_grid.py")


if __name__ == "__main__":
    main()
