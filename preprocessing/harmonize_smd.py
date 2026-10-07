"""
Harmonize the Server Machine Dataset (SMD) into a 3-environment
pipeline compatible with the existing Single/Multi/LODO diversity-
training design used throughout this project.
=========================================================================
SMD's 38 dimensions are anonymized machine metrics with NO relationship
to the 18-feature IDS space -- there is no column mapping, no shared
IQR scaler, and no inherited log1p transform list. This script fits an
ENTIRELY NEW scaler specific to SMD's own data, following the same
methodological philosophy (IQR scaling, [-10, 10] clipping, exact-row-
match deduplication) as the IDS pipeline, but with its own numbers.

SMD's own values are already close to [0, 1] (min-max normalized by
the original authors), so no log1p pre-transform is applied.

Environment design, mapping onto Single/Multi/LODO:
  - Group 1 machines  ~ "environment A"
  - Group 2 machines  ~ "environment B"
  - Group 3 machines  ~ "environment C"
Each group's training data is pooled and split into train/val. Each
group's test data becomes a separate held-out evaluation target file,
so any group can serve as the "unseen target" under a LODO-style
design.

Run from project root:
    python preprocessing\\harmonize_smd.py
"""

import os
import numpy as np
import pandas as pd

SMD_ROOT = r"C:\MLProject\zero_day_project\data\raw\SMD\ServerMachineDataset"
TRAIN_DIR = os.path.join(SMD_ROOT, "train")
TEST_DIR = os.path.join(SMD_ROOT, "test")
TEST_LABEL_DIR = os.path.join(SMD_ROOT, "test_label")

OUTPUT_DIR = r"C:\MLProject\zero_day_project\artefacts_v2\dedup_smd"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_FEATURES = 38
VAL_FRACTION = 0.15


def list_machines_by_group():
    all_files = sorted(f for f in os.listdir(TRAIN_DIR) if f.endswith(".txt"))
    groups = {1: [], 2: [], 3: []}
    for fname in all_files:
        parts = fname.replace(".txt", "").split("-")
        group = int(parts[1])
        groups[group].append(fname)
    return groups


def load_machine_train(fname):
    path = os.path.join(TRAIN_DIR, fname)
    return pd.read_csv(path, header=None).values.astype(np.float64)


def load_machine_test(fname):
    test_path = os.path.join(TEST_DIR, fname)
    label_path = os.path.join(TEST_LABEL_DIR, fname)
    X = pd.read_csv(test_path, header=None).values.astype(np.float64)
    y = pd.read_csv(label_path, header=None).values.astype(int).flatten()
    return X, y


def dedupe(X, y=None):
    n_before = len(X)
    df = pd.DataFrame(X)
    if y is not None:
        df["__label__"] = y
        df = df.drop_duplicates(keep="first").reset_index(drop=True)
        y_out = df["__label__"].values.astype(np.int64)
        X_out = df.drop(columns=["__label__"]).values.astype(np.float32)
    else:
        df = df.drop_duplicates(keep="first").reset_index(drop=True)
        X_out = df.values.astype(np.float32)
        y_out = None
    n_after = len(X_out)
    print(f"    Dedup: {n_before} -> {n_after} rows ({n_before - n_after} removed, "
          f"{(n_before - n_after) / n_before * 100:.2f}%)")
    return X_out, y_out


def main():
    print("Scanning SMD machines by group...")
    groups = list_machines_by_group()
    for g, machines in groups.items():
        print(f"  Group {g}: {len(machines)} machines")

    group_train_raw = {}
    for g, machines in groups.items():
        print(f"\nLoading Group {g} training data ({len(machines)} machines)...")
        blocks = [load_machine_train(m) for m in machines]
        pooled = np.vstack(blocks)
        print(f"  Pooled shape: {pooled.shape}")
        group_train_raw[g] = pooled

    print(f"\n{'=' * 70}\nFitting SMD-specific IQR scaler (all 3 groups combined)\n{'=' * 70}")
    all_train_raw = np.vstack([group_train_raw[g] for g in (1, 2, 3)])
    q1 = np.percentile(all_train_raw, 25, axis=0)
    q2 = np.percentile(all_train_raw, 50, axis=0)
    q3 = np.percentile(all_train_raw, 75, axis=0)
    iqr = q3 - q1
    iqr_safe = np.where(iqr == 0, 1.0, iqr)
    print(f"  Fit on {len(all_train_raw)} total rows, {N_FEATURES} features.")
    n_zero_iqr = int((iqr == 0).sum())
    if n_zero_iqr > 0:
        print(f"  WARNING: {n_zero_iqr} feature(s) have zero IQR (constant in training data) -- "
              f"these will be scaled using a fallback divisor of 1.0.")

    np.savez(os.path.join(OUTPUT_DIR, "smd_iqr_scaler.npz"), q1=q1, q2=q2, q3=q3, iqr=iqr)
    print(f"  Saved scaler -> {os.path.join(OUTPUT_DIR, 'smd_iqr_scaler.npz')}")

    def scale_and_clip(X_raw):
        X_scaled = (X_raw - q2) / iqr_safe
        return np.clip(X_scaled, -10, 10).astype(np.float32)

    print(f"\n{'=' * 70}\nScaling, train/val splitting, deduplicating each group\n{'=' * 70}")
    train_blocks, val_blocks = [], []
    for g in (1, 2, 3):
        X_scaled = scale_and_clip(group_train_raw[g])
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(X_scaled))
        n_val = int(len(X_scaled) * VAL_FRACTION)
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        X_train_g, X_val_g = X_scaled[train_idx], X_scaled[val_idx]

        print(f"  Group {g}: {len(X_train_g)} train rows, {len(X_val_g)} val rows before dedup")
        X_train_g, _ = dedupe(X_train_g)
        X_val_g, _ = dedupe(X_val_g)
        train_blocks.append(X_train_g)
        val_blocks.append(X_val_g)

    min_train = min(len(b) for b in train_blocks)
    min_val = min(len(b) for b in val_blocks)
    print(f"\n  Truncating all groups to equal size for stacking: "
          f"train={min_train}/group, val={min_val}/group")
    X_train_final = np.vstack([b[:min_train] for b in train_blocks])
    X_val_final = np.vstack([b[:min_val] for b in val_blocks])

    np.savez(os.path.join(OUTPUT_DIR, "train_smd_dedup.npz"), X_train=X_train_final)
    np.savez(os.path.join(OUTPUT_DIR, "val_smd_dedup.npz"), X_val=X_val_final)
    print(f"  Saved train_smd_dedup.npz: {X_train_final.shape}")
    print(f"  Saved val_smd_dedup.npz: {X_val_final.shape}")

    print(f"\n{'=' * 70}\nProcessing each group's TEST data as a held-out evaluation target\n{'=' * 70}")
    for g, machines in groups.items():
        print(f"\nGroup {g} test data ({len(machines)} machines)...")
        X_blocks, y_blocks = [], []
        for m in machines:
            X_m, y_m = load_machine_test(m)
            X_blocks.append(X_m)
            y_blocks.append(y_m)
        X_test_raw = np.vstack(X_blocks)
        y_test = np.concatenate(y_blocks)
        print(f"  Pooled: {len(X_test_raw)} rows ({(y_test == 0).sum()} benign, "
              f"{(y_test == 1).sum()} anomalous)")

        X_test_scaled = scale_and_clip(X_test_raw)
        X_test_final, y_test_final = dedupe(X_test_scaled, y_test)

        out_path = os.path.join(OUTPUT_DIR, f"test_smd_group{g}_dedup.npz")
        np.savez(out_path, X=X_test_final, y=y_test_final)
        print(f"  Saved {out_path}: X={X_test_final.shape}, y={y_test_final.shape}")

    print(f"\n{'=' * 70}\nDONE. All files saved to {OUTPUT_DIR}\n{'=' * 70}")
    print("Files:")
    print("  smd_iqr_scaler.npz         (q1/q2/q3/iqr, fit on all 3 groups combined)")
    print("  train_smd_dedup.npz        (X_train, 3 groups stacked in fixed order)")
    print("  val_smd_dedup.npz          (X_val, 3 groups stacked in fixed order)")
    print("  test_smd_group1_dedup.npz  (X, y -- Group 1 as held-out target)")
    print("  test_smd_group2_dedup.npz  (X, y -- Group 2 as held-out target)")
    print("  test_smd_group3_dedup.npz  (X, y -- Group 3 as held-out target)")


if __name__ == "__main__":
    main()
