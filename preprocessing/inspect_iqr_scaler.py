"""
Step 1: Inspect iqr_scaler.npz's format
=========================================================================
Before harmonizing the IEC 60870-5-104 dataset into the existing pipeline,
we need to know exactly how the saved IQR scaler is structured (which
keys, what shape, what order) so the new dataset is scaled identically
to CIC-IDS2017/CSE-CIC-IDS2018/UNSW-NB15 rather than guessed at.

This script only reads and prints iqr_scaler.npz and selected_features.csv
-- no data is written or modified.

Run from project root:
    python paper2\\pilot\\inspect_iqr_scaler.py
"""

import os
import numpy as np

DEDUP_DIR = r"C:\MLProject\zero_day_project\artefacts_v2\dedup"  # TODO confirm
ARTEFACTS_V2_DIR = r"C:\MLProject\zero_day_project\artefacts_v2"  # scaler lives here, one level up from dedup


def main():
    scaler_path = os.path.join(ARTEFACTS_V2_DIR, "iqr_scaler.npz")
    features_path = os.path.join(DEDUP_DIR, "selected_features.csv")

    print(f"{'=' * 70}\nInspecting {scaler_path}\n{'=' * 70}")
    if not os.path.exists(scaler_path):
        print(f"NOT FOUND at this path. Please locate iqr_scaler.npz and update DEDUP_DIR.")
    else:
        d = np.load(scaler_path, allow_pickle=True)
        print(f"Keys found: {list(d.keys())}")
        for key in d.keys():
            arr = d[key]
            print(f"\n  '{key}': shape={arr.shape}, dtype={arr.dtype}")
            if arr.size <= 30:
                print(f"    values: {arr}")
            else:
                print(f"    first 5 values: {arr.flatten()[:5]}")
                print(f"    last 5 values:  {arr.flatten()[-5:]}")

    print(f"\n{'=' * 70}\nInspecting {features_path}\n{'=' * 70}")
    if not os.path.exists(features_path):
        print(f"NOT FOUND at this path. Trying alternate locations...")
        for alt_dir in [ARTEFACTS_V2_DIR,
                        r"C:\MLProject\zero_day_project\artefacts",
                        r"C:\MLProject\zero_day_project"]:
            alt_path = os.path.join(alt_dir, "selected_features.csv")
            if os.path.exists(alt_path):
                print(f"  Found instead at: {alt_path}")
                features_path = alt_path
                break
        else:
            print(f"  Not found in any checked location.")
            return
    with open(features_path) as f:
        content = f.read()
    print(content[:2000])
    if len(content) > 2000:
        print(f"... ({len(content)} total characters, truncated)")


if __name__ == "__main__":
    main()
