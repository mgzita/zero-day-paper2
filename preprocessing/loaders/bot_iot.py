"""
=============================================================================
loaders/bot_iot.py — load BoT-IoT zero-day evaluation data

IMPORTANT DIFFERENCE from every other loader so far: BoT-IoT is a
ZERO-DAY dataset, not a source (pretraining) dataset. Per config.py:
  - ZERO_DAY_HAS_BENIGN["bot_iot"] = False (attack-only, per project design)
  - It is withheld ENTIRELY from pretraining/validation/calibration
  - No benign filtering happens here — we want ALL rows (attack traffic
    included), since this dataset is used purely for zero-day detection
    rate evaluation, not training

This is architecturally different from the cic_ids2017/unsw_nb15/ton_iot/
ugr16/ait_netflow loaders, which all filter DOWN to benign-only rows.
Do not add a benign filter here — that would defeat the purpose of a
zero-day test set.

Sanity check included: verify whether any benign rows exist in this data
at all (attack==0), since config.py's ZERO_DAY_HAS_BENIGN assumption
should be empirically confirmed, not just assumed from prior documentation.
=============================================================================
"""

import os
import glob
import pandas as pd


def load_bot_iot_file(path: str) -> pd.DataFrame:
    """Load a single BoT-IoT partition file. NO FILTERING — this is a
    zero-day dataset, we want everything (attack traffic included)."""
    df = pd.read_csv(path, low_memory=False)

    required_cols = ["saddr", "daddr", "sport", "dport", "stime", "attack", "category"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Expected columns missing from {path}: {missing}. Check this "
            f"is really a BoT-IoT 'Full5pc' partition file with the "
            f"standard 46-column Argus-derived schema."
        )

    n_benign = (df["attack"] == 0).sum()
    n_attack = (df["attack"] == 1).sum()
    print(f"  {os.path.basename(path)}: {len(df)} rows total "
          f"({n_benign} benign, {n_attack} attack) — NO filtering applied, "
          f"loading all rows since this is a zero-day evaluation set.")

    return df


def load_bot_iot(raw_data_dir: str) -> pd.DataFrame:
    """Load and pool all BoT-IoT partition files. NO benign filtering —
    see module docstring for why this differs from every other loader."""
    target_files = glob.glob(os.path.join(raw_data_dir, "**", "*.csv"), recursive=True)

    if not target_files:
        raise FileNotFoundError(
            f"No CSV files found under {raw_data_dir}. Check that the 4 "
            f"partition files (UNSW_2018_IoT_Botnet_Full5pc_1.csv..._4.csv) "
            f"were downloaded and placed correctly."
        )

    print(f"Loading {len(target_files)} BoT-IoT partition file(s) "
          f"(zero-day set — no benign filtering):")
    frames = [load_bot_iot_file(f) for f in sorted(target_files)]
    combined = pd.concat(frames, ignore_index=True)

    n_benign_total = (combined["attack"] == 0).sum()
    n_attack_total = (combined["attack"] == 1).sum()
    pct_benign = 100 * n_benign_total / len(combined) if len(combined) > 0 else 0

    print(f"\nTotal pooled rows: {len(combined)} "
          f"({n_benign_total} benign [{pct_benign:.2f}%], {n_attack_total} attack)")

    # Empirical check against config.py's ZERO_DAY_HAS_BENIGN["bot_iot"] = False
    # assumption — confirm it matches what's actually in the data, rather
    # than silently trusting prior documentation.
    if n_benign_total == 0:
        print("CONFIRMED: attack-only, matches config.py's "
              "ZERO_DAY_HAS_BENIGN['bot_iot'] = False assumption exactly.")
    elif pct_benign < 1:
        print(f"NOTE: {n_benign_total} benign rows found ({pct_benign:.2f}%) — "
              f"technically not zero, but negligible. config.py's "
              f"ZERO_DAY_HAS_BENIGN=False treatment (report detection rate "
              f"only, no AUC/F1/etc.) is still the right call given how "
              f"small this fraction is.")
    else:
        print(f"WARNING: {pct_benign:.2f}% of rows are benign — this "
              f"contradicts config.py's ZERO_DAY_HAS_BENIGN['bot_iot'] = "
              f"False assumption more than expected. Worth revisiting "
              f"whether the full metric suite (not just detection rate) "
              f"should be computed for this dataset after all.")

    return combined


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from config import CLEANED_DATA_DIR

    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "../raw_data/bot_iot"
    df = load_bot_iot(raw_dir)
    print(f"\nFinal shape: {df.shape}")
    print(f"Columns: {list(df.columns)[:10]} ... ({len(df.columns)} total)")

    os.makedirs(CLEANED_DATA_DIR, exist_ok=True)
    # Named differently from the source-dataset outputs (no '_benign'
    # suffix) since this file intentionally contains attack traffic too.
    output_path = os.path.join(CLEANED_DATA_DIR, "bot_iot_full.parquet")
    df.to_parquet(output_path, index=False)
    print(f"\nSaved cleaned data to: {os.path.abspath(output_path)}")
