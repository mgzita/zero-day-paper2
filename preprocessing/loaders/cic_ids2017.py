"""
=============================================================================
loaders/cic_ids2017.py — load CIC-IDS2017 source data (corrected/
regenerated distribution, Engelen et al. 2021 — retains Flow ID/Src IP/
Dst IP/Timestamp, unlike the original MachineLearningCSV.zip release).

Simpler than the AIT NetFlow loader:
  - Proper header row already present, no column-name cleanup needed
  - Single 'Label' column distinguishes benign ('BENIGN') from attack
  - No comment-marker artifacts, no protocol-specific schema differences
=============================================================================
"""

import os
import glob
import pandas as pd


def load_cic_ids2017_file(path: str) -> pd.DataFrame:
    """Load a single CIC-IDS2017 day-file, filtering to benign rows only."""
    df = pd.read_csv(path, low_memory=False)

    # Sanity check: confirm expected columns are present before proceeding.
    # This dataset previously had a version WITHOUT these columns (the
    # original MachineLearningCSV.zip) — fail loudly if we've somehow
    # picked up the wrong file/distribution rather than silently
    # continuing without Src IP/Dst IP.
    required_cols = ["Flow ID", "Src IP", "Dst IP", "Src Port", "Dst Port", "Timestamp", "Label"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Expected columns missing from {path}: {missing}. "
            f"This likely means the wrong CIC-IDS2017 distribution was used "
            f"(e.g. the original MachineLearningCSV.zip, which lacks IPs, "
            f"instead of the corrected Engelen et al. 2021 release)."
        )

    # Filter to benign rows only (SOURCE dataset — no attack traffic
    # allowed at any stage, per config.py's SOURCE_DATASETS rule).
    n_before = len(df)
    df = df[df["Label"].astype(str).str.upper() == "BENIGN"].copy()
    n_after = len(df)
    n_removed = n_before - n_after

    print(f"  {os.path.basename(path)}: {n_before} rows -> {n_after} rows "
          f"benign ({n_removed} attack rows removed, "
          f"{100 * n_removed / n_before:.1f}%)")

    return df


def load_cic_ids2017(raw_data_dir: str) -> pd.DataFrame:
    """Load and pool all CIC-IDS2017 day-files (benign-filtered)."""
    target_files = glob.glob(os.path.join(raw_data_dir, "**", "*.csv"), recursive=True)

    if not target_files:
        raise FileNotFoundError(
            f"No CSV files found under {raw_data_dir}. Check that the "
            f"5 day-files (Monday/Tuesday/Wednesday/Thursday/Friday-"
            f"WorkingHours.csv) were placed correctly."
        )

    print(f"Loading {len(target_files)} CIC-IDS2017 file(s):")
    frames = [load_cic_ids2017_file(f) for f in sorted(target_files)]
    combined = pd.concat(frames, ignore_index=True)

    print(f"\nTotal pooled benign rows: {len(combined)}")
    return combined


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from config import CLEANED_DATA_DIR

    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "../raw_data/cic_ids2017"
    df = load_cic_ids2017(raw_dir)
    print(f"\nFinal shape: {df.shape}")
    print(f"Columns: {list(df.columns)[:10]} ... ({len(df.columns)} total)")

    # SAVE THE OUTPUT — see ait_netflow.py's loader for why this uses
    # CLEANED_DATA_DIR from config.py rather than a recomputed path.
    os.makedirs(CLEANED_DATA_DIR, exist_ok=True)
    output_path = os.path.join(CLEANED_DATA_DIR, "cic_ids2017_benign.parquet")
    df.to_parquet(output_path, index=False)
    print(f"\nSaved cleaned data to: {os.path.abspath(output_path)}")
