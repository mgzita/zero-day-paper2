"""
=============================================================================
check_ait_netflow.py — one-off inspection, NOT part of the official pipeline

Run this after extracting russellmitchell_netflows.zip and placing the CSV
into ../raw_data/ait_netflow/. This is deliberately separate from
inspect_raw_columns.py / config.py's SOURCE_DATASETS so we can look at the
columns BEFORE deciding whether to actually swap this in for
cse_cic_ids2018 (or add it as an 8th dataset).
=============================================================================
"""

import os
import glob
import pandas as pd

AIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw_data", "ait_netflow")


def find_csvs():
    return glob.glob(os.path.join(AIT_DIR, "**", "*.csv"), recursive=True)


if __name__ == "__main__":
    print(f"Looking in: {os.path.abspath(AIT_DIR)}\n")
    csvs = find_csvs()

    if not csvs:
        print("No CSV files found. Did you extract russellmitchell_netflows.zip")
        print("and place the extracted CSV(s) into this folder?")
    else:
        print(f"Found {len(csvs)} CSV file(s):")
        for f in csvs:
            size_mb = os.path.getsize(f) / (1024 * 1024)
            print(f"  - {os.path.relpath(f, AIT_DIR)}  ({size_mb:.1f} MB)")

        sample = csvs[0]
        print(f"\nInspecting: {sample}\n")
        try:
            df = pd.read_csv(sample, nrows=5, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(sample, nrows=5, low_memory=False, encoding="latin-1")

        print(f"Shape (first 5 rows): {df.shape}")
        print(f"Columns ({len(df.columns)}):")
        for col in df.columns:
            print(f"  - {col!r}  (dtype: {df[col].dtype})")

        print("\nFirst row:")
        print(df.iloc[0].to_string())
