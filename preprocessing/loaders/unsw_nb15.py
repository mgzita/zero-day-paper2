"""
=============================================================================
loaders/unsw_nb15.py — load UNSW-NB15 source data

Uses the 4 raw chronological partition files (UNSW-NB15_1.csv..._4.csv,
headerless), NOT the training-set/testing-set files (which have proper
headers but are a shuffled, non-chronological ML-benchmark resampling —
wrong for building temporal window sequences; see project design notes).

Handles:
  1. No header row in the raw partition files — manual column assignment
     (49 columns, per NUSW-NB15_features.csv documentation)
  2. Non-UTF-8 encoding (confirmed via direct inspection earlier)
  3. Distinguishing raw partition files from training-set/testing-set
     files that may sit in the same folder
  4. Filtering to benign rows only (Label == 0)
=============================================================================
"""

import os
import re
import glob
import pandas as pd

UNSW_NB15_COLUMN_NAMES = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes",
    "dbytes", "sttl", "dttl", "sloss", "dloss", "service", "Sload", "Dload",
    "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz",
    "trans_depth", "res_bdy_len", "Sjit", "Djit", "Stime", "Ltime",
    "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports",
    "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd",
    "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat",
    "Label",
]


def _is_raw_partition_file(path: str) -> bool:
    """True for UNSW-NB15_1.csv .. _4.csv (headerless, chronological),
    False for UNSW_NB15_training-set.csv / _testing-set.csv (has headers,
    but shuffled/non-chronological — wrong for temporal windowing) or
    NUSW-NB15_features.csv / NUSW-NB15_GT.csv (metadata, not flow data)."""
    basename = os.path.basename(path).lower()
    return (
        basename.startswith("unsw-nb15_")
        and basename.replace("unsw-nb15_", "").replace(".csv", "").isdigit()
    )


def _read_with_encoding_fallback(path: str, **kwargs):
    last_error = None
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs), encoding
        except (UnicodeDecodeError, UnicodeError) as e:
            last_error = e
            continue
    raise last_error


def load_unsw_nb15_file(path: str) -> pd.DataFrame:
    """Load a single UNSW-NB15 raw partition file, filtering to benign
    rows only."""
    df, encoding_used = _read_with_encoding_fallback(
        path, header=None, names=UNSW_NB15_COLUMN_NAMES, low_memory=False
    )

    if len(df.columns) != len(UNSW_NB15_COLUMN_NAMES):
        raise ValueError(
            f"Column count mismatch for {path}: got {len(df.columns)}, "
            f"expected {len(UNSW_NB15_COLUMN_NAMES)}. The raw file's "
            f"actual column count doesn't match NUSW-NB15_features.csv's "
            f"documented 49-column order — check the file isn't corrupted "
            f"or from a different distribution."
        )

    # Filter to benign rows only (SOURCE dataset — no attack traffic
    # allowed at any stage, per config.py's SOURCE_DATASETS rule).
    # Label: 0 = benign, 1 = attack (attack_cat is blank/NaN for benign).
    n_before = len(df)
    df = df[df["Label"] == 0].copy()
    n_after = len(df)
    n_removed = n_before - n_after

    print(f"  {os.path.basename(path)} (encoding: {encoding_used}): "
          f"{n_before} rows -> {n_after} rows benign "
          f"({n_removed} attack rows removed, "
          f"{100 * n_removed / n_before:.1f}%)")

    return df


def load_unsw_nb15(raw_data_dir: str) -> pd.DataFrame:
    """Load and pool the 4 UNSW-NB15 raw partition files (benign-filtered),
    explicitly skipping training-set/testing-set files and metadata files
    that may be present in the same folder tree, and deduplicating by
    filename in case the same partition file exists in multiple locations
    (e.g. both raw_data_dir/ directly and a 'Training and Testing Sets'
    subfolder — confirmed to happen in practice)."""
    all_csvs = glob.glob(os.path.join(raw_data_dir, "**", "*.csv"), recursive=True)
    candidate_files = [f for f in all_csvs if _is_raw_partition_file(f)]

    skipped = [f for f in all_csvs if not _is_raw_partition_file(f)]
    if skipped:
        print(f"Skipping {len(skipped)} non-partition file(s) (metadata or "
              f"training-set/testing-set — wrong distribution for temporal "
              f"windowing):")
        for f in skipped:
            print(f"    - {os.path.relpath(f, raw_data_dir)}")

    # Deduplicate by filename — keep only the first occurrence of each
    # unique basename (e.g. UNSW-NB15_1.csv), even if it exists at
    # multiple paths. Without this, a file present in two locations gets
    # loaded and counted twice.
    seen_basenames = {}
    duplicates_found = []
    for f in sorted(candidate_files):
        basename = os.path.basename(f)
        if basename in seen_basenames:
            duplicates_found.append((basename, seen_basenames[basename], f))
        else:
            seen_basenames[basename] = f
    target_files = list(seen_basenames.values())

    if duplicates_found:
        print(f"\nWARNING: {len(duplicates_found)} duplicate filename(s) found "
              f"at multiple paths — using only the FIRST occurrence of each, "
              f"skipping the rest:")
        for basename, kept_path, skipped_path in duplicates_found:
            print(f"    {basename}: keeping {os.path.relpath(kept_path, raw_data_dir)}, "
                  f"skipping {os.path.relpath(skipped_path, raw_data_dir)}")

    if not target_files:
        raise FileNotFoundError(
            f"No UNSW-NB15_1.csv..._4.csv raw partition files found under "
            f"{raw_data_dir}. Check that these 4 files (not just "
            f"training-set/testing-set) were downloaded and placed correctly."
        )
    if len(target_files) != 4:
        print(f"WARNING: expected 4 unique raw partition files, found "
              f"{len(target_files)}. Continuing anyway, but check whether "
              f"all 4 (_1 through _4) are actually present.")

    print(f"\nLoading {len(target_files)} UNSW-NB15 raw partition file(s):")
    frames = [load_unsw_nb15_file(f) for f in sorted(target_files)]
    combined = pd.concat(frames, ignore_index=True)

    # Fix mixed-type columns before saving. UNSW-NB15's raw CSVs are known
    # to have inconsistent representations across several columns — hex
    # port values (e.g. '0x0303') in sport/dsport, and stray
    # whitespace-padded or string-typed values in other nominally-numeric
    # columns (e.g. ct_ftp_cmd, is_ftp_login, ct_flw_http_mthd — confirmed
    # via actual pyarrow errors during testing, one column at a time,
    # before switching to this proactive approach). Rather than fix these
    # reactively as pyarrow rejects each one individually, coerce every
    # column NOT in the known categorical/string list to numeric in one
    # pass, using pd.to_numeric (which handles plain numeric strings and
    # NaN-fills anything genuinely unparseable) with special handling for
    # hex-formatted values.
    CATEGORICAL_COLUMNS = {"srcip", "dstip", "proto", "state", "service", "attack_cat"}
    numeric_candidate_cols = [c for c in combined.columns if c not in CATEGORICAL_COLUMNS]

    for col in numeric_candidate_cols:
        if pd.api.types.is_numeric_dtype(combined[col]):
            continue  # already clean, skip

        as_str = combined[col].astype(str)
        n_hex = as_str.str.strip().str.lower().str.startswith("0x").sum()
        if n_hex > 0:
            print(f"Found {n_hex} hex-formatted values in '{col}' — converting to integer.")
            combined[col] = as_str.apply(
                lambda x: int(x.strip(), 16) if x.strip().lower().startswith("0x")
                else pd.to_numeric(x.strip(), errors="coerce")
            )
        else:
            n_before_numeric = combined[col].notna().sum()
            combined[col] = pd.to_numeric(as_str.str.strip(), errors="coerce")
            n_after_numeric = combined[col].notna().sum()
            n_lost = n_before_numeric - n_after_numeric
            if n_lost > 0:
                print(f"WARNING: '{col}' had {n_lost} value(s) that could not be "
                      f"converted to numeric (set to NaN) — check this column's "
                      f"raw values if this count seems high relative to "
                      f"{len(combined)} total rows.")

        combined[col] = pd.to_numeric(combined[col], errors="coerce")
        # Use Int64 (nullable int) only if no fractional values are present,
        # otherwise leave as float64 (e.g. dur, Sjit, Djit are legitimately
        # non-integer).
        if combined[col].dropna().apply(lambda x: float(x).is_integer()).all():
            combined[col] = combined[col].astype("Int64")

    print(f"\nTotal pooled benign rows: {len(combined)}")
    return combined


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from config import CLEANED_DATA_DIR

    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "../raw_data/unsw_nb15"
    df = load_unsw_nb15(raw_dir)
    print(f"\nFinal shape: {df.shape}")
    print(f"Columns: {list(df.columns)[:10]} ... ({len(df.columns)} total)")

    os.makedirs(CLEANED_DATA_DIR, exist_ok=True)
    output_path = os.path.join(CLEANED_DATA_DIR, "unsw_nb15_benign.parquet")
    df.to_parquet(output_path, index=False)
    print(f"\nSaved cleaned data to: {os.path.abspath(output_path)}")
