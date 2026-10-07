"""
=============================================================================
loaders/ugr16.py — load UGR16 source data (CAL/calibration set only)

Simpler than most other loaders:
  - No attack filtering needed: the CAL set is background/benign traffic
    by construction (the official site's own description) — NOT the
    TEST set, which mixes background with synthetic attacks. If any
    non-CAL file is accidentally present, this loader does not filter
    attacks out of it — see the sanity check below instead.
  - No header row — manual column assignment (13 columns, fixed order
    per the official UGR16 format).
  - One known data-quality caveat, stated on the OFFICIAL site itself:
    "although this data set is clean from synthetically generated
    attacks... some other traffic anomalies appear in this period." This
    loader does NOT attempt to filter these out (no reliable signal to
    do so) — flagged here as a known, accepted limitation, not a bug.
=============================================================================
"""

import os
import glob
import pandas as pd

UGR16_COLUMN_NAMES = [
    "timestamp", "duration", "src_ip", "dst_ip", "src_port", "dst_port",
    "protocol", "flags", "forwarding_status", "tos", "packets", "bytes",
    "label",
]

# Explicit, memory-efficient dtypes — specifying these upfront avoids
# pandas' default (memory-hungry) type inference and object-dtype
# strings, and is REQUIRED (not just an optimization) for at least one
# real UGR16 file, which is large enough that pd.read_csv with default
# settings raised "ParserError: ... out of memory" trying to tokenize it
# in one pass.
UGR16_DTYPES = {
    "timestamp": "str",
    "duration": "float32",
    "src_ip": "str",
    "dst_ip": "str",
    "src_port": "int32",
    "dst_port": "int32",
    "protocol": "category",
    "flags": "category",
    "forwarding_status": "int8",
    "tos": "int8",
    "packets": "int64",
    "bytes": "int64",
    "label": "category",
}

CHUNK_SIZE = 500_000  # rows per chunk during reading


def _read_with_encoding_fallback(path: str, max_hours: float = None, **kwargs):
    """Read a (potentially large) CSV in chunks with explicit dtypes,
    trying encodings in order. Returns a single concatenated DataFrame
    and which encoding worked.

    If max_hours is set, filters to only the first `max_hours` of
    CONTIGUOUS time from the start of the file, discarding the rest
    during chunked reading (not after — keeps peak memory low). This is
    NOT random sampling: UGR16's full week is ~240M rows / 43GB, roughly
    100x larger than every other source dataset combined, and random
    row sampling would destroy the temporal continuity windowing depends
    on. A contiguous time slice preserves real, orderable sequences.
    """
    last_error = None
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            chunks = []
            reader = pd.read_csv(
                path, encoding=encoding, chunksize=CHUNK_SIZE,
                dtype=UGR16_DTYPES, **kwargs,
            )
            cutoff_time = None
            total_rows_seen = 0
            for chunk in reader:
                total_rows_seen += len(chunk)
                if max_hours is not None:
                    chunk_ts = pd.to_datetime(chunk["timestamp"], errors="coerce")
                    if cutoff_time is None:
                        # Establish cutoff from the very first timestamp seen
                        first_valid = chunk_ts.dropna()
                        if len(first_valid) > 0:
                            cutoff_time = first_valid.iloc[0] + pd.Timedelta(hours=max_hours)
                    if cutoff_time is not None:
                        chunk = chunk[chunk_ts <= cutoff_time]
                        if len(chunk) == 0 and total_rows_seen > CHUNK_SIZE:
                            # Past the cutoff and this chunk contributed
                            # nothing — stop reading further chunks entirely
                            # rather than scanning the remaining ~95% of a
                            # 43GB file for no reason.
                            break
                chunks.append(chunk)
            result = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=UGR16_COLUMN_NAMES)
            if max_hours is not None:
                print(f"    Scanned {total_rows_seen} total rows in file, "
                      f"kept {len(result)} within the first {max_hours} "
                      f"contiguous hour(s).")
            return result, encoding
        except (UnicodeDecodeError, UnicodeError) as e:
            last_error = e
            continue
    raise last_error


def load_ugr16_file(path: str, max_hours: float = None) -> pd.DataFrame:
    """Load a single UGR16 CAL-set week file. No attack filtering — the
    CAL set is background traffic by construction, not a mixed set
    requiring row-level label filtering (unlike every other source
    dataset in this project).

    max_hours: if set, keep only the first N contiguous hours of the
    file (see module-level note on why this matters for UGR16
    specifically — its full-week volume is ~100x larger than every
    other source dataset combined)."""
    # Check the raw field count BEFORE assigning names — same lesson as
    # the ToN-IoT loader. Skip any leading blank line(s) rather than
    # blindly reading line 1, since at least one real UGR16 file has been
    # confirmed to start with an empty line before the actual data.
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
        while first_line.strip() == "" and first_line != "":
            first_line = f.readline()
    actual_field_count = len(first_line.rstrip("\n").split(","))
    if actual_field_count != len(UGR16_COLUMN_NAMES):
        raise ValueError(
            f"Column count mismatch for {path}: first line has "
            f"{actual_field_count} comma-separated fields, expected "
            f"{len(UGR16_COLUMN_NAMES)}. Check this is really a CAL-set "
            f"netflow CSV in the standard 13-column format, not e.g. one "
            f"of the pre-extracted attack-type subset files "
            f"(blacklist_*, spam_*, sshscan_*, udpscan_*)."
        )

    df, encoding_used = _read_with_encoding_fallback(
        path, max_hours=max_hours, header=None, names=UGR16_COLUMN_NAMES
    )

    # Sanity check (not a filter): confirm the 'label' column looks like
    # what's expected for the CAL set. UGR16's CAL set uses 'background'
    # as its label value. If a substantial fraction of rows have a
    # DIFFERENT label, this may not actually be a CAL-set file (e.g. the
    # TEST set was used by mistake instead).
    if "label" in df.columns and len(df) > 0:
        label_counts = df["label"].value_counts()
        non_background = df[df["label"].astype(str).str.lower() != "background"]
        pct_non_background = 100 * len(non_background) / len(df) if len(df) > 0 else 0
        if pct_non_background > 5:
            print(f"  WARNING: {pct_non_background:.1f}% of rows in "
                  f"{os.path.basename(path)} have a label OTHER than "
                  f"'background' ({dict(label_counts)}). This may not be "
                  f"a CAL-set file — check you're not accidentally using "
                  f"a TEST-set file, which mixes background with "
                  f"synthetic attacks.")

    print(f"  {os.path.basename(path)} (encoding: {encoding_used}): "
          f"{len(df)} rows loaded (CAL set — no attack filtering applied, "
          f"benign by construction).")

    return df


def load_ugr16(raw_data_dir: str, max_hours: float = 2.0) -> pd.DataFrame:
    """Load and pool all UGR16 CAL-set week files found under raw_data_dir.

    max_hours (default 2.0): keep only the first N contiguous hours PER
    FILE, not random sampling. UGR16's full week is ~240M rows / 43GB —
    roughly 100x larger than every other source dataset combined — and
    windowing needs contiguous, orderable time ranges, which random
    sampling would destroy. 2 hours was chosen to land in the same
    order of magnitude as this project's other source datasets
    (CIC-IDS2017: 1.66M rows, UNSW-NB15: 2.2M rows), based on an
    observed rate of ~1.4M rows/hour in the real data. Pass None to keep
    the full file (not recommended given the size).
    """
    target_files = glob.glob(os.path.join(raw_data_dir, "**", "*.csv"), recursive=True)

    # Explicitly exclude the pre-extracted attack-type subset files
    # (blacklist_*, spam_*, sshscan_*, udpscan_*) mentioned on the
    # official site — these are NOT what we want for source pretraining.
    EXCLUDE_PATTERNS = ["blacklist_", "spam_", "sshscan_", "udpscan_"]
    excluded = [
        f for f in target_files
        if any(p in os.path.basename(f).lower() for p in EXCLUDE_PATTERNS)
    ]
    target_files = [f for f in target_files if f not in excluded]

    if excluded:
        print(f"Excluding {len(excluded)} pre-extracted attack-type subset "
              f"file(s) (not the full CAL set):")
        for f in excluded:
            print(f"    - {os.path.relpath(f, raw_data_dir)}")

    if not target_files:
        raise FileNotFoundError(
            f"No CAL-set CSV files found under {raw_data_dir}. Check that "
            f"week files (e.g. march_week3.csv) were downloaded and placed "
            f"correctly, and are named/extracted as .csv."
        )

    if max_hours is not None:
        print(f"NOTE: limiting each file to its first {max_hours} contiguous "
              f"hour(s) — see load_ugr16()'s docstring for why. Pass "
              f"max_hours=None to override and load full files instead.")

    print(f"\nLoading {len(target_files)} UGR16 CAL-set week file(s):")
    frames = [load_ugr16_file(f, max_hours=max_hours) for f in sorted(target_files)]
    combined = pd.concat(frames, ignore_index=True)

    print(f"\nTotal pooled rows: {len(combined)}")
    print(f"NOTE: per the official UGR16 site, the CAL set is 'clean from "
          f"synthetically generated attacks' but may still contain some "
          f"real-world traffic anomalies (their own documented caveat) — "
          f"not filtered here, as there's no reliable signal to do so.")
    return combined


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from config import CLEANED_DATA_DIR

    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "../raw_data/ugr16"
    # Second CLI arg overrides max_hours (default 2.0). Pass 'none' to
    # load the full file(s) instead — not recommended given the size.
    if len(sys.argv) > 2:
        max_hours = None if sys.argv[2].lower() == "none" else float(sys.argv[2])
    else:
        max_hours = 2.0
    df = load_ugr16(raw_dir, max_hours=max_hours)
    print(f"\nFinal shape: {df.shape}")
    print(f"Columns: {list(df.columns)} ({len(df.columns)} total)")

    os.makedirs(CLEANED_DATA_DIR, exist_ok=True)
    output_path = os.path.join(CLEANED_DATA_DIR, "ugr16_benign.parquet")
    df.to_parquet(output_path, index=False)
    print(f"\nSaved cleaned data to: {os.path.abspath(output_path)}")
