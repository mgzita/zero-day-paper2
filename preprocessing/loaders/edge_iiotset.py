"""
=============================================================================
loaders/edge_iiotset.py — load Edge-IIoTset zero-day evaluation data

Like BoT-IoT, this is a ZERO-DAY dataset, not a source (pretraining)
dataset — no benign filtering happens here. UNLIKE BoT-IoT, config.py
marks this as a FULL-METRIC-SUITE zero-day set
(ZERO_DAY_HAS_BENIGN["edge_iiotset"] = True), since it has substantial
genuine benign traffic (not a negligible ~0.01% fraction like BoT-IoT).

IMPORTANT SCOPE NOTE: this loader does NOT implement the custom
flow_count-as-unique-5-tuples logic or the per-5-tuple duration proxy
from FEATURE_HARMONIZATION_MAP — those belong in the windowing step,
which operates on packet rows directly. This loader's job is just to
read, clean, and save the packet-level data as-is.

Uses chunked reading defensively (not because we've confirmed a memory
problem here, but because this file is ~1.19GB with 63 columns, and
we've already hit one real out-of-memory crash this project on a
different large file — cheaper to build this in from the start than
debug it reactively again).
=============================================================================
"""

import os
import glob
import pandas as pd

CHUNK_SIZE = 200_000  # rows per chunk — smaller than UGR16's, since this
                       # file has 63 columns (much wider) vs UGR16's 13


def load_edge_iiotset_file(path: str) -> pd.DataFrame:
    """Load the Edge-IIoTset merged DNN dataset file in chunks. NO
    filtering — this is a zero-day dataset with a full-metric-suite
    design (both benign and attack traffic needed)."""
    chunks = []
    total_rows = 0
    reader = pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False)
    for chunk in reader:
        chunks.append(chunk)
        total_rows += len(chunk)

    df = pd.concat(chunks, ignore_index=True)

    # Fix mixed-type columns before saving. Same category of issue as
    # UNSW-NB15/ToN-IoT's loaders: this file has 63 columns, many
    # protocol-specific (e.g. http.file_data, mqtt.len) that are NaN for
    # irrelevant protocols — if one chunk happens to be ALL-NaN for such
    # a column (parsed as float64) while another chunk has real numeric
    # values represented differently (parsed as object/string), pd.concat
    # produces a column pyarrow can't convert. Confirmed via an actual
    # ArrowInvalid error on 'http.file_data' during testing on real data.
    # Rather than fix these one at a time as they surface, proactively
    # coerce every column known to be genuinely numeric based on the
    # confirmed schema from direct inspection (inspect_raw_columns.py).
    #
    # Categorical/text columns — confirmed via direct inspection of the
    # actual DNN-EdgeIIoT-dataset.csv to be genuinely string-valued, not
    # numeric-with-occasional-text:
    CATEGORICAL_COLUMNS = {
        "frame.time", "ip.src_host", "ip.dst_host", "tcp.options",
        "tcp.payload", "mqtt.conack.flags", "mqtt.msg", "mqtt.protoname",
        "mqtt.topic", "Attack_type",
    }
    numeric_candidate_cols = [c for c in df.columns if c not in CATEGORICAL_COLUMNS]

    n_columns_fixed = 0
    for col in numeric_candidate_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue  # already clean, skip
        as_str = df[col].astype(str)
        df[col] = pd.to_numeric(as_str.str.strip(), errors="coerce")
        n_columns_fixed += 1

    # ALSO clean the categorical/text columns — same underlying problem
    # can hit these too: if a whole chunk is NaN for a text column (e.g.
    # tcp.options is absent for non-TCP packets), that chunk parses the
    # column as float64 (all-NaN), while another chunk with genuine
    # string content (e.g. hex-encoded TCP option bytes) parses as
    # object/string. pd.concat then produces a column mixing real Python
    # str objects with bare Python float NaN objects, which pyarrow
    # cannot write as a string column (confirmed via an actual
    # ArrowTypeError on 'tcp.options' during testing on real data).
    # Fix: force every value to a clean string, preserving true nulls
    # (not converting NaN itself to the text "nan").
    n_categorical_fixed = 0
    for col in CATEGORICAL_COLUMNS:
        if col not in df.columns:
            continue
        df[col] = df[col].apply(lambda x: str(x) if pd.notna(x) else None)
        n_categorical_fixed += 1

    if n_columns_fixed > 0 or n_categorical_fixed > 0:
        print(f"  Coerced {n_columns_fixed} column(s) to numeric and "
              f"cleaned {n_categorical_fixed} text column(s) to resolve "
              f"cross-chunk dtype inconsistencies (protocol-specific "
              f"fields that are NaN for irrelevant protocols in some rows).")

    required_cols = ["ip.src_host", "ip.dst_host", "Attack_label", "Attack_type"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Expected columns missing from {path}: {missing}. Check this "
            f"is really the DNN-EdgeIIoT-dataset.csv file, not one of the "
            f"per-attack raw files (Attack traffic/*.csv) or the ML-ready "
            f"version (which drops IP columns)."
        )

    n_benign = (df["Attack_label"] == 0).sum()
    n_attack = (df["Attack_label"] == 1).sum()
    pct_benign = 100 * n_benign / len(df) if len(df) > 0 else 0

    print(f"  {os.path.basename(path)}: {len(df)} rows total "
          f"({n_benign} benign [{pct_benign:.1f}%], {n_attack} attack) — "
          f"NO filtering applied, loading all rows since this is a "
          f"zero-day evaluation set.")

    return df


def load_edge_iiotset(raw_data_dir: str) -> pd.DataFrame:
    """Load Edge-IIoTset. Expects the merged DNN-EdgeIIoT-dataset.csv,
    NOT the per-attack raw files or the ML-ready (IP-stripped) version."""
    # Prefer the DNN merged file specifically, same logic as
    # inspect_raw_columns.py's PREFER_PATTERNS — explicitly avoid picking
    # up one of the per-attack files (Attack traffic/*.csv) or the
    # ML-ready version if multiple candidates exist in the folder.
    all_csvs = glob.glob(os.path.join(raw_data_dir, "**", "*.csv"), recursive=True)
    dnn_files = [f for f in all_csvs if "dnn-edge" in os.path.basename(f).lower()]

    if not dnn_files:
        raise FileNotFoundError(
            f"No DNN-EdgeIIoT-dataset.csv found under {raw_data_dir}. "
            f"Found {len(all_csvs)} other CSV file(s) instead — check "
            f"you have the merged DNN dataset file, not just the "
            f"per-attack raw files or the ML-ready (IP-stripped) version."
        )
    if len(dnn_files) > 1:
        print(f"WARNING: found {len(dnn_files)} files matching 'dnn-edge' — "
              f"using only the first: {dnn_files[0]}")

    target_file = sorted(dnn_files)[0]
    print(f"Loading Edge-IIoTset from: {os.path.relpath(target_file, raw_data_dir)}")
    df = load_edge_iiotset_file(target_file)

    print(f"\nTotal rows: {len(df)}")
    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from config import CLEANED_DATA_DIR

    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "../raw_data/edge_iiotset"
    df = load_edge_iiotset(raw_dir)
    print(f"\nFinal shape: {df.shape}")
    print(f"Columns: {list(df.columns)[:10]} ... ({len(df.columns)} total)")

    os.makedirs(CLEANED_DATA_DIR, exist_ok=True)
    # No '_benign' suffix — intentionally contains attack traffic too,
    # same naming convention as bot_iot_full.parquet.
    output_path = os.path.join(CLEANED_DATA_DIR, "edge_iiotset_full.parquet")
    df.to_parquet(output_path, index=False)
    print(f"\nSaved cleaned data to: {os.path.abspath(output_path)}")
