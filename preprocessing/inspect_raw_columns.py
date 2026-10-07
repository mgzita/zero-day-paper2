"""
=============================================================================
inspect_raw_columns.py
Run this AFTER placing raw dataset files into ../raw_data/<dataset_name>/

Purpose: print the actual column names/dtypes for each dataset so we can
fill in FEATURE_HARMONIZATION_MAP in config.py with real source columns,
rather than guessing. This mirrors the "testing.py" column-comparison step
already used in Paper 1's preprocessing.
=============================================================================
"""

import os
import glob
import pandas as pd
from config import RAW_DATA_DIR, SOURCE_DATASETS, ZERO_DAY_DATASETS

ALL_DATASETS = SOURCE_DATASETS + ZERO_DAY_DATASETS

# UGR16's raw files have NO header row — assign the known fixed column
# order manually, or pandas will treat the first data row as headers.
UGR16_COLUMN_NAMES = [
    "timestamp", "duration", "src_ip", "dst_ip", "src_port", "dst_port",
    "protocol", "flags", "forwarding_status", "tos", "packets", "bytes",
    "label",
]

# UNSW-NB15's raw partition files (UNSW-NB15_1.csv through _4.csv) ALSO
# have no header row — same issue, different dataset. Official 49-column
# order per NUSW-NB15_features.csv documentation.
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

# ToN-IoT's per-scenario conn.csv files (Zeek conn.log format) also have
# NO header row. This is the CORRECT file to use instead of the merged
# train_test_network.csv, which drops the ts field entirely. Standard
# 21-column Zeek conn.log schema, confirmed against an actual uploaded
# sample.
TON_IOT_CONN_COLUMN_NAMES = [
    "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
    "proto", "service", "duration", "orig_bytes", "resp_bytes",
    "conn_state", "local_orig", "local_resp", "missed_bytes", "history",
    "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
    "tunnel_parents",
]


def find_first_data_file(dataset_dir: str, dataset_name: str = ""):
    """Find the best candidate data file in a dataset directory (recursive).

    Skips known non-data files (feature dictionaries, ground-truth-only
    files) and prefers pre-merged/combined datasets over single-attack-type
    or per-file splits, since those are what the pipeline actually needs.
    """
    # Filenames to actively SKIP — these are metadata/dictionary files that
    # look like data but aren't (caught via unsw_nb15's NUSW-NB15_features.csv).
    SKIP_PATTERNS = ["_features.csv", "_gt.csv", "_ground_truth", "readme"]

    # Filenames to PREFER if present — merged/combined datasets are more
    # useful than single-attack-type raw exports (caught via edge_iiotset's
    # Attack traffic/Backdoor_attack.csv).
    #
    # NOTE: ton_iot is a deliberate EXCEPTION — "conn" (Zeek's conn.log,
    # which retains the `ts` timestamp) is prioritized OVER "train_test"
    # (the merged train_test_network.csv, which drops `ts` entirely).
    # This is why preference is dataset-aware rather than one global list.
    if dataset_name == "ton_iot":
        PREFER_PATTERNS = ["conn"]
    elif dataset_name == "ait_netflow":
        # Prefer tcp_complete/udp_complete (finished flows, reliable
        # duration/byte stats) over tcp_nocomplete (still-open connections
        # at capture end). Substrings are distinct enough that
        # "tcp_nocomplete.csv" does NOT match "tcp_complete".
        PREFER_PATTERNS = ["tcp_complete", "udp_complete"]
    else:
        PREFER_PATTERNS = ["ml-edge", "dnn-edge", "train_test", "_combined", "merged"]

    all_candidates = []
    for ext in ("*.csv", "*.parquet", "*.pcap_ISCX.csv", "*.tsv"):
        all_candidates.extend(
            glob.glob(os.path.join(dataset_dir, "**", ext), recursive=True)
        )

    if not all_candidates:
        return None

    # Filter out known metadata files
    filtered = [
        f for f in all_candidates
        if not any(p in os.path.basename(f).lower() for p in SKIP_PATTERNS)
    ]
    if not filtered:
        filtered = all_candidates  # fall back if everything got filtered

    # Prefer merged/combined files if any exist
    preferred = [
        f for f in filtered
        if any(p in os.path.basename(f).lower() for p in PREFER_PATTERNS)
    ]
    if preferred:
        return sorted(preferred)[0]

    return sorted(filtered)[0]


def read_with_encoding_fallback(path: str, nrows: int = 5):
    """Try utf-8 first, fall back to latin-1/cp1252 for older/non-UTF-8 exports
    (e.g. UNSW-NB15's files, which are not UTF-8 encoded)."""
    last_error = None
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, nrows=nrows, low_memory=False, encoding=encoding), encoding
        except (UnicodeDecodeError, UnicodeError) as e:
            last_error = e
            continue
    raise last_error


def _read_headerless(path: str, column_names: list, nrows: int = 5):
    """Read a CSV that has no header row, assigning known column names,
    trying multiple encodings."""
    last_error = None
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(
                path, nrows=nrows, low_memory=False,
                header=None, names=column_names, encoding=encoding,
            )
            return df, encoding
        except (UnicodeDecodeError, UnicodeError) as e:
            last_error = e
            continue
    raise last_error


def _is_unsw_raw_partition_file(path: str) -> bool:
    """True for UNSW-NB15_1.csv .. _4.csv (no header), False for
    UNSW_NB15_training-set.csv / _testing-set.csv (already have headers)."""
    basename = os.path.basename(path).lower()
    return (
        basename.startswith("unsw-nb15_")
        and basename.replace("unsw-nb15_", "").replace(".csv", "").isdigit()
    )


def list_all_files(dataset_dir: str):
    """List every file (any extension) under a dataset directory, recursive."""
    all_files = []
    for root, _dirs, files in os.walk(dataset_dir):
        for f in files:
            all_files.append(os.path.join(root, f))
    return all_files


def inspect_dataset(name: str):
    dataset_dir = os.path.join(RAW_DATA_DIR, name)
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")

    if not os.path.isdir(dataset_dir):
        print(f"  Directory not found: {dataset_dir}")
        return

    sample_file = find_first_data_file(dataset_dir, name)
    if sample_file is None:
        # No usable data file found — check WHAT is actually there instead
        # of just saying "not found", since it's often a zip that wasn't
        # extracted, or an unexpected file extension.
        existing_files = list_all_files(dataset_dir)
        if not existing_files:
            print(f"  Folder is empty: {dataset_dir}")
            print(f"  -> Place raw {name} files here before re-running this script.")
            return

        print(f"  No .csv/.parquet/.tsv files found, but {len(existing_files)} "
              f"file(s) DO exist here:")
        zips_found = []
        for f in existing_files[:20]:
            rel = os.path.relpath(f, dataset_dir)
            size_mb = os.path.getsize(f) / (1024 * 1024)
            print(f"    - {rel}  ({size_mb:.1f} MB)")
            if f.lower().endswith((".zip", ".7z", ".rar", ".gz", ".tar")):
                zips_found.append(f)
        if len(existing_files) > 20:
            print(f"    ... and {len(existing_files) - 20} more")

        if zips_found:
            print(f"\n  -> Looks like an unextracted archive. Extract it into")
            print(f"     {dataset_dir} (or a subfolder) and re-run this script:")
            for z in zips_found:
                print(f"       {z}")
        else:
            print(f"\n  -> These files have extensions this script doesn't scan for.")
            print(f"     Check if they're actually data files with an unusual")
            print(f"     extension, or the wrong files entirely.")
        return

    print(f"  Sample file: {sample_file}")

    # Warn if there are OTHER candidate files too, so the person knows this
    # is a choice, not the only file that exists.
    all_csvs = []
    for ext in ("*.csv", "*.parquet", "*.tsv"):
        all_csvs.extend(glob.glob(os.path.join(dataset_dir, "**", ext), recursive=True))
    if len(all_csvs) > 1:
        print(f"  NOTE: {len(all_csvs)} total data files found in this folder — "
              f"only inspecting the one above. Others include:")
        for f in sorted(all_csvs)[:5]:
            if f != sample_file:
                print(f"    - {os.path.relpath(f, dataset_dir)}")
        if len(all_csvs) > 6:
            print(f"    ... and {len(all_csvs) - 6} more")

    try:
        if sample_file.endswith(".parquet"):
            df = pd.read_parquet(sample_file).head(5)
            encoding_used = "n/a (parquet)"
        elif name == "ugr16":
            df, encoding_used = _read_headerless(sample_file, UGR16_COLUMN_NAMES)
            print(f"  NOTE: UGR16 has no header row — columns assigned manually")
            print(f"  from the known fixed format ({len(UGR16_COLUMN_NAMES)} cols).")
            print(f"  Verify column count matches before trusting this.")
        elif name == "unsw_nb15" and _is_unsw_raw_partition_file(sample_file):
            # Only the 4 raw partition files (UNSW-NB15_1.csv.._4.csv) lack
            # a header — the training-set/testing-set files already have
            # proper column names and should be read normally instead.
            df, encoding_used = _read_headerless(sample_file, UNSW_NB15_COLUMN_NAMES)
            print(f"  NOTE: This is a raw UNSW-NB15 partition file (no header row)")
            print(f"  — columns assigned manually from NUSW-NB15_features.csv order")
            print(f"  ({len(UNSW_NB15_COLUMN_NAMES)} cols). Verify column count matches.")
        elif name == "ton_iot" and "conn" in os.path.basename(sample_file).lower():
            # ToN-IoT's per-scenario conn.csv (Zeek conn.log) has no header
            # row either. This is the CORRECT file — it retains `ts`,
            # unlike the merged train_test_network.csv.
            df, encoding_used = _read_headerless(sample_file, TON_IOT_CONN_COLUMN_NAMES)
            print(f"  NOTE: This is a ToN-IoT Zeek conn.log file (no header row)")
            print(f"  — columns assigned manually from standard Zeek conn.log order")
            print(f"  ({len(TON_IOT_CONN_COLUMN_NAMES)} cols). Verify column count matches.")
        else:
            df, encoding_used = read_with_encoding_fallback(sample_file, nrows=5)
    except Exception as e:
        print(f"  ERROR reading file: {e}")
        return

    print(f"  Encoding used: {encoding_used}")
    print(f"  Shape (first 5 rows): {df.shape}")
    print(f"  Columns ({len(df.columns)}):")
    for col in df.columns:
        print(f"    - {col!r}  (dtype: {df[col].dtype})")


if __name__ == "__main__":
    print(f"Raw data root: {os.path.abspath(RAW_DATA_DIR)}\n")
    for name in ALL_DATASETS:
        inspect_dataset(name)

    print(f"\n{'=' * 70}")
    print("Next step: fill in FEATURE_HARMONIZATION_MAP in config.py using")
    print("the column names printed above, mapping each dataset's raw")
    print("columns to the 13 WINDOW_FEATURES defined in config.py.")
    print(f"{'=' * 70}")
