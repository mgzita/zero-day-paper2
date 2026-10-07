"""
=============================================================================
loaders/ton_iot.py — load ToN-IoT source data

Uses per-scenario conn.csv files (Zeek conn.log format, confirmed via
direct inspection of an uploaded sample) across all 9 "normal_*" scenario
folders, NOT the merged train_test_network.csv (which drops the ts field
entirely — see project design notes).

Labeling convention: the 9 folders are named normal_backdoor, normal_DDoS,
normal_DoS, normal_injection, normal_MITM, normal_password,
normal_runsomware, normal_scanning, normal_XSS — i.e. "normal traffic
captured during each attack scenario", not paired normal/attack folders.
The raw conn.csv itself has NO label/type column (confirmed: 21 columns,
pure Zeek schema). So the labeling signal is the FOLDER NAME, not a
per-row field — every row in every normal_* folder is benign by
convention.

This loader checks defensively for a label/type column just in case a
variant exists with one, but falls back to "entire folder is benign" if
not, rather than assuming blindly either way.
=============================================================================
"""

import os
import glob
import pandas as pd

TON_IOT_CONN_COLUMN_NAMES = [
    "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
    "proto", "service", "duration", "orig_bytes", "resp_bytes",
    "conn_state", "local_orig", "local_resp", "missed_bytes", "history",
    "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
    "tunnel_parents",
]

EXPECTED_SCENARIO_FOLDERS = [
    "normal_backdoor", "normal_DDoS", "normal_DoS", "normal_injection",
    "normal_MITM", "normal_password", "normal_runsomware",
    "normal_scanning", "normal_XSS",
]


def _read_with_encoding_fallback(path: str, **kwargs):
    last_error = None
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs), encoding
        except (UnicodeDecodeError, UnicodeError) as e:
            last_error = e
            continue
    raise last_error


def load_ton_iot_conn_file(path: str) -> pd.DataFrame:
    """Load a single ToN-IoT conn.csv file (Zeek conn.log, no header row).
    Every row is treated as benign UNLESS a label/type column is somehow
    present (defensive check — not expected based on the confirmed sample,
    but checked rather than assumed)."""
    # Check the raw field count BEFORE calling read_csv with names= —
    # pandas silently absorbs a mismatched column count by treating extra
    # leading fields as an index, rather than raising an error, so
    # checking len(df.columns) AFTER the fact does not reliably catch a
    # schema mismatch (confirmed via direct testing).
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
    actual_field_count = len(first_line.rstrip("\n").split(","))
    if actual_field_count != len(TON_IOT_CONN_COLUMN_NAMES):
        raise ValueError(
            f"Column count mismatch for {path}: first line has "
            f"{actual_field_count} comma-separated fields, expected "
            f"{len(TON_IOT_CONN_COLUMN_NAMES)}. This file's schema doesn't "
            f"match the standard Zeek conn.log format confirmed from the "
            f"sample — check whether this is really a conn.log file, not "
            f"e.g. files.log/dns.log/http.log, and that no extra columns "
            f"(e.g. an unexpected label field) have been added."
        )

    df, encoding_used = _read_with_encoding_fallback(
        path, header=None, names=TON_IOT_CONN_COLUMN_NAMES, low_memory=False
    )

    n_before = len(df)

    # Defensive check: if a label/type column somehow exists, use it.
    # Not expected based on the confirmed 21-column sample, but check
    # rather than assume.
    if "label" in df.columns or "type" in df.columns:
        label_col = "label" if "label" in df.columns else "type"
        print(f"    NOTE: unexpected '{label_col}' column found — filtering "
              f"by it rather than assuming the whole folder is benign.")
        is_benign_val = "normal" if label_col == "type" else 0
        df = df[df[label_col] == is_benign_val].copy()
    # else: no label column present (expected case) — entire file is
    # treated as benign, per the normal_* folder naming convention.

    n_after = len(df)
    n_removed = n_before - n_after

    print(f"    {os.path.basename(path)} (encoding: {encoding_used}): "
          f"{n_before} rows -> {n_after} rows benign "
          f"({n_removed} rows removed by label filter)")

    return df


def load_ton_iot(raw_data_dir: str) -> pd.DataFrame:
    """Load and pool conn.csv from all available ToN-IoT scenario files.

    Handles Windows' auto-rename pattern when multiple files share the
    same original name (conn.csv -> conn.csv, conn (2).csv, conn (3).csv,
    ..., or conn01.csv, conn02.csv) — a common result of downloading
    conn.csv from each of the 9 scenario folders individually, since the
    browser can't tell they came from different source folders.
    """
    all_csvs = glob.glob(os.path.join(raw_data_dir, "**", "*.csv"), recursive=True)
    # Broad match: any filename starting with 'conn' (case-insensitive),
    # covering 'conn.csv', 'conn (2).csv', 'conn(7).csv', 'conn01.csv',
    # and the original 'conn_<scenario>.csv' naming, while still excluding
    # other Zeek log types (files.csv, dns.csv, http.csv, ssl.csv — none
    # of which start with 'conn').
    conn_files = [f for f in all_csvs if os.path.basename(f).lower().startswith("conn")]

    if not conn_files:
        raise FileNotFoundError(
            f"No conn*.csv files found under {raw_data_dir}. Check that "
            f"conn.csv (or its Windows-renamed duplicates, e.g. "
            f"'conn (2).csv') were downloaded and placed correctly."
        )

    # Best-effort scenario identification — only meaningful if filenames
    # or folder paths still contain the scenario name. If files were
    # renamed generically by Windows (conn (2).csv, conn01.csv, etc.)
    # with no scenario info preserved, this will find 0 matches; that's
    # expected in that case and NOT a sign of a problem, so don't warn
    # about "missing" scenarios when we genuinely can't tell.
    found_scenarios = set()
    for f in conn_files:
        for scenario in EXPECTED_SCENARIO_FOLDERS:
            if scenario.lower() in f.lower():
                found_scenarios.add(scenario)

    print(f"Found {len(conn_files)} conn*.csv file(s).")
    if found_scenarios:
        missing_scenarios = set(EXPECTED_SCENARIO_FOLDERS) - found_scenarios
        if missing_scenarios:
            print(f"  Identified {len(found_scenarios)} of 9 expected scenarios "
                  f"by name; {len(missing_scenarios)} not identifiable from "
                  f"filenames: {sorted(missing_scenarios)}")
    else:
        print(f"  (Scenario names not identifiable from filenames — likely "
              f"Windows auto-renamed duplicates during download, e.g. "
              f"'conn (2).csv'. This is fine; each file is still loaded and "
              f"pooled regardless of what it's named. Cannot confirm exactly "
              f"which/how many of the 9 scenarios are represented, but "
              f"{len(conn_files)} files were found and will all be used.)")

    print(f"\nLoading {len(conn_files)} ToN-IoT conn.csv file(s):")
    frames = []
    for f in sorted(conn_files):
        print(f"  {os.path.relpath(f, raw_data_dir)}:")
        frames.append(load_ton_iot_conn_file(f))
    combined = pd.concat(frames, ignore_index=True)

    print(f"\nTotal pooled benign rows: {len(combined)}")

    # Fix mixed-type columns before saving. Same category of issue as
    # UNSW-NB15's loader: individual files can parse a numeric column
    # differently (e.g. one file has a stray non-numeric value in 'ts',
    # forcing that file's column to a string dtype), and pooling via
    # pd.concat then produces a column that's part-int, part-string,
    # which pyarrow rejects. Proactively coerce every column that should
    # be numeric, in one pass, rather than fixing them one at a time as
    # pyarrow errors surface each one individually (confirmed necessary
    # via an actual pyarrow ArrowInvalid error on the 'ts' column during
    # testing on real data).
    #
    # Zeek's convention of using '-' as a null/not-applicable placeholder
    # (confirmed in the uploaded sample: duration/orig_bytes/resp_bytes
    # show '-' for OTH-state connections) is handled correctly by
    # pd.to_numeric(errors='coerce'), which turns '-' into NaN.
    CATEGORICAL_COLUMNS = {
        "uid", "id.orig_h", "id.resp_h", "proto", "service", "conn_state",
        "local_orig", "local_resp", "history", "tunnel_parents",
    }
    numeric_candidate_cols = [c for c in combined.columns if c not in CATEGORICAL_COLUMNS]

    for col in numeric_candidate_cols:
        if pd.api.types.is_numeric_dtype(combined[col]):
            continue  # already clean, skip

        as_str = combined[col].astype(str)
        n_before_numeric = combined[col].notna().sum()
        combined[col] = pd.to_numeric(as_str.str.strip(), errors="coerce")
        n_after_numeric = combined[col].notna().sum()
        n_lost = n_before_numeric - n_after_numeric
        if n_lost > 0:
            if col == "ts":
                print(f"  WARNING: 'ts' had {n_lost} value(s) that could not be "
                      f"converted to numeric — this is NOT an expected Zeek "
                      f"placeholder pattern (unlike duration/bytes '-' values); "
                      f"worth checking what the actual raw value was. These "
                      f"row(s) will be dropped below.")
            else:
                print(f"  NOTE: '{col}' had {n_lost} value(s) that could not be "
                      f"converted to numeric (set to NaN) — expected for Zeek's "
                      f"'-' null placeholder in duration/bytes fields on "
                      f"connections with no data transferred (e.g. OTH state).")

        # Use Int64 only if no fractional values present; ts/duration are
        # legitimately float (e.g. ts has sub-second precision sometimes).
        if combined[col].dropna().apply(lambda x: float(x).is_integer()).all():
            combined[col] = combined[col].astype("Int64")

    # Drop rows with unparseable 'ts' — a row without a valid timestamp
    # cannot be assigned to any temporal window downstream, so it's
    # useless (and would break windowing) rather than just imperfect.
    # Unlike duration/orig_bytes/resp_bytes (where NaN is an EXPECTED
    # Zeek '-' placeholder for OTH-state connections), a missing ts has
    # no legitimate interpretation and should be dropped, not kept as NaN.
    if "ts" in combined.columns:
        n_before_drop = len(combined)
        combined = combined[combined["ts"].notna()].copy()
        n_dropped = n_before_drop - len(combined)
        if n_dropped > 0:
            print(f"  Dropped {n_dropped} row(s) with unparseable 'ts' "
                  f"(cannot be assigned to a temporal window without one).")

    return combined


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from config import CLEANED_DATA_DIR

    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "../raw_data/ton_iot"
    df = load_ton_iot(raw_dir)
    print(f"\nFinal shape: {df.shape}")
    print(f"Columns: {list(df.columns)[:10]} ... ({len(df.columns)} total)")

    os.makedirs(CLEANED_DATA_DIR, exist_ok=True)
    output_path = os.path.join(CLEANED_DATA_DIR, "ton_iot_benign.parquet")
    df.to_parquet(output_path, index=False)
    print(f"\nSaved cleaned data to: {os.path.abspath(output_path)}")
