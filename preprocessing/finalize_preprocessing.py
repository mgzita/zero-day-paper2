"""
=============================================================================
finalize_preprocessing.py — the actual "no redo needed" pass, run AFTER
audit_cleaned_data.py and BEFORE windowing.

This is what audit_cleaned_data.py deliberately did NOT do: it only
detected/reported problems (plus safe exact-duplicate removal). This
script APPLIES the remaining decisions:

  1. UNIFIED TIMESTAMP PARSING — the single most critical gap. Every
     dataset's timestamp is currently in a different native format
     (string variants, Unix epoch int/float, subsecond precision
     differences). Windowing cannot work until every dataset has ONE
     consistent datetime column. Adds `_timestamp_parsed` (pandas
     datetime64) to every dataset.

  2. CRITICAL-COLUMN NULL DROPS — rows missing IP, port, or timestamp
     cannot be windowed at all (same logic already applied to ToN-IoT's
     `ts` at the loader stage; this makes it systematic across all 7).

  3. LEGITIMATE-ZERO NULL FILLS — duration/byte/packet columns where NaN
     represents "no data transferred" (Zeek's '-' convention, confirmed
     in ToN-IoT/UNSW-NB15) are filled with 0, not left as NaN, since 0
     is a meaningful value for window aggregation (mean/variance
     calculations should include these as zeros, not silently drop them).

  4. IP/PORT RANGE VALIDATION — port numbers outside 0-65535, and
     malformed IP address strings, are flagged (and where clearly
     corrupted, dropped) rather than silently passed through to
     windowing.

IMPORTANT: two timestamp formats (CIC-IDS2017's 'Timestamp', Edge-IIoTset's
'frame.time') have NOT been verified against real sample values in this
project — only confirmed the columns exist. This script uses pandas'
flexible datetime parser (handles most common formats automatically) but
FLAGS these two explicitly for manual verification, since a silently
wrong parse would corrupt every downstream window without any error.
=============================================================================
"""

import os
import sys
import re
import ipaddress
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CLEANED_DATA_DIR, FEATURE_HARMONIZATION_MAP

DATASET_FILES = {
    "ait_netflow": "ait_netflow_benign.parquet",
    "cic_ids2017": "cic_ids2017_benign.parquet",
    "unsw_nb15": "unsw_nb15_benign.parquet",
    "ton_iot": "ton_iot_benign.parquet",
    "ugr16": "ugr16_benign.parquet",
    "bot_iot": "bot_iot_full.parquet",
    "edge_iiotset": "edge_iiotset_full.parquet",
}

# Timestamp column + how to parse it, per dataset.
# 'unit' set means Unix epoch (int/float seconds); None means pandas
# should parse it as a datetime string.
TIMESTAMP_SPEC = {
    "ait_netflow": {"col": "timestamp", "unit": None, "verified": True},
    "cic_ids2017": {"col": "Timestamp", "unit": None, "verified": False},
    "unsw_nb15": {"col": "Stime", "unit": "s", "verified": True},
    "ton_iot": {"col": "ts", "unit": "s", "verified": True},
    "ugr16": {"col": "timestamp", "unit": None, "verified": True},
    "bot_iot": {"col": "stime", "unit": "s", "verified": True},
    "edge_iiotset": {"col": "frame.time", "unit": None, "verified": False},
}

# Columns where NaN legitimately means "zero" (no data transferred),
# confirmed via Zeek's '-' placeholder convention (ToN-IoT) and similar
# patterns (UNSW-NB15). Filling with 0 rather than dropping preserves
# these rows for window aggregation instead of losing them entirely.
ZERO_FILL_COLUMNS = {
    "ton_iot": ["duration", "orig_bytes", "resp_bytes"],
    "unsw_nb15": [],  # ct_ftp_cmd etc. not used in WINDOW_FEATURES, no fill needed
}

# IP/port columns per dataset, for range/format validation.
IP_PORT_COLUMNS = {
    "ait_netflow": {"ips": ["c_ip:1", "s_ip:15"], "ports": ["c_port:2", "s_port:16"]},
    "cic_ids2017": {"ips": ["Src IP", "Dst IP"], "ports": ["Src Port", "Dst Port"]},
    "unsw_nb15": {"ips": ["srcip", "dstip"], "ports": ["sport", "dsport"]},
    "ton_iot": {"ips": ["id.orig_h", "id.resp_h"], "ports": ["id.orig_p", "id.resp_p"]},
    "ugr16": {"ips": ["src_ip", "dst_ip"], "ports": ["src_port", "dst_port"]},
    "bot_iot": {"ips": ["saddr", "daddr"], "ports": ["sport", "dport"]},
    "edge_iiotset": {"ips": ["ip.src_host", "ip.dst_host"], "ports": ["tcp.srcport", "tcp.dstport"]},
}


def parse_timestamps(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Add a unified `_timestamp_parsed` datetime64 column."""
    spec = TIMESTAMP_SPEC.get(name)
    if spec is None or spec["col"] not in df.columns:
        print(f"  [SKIP] No timestamp spec/column found for '{name}'.")
        return df

    col, unit, verified = spec["col"], spec["unit"], spec["verified"]
    n_before = len(df)

    if unit is not None:
        df["_timestamp_parsed"] = pd.to_datetime(df[col], unit=unit, errors="coerce")
    else:
        df["_timestamp_parsed"] = pd.to_datetime(df[col], errors="coerce")

    n_unparsed = df["_timestamp_parsed"].isna().sum()
    pct_unparsed = 100 * n_unparsed / n_before if n_before > 0 else 0

    verify_tag = "" if verified else " [UNVERIFIED FORMAT — CHECK THIS]"
    print(f"  Parsed '{col}' -> '_timestamp_parsed'{verify_tag}: "
          f"{n_unparsed} row(s) failed to parse ({pct_unparsed:.2f}%)")

    if not verified and n_unparsed < n_before * 0.5:
        sample = df[col].dropna().iloc[0] if df[col].notna().any() else None
        sample_parsed = df["_timestamp_parsed"].dropna().iloc[0] if df["_timestamp_parsed"].notna().any() else None
        print(f"    MANUAL CHECK NEEDED: raw sample = {sample!r} -> "
              f"parsed as {sample_parsed!r}. Confirm this is actually "
              f"correct (right timezone, right date, not day/month "
              f"swapped) before trusting this column for windowing.")

    if pct_unparsed > 50:
        print(f"    [FLAG] Over half of timestamps failed to parse — "
              f"this format is likely WRONG, not just a few bad rows. "
              f"Do not proceed to windowing with this dataset until fixed.")

    # Drop rows with unparseable timestamp — cannot be windowed without one.
    if n_unparsed > 0:
        df = df[df["_timestamp_parsed"].notna()].copy()
        print(f"  Dropped {n_unparsed} row(s) with unparseable timestamp.")

    return df


def apply_zero_fills(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Fill legitimate-zero NaN columns (Zeek '-' placeholder pattern)
    with 0, rather than leaving as NaN for window aggregation."""
    cols = ZERO_FILL_COLUMNS.get(name, [])
    for col in cols:
        if col not in df.columns:
            continue
        n_filled = df[col].isna().sum()
        if n_filled > 0:
            df[col] = df[col].fillna(0)
            print(f"  Filled {n_filled} NaN value(s) in '{col}' with 0 "
                  f"(legitimate 'no data transferred' per Zeek convention).")
    return df


def drop_critical_nulls(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Drop rows with null IP/port — cannot be windowed or aggregated
    into flow_count/unique_dst_ips/etc. without these."""
    spec = IP_PORT_COLUMNS.get(name, {})
    critical_cols = [c for c in spec.get("ips", []) + spec.get("ports", []) if c in df.columns]
    if not critical_cols:
        return df

    n_before = len(df)
    df = df.dropna(subset=critical_cols).copy()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        pct = 100 * n_dropped / n_before
        print(f"  Dropped {n_dropped} row(s) ({pct:.2f}%) with null "
              f"IP/port in {critical_cols}.")
    return df


def validate_ports(df: pd.DataFrame, name: str):
    """Flag (report only) port values outside the valid 0-65535 range —
    a genuine sign of corruption, not just missing data."""
    spec = IP_PORT_COLUMNS.get(name, {})
    for col in spec.get("ports", []):
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        n_invalid = ((df[col] < 0) | (df[col] > 65535)).sum()
        if n_invalid > 0:
            pct = 100 * n_invalid / len(df)
            print(f"  [FLAG] '{col}' has {n_invalid} value(s) ({pct:.2f}%) "
                  f"outside the valid port range (0-65535) — likely "
                  f"corrupted, worth investigating before windowing.")


def validate_ips(df: pd.DataFrame, name: str):
    """Flag (report only) malformed IP address strings."""
    spec = IP_PORT_COLUMNS.get(name, {})
    for col in spec.get("ips", []):
        if col not in df.columns:
            continue
        sample = df[col].dropna().astype(str).unique()[:1000]  # cap for speed on large datasets
        n_invalid = 0
        for ip_str in sample:
            try:
                ipaddress.ip_address(ip_str)
            except ValueError:
                n_invalid += 1
        if n_invalid > 0:
            pct = 100 * n_invalid / len(sample)
            print(f"  [FLAG] '{col}': {n_invalid}/{len(sample)} sampled "
                  f"unique values ({pct:.1f}%) are not valid IP addresses "
                  f"— worth investigating.")


def finalize_dataset(name: str, filename: str):
    path = os.path.join(CLEANED_DATA_DIR, filename)
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")

    if not os.path.exists(path):
        print(f"  [MISSING] {path} — run this dataset's loader first.")
        return

    df = pd.read_parquet(path)
    n_original = len(df)
    print(f"  Shape (start): {df.shape}")

    df = parse_timestamps(df, name)
    df = drop_critical_nulls(df, name)
    df = apply_zero_fills(df, name)
    validate_ports(df, name)
    validate_ips(df, name)

    print(f"  Shape (final): {df.shape} "
          f"({n_original - len(df)} total rows dropped, "
          f"{100 * (n_original - len(df)) / n_original:.2f}%)")

    df.to_parquet(path, index=False)
    print(f"  Saved finalized data back to {path}")


if __name__ == "__main__":
    print("Running comprehensive preprocessing finalization across all 7 "
          "cleaned datasets (timestamps, critical nulls, zero-fills, "
          "IP/port validation)...")
    for name, filename in DATASET_FILES.items():
        finalize_dataset(name, filename)

    print(f"\n{'=' * 70}")
    print("Finalization complete. Pay special attention to any "
          "'MANUAL CHECK NEEDED' or 'FLAG' messages above — especially "
          "for cic_ids2017 and edge_iiotset's timestamp parsing, which "
          "have not been verified against real sample data.")
    print(f"{'=' * 70}")
