"""
=============================================================================
audit_cleaned_data.py — actual data quality audit across all 7 cleaned
datasets, run AFTER loading but BEFORE windowing.

This checks what the loaders did NOT check: exact duplicate rows,
null rates on the columns FEATURE_HARMONIZATION_MAP actually depends on,
basic physical-plausibility sanity ranges (negative durations/bytes/
packets), and a concrete check on the two institutional-lineage pairs
config.py already flags as a documented limitation.

Run this before building the windowing code — a null/duplicate/range
problem is much cheaper to fix now than after windows are built on top
of it.
=============================================================================
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CLEANED_DATA_DIR, FEATURE_HARMONIZATION_MAP, LEAKAGE_CHECK_PAIRS

# Map dataset name -> (parquet filename, is_source_dataset)
DATASET_FILES = {
    "ait_netflow": ("ait_netflow_benign.parquet", True),
    "cic_ids2017": ("cic_ids2017_benign.parquet", True),
    "unsw_nb15": ("unsw_nb15_benign.parquet", True),
    "ton_iot": ("ton_iot_benign.parquet", True),
    "ugr16": ("ugr16_benign.parquet", True),
    "bot_iot": ("bot_iot_full.parquet", False),
    "edge_iiotset": ("edge_iiotset_full.parquet", False),
}


def audit_and_dedupe(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Check for exact duplicate rows AND actually remove them (safe to
    do automatically — two byte-identical rows are essentially always
    an artifact, e.g. a file being present in two source locations, not
    genuinely distinct data). Returns the deduplicated dataframe.

    Does NOT auto-remove partial-key near-duplicates (e.g. same 5-tuple
    + timestamp but differing in other fields) — reports these instead,
    since they could be genuinely distinct data (e.g. two connections
    that coincidentally share a timestamp at 1-second resolution) rather
    than true duplicates, and require human judgment to resolve safely.
    """
    n_before = len(df)
    n_exact_dupes = df.duplicated().sum()
    pct = 100 * n_exact_dupes / n_before if n_before > 0 else 0

    if n_exact_dupes > 0:
        df = df.drop_duplicates(keep="first").reset_index(drop=True)
        print(f"  [FIXED] Removed {n_exact_dupes} exact duplicate row(s) "
              f"({pct:.2f}%) — kept first occurrence of each.")
    else:
        print(f"  [OK] No exact duplicate rows found.")

    return df


def audit_near_duplicates(df: pd.DataFrame, name: str):
    """Report (not remove) rows sharing a likely identifying key
    (src/dst IP + src/dst port + timestamp, using whatever column names
    this dataset actually has) — these are worth a human look, not an
    automatic decision, since they may be genuinely distinct connections
    that happen to share a timestamp at low time resolution."""
    KEY_COLUMNS = {
        "ait_netflow": ["c_ip:1", "s_ip:15", "c_port:2", "s_port:16", "timestamp"],
        "cic_ids2017": ["Src IP", "Dst IP", "Src Port", "Dst Port", "Timestamp"],
        "unsw_nb15": ["srcip", "dstip", "sport", "dsport", "Stime"],
        "ton_iot": ["id.orig_h", "id.resp_h", "id.orig_p", "id.resp_p", "ts"],
        "ugr16": ["src_ip", "dst_ip", "src_port", "dst_port", "timestamp"],
        "bot_iot": ["saddr", "daddr", "sport", "dport", "stime"],
        "edge_iiotset": ["ip.src_host", "ip.dst_host", "tcp.srcport", "tcp.dstport", "frame.time"],
    }

    if name not in KEY_COLUMNS:
        return
    key_cols = [c for c in KEY_COLUMNS[name] if c in df.columns]
    if len(key_cols) < 3:
        print(f"  [SKIP] Not enough identifying columns present to check "
              f"near-duplicates for this dataset.")
        return

    n_near_dupes = df.duplicated(subset=key_cols).sum()
    pct = 100 * n_near_dupes / len(df) if len(df) > 0 else 0
    if n_near_dupes > 0:
        print(f"  [FLAG] {n_near_dupes} row(s) ({pct:.1f}%) share identical "
              f"{key_cols} with another row but differ elsewhere — NOT "
              f"auto-removed (could be genuinely distinct connections at "
              f"coarse timestamp resolution). Worth a manual look if this "
              f"percentage seems high.")
    else:
        print(f"  [OK] No near-duplicates found on key columns {key_cols}.")


def audit_nulls(df: pd.DataFrame, name: str, harmonization_map: dict):
    """Check null rates specifically on columns referenced in
    FEATURE_HARMONIZATION_MAP for this dataset — these are the columns
    that actually matter for windowing, not every raw column."""
    if name not in harmonization_map:
        print(f"  [SKIP] No FEATURE_HARMONIZATION_MAP entry for '{name}'")
        return {}

    relevant_cols = set()
    for feature, spec in harmonization_map[name].items():
        if not isinstance(spec, dict):
            continue  # skip non-feature entries like 'timestamp_column', 'benign_filter'
        cols = spec.get("source_columns")
        if cols:
            relevant_cols.update(cols)
    ts_col = harmonization_map[name].get("timestamp_column")
    if isinstance(ts_col, str) and not ts_col.startswith("BLOCKED"):
        relevant_cols.add(ts_col)

    relevant_cols = {c for c in relevant_cols if c in df.columns}
    if not relevant_cols:
        print(f"  [WARN] None of the harmonization map's source columns "
              f"were found in the actual dataframe — check column naming "
              f"consistency between the loader and config.py.")
        return {}

    null_report = {}
    print(f"  Null rates on the {len(relevant_cols)} column(s) actually used "
          f"by FEATURE_HARMONIZATION_MAP:")
    for col in sorted(relevant_cols):
        n_null = df[col].isna().sum()
        pct_null = 100 * n_null / len(df) if len(df) > 0 else 0
        null_report[col] = pct_null
        flag = "FLAG" if pct_null > 20 else ("NOTE" if pct_null > 0 else "OK")
        print(f"    [{flag}] {col}: {n_null} null ({pct_null:.1f}%)")

    return null_report


def audit_sanity_ranges(df: pd.DataFrame, name: str, harmonization_map: dict):
    """Basic physical-plausibility checks: negative durations, negative
    byte/packet counts. These would indicate corrupted data, not just
    missing data."""
    if name not in harmonization_map:
        return

    checks_run = 0
    for feature in ("duration_mean", "byte_count", "packet_count"):
        spec = harmonization_map[name].get(feature, {})
        cols = spec.get("source_columns") or []
        for col in cols:
            if col not in df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            n_negative = (df[col] < 0).sum()
            checks_run += 1
            if n_negative > 0:
                pct = 100 * n_negative / len(df)
                print(f"  [FLAG] '{col}' has {n_negative} negative value(s) "
                      f"({pct:.2f}%) — physically implausible for a "
                      f"{feature.replace('_', ' ')}, worth investigating.")

    if checks_run == 0:
        print(f"  [SKIP] No numeric duration/byte/packet columns found to "
              f"range-check for this dataset.")
    else:
        print(f"  [OK] Ran {checks_run} range check(s), no negative values "
              f"reported above means all clean.")


def audit_dataset(name: str, filename: str):
    path = os.path.join(CLEANED_DATA_DIR, filename)
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")

    if not os.path.exists(path):
        print(f"  [MISSING] {path} does not exist — run this dataset's "
              f"loader first.")
        return None

    df = pd.read_parquet(path)
    n_original = len(df)
    print(f"  Shape (before dedup): {df.shape}")

    df = audit_and_dedupe(df, name)
    audit_near_duplicates(df, name)
    audit_nulls(df, name, FEATURE_HARMONIZATION_MAP)
    audit_sanity_ranges(df, name, FEATURE_HARMONIZATION_MAP)

    # Save back only if something was actually removed, so the fix
    # persists rather than only existing for this run.
    if len(df) != n_original:
        df.to_parquet(path, index=False)
        print(f"  Saved deduplicated data back to {path} "
              f"(new shape: {df.shape}).")

    return df


def audit_leakage_pairs(loaded_dfs: dict):
    """Concretely check the institutional-lineage pairs config.py already
    flags — not just accept the documented caveat, but actually look for
    overlapping IP addresses between the flagged dataset pairs."""
    print(f"\n{'=' * 70}\nLeakage pair checks (config.py's LEAKAGE_CHECK_PAIRS)\n{'=' * 70}")

    # Map dataset name -> which column holds destination/server IPs, for
    # a simple overlap check. Best-effort — schemas differ per dataset.
    IP_COLUMNS = {
        "unsw_nb15": ("srcip", "dstip"),
        "bot_iot": ("saddr", "daddr"),
        "ton_iot": ("id.orig_h", "id.resp_h"),
    }

    for name_a, name_b in LEAKAGE_CHECK_PAIRS:
        print(f"\n  Checking {name_a} <-> {name_b}:")
        if name_a not in loaded_dfs or name_b not in loaded_dfs:
            print(f"    [SKIP] One or both datasets not loaded/available.")
            continue
        if name_a not in IP_COLUMNS or name_b not in IP_COLUMNS:
            print(f"    [SKIP] IP column mapping not defined for this pair "
                  f"in this audit script.")
            continue

        df_a, df_b = loaded_dfs[name_a], loaded_dfs[name_b]
        src_a, dst_a = IP_COLUMNS[name_a]
        src_b, dst_b = IP_COLUMNS[name_b]

        ips_a = set(df_a[src_a].astype(str)) | set(df_a[dst_a].astype(str))
        ips_b = set(df_b[src_b].astype(str)) | set(df_b[dst_b].astype(str))
        overlap = ips_a & ips_b
        pct_of_b = 100 * len(overlap) / len(ips_b) if len(ips_b) > 0 else 0

        flag = "FLAG" if pct_of_b > 5 else "OK"
        print(f"    [{flag}] {len(overlap)} shared IP address(es) between "
              f"the two datasets ({pct_of_b:.1f}% of {name_b}'s unique IPs). "
              f"Note: private/testbed IP ranges (e.g. 192.168.x.x) commonly "
              f"overlap by coincidence across unrelated datasets — check "
              f"whether overlapping IPs are private/reused ranges before "
              f"treating this as concerning.")


if __name__ == "__main__":
    print("Running full data quality audit across all 7 cleaned datasets...")
    loaded = {}
    for name, (filename, is_source) in DATASET_FILES.items():
        df = audit_dataset(name, filename)
        if df is not None:
            loaded[name] = df

    audit_leakage_pairs(loaded)

    print(f"\n{'=' * 70}")
    print("Audit complete. Review FLAG items above before proceeding to "
          "windowing — NOTE items are informational, OK items need no action.")
    print(f"{'=' * 70}")
