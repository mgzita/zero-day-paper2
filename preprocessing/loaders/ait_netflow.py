"""
=============================================================================
loaders/ait_netflow.py — load and clean AIT NetFlow source data

Handles:
  1. tstat's comment-marker header artifact (columns like '#15#c_ip:1'
     instead of clean 'c_ip:1') — this is tstat writing its header as a
     commented line (e.g. '#15# c_ip ...'), which pandas reads literally
     when the file has no true header row it recognizes as such.
  2. Filtering to benign-only rows via role_cli/role_serv, required
     because this is a SOURCE (pretraining) dataset per config.py.
  3. Loading both tcp_complete.csv and udp_complete.csv (NOT
     tcp_nocomplete.csv — still-open connections at capture end, unreliable
     duration/byte stats).
=============================================================================
"""

import os
import re
import glob
import pandas as pd


def _clean_column_name(col: str) -> str:
    """Strip tstat's leading comment-marker artifact from a column name.
    Two formats observed in practice:
      - TCP files: '#15#c_ip:1' -> 'c_ip:1'  (number between two '#'s)
      - UDP files: '#c_ip:1'    -> 'c_ip:1'  (single leading '#', no number)
    Leaves already-clean column names (e.g. 'c_port:2') untouched.
    """
    return re.sub(r"^#\d*#?", "", col)


def load_ait_netflow_file(path: str) -> pd.DataFrame:
    """Load a single AIT NetFlow CSV (tcp_complete.csv or udp_complete.csv),
    cleaning the tstat header artifact and filtering to benign rows only.
    """
    df = pd.read_csv(path, low_memory=False)

    # Fix 1: strip tstat's '#15#' comment-marker artifact from column names
    df.columns = [_clean_column_name(c) for c in df.columns]

    # Sanity check: confirm the fix actually worked before proceeding —
    # fail loudly rather than silently continuing with a malformed schema.
    if any(col.startswith("#") for col in df.columns):
        bad_cols = [c for c in df.columns if c.startswith("#")]
        raise ValueError(
            f"Column name cleanup failed for {path} — still has "
            f"comment-marker artifacts: {bad_cols}. Check the regex "
            f"pattern against this file's actual header format."
        )

    # Tag protocol explicitly — needed because TCP and UDP files have
    # different schemas (UDP has no SYN/RST/handshake columns at all),
    # and pooling them via pd.concat fills the gap with NaN. Downstream
    # code needs to know which rows are which protocol to handle this
    # correctly rather than silently propagating NaN into aggregates.
    is_udp = "udp" in os.path.basename(path).lower()
    df["_protocol_source"] = "udp" if is_udp else "tcp"

    # Fix 3: UDP has no SYN/RST concept (connectionless) — these columns
    # will be NaN after pooling with TCP. Set them to 0 explicitly for UDP
    # rows so failed_connection_rate treats UDP as "not applicable / never
    # a failed handshake" rather than leaving NaN to corrupt window
    # aggregates that mix TCP and UDP flows.
    tcp_only_conn_cols = ["c_syn_cnt:13", "s_syn_cnt:27", "c_rst_cnt:4", "s_rst_cnt:18"]
    if is_udp:
        for col in tcp_only_conn_cols:
            if col in df.columns:
                df[col] = 0
            else:
                # Column may not exist at all in UDP file's raw schema —
                # create it as 0 so downstream code can rely on it existing.
                df[col] = 0

    # Fix 2: filter to benign rows only (SOURCE dataset — no attack traffic
    # allowed at any stage, per config.py's SOURCE_DATASETS rule).
    n_before = len(df)
    is_attack = (
        df["role_cli"].astype(str).str.startswith("attacker_")
        | df["role_serv"].astype(str).str.startswith("attacker_")
    )
    df = df[~is_attack].copy()
    n_after = len(df)
    n_removed = n_before - n_after

    print(f"  {os.path.basename(path)}: {n_before} rows -> {n_after} rows "
          f"benign ({n_removed} attack rows removed, "
          f"{100 * n_removed / n_before:.1f}%)")

    return df


def load_ait_netflow(raw_data_dir: str) -> pd.DataFrame:
    """Load and pool AIT NetFlow's tcp_complete + udp_complete files
    (benign-filtered), across all available testbed subfolders/files.

    Deliberately EXCLUDES tcp_nocomplete.csv per config.py's documented
    rationale.
    """
    target_files = []
    for pattern in ("*tcp_complete*.csv", "*udp_complete*.csv"):
        target_files.extend(
            glob.glob(os.path.join(raw_data_dir, "**", pattern), recursive=True)
        )

    if not target_files:
        raise FileNotFoundError(
            f"No tcp_complete/udp_complete CSV files found under {raw_data_dir}. "
            f"Check that files were extracted and placed correctly."
        )

    print(f"Loading {len(target_files)} AIT NetFlow file(s):")
    frames = [load_ait_netflow_file(f) for f in sorted(target_files)]
    combined = pd.concat(frames, ignore_index=True)

    print(f"\nTotal pooled benign rows: {len(combined)}")
    return combined


if __name__ == "__main__":
    # Quick standalone test — adjust path as needed
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from config import CLEANED_DATA_DIR

    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "../raw_data/ait_netflow"
    df = load_ait_netflow(raw_dir)
    print(f"\nFinal shape: {df.shape}")
    print(f"Columns: {list(df.columns)[:10]} ... ({len(df.columns)} total)")

    # SAVE THE OUTPUT — without this, the cleaned data only exists in
    # memory for the duration of this script run and is lost afterward.
    # Uses CLEANED_DATA_DIR from config.py (project root's cleaned_data/,
    # alongside raw_data/) rather than recomputing the path here — a
    # previous version of this script incorrectly saved into
    # preprocessing/cleaned_data/ instead, one directory level off.
    os.makedirs(CLEANED_DATA_DIR, exist_ok=True)
    output_path = os.path.join(CLEANED_DATA_DIR, "ait_netflow_benign.parquet")
    df.to_parquet(output_path, index=False)
    print(f"\nSaved cleaned data to: {os.path.abspath(output_path)}")
