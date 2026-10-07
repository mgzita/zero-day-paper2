"""
=============================================================================
config.py — Paper 2 (MT-SST) project configuration
=============================================================================
Encodes the frozen design decisions from paper2_final_design.md so the
pipeline can't silently drift from the protocol. If you need to change
something here, check whether it also needs to change in the paper's
Methodology/Experimental Setup sections.
=============================================================================
"""

import os

# -----------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------
PREPROCESSING_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(PREPROCESSING_DIR, "..", "raw_data")
WINDOWED_DATA_DIR = os.path.join(PREPROCESSING_DIR, "..", "windowed_data")
CLEANED_DATA_DIR = os.path.join(PREPROCESSING_DIR, "..", "cleaned_data")
# CLEANED_DATA_DIR holds each dataset's output AFTER loading + benign
# filtering, BEFORE temporal windowing. This is the actual checkpoint of
# each loader's work — without saving here, re-running a loader script is
# the only way to get the cleaned data back, and nothing persists between
# sessions. Saved as parquet (preserves dtypes, much faster than CSV for
# these row counts).
SPLITS_DIR = os.path.join(PREPROCESSING_DIR, "..", "splits")
MODELS_DIR = os.path.join(PREPROCESSING_DIR, "..", "models")
OUTPUTS_DIR = os.path.join(PREPROCESSING_DIR, "..", "outputs")
LOGS_DIR = os.path.join(PREPROCESSING_DIR, "..", "logs")

# -----------------------------------------------------------------------
# Dataset roles — DO NOT mix these lists. Zero-day sets must never appear
# in pretraining, model selection, or detector fitting, at any stage.
# -----------------------------------------------------------------------
SOURCE_DATASETS = [
    "cic_ids2017",
    "ait_netflow",   # replaces cse_cic_ids2018 — see FEATURE_HARMONIZATION_MAP
                      # entry below for full rationale. Independent institution
                      # (Austrian Institute of Technology), retains IP addresses
                      # (unlike cse_cic_ids2018's published CSVs), and reduces
                      # the CIC-IDS2017/CSE-CIC-IDS2018 shared-lineage overlap
                      # that previously existed among source datasets.
    "unsw_nb15",
    "ton_iot",      # benign network flow traffic ONLY — never attack traffic
    "ugr16",        # benign traffic isolated where reliably identifiable
]

ZERO_DAY_DATASETS = [
    "bot_iot",       # ATTACK-ONLY — no benign traffic. Detection rate only.
    "edge_iiotset",  # Full metric suite. Ferrag et al. (2022, IEEE Access) —
                      # institutionally independent (Guelma, Algiers, De
                      # Montfort, TII), custom testbed, no CIC/UNSW/Granada
                      # lineage. Attack taxonomy includes explicit BACKDOOR
                      # and RECONNAISSANCE (port/OS/vulnerability scanning)
                      # categories — closer match to RQ1's low-and-slow
                      # target than TII-SSRC-23 (which lacked backdoor
                      # entirely). Still no explicit "worm" category — see
                      # Limitations. Second replacement after CIC-ToN-IoT
                      # (packet-level overlap with ton_iot) and TII-SSRC-23
                      # (no backdoor category, partial ISCX2012 benign reuse).
]

# Explicit flag: which zero-day sets have benign traffic at all.
# BoT-IoT does not — AUC/specificity/FPR/F1 are undefined for it.
ZERO_DAY_HAS_BENIGN = {
    "bot_iot": False,
    "edge_iiotset": True,
}

# Dataset pairs needing an explicit provenance/leakage check + caveat in the
# paper, regardless of verification outcome.
#   - ton_iot/cic_ton_iot pair REMOVED (Sep 2026) — CIC-ToN-IoT replaced.
#   - cic_ids2017-tii_ssrc23 / cse_cic_ids2018-tii_ssrc23 pairs REMOVED
#     (Sep 2026) — TII-SSRC-23 replaced by edge_iiotset, which has no known
#     benign-traffic reuse from any source-dataset lineage (custom testbed).
#
# INSTITUTIONAL-LINEAGE pairs (NOT literal packet overlap — different
# testbeds, different years, different tooling — but same lab/research
# group, so a model could partly learn "how this lab builds datasets"
# rather than genuine attack behavior). Both source datasets below and the
# zero-day BoT-IoT set were built by UNSW Canberra Cyber (Moustafa et al.):
LEAKAGE_CHECK_PAIRS = [
    ("unsw_nb15", "bot_iot"),   # same lab; different testbed/capture
    ("ton_iot", "bot_iot"),     # same lab; different testbed/capture, later
                                 # generation, different platform (NSX vCloud NFV)
]

# -----------------------------------------------------------------------
# Temporal windowing (Methodology Section 3.1 / Experimental Setup 4.2)
# -----------------------------------------------------------------------
WINDOW_DURATIONS_MINUTES = [1, 2, 3, 5]   # compared in window sensitivity analysis
DEFAULT_WINDOW_MINUTES = 5                # main experiments use this unless swept
SEQUENCE_LENGTH = 10                      # L — fixed across all experiments

# Aggregate window features (Methodology Section 3.1). Every dataset's raw
# columns must be mapped to exactly this feature set before windowing —
# see FEATURE_HARMONIZATION_MAP below, which MUST be filled in per dataset
# before running preprocessing on real data.
WINDOW_FEATURES = [
    "flow_count",
    "packet_count",
    "byte_count",
    "unique_dst_ips",
    "unique_dst_ports",
    "connection_rate",
    "failed_connection_rate",
    "duration_mean",
    "duration_variance",
    "byte_rate_variance",
    "packet_rate_variance",
    "dst_ip_entropy",
    "dst_port_entropy",
]

# -----------------------------------------------------------------------
# Feature harmonization map — PLACEHOLDER, must be completed per dataset
# before running real preprocessing (Methodology Section 3.2). Each entry
# maps a WINDOW_FEATURES name to the raw source column(s) and any
# unit/direction conversion needed for that specific dataset's export
# format (CICFlowMeter vs Bro/Zeek vs IoT-testbed exports).
#
# Structure: FEATURE_HARMONIZATION_MAP[dataset_name][window_feature] =
#     {"source_columns": [...], "notes": "..."}
#
# This dict is intentionally empty here — fill it in as you inspect each
# dataset's actual raw columns (see inspect_raw_columns.py).
# -----------------------------------------------------------------------
FEATURE_HARMONIZATION_MAP = {
    # ------------------------------------------------------------------
    # CIC-IDS2017 — clean mapping, all 13 WINDOW_FEATURES computable.
    # Source: corrected/regenerated distribution (Engelen et al. 2021),
    # which retains Flow ID / Src IP / Dst IP / Timestamp unlike the
    # original MachineLearningCSV.zip release.
    # ------------------------------------------------------------------
    "cic_ids2017": {
        "flow_count": {"source_columns": None, "notes": "1 row = 1 flow; count rows per window"},
        "packet_count": {"source_columns": ["Total Fwd Packet", "Total Bwd packets"], "notes": "sum both directions"},
        "byte_count": {"source_columns": ["Total Length of Fwd Packet", "Total Length of Bwd Packet"], "notes": "sum both directions"},
        "unique_dst_ips": {"source_columns": ["Dst IP"], "notes": "count distinct per window"},
        "unique_dst_ports": {"source_columns": ["Dst Port"], "notes": "count distinct per window"},
        "connection_rate": {"source_columns": None, "notes": "flow_count / window duration"},
        "failed_connection_rate": {
            "source_columns": ["RST Flag Count", "Total Bwd packets"],
            "notes": "APPROXIMATE — CICFlowMeter has no native connection-state field. "
                     "Heuristic: flag as failed if RST Flag Count > 0 OR Total Bwd packets == 0 "
                     "(no response received). This heuristic differs in kind from the native "
                     "conn_state/state fields available in ToN-IoT/UNSW-NB15/BoT-IoT — flag this "
                     "inconsistency explicitly in the paper's feature-harmonization table.",
        },
        "duration_mean": {"source_columns": ["Flow Duration"], "notes": "microseconds — convert to seconds for cross-dataset consistency"},
        "duration_variance": {"source_columns": ["Flow Duration"], "notes": "variance within window"},
        "byte_rate_variance": {"source_columns": ["Flow Bytes/s"], "notes": "variance within window"},
        "packet_rate_variance": {"source_columns": ["Flow Packets/s"], "notes": "variance within window"},
        "dst_ip_entropy": {"source_columns": ["Dst IP"], "notes": "Shannon entropy of dst IP distribution within window"},
        "dst_port_entropy": {"source_columns": ["Dst Port"], "notes": "Shannon entropy of dst port distribution within window"},
        "timestamp_column": "Timestamp",
    },

    # ------------------------------------------------------------------
    # AIT NetFlow — RESOLVED, replaces cse_cic_ids2018. Austrian Institute
    # of Technology, genuinely independent institution (not CIC/UNSW/
    # Granada/Guelma-Algiers-DMU lineage) — reduces the shared-lineage
    # overlap that previously existed between cic_ids2017 and
    # cse_cic_ids2018. Confirmed via direct inspection of tcp_complete.csv
    # (tstat-format output, 138 columns). All 13 WINDOW_FEATURES
    # computable, and native per-direction SYN/FIN/RST counts give a
    # MORE granular failed-connection signal than the single categorical
    # state/conn_state field other datasets provide.
    #
    # CRITICAL FILTERING STEP before use: this dataset labels attacker
    # hosts explicitly via role_cli/role_serv (e.g. 'attacker_0'). Since
    # this is a SOURCE (pretraining) dataset, filter OUT any row where
    # role_cli or role_serv starts with 'attacker_' before use — confirmed
    # necessary from the very first inspected row, which was attack traffic.
    #
    # Three files available: tcp_complete.csv, tcp_nocomplete.csv,
    # udp_complete.csv. Use tcp_complete + udp_complete (finished flows
    # with reliable duration/byte stats); tcp_nocomplete is still-open
    # connections at capture end and likely has unreliable duration.
    # ------------------------------------------------------------------
    "ait_netflow": {
        "flow_count": {"source_columns": None, "notes": "1 row = 1 flow; count rows per window"},
        "packet_count": {"source_columns": ["c_pkts_all:3", "s_pkts_all:17"], "notes": "sum client+server directions"},
        "byte_count": {"source_columns": ["c_bytes_all:9", "s_bytes_all:23"], "notes": "sum client+server directions"},
        "unique_dst_ips": {"source_columns": ["s_ip:15"], "notes": "count distinct per window; server IP treated as 'destination' (client initiates connection)"},
        "unique_dst_ports": {"source_columns": ["s_port:16"], "notes": "count distinct per window"},
        "connection_rate": {"source_columns": None, "notes": "flow_count / window duration"},
        "failed_connection_rate": {
            "source_columns": ["c_syn_cnt:13", "s_syn_cnt:27", "c_rst_cnt:4", "s_rst_cnt:18"],
            "notes": "NATIVE per-direction signal, more granular than other datasets' "
                     "categorical state field — treat as failed if RST count > 0 on either "
                     "side, or SYN sent (c_syn_cnt > 0) with no corresponding server SYN "
                     "(s_syn_cnt == 0), indicating an unanswered connection attempt. "
                     "CONFIRMED VIA ACTUAL LOADER RUN: these columns are TCP-only — UDP "
                     "(udp_complete.csv) has no SYN/RST concept at all, so after pooling "
                     "tcp_complete+udp_complete, UDP rows have NaN here. Handle explicitly: "
                     "UDP rows should be treated as 0/not-applicable for this feature "
                     "(UDP has no failed-handshake concept), NOT left as NaN, which would "
                     "corrupt window aggregates that mix TCP and UDP flows.",
        },
        "duration_mean": {"source_columns": ["durat:31"], "notes": "VERIFY UNITS before use — sample row shows durat=200.708 alongside first/last in epoch-milliseconds; confirm whether this is already seconds or needs conversion"},
        "duration_variance": {"source_columns": ["durat:31"], "notes": "variance within window"},
        "byte_rate_variance": {"source_columns": ["c_bytes_all:9", "s_bytes_all:23", "durat:31"], "notes": "derive rate, then variance within window"},
        "packet_rate_variance": {"source_columns": ["c_pkts_all:3", "s_pkts_all:17", "durat:31"], "notes": "derive rate, then variance within window"},
        "dst_ip_entropy": {"source_columns": ["s_ip:15"], "notes": "Shannon entropy of dst IP distribution within window"},
        "dst_port_entropy": {"source_columns": ["s_port:16"], "notes": "Shannon entropy of dst port distribution within window"},
        "timestamp_column": "timestamp",
        "benign_filter": "role_cli != 'attacker_*' AND role_serv != 'attacker_*' (exclude any row where either role starts with 'attacker_')",
    },

    # ------------------------------------------------------------------
    # UNSW-NB15 — clean mapping, all 13 WINDOW_FEATURES computable.
    # Uses Stime (flow start, unix epoch) as the timestamp for windowing.
    # ------------------------------------------------------------------
    "unsw_nb15": {
        "flow_count": {"source_columns": None, "notes": "1 row = 1 flow; count rows per window"},
        "packet_count": {"source_columns": ["Spkts", "Dpkts"], "notes": "sum both directions"},
        "byte_count": {"source_columns": ["sbytes", "dbytes"], "notes": "sum both directions"},
        "unique_dst_ips": {"source_columns": ["dstip"], "notes": "count distinct per window"},
        "unique_dst_ports": {"source_columns": ["dsport"], "notes": "count distinct per window"},
        "connection_rate": {"source_columns": None, "notes": "flow_count / window duration"},
        "failed_connection_rate": {
            "source_columns": ["state"],
            "notes": "native connection-state field — treat state in {'RST','INT'} as failed "
                     "(exact set to confirm against UNSW-NB15 documentation before finalizing)",
        },
        "duration_mean": {"source_columns": ["dur"], "notes": "already in seconds"},
        "duration_variance": {"source_columns": ["dur"], "notes": "variance within window"},
        "byte_rate_variance": {"source_columns": ["sbytes", "dbytes", "dur"], "notes": "derive rate = bytes/dur, then variance within window"},
        "packet_rate_variance": {"source_columns": ["Spkts", "Dpkts", "dur"], "notes": "derive rate = packets/dur, then variance within window"},
        "dst_ip_entropy": {"source_columns": ["dstip"], "notes": "Shannon entropy of dst IP distribution within window"},
        "dst_port_entropy": {"source_columns": ["dsport"], "notes": "Shannon entropy of dst port distribution within window"},
        "timestamp_column": "Stime",
    },

    # ------------------------------------------------------------------
    # ToN-IoT — RESOLVED. Uses per-scenario conn.csv (Zeek conn.log format,
    # confirmed against an actual uploaded sample) instead of the merged
    # train_test_network.csv, which drops the ts field entirely. All 9
    # scenario folders (normal_backdoor, normal_DDoS, ..., normal_XSS) use
    # this same conn.csv schema — pool and filter to conn rows only
    # (dns.csv/files.csv/http.csv/ssl.csv are other Zeek log types in the
    # same folders and are NOT used for these window features).
    # ------------------------------------------------------------------
    "ton_iot": {
        "flow_count": {"source_columns": None, "notes": "1 row = 1 flow; count rows per window"},
        "packet_count": {"source_columns": ["orig_pkts", "resp_pkts"], "notes": "sum both directions"},
        "byte_count": {"source_columns": ["orig_ip_bytes", "resp_ip_bytes"], "notes": "sum both directions; use *_ip_bytes (includes headers) for consistency with other datasets' byte counts, not orig_bytes/resp_bytes (payload only)"},
        "unique_dst_ips": {"source_columns": ["id.resp_h"], "notes": "count distinct per window"},
        "unique_dst_ports": {"source_columns": ["id.resp_p"], "notes": "count distinct per window"},
        "connection_rate": {"source_columns": None, "notes": "flow_count / window duration"},
        "failed_connection_rate": {
            "source_columns": ["conn_state"],
            "notes": "native Zeek connection-state field — treat conn_state in "
                     "{'S0','REJ','RSTO','RSTOS0','RSTRH','OTH'} as failed (confirm exact "
                     "Zeek state codes against documentation before finalizing; 'OTH' seen "
                     "in the sample file and worth checking whether it should count as failed "
                     "or as a distinct 'no clear start' category)",
        },
        "duration_mean": {"source_columns": ["duration"], "notes": "already in seconds; NOTE: '-' placeholder seen for connections with no duration (e.g. OTH state) — must be handled/imputed before aggregation"},
        "duration_variance": {"source_columns": ["duration"], "notes": "variance within window"},
        "byte_rate_variance": {"source_columns": ["orig_ip_bytes", "resp_ip_bytes", "duration"], "notes": "derive rate, then variance within window"},
        "packet_rate_variance": {"source_columns": ["orig_pkts", "resp_pkts", "duration"], "notes": "derive rate, then variance within window"},
        "dst_ip_entropy": {"source_columns": ["id.resp_h"], "notes": "Shannon entropy of dst IP distribution within window"},
        "dst_port_entropy": {"source_columns": ["id.resp_p"], "notes": "Shannon entropy of dst port distribution within window"},
        "timestamp_column": "ts",
    },

    # ------------------------------------------------------------------
    # UGR16 — clean mapping. Unidirectional/NetFlow-style (no fwd/bwd
    # split), so packet_count/byte_count are single fields, not sums.
    # ------------------------------------------------------------------
    "ugr16": {
        "flow_count": {"source_columns": None, "notes": "1 row = 1 flow; count rows per window"},
        "packet_count": {"source_columns": ["packets"], "notes": "unidirectional NetFlow — single field, no fwd/bwd split"},
        "byte_count": {"source_columns": ["bytes"], "notes": "unidirectional NetFlow — single field, no fwd/bwd split"},
        "unique_dst_ips": {"source_columns": ["dst_ip"], "notes": "count distinct per window"},
        "unique_dst_ports": {"source_columns": ["dst_port"], "notes": "count distinct per window"},
        "connection_rate": {"source_columns": None, "notes": "flow_count / window duration"},
        "failed_connection_rate": {
            "source_columns": ["flags"],
            "notes": "APPROXIMATE — no native connection-state field, only raw TCP flags string "
                     "(e.g. '.A....'). Heuristic: treat SYN-only (no ACK response pair observed) "
                     "as failed. Same category of approximation as cic_ids2017 (both lack a "
                     "native connection-state field, unlike unsw_nb15/ton_iot/bot_iot's "
                     "categorical state field, or ait_netflow's native per-direction SYN/RST counts).",
        },
        "duration_mean": {"source_columns": ["duration"], "notes": "already in seconds"},
        "duration_variance": {"source_columns": ["duration"], "notes": "variance within window"},
        "byte_rate_variance": {"source_columns": ["bytes", "duration"], "notes": "derive rate, then variance within window"},
        "packet_rate_variance": {"source_columns": ["packets", "duration"], "notes": "derive rate, then variance within window"},
        "dst_ip_entropy": {"source_columns": ["dst_ip"], "notes": "Shannon entropy of dst IP distribution within window"},
        "dst_port_entropy": {"source_columns": ["dst_port"], "notes": "Shannon entropy of dst port distribution within window"},
        "timestamp_column": "timestamp",
    },

    # ------------------------------------------------------------------
    # BoT-IoT — clean mapping, all 13 WINDOW_FEATURES computable.
    # ------------------------------------------------------------------
    "bot_iot": {
        "flow_count": {"source_columns": None, "notes": "1 row = 1 flow; count rows per window"},
        "packet_count": {"source_columns": ["spkts", "dpkts"], "notes": "sum both directions (or use pre-summed 'pkts')"},
        "byte_count": {"source_columns": ["sbytes", "dbytes"], "notes": "sum both directions (or use pre-summed 'bytes')"},
        "unique_dst_ips": {"source_columns": ["daddr"], "notes": "count distinct per window"},
        "unique_dst_ports": {"source_columns": ["dport"], "notes": "count distinct per window"},
        "connection_rate": {"source_columns": None, "notes": "flow_count / window duration"},
        "failed_connection_rate": {
            "source_columns": ["state"],
            "notes": "native Argus connection-state field — treat state in {'RST','INT'} as "
                     "failed (confirm exact Argus state codes before finalizing)",
        },
        "duration_mean": {"source_columns": ["dur"], "notes": "already in seconds"},
        "duration_variance": {"source_columns": ["dur"], "notes": "variance within window"},
        "byte_rate_variance": {"source_columns": ["rate"], "notes": "Argus provides a pre-computed rate field — verify definition matches bytes/sec before using directly"},
        "packet_rate_variance": {"source_columns": ["spkts", "dpkts", "dur"], "notes": "derive rate, then variance within window"},
        "dst_ip_entropy": {"source_columns": ["daddr"], "notes": "Shannon entropy of dst IP distribution within window"},
        "dst_port_entropy": {"source_columns": ["dport"], "notes": "Shannon entropy of dst port distribution within window"},
        "timestamp_column": "stime",
    },

    # ------------------------------------------------------------------
    # Edge-IIoTset — RESOLVED via option (b): packet-level data, no
    # pre-existing flow unit. flow_count is redefined as "unique 5-tuples
    # observed per window" rather than "count of completed flow objects"
    # (the definition used by the other 6 datasets). This is a genuine,
    # documented methodological difference — state it explicitly in the
    # paper's feature-harmonization table, not silently.
    #
    # duration_mean/duration_variance ARE computable despite being
    # packet-level: group packets by 5-tuple WITHIN each window, take
    # (max ts - min ts) per group as a duration proxy, then compute
    # mean/variance of that across all 5-tuple groups in the window. This
    # is an approximation (no real flow-timeout logic, no cross-window
    # flow continuity), but avoids leaving these two features blocked.
    # ------------------------------------------------------------------
    "edge_iiotset": {
        "flow_count": {
            "source_columns": ["ip.src_host", "ip.dst_host", "tcp.srcport", "tcp.dstport", "udp.port"],
            "notes": "REDEFINED for this dataset only: count of unique 5-tuples "
                     "(src IP, dst IP, src port, dst port, protocol) observed within "
                     "the window, NOT count of completed flow objects. State this "
                     "explicitly in the feature-harmonization table.",
        },
        "packet_count": {"source_columns": None, "notes": "= row count per window (each row is one packet)"},
        "byte_count": {"source_columns": ["tcp.len"], "notes": "sum per window; udp.* has no equivalent length field in this schema — verify whether UDP packet bytes are undercounted as a result"},
        "unique_dst_ips": {"source_columns": ["ip.dst_host"], "notes": "count distinct per window"},
        "unique_dst_ports": {"source_columns": ["tcp.dstport", "udp.port"], "notes": "combine both — only one populated per row depending on protocol"},
        "connection_rate": {"source_columns": None, "notes": "redefined flow_count (unique 5-tuples) / window duration"},
        "failed_connection_rate": {
            "source_columns": ["tcp.connection.rst", "tcp.connection.syn", "tcp.connection.synack"],
            "notes": "packet-level TCP connection flags are actually MORE granular here than "
                     "other datasets' flow-level approximations — a genuinely native "
                     "failed-connection signal, unlike the heuristics needed elsewhere",
        },
        "duration_mean": {
            "source_columns": ["frame.time", "ip.src_host", "ip.dst_host", "tcp.srcport", "tcp.dstport"],
            "notes": "APPROXIMATION: group packets by 5-tuple within the window, compute "
                     "(max frame.time - min frame.time) per group, take mean across groups. "
                     "Not a true flow duration (no timeout/termination logic) — document as "
                     "an approximation specific to this dataset.",
        },
        "duration_variance": {
            "source_columns": ["frame.time", "ip.src_host", "ip.dst_host", "tcp.srcport", "tcp.dstport"],
            "notes": "same approximation as duration_mean — variance across per-5-tuple durations within the window",
        },
        "byte_rate_variance": {"source_columns": ["tcp.len"], "notes": "derivable directly from packet rows within window"},
        "packet_rate_variance": {"source_columns": None, "notes": "derivable directly from packet rows within window"},
        "dst_ip_entropy": {"source_columns": ["ip.dst_host"], "notes": "Shannon entropy of dst IP distribution within window"},
        "dst_port_entropy": {"source_columns": ["tcp.dstport", "udp.port"], "notes": "Shannon entropy of dst port distribution within window"},
        "timestamp_column": "frame.time",
    },
}

# -----------------------------------------------------------------------
# Leakage-aware splitting (Methodology Section 3.3)
# Split fractions are applied at the session/day/scenario level, NEVER by
# randomly assigning individual windows (overlapping windows would leak).
# -----------------------------------------------------------------------
SPLIT_FRACTIONS = {
    "pretrain": 0.70,              # source datasets only
    "source_validation": 0.15,     # source datasets only — hyperparameter tuning
    "threshold_calibration": 0.15, # source datasets only — GLOBAL threshold,
                                     # never from zero-day/target data
}
# Zero-day datasets are held out ENTIRELY — no split needed, they are
# 100% test data, used exactly once for final reporting.

RANDOM_SEED = 42
N_SEEDS_FOR_CI = 5   # per-dataset 95% CI (t-distribution), matching Paper 1

# -----------------------------------------------------------------------
# Threshold calibration protocol — RESOLVED (see final design doc).
# Global, source-benign-only. Do not change this to local/target
# calibration without updating the paper's claims accordingly.
# -----------------------------------------------------------------------
THRESHOLD_CALIBRATION_SCOPE = "source_benign_only"  # NOT "target_local"
SCORE_NORMALIZATION_PERCENTILES = (1, 99)  # 1st/99th percentile bounds
SCORE_NORMALIZATION_EPSILON = 1e-8

# -----------------------------------------------------------------------
# Anomaly detectors (Methodology Section 3.6)
# -----------------------------------------------------------------------
PRIMARY_DETECTORS = ["deep_svdd", "oc_nn", "isolation_forest"]
SECONDARY_DETECTORS = ["mahalanobis", "energy_based", "lof", "oc_svm"]
ALL_DETECTORS = PRIMARY_DETECTORS + SECONDARY_DETECTORS  # 7 individual + fusion = 8

FUSION_TOP_K = 3  # number of top detectors (by source-validation AUC) used in fusion

# -----------------------------------------------------------------------
# Representation variants (Methodology Section 3.5)
# -----------------------------------------------------------------------
REPRESENTATION_VARIANTS = [
    "raw_features",
    "denoising_ae",
    "single_env_mtsst",
    "multi_env_mtsst",       # PRIMARY / headline model
    "multi_env_mtsst_coral",
    "multi_env_mtsst_mmd",
]

if __name__ == "__main__":
    print(f"Source datasets ({len(SOURCE_DATASETS)}): {SOURCE_DATASETS}")
    print(f"Zero-day datasets ({len(ZERO_DAY_DATASETS)}): {ZERO_DAY_DATASETS}")
    print(f"Window durations to sweep: {WINDOW_DURATIONS_MINUTES} min")
    print(f"Sequence length L: {SEQUENCE_LENGTH}")
    print("\nFeature harmonization map is EMPTY — fill in before real preprocessing.")
