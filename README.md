# Paper 2 — MT-SST Project

Separate from the Paper 1 codebase (`zero_day_project`) since the temporal
windowing pipeline is fundamentally different from Paper 1's per-flow
preprocessing. Design reference: `paper2_final_design.md` (keep a copy in
this folder — it's the source of truth for every decision below).

## Folder structure

```
zero_day_paper2_mtsst/
├── raw_data/              # place raw dataset files here (see below)
│   ├── cic_ids2017/       # SOURCE (pretraining)
│   ├── cse_cic_ids2018/   # SOURCE
│   ├── unsw_nb15/         # SOURCE
│   ├── ton_iot/           # SOURCE — benign traffic ONLY
│   ├── ugr16/             # SOURCE
│   ├── bot_iot/           # ZERO-DAY — attack-only, withheld entirely
│   └── cic_ton_iot/       # ZERO-DAY — withheld entirely
├── preprocessing/
│   ├── config.py          # ALL frozen design decisions live here
│   └── inspect_raw_columns.py
├── windowed_data/
│   ├── source/             # windowed sequences from source datasets
│   └── zero_day/           # windowed sequences from zero-day datasets
├── splits/                 # pretrain / source-validation / threshold-
│                            # calibration split manifests (session/day-level)
├── models/                  # trained MT-SST checkpoints (per variant)
├── outputs/                 # results, figures, master results file
├── notebooks/                # exploratory work
└── logs/
```

## Setup workflow

1. **Copy `paper2_final_design.md` into this folder** so `config.py` and
   the paper stay in sync as you implement.

2. **Place raw dataset files** into the matching `raw_data/<name>/`
   subfolder. Source datasets you already have from Paper 1
   (CIC-IDS2017, CSE-CIC-IDS2018, UNSW-NB15) can be copied or symlinked
   from `C:\MLProject\zero_day_project\`. ToN-IoT, UGR16, and CIC-ToN-IoT
   are new for Paper 2.

3. **Run column inspection**:
   ```
   cd preprocessing
   python inspect_raw_columns.py
   ```
   This prints the actual raw columns for whatever files you've placed,
   so `FEATURE_HARMONIZATION_MAP` in `config.py` can be filled in with
   real column names rather than guesses — same idea as the CIC2017 vs
   CIC2018 column-mapping exercise from Paper 1.

4. **Fill in `FEATURE_HARMONIZATION_MAP`** in `config.py` for each
   dataset, mapping the 13 `WINDOW_FEATURES` to real source columns.
   This directly produces the feature-mapping table required in
   Methodology Section 3.2 of the paper.

5. From there: temporal windowing → leakage-aware splitting → MT-SST
   pretraining. Not yet built — next steps once the harmonization map is
   filled in.

## Things `config.py` already encodes so they can't drift

- **BoT-IoT is attack-only** (`ZERO_DAY_HAS_BENIGN["bot_iot"] = False`) —
  don't compute AUC/F1/FPR for it anywhere in the pipeline.
- **Zero-day datasets never enter pretraining, validation, or threshold
  calibration** — enforced by keeping `SOURCE_DATASETS` and
  `ZERO_DAY_DATASETS` as separate lists that downstream code must respect.
- **Threshold calibration is global, source-benign-only** — not local
  per-target calibration. This is a resolved decision, not a placeholder.
- **ToN-IoT / CIC-ToN-IoT shared-origin leakage check** is flagged in
  `LEAKAGE_CHECK_PAIRS` — the disjointness check this implies still needs
  to be run once both datasets are in place.
- **Splits are session/day/scenario-level, not random window-level** —
  `SPLIT_FRACTIONS` apply to that grouping, not to individual windows.
