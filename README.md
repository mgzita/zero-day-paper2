# Architecture-by-Target Mechanisms of Training-Environment Diversity in Unsupervised Anomaly Detection

Code for:

M. Zita, M. U. Rehman, M. Abrar, M. Kazim, and S. Khalid. *Architecture-by-Target Mechanisms of Training-Environment Diversity in Unsupervised Anomaly Detection.* Manuscript in preparation, De Montfort University, 2026.

This study follows [zero-day-ids-ensemble](https://github.com/mgzita/zero-day-ids-ensemble). Here we ask whether training on more than one environment helps unsupervised anomaly detectors on an unseen target, and how the effect varies. Three kinds of evidence are compared: feature attribution (KernelSHAP and Integrated Gradients), functional feature ablation, and representation-level distance (MMD² and sliced Wasserstein).

**Release:** tag `v1.0-paper2`, commit `9898cb7`. The run manifest is `RUN_MANIFEST.json` in the repository root.

## Design

| Element | Setting |
|---|---|
| Models | Autoencoder (AE), Denoising Autoencoder (DAE), Deep SVDD, Normalizing Flow |
| Target domains (never used for training) | UNSW-NB15, IEC-104 (industrial control traffic), SMD (server telemetry, native 38-feature space) |
| Training conditions | Single-A (source A only), Single-B (source B only), Diverse (A and B, twice the data), Mixed-matched (A and B, size-matched to one source) |
| Network training sources | A: CIC-IDS2017; B: CSE-CIC-IDS2018 |
| SMD sources | A: Group 1; B: Group 3; target: Group 2 |
| Seeds | Ten per model and target combination |
| Primary metric | Raw AUROC; AUPRC and recall at 1% and 5% FPR are also reported. |
| Statistics | Paired t-test (Wilcoxon signed-rank as sensitivity check), paired Cohen's d, Benjamini-Hochberg FDR over 48 confirmatory tests |

## Key results

- Two-source training did not beat the best single source in any model and target cell. Its apparent benefit on SMD comes from a weak source A.
- 12 of 48 confirmatory tests survive joint FDR correction: 7 functional-ablation (raw AUROC), 3 KernelSHAP and 2 representation (MMD²). None of the 12 Integrated Gradients comparisons is significant.
- The effect of feature-group ablation depends on both model and target (model x target interaction F(6, 79) = 15.5 on raw AUROC). Only the AE reverses direction across targets on both raw AUROC and |AUROC - 0.5|; other apparent reversals appear only under |AUROC - 0.5|, which hides inverted scores.
- Many full models score below chance on at least one target.
- Feature groups are built from unlabelled feature distributions of all three environments, including the target, so the mechanism analyses are not fully target-independent (see the paper, Section 3.3).

## Datasets

Place raw files under `raw_data/` (not tracked):

- CIC-IDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
- CSE-CIC-IDS2018: https://www.unb.ca/cic/datasets/ids-2018.html
- UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset
- IEC-104: see the dataset reference in the paper
- SMD (Server Machine Dataset): https://github.com/NetManAIOps/OmniAnomaly

Paths and frozen design settings are in `preprocessing/config.py`.

## Setup

```
python -m venv .venv
.venv\Scripts\activate          # Windows; use source .venv/bin/activate on Linux or macOS
pip install -r requirements.txt
```

## Reproducing the paper

Run all commands from the repository root.

### 1. Preprocessing

| Step | Script |
|---|---|
| Dataset loaders | `preprocessing/loaders/*.py` |
| IEC-104 and SMD harmonization | `preprocessing/harmonize_iec104.py`, `preprocessing/harmonize_smd.py` |
| Final preprocessing and audit | `preprocessing/finalize_preprocessing.py`, `preprocessing/audit_cleaned_data.py` |

### 2. Training and full pipelines

| Target | AE, DAE, Deep SVDD | Normalizing Flow |
|---|---|---|
| UNSW-NB15 | `train_ae_dae_lodo3_checkpoints.py`, `extend_ae_dae_svdd_unsw_h1_repr.py` | `train_normalizing_flow.py`, `flow_full_pipeline.py` |
| IEC-104 | `ae_dae_svdd_pipeline_iec104.py` | `flow_full_pipeline_iec104.py` |
| SMD | `ae_dae_svdd_pipeline_smd.py` | `flow_full_pipeline_smd.py` |

### 3. Analyses

| Paper result | Script |
|---|---|
| Single-B and Mixed-matched baselines | `single_source_baselines_v3.py` |
| AUROC, AUPRC, recall at fixed FPR; permutation | `analyze_eval_extra.py`, `paper2_followup_tests.py` |
| Integrated Gradients attribution | `compute_ig_attribution_all.py`, `analyze_ig_results.py` |
| SHAP versus IG, paired per seed | `shap_vs_ig_paired.py`, `shap_ig_diff.py` |
| Attribution tables | `generate_h1_latex_tables.py` |
| Functional ablation | `ablation_h3_lodo3.py`, `ablation_h3_lodo3_extra_seeds.py`, `combine_h3_lodo3_seeds.py` |
| Raw-AUROC ablation and interaction | `rerun_h3_save_raw.py`, `h3_interaction_raw_auroc.py` (output `h3_interaction_raw_auroc.json`) |
| Decorrelated feature groups | `ablation_h4_v2_decorrelated.py` |
| SMD group sensitivity | `smd_ess_sensitivity_sweep.py` |
| Representation distance (MMD²) | `compute_flow_representation_stats.py` |
| Representation distance (sliced Wasserstein) | `compute_flow_representation_stats_wasserstein.py`, `wasserstein_fdr_all_cells.py`, `representation_interaction_wasserstein.py` |
| MMD² bandwidth artifact diagnosis | `flow_iec104_mmd_bandwidth_sweep.py`, `flow_iec104_repr_diagnostic.py` |
| SMD statistics | `analyze_smd_results.py`, `analyze_flow_smd_results.py` |
| Fallback scorer usage | `fallback_usage_tally.py` |
| Combine all results | `final_combine_all_results.py` |
| Check every number in the manuscript | `verify_all_manuscript_numbers.py` |
| Run manifest (versions, commit, hashes, seeds) | `make_run_manifest.py` (writes `RUN_MANIFEST.json`) |

## License

MIT License, see `LICENSE`.

## Contact

Margaret Zita, De Montfort University, Leicester, United Kingdom
