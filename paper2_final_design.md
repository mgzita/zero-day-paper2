# Paper 2 — Final Design Summary

**Title (finalized):** Self-Supervised Domain-Invariant Temporal
Representation Learning for Cross-Domain Zero-Day Intrusion Detection

*Decided after four external review rounds — this framing emphasizes
self-supervision, invariance, and temporal representation as the paper's
scientific contribution, without making "Transformer" the headline claim,
consistent with the "lead with the finding, not the architecture" guidance
above.*

**Research narrative:** Paper 1 showed generalization does not naturally emerge
from combined multi-environment benign training. Paper 2 tests whether it can
be achieved through temporal, self-supervised, multi-environment representation
learning — and whether a good representation reduces the need for the kind of
ensemble fusion Paper 1 relied on for robustness.

**Framing the contribution (per external review)**: lead with the scientific
finding, not the architecture. None of Transformer, self-supervised
objectives, or multi-domain pretraining are individually novel — the
contribution is *"multi-environment self-supervised learning produces
measurable representation invariance, which predicts zero-day robustness,"*
not *"we built a transformer for IDS."* This framing should be explicit in
the abstract and introduction, not left implicit.

---

## ⚠️ One open decision, unresolved from the roadmap discussion

Your original Paper1/Paper2/Thesis component table marked **Self-Supervised** as
Thesis-only (X for Paper 2). Everything below treats MT-SST as fully
self-supervised in Paper 2, per how the design has evolved through this
conversation. If you want to keep the original roadmap intact, this needs
revisiting before implementation — and the Thesis will need re-scoping either
way, since Paper 2 now also touches "partial adaptive fusion" (as a comparison,
not a committed pipeline stage — see below).

---

## 1. Architecture

**Primary encoder:** Transformer encoder $g_\phi$, shared across all source
environments, with three self-supervised pretraining heads:
- Masked window prediction ($\mathcal{L}_{\mathrm{mw}}$)
- Masked feature prediction ($\mathcal{L}_{\mathrm{mf}}$)
- Next-window prediction ($\mathcal{L}_{\mathrm{nw}}$)

Joint objective: $\mathcal{L} = \mathcal{L}_{\mathrm{mw}} + \lambda_{\mathrm{mf}}
\mathcal{L}_{\mathrm{mf}} + \lambda_{\mathrm{nw}} \mathcal{L}_{\mathrm{nw}}$,
optimized over environment-balanced batches (no explicit MI-decorrelation term
— invariance is induced implicitly via balanced multi-environment batching).

**No secondary encoder ablation.** Mamba was considered and dropped — both
external reviews flagged it as a risk of making the paper look like it has two
unrelated architecture claims, without meaningfully strengthening the central
domain-invariance argument. The Transformer encoder is the paper's only
architecture claim.

## 2. Temporal Windowing

- Window duration $\Delta t \in \{1,2,3,5\}$ min (compared in window
  sensitivity analysis)
- Sequence length $L=10$ (fixed across all experiments)
- Features standardized **per-environment**, using only that environment's
  benign training windows
- **Feature harmonization table required**: explicit mapping showing how each
  aggregate feature (flow count, packet count, unique dest IPs/ports,
  connection/failed-connection rate, duration stats, byte/packet-rate
  variance, entropy measures) is computed identically across all 5 source +
  2 zero-day environments, despite differing raw capture formats. Direct
  response to Paper 1 feedback on feature harmonization.
- **CSE-CIC-IDS2018 replaced with AIT NetFlow**: the accepted-limitation
  approach (documenting a missing-IP gap) was superseded once a better
  option was found. Neither the Kaggle mirror nor the official AWS-hosted
  release of CSE-CIC-IDS2018 includes source/destination IP columns
  (confirmed by direct inspection of both), and re-running CICFlowMeter on
  the raw PCAPs (~477GB across 10 days) was rejected as disproportionate
  cost. Instead, CSE-CIC-IDS2018 is replaced by **AIT NetFlow** (Austrian
  Institute of Technology, derived from AIT-LDS-v2.0), confirmed via direct
  inspection to retain IP addresses (`c_ip`/`s_ip`), timestamps, and native
  per-direction SYN/FIN/RST counts — a more granular failed-connection
  signal than the categorical state fields other datasets provide. This
  also resolves the CIC-IDS2017/CSE-CIC-IDS2018 shared-lineage overlap
  (both were CIC/UNB products) by introducing a genuinely independent
  institution instead. One required preprocessing step: AIT NetFlow labels
  attacker hosts explicitly via `role_cli`/`role_serv` (e.g. `attacker_0`)
  — since this is a source (pretraining) dataset, any row where either role
  starts with `attacker_` must be filtered out before use.
- **Session/day/scenario-based splits, not random window-level splits** —
  overlapping windows carry temporal correlation; random splitting would leak
  train/test information across adjacent windows. This is a sharper version of
  the leakage issue Paper 1's feedback caught at the flow level.

## 3. Multi-Environment Pretraining Protocol

- Source (pretraining): CIC-IDS2017, AIT NetFlow, UNSW-NB15, ToN-IoT
  (benign only), UGR16
- Zero-day (withheld entirely): BoT-IoT, Edge-IIoTset
- **Edge-IIoTset replaces the original CIC-ToN-IoT choice** (via an
  intermediate TII-SSRC-23 candidate, also rejected). History: CIC-ToN-IoT
  was dropped after discovering it shares literal raw pcap files with
  ToN-IoT, a source dataset — the same underlying packet captures
  re-extracted with a different feature tool, not just shared testbed
  origin. TII-SSRC-23 was then considered but rejected because its attack
  taxonomy (DoS, bruteforce, information gathering, Mirai botnet) has no
  backdoor category, and its benign traffic partially reuses ISCX 2012 data
  (a weak provenance link to CIC-IDS2017, which was a source dataset at
  the time this decision was made — noted here for historical accuracy
  even though CSE-CIC-IDS2018 has since been replaced by AIT NetFlow).
  **Edge-IIoTset**
  (Ferrag, Friha, Hamouda, Maglaras, Janicke; IEEE Access, 2022) resolves
  both issues: institutionally independent (University of Guelma, University
  of Algiers, De Montfort University, TII — no CIC/UNSW/Granada lineage),
  custom-built testbed with no known benign-traffic reuse from any source
  dataset, and its attack taxonomy explicitly includes **Backdoor** and
  **Reconnaissance** (port scanning, OS fingerprinting, vulnerability
  scanning) categories, alongside DDoS, MITM, injection, password cracking,
  and ransomware. Still no explicit "worm" category — worm-as-category is
  largely absent from post-2020 IDS datasets generally (see Limitations) —
  so RQ1's low-and-slow claim covers backdoors and reconnaissance/scanning,
  not worms specifically.
- Hyperparameter tuning (mask ratios, loss weights, architecture) uses only a
  held-out **source**-environment validation split. No OOD data touches model
  selection at any point.

## 4. Downstream Anomaly Detection

**Eight detectors evaluated (all in the experimental pipeline), presented as
primary + secondary to keep the main narrative focused** — per external
review: too many detectors in full prose risks reading as benchmarking rather
than scientific investigation. All eight are still run and reported (RQ3
needs the full comparison), but the write-up foregrounds four:

**Primary table (main text):**
1. Deep SVDD — modern deep one-class learning
2. One-Class Neural Network (OC-NN) — directly answers Paper 1 feedback's
   request for a modern deep one-class baseline
3. Isolation Forest — strong classical baseline
4. Fusion (best single detector vs. equal-weight vs. learned-weight) —
   connects directly to Paper 1's central ensemble-fusion finding

These four carry the paper's scientific story on their own.

**Secondary table (supplementary/appendix):**
5. Mahalanobis Distance
6. Energy-based detection
7. Local Outlier Factor (LOF)
8. One-Class SVM (OC-SVM)

Reported as "detector robustness" evidence — same protocol, same embeddings,
lower narrative weight.

**Excluded, with justification stated in text:** Deep SAD-style models —
require partial attack labels during training, incompatible with this paper's
fully unsupervised, zero-label protocol. Named explicitly rather than silently
dropped.

**Fusion — framed as a comparison, not a committed pipeline stage:**
Added as extra rows in the same RQ3 comparison table, not a new pipeline
component:
- Best single detector (from the 7-way comparison)
- Equal-weight fusion of top-$k$ detectors (naive baseline — tests whether
  *any* fusion helps)
- Learned-weight fusion of top-$k$ detectors, weights selected via
  worst-domain **source-validation** AUC (never the zero-day test sets),
  using the same percentile-bound + $\epsilon$ score normalization established
  in Paper 1's revision

This directly answers Paper 1's reviewer criticism that its (0.4, 0.4, 0.2)
weights were subjectively chosen. Both possible outcomes are useful:
fusion wins → report as an add-on; fusion doesn't win → stronger narrative
("a good representation makes ensemble fusion unnecessary — Paper 1 needed
fusion because its representation was weaker").

## 5. Baselines (separate from the 7-detector ablation — different question)

The detector ablation asks *which scorer works best on MT-SST's embedding*.
Baselines ask *does MT-SST's embedding beat other full pipelines*. Kept
distinct per our discussion.

| Baseline | Isolates |
|---|---|
| Paper 1's AE–VAE–Deep SVDD ensemble (flow-level) | Temporal windowing vs. flow-level |
| Denoising Autoencoder on temporal windows (reconstruction only, no pretext tasks, no multi-env joint objective) | Self-supervised pretext objectives vs. plain reconstruction |
| Single-environment MT-SST (same architecture, one dataset) | Multi-environment training vs. architecture — extends naturally into the environment-count ablation below |
| Deep SVDD trained end-to-end on flow-level features (no windowing, no self-supervision) | A modern deep one-class baseline, matching Paper 1 feedback's explicit request |
| PPT-GNN / TCG-IDS (reported numbers, with explicit protocol-difference caveat) | Positioning against closest published work; full reproduction optional given cost |

## 6. Data-Diversity-vs-Architecture Ablation (new, directly from Paper 1 feedback)

Same encoder architecture, trained under 1, 2, 3, and 5 source environments
(plus existing LODO), to cleanly separate "does more diverse training data
help" from "does the architecture/self-supervision help" — the exact
confound Paper 1's feedback flagged in its single-env-AE-vs-multi-env-ensemble
comparison.

## 7. Evaluation Protocol (direct fixes from Paper 1 feedback)

- **BoT-IoT is attack-only** — report detection rate only; AUC/specificity/FPR
  undefined there. Do not report F1=1.0 as "perfect detection."
- **Edge-IIoTset and all other datasets with both classes**: AUC, PR-AUC,
  Precision, Recall, F1, Balanced Accuracy, FPR, FNR — not AUC/Recall/F1/FNR
  alone. PR-AUC specifically required given class imbalance.
- **Explicit split table per environment**: train / source-validation /
  threshold-calibration / test, stated plainly, with a note on whether attack
  data influenced any design decision.
- **Threshold protocol — resolved**: global threshold calibrated only from
  source-benign validation data (never target-environment benign traffic).
  Three independent review rounds converged on this recommendation: it gives
  the strongest zero-shot claim, and reviewers generally weight methodological
  rigor over marginally higher numbers from local calibration. Expect this to
  produce somewhat lower absolute performance on the zero-day environments
  than local calibration would — report that honestly rather than switching
  protocols to chase a better number.
- **Per-attack-family results** (esp. low-and-slow: backdoors, reconnaissance,
  worms, scanning) with sample counts and CIs — avoid overinterpreting 100%
  recall on tiny classes, per Paper 1 feedback.
- **False alarms per 1,000/10,000 benign flows** as a practical-deployment
  metric alongside FPR.
- **All numbers generated from one master results file** — abstract, tables,
  figures, discussion, conclusion all pull from the same source, avoiding
  Paper 1's cross-section numerical mismatches.
- **Statistical significance test across LODO domains**: Wilcoxon
  signed-rank test for pairwise comparisons (e.g., MT-SST vs. Paper 1
  ensemble), or a Friedman test with Nemenyi post-hoc analysis if comparing
  more than two methods across the LODO domains simultaneously — standard
  practice for multi-dataset method comparison, and demonstrates observed
  improvements aren't due to random variation. Complements, not replaces,
  the per-dataset confidence intervals already planned.

## 8. Representation Invariance & Embedding Quality Analysis (new)

Added in response to external review: downstream detection performance is
*indirect* evidence of domain invariance. A paper whose central claim is
"domain-invariant representation" needs to measure that directly, not just
infer it from AUC.

**Presented as a chronological causal chain, not a loose collection of
analyses** — per external review, this reads as *learning → representation
change → generalization improvement*, which is easier for a reviewer to
follow than analyses presented out of order:

1. **Raw features** (baseline: no learning applied)
2. **Self-supervised MT-SST embeddings** (after multi-environment pretraining;
   implicit invariance variant, and explicit CORAL/MMD variants, all compared
   at this step)
3. **Linear probe domain classifier**: train a simple classifier to predict
   which source environment a benign window sequence came from, using only
   its embedding. *Lower* accuracy (closer to chance across 5 classes)
   indicates *greater* invariance. Run on both (1) and (2) to show
   pretraining's effect directly.
4. **MMD / CORAL distribution alignment**: Maximum Mean Discrepancy and/or
   CORAL distance between environment-wise embedding distributions, again
   on both (1) and (2).
5. **Embedding visualization**: t-SNE / UMAP of benign embeddings across all
   5 source environments, before vs. after pretraining — visual evidence
   environments overlap rather than cluster separately post-training.
6. **Anomaly detectors** (Section 4): the 7-detector + fusion comparison,
   now understood as operating on a representation whose invariance has
   already been independently measured in steps 3–5.
7. **Zero-day performance** (Section 7): the payoff step — connecting back to
   steps 3–4 via the signature "domain classifier accuracy vs. zero-day AUC"
   table (Experiment 8, Section 10).

**Explicit invariance-enforcement ablation** — added in response to external
review's strongest anticipated criticism: measuring invariance is not the
same as *causing* it, and a skeptical reviewer can argue any gain comes from
more data exposure rather than genuine invariance. Rather than defending this
in prose, it becomes another controlled comparison, consistent with the
paper's whole design philosophy:

- **MT-SST + CORAL**: $\mathcal{L} = \mathcal{L}_{\mathrm{SSL}} +
  \lambda_{\mathrm{coral}} \mathcal{L}_{\mathrm{CORAL}}$, covariance
  alignment term added to the existing joint pretraining objective, computed
  over the same environment-balanced batches already in use.
- **MT-SST + MMD**: $\mathcal{L} = \mathcal{L}_{\mathrm{SSL}} +
  \lambda_{\mathrm{mmd}} \mathcal{L}_{\mathrm{MMD}}$, same batching.

Both are cheap additions — no adversarial minimax training loop, no new
infrastructure, same batches and evaluation pipeline as the base model.
**MT-SST remains the headline contribution**; CORAL/MMD variants are framed
explicitly as an "explicit invariance enforcement ablation," not a
competing architecture.

The full representation comparison for Section 8 becomes: Raw Features → AE
→ Single-Env MT-SST → Multi-Env MT-SST (implicit invariance) → Multi-Env
MT-SST + CORAL → Multi-Env MT-SST + MMD.

Every possible outcome of this comparison is publishable — the point is not
to bet on one result:

| Observation | Interpretation |
|---|---|
| Multi-Env MT-SST > Single-Env MT-SST | Data diversity helps |
| Multi-Env MT-SST reduces domain-classifier accuracy vs. raw/AE | Implicit invariance emerges from SSL alone |
| CORAL/MMD further reduces domain accuracy but does *not* improve zero-day AUC | Explicit alignment is unnecessary — SSL alone suffices |
| CORAL/MMD *does* improve zero-day AUC beyond implicit MT-SST | Invariance is causal, not just correlated — explicit alignment adds real value |

A reviewer cannot attack either outcome, because the ablation is designed to
be informative regardless of which way it lands.

**Embedding quality analysis** (detector-agnostic, closes the loop on RQ2/RQ3
separation of concerns): linear probe accuracy for benign/attack
separability on labeled source-validation data, plus k-NN retrieval or
clustering purity as an alternative quality signal.

This chain is what actually justifies calling the representation
"domain-invariant" in the paper's title and abstract, rather than inferring it
from downstream numbers alone.

## 9. Research Questions (final)

- **RQ1:** Does temporal windowing improve low-and-slow attack detection over
  flow-level representations?
- **RQ2:** Does self-supervised multi-environment pretraining produce
  measurably more domain-invariant representations (Section 8) than
  flow-level or single-environment training; is any generalization gain
  attributable to data diversity, architecture, or both (Section 6
  ablation); and does *explicit* invariance enforcement (CORAL/MMD) improve
  on the *implicit* invariance that emerges from self-supervision alone?
- **RQ3:** Which anomaly-scoring approach — single detector, equal-weight
  fusion, or learned-weight fusion — performs best on MT-SST embeddings, and
  does fusion meaningfully outperform the best single detector?
- **RQ4:** Do domain-invariant temporal representations reduce the
  threshold-transfer failures identified in Paper 1's LODO evaluation?

## 10. Full Experiment List

1. Main performance evaluation (source environments, held-out benign/attack)
2. Zero-day evaluation (BoT-IoT — detection rate only; Edge-IIoTset — full
   metric suite)
3. LODO validation
4. Detector + fusion comparison (RQ3, 8 detectors + 2 fusion variants;
   Deep SVDD/OC-NN/Isolation Forest/Fusion reported as primary results,
   Mahalanobis/Energy/LOF/OC-SVM as secondary/supplementary)
5. Temporal window sensitivity ($\Delta t \in \{1,2,3,5\}$ min)
6. Data-diversity-vs-architecture ablation (1/2/3/5 source environments)
7. Baseline comparison (Paper 1 ensemble, denoising AE, single-env MT-SST,
   flow-level Deep SVDD, PPT-GNN/TCG-IDS reported numbers)
8. Representation invariance & embedding quality analysis (linear probe
   domain classifier, t-SNE/UMAP, MMD/CORAL, benign/attack linear probe,
   k-NN retrieval or clustering purity) across Raw Features → AE →
   Single-Env MT-SST → Multi-Env MT-SST → Multi-Env MT-SST+CORAL →
   Multi-Env MT-SST+MMD — Section 8, directly evidencing the paper's central
   "domain-invariant" claim and pre-empting the "invariance is measured but
   never optimized" criticism. Include the "domain classifier accuracy vs.
   zero-day AUC" table across all six representations as the paper's
   signature result.

## 11. Limitations (to state explicitly in the paper)

- The primary MT-SST model induces invariance through balanced
  multi-environment self-supervised learning rather than adversarial domain
  confusion or information-theoretic regularization. Explicit alignment
  variants (CORAL and MMD, Section 8) are evaluated as ablations to test
  whether explicit enforcement adds anything beyond this, but they are not
  the paper's primary method — MT-SST (implicit) remains the headline
  contribution regardless of which variant wins empirically.
- Only tabular temporal aggregate features are considered, not raw packets
  or payload content.
- Evaluation is restricted to the selected benchmark datasets (5 source + 2
  zero-day); generalization claims are bounded by that scope.
- **Edge-IIoTset's attack taxonomy does not include an explicit "worm"
  category** (Backdoor, DDoS, Reconnaissance/scanning, MITM, injection,
  password cracking, and ransomware only); the low-and-slow attack claim in
  RQ1 covers backdoors and reconnaissance/scanning specifically, not worms.
  This reflects a broader shift in the field rather than a gap specific to
  this dataset choice: "worm" as a distinct attack category is largely
  absent from post-2020 IDS datasets generally, having been a more
  prominent threat model in earlier-2000s-era benchmarks such as UNSW-NB15.
- **Edge-IIoTset is an IoT/IIoT-specific dataset**, generated from a
  purpose-built testbed of over 10 IoT/IIoT device types rather than
  general enterprise network traffic; results on it should be interpreted
  as zero-day generalization to an IoT-specific environment, consistent
  with BoT-IoT's IoT focus, rather than to enterprise network traffic
  generally.
- **Institutional-lineage overlap between BoT-IoT and two of the five
  source datasets.** UNSW-NB15 and ToN-IoT are both source datasets built
  by the same research group (UNSW Canberra Cyber, Moustafa et al.) that
  also built BoT-IoT, the primary zero-day evaluation set. These are
  confirmed to be genuinely different testbeds, capture years, and tooling
  — not literal packet overlap, unlike the CIC-ToN-IoT case this design
  already avoided. The risk is methodological rather than data-level: a
  model could partly learn conventions common to how this lab constructs
  and labels datasets (traffic generation patterns, feature extraction
  choices) rather than purely generalizable attack behavior, which would
  inflate BoT-IoT zero-day performance without reflecting genuine
  novel-environment generalization. This is the same category of concern
  already acknowledged for UNSW-NB15/BoT-IoT specifically; it now applies
  to two of five source datasets rather than one, since ToN-IoT joined the
  source set for Paper 2.
- Large foundation models for network telemetry are outside this paper's
  scope (noted as Future Work, not attempted here).

---

## Status: design frozen — ready for implementation

Four independent external review rounds have now converged on this design
with no new structural objections, and each explicitly recommended freezing
the design and moving to implementation. The title decision is resolved
(see top of document): **Self-Supervised Domain-Invariant Temporal
Representation Learning for Cross-Domain Zero-Day Intrusion Detection.**

The remaining uncertainty is no longer the design — it's what the
experiments actually reveal, particularly whether the invariance →
generalization relationship in Section 8 holds up empirically.

---

## Next step

This supersedes the "Experiments" list in `paper2_experimental_setup.tex` and
adds new subsections to `paper2_methodology.tex` (downstream detection +
fusion, invariance analysis chain, baselines) and updates to
`paper2_introduction.tex` (RQ2, RQ3 wording) and `paper2_discussion.tex`
(RQ3 fusion interpretation). Once you pick a title, say the word and I'll
update all four `.tex` files to match this document exactly.
