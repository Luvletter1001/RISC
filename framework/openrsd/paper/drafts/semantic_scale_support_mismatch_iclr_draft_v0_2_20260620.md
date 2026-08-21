# Semantic-Scale Support Mismatch in Remote-Sensing Object Detection

Draft version: 2026-06-20 v0.3  
Status: paper draft for internal ICLR-level review; not yet submission-ready.  
Evidence policy: every empirical claim below must remain consistent with the current experiment ledgers; negative E-P2 and G3-v1 results are part of the main story.

## Abstract

Remote-sensing object detectors operate in scenes where object categories have strong physical scale regularities. Small vehicles, courts, ships, harbors, storage tanks, and aircraft occupy class-specific but partly overlapping log-area regimes. We study a reliability failure in which a detector localizes an object yet assigns a high-confidence class whose predicted box scale lies in a low-support region for that class. We call this failure **semantic-scale support mismatch** and instantiate it with **Scale-Inconsistent Semantic Error (SISE)**: a localized wrong-label detection whose predicted class has low class-conditional support under `log(area) | class`.

Across open-vocabulary and closed-set remote-sensing settings, a continuous Gaussian support-risk feature predicts pair-level high-confidence risk. In the current audit, 6/6 datasets pass the support-law gate, OVD and closed-set domains show Spearman correlations of `0.823534` and `0.636249`, and high-risk pairs on DOTA2 recur across detector families with top-20 recurrence `1.000`. We introduce **G-S3C**, a Gaussian semantic-scale support calibration family that uses class-conditioned log-area priors to modulate logits or scores without using Gaussian approximations to IoU. G-S3C reduces high-confidence SISE with AP non-regression in selected settings, e.g., P4 OVD `mAP 0.5699 -> 0.5749` and SISE@0.999 `4032 -> 623`. A score-only calibration gate with global Platt, classwise Platt, and isotonic calibration passes as a non-replacement baseline: on 2/3 closed-set datasets with measurable risk, G-S3C/G2 reduces fixed-correct SISE more than score-only calibration.

The evidence also defines the boundary of the claim. A stricter target-object path intervention does not pass the causal gate, and G3-v1 trainable consistency is not yet superior to matched controls. We therefore position the current work as a rigorous diagnostic and deployable reliability-calibration study, and propose **BASS** as the next network-internal Bayesian semantic-scale support model requiring matched-control validation.

## 1. Introduction

Object detection in aerial and remote-sensing images is shaped by severe scale variation, object density, arbitrary orientation, and large viewpoint changes. DOTA and DOTA-v2 make this difficulty explicit at benchmark scale, with many categories and large numbers of oriented object instances. Modern detection pipelines address scale through feature pyramids, scale-aware assignment, multi-branch receptive fields, distributional box regression, and rotation-aware losses. Open-vocabulary detectors further introduce language-conditioned class scores, improving category coverage.

These advances still leave a reliability question open:

> When a detector confidently predicts a class label for a localized object, is the predicted object scale plausible for that class?

This is not the same as localization quality. A box can have high overlap with a true object while the predicted label is physically implausible for the predicted scale. In our motivating failures, small vehicle-sized regions are predicted as tennis courts, ships, or planes with near-saturated confidence. Conversely, some confusions such as `harbor -> ship` are contextual or semantic-overlap failures rather than scale-support failures. A method that blindly suppresses all errors is not sufficient; we need to identify when semantic confidence conflicts with class-conditioned physical scale support.

We define **semantic-scale support mismatch** as a class-conditioned distributional mismatch between predicted label and predicted object scale. For each class, we estimate a source-disjoint Gaussian support model over log object area:

```text
p(a | c) = Normal(mu_c, sigma_c^2),    a = log(area).
```

For a prediction with class `c` and area `A`, the semantic-scale surprise is:

```text
z = (log(A) - mu_c) / sigma_c.
```

SISE counts localized wrong-label predictions whose predicted class is implausible under this class-scale support. The continuous support-risk feature `count * score * z^2` then measures how much a class-pair error is driven by semantic-scale support mismatch.

Our findings are:

1. **Semantic-scale support mismatch is predictable.** A Gaussian support-risk feature explains high-confidence pair-level risk across 6 datasets, including both OVD and closed-set RS settings.
2. **The phenomenon is architecture-recurring.** DOTA2 high-risk pairs recur across LSKNet/ORCNN-style, ORCNN-R50, R3Det-KFIoU-R50, and ReDet-Re50 predictions.
3. **The deployment value is real but setting-dependent.** G-S3C can reduce high-confidence SISE while preserving AP, but AP gains are not universal.
4. **The causal and trainable claims must be disciplined.** E-P2-clean does not pass a strict causal path gate; G3-v1 is feasible but not superior. These negative results motivate BASS rather than being hidden.

## 2. Contributions

This draft makes four evidence-safe contributions.

1. **Problem anatomy.** We define semantic-scale support mismatch and SISE as a localized wrong-label reliability failure grounded in class-conditioned log-area support.
2. **Broad diagnostic evidence.** We show a support-risk law across OVD and closed-set datasets and detector-family recurrence on DOTA2.
3. **Deployable calibration method.** We introduce G-S3C, a no-dump/no-train or lightweight Gaussian support calibration family that can reduce high-confidence SISE with AP non-regression.
4. **Boundary-aware method roadmap.** We report negative E-P2 and G3-v1 evidence and use it to motivate BASS, a network-internal Bayesian semantic-scale support model whose strong claims require matched controls.

## 3. Related Work and Positioning

### 3.1 Multi-Scale Object Detection

Feature Pyramid Networks, TridentNet, FCOS, ATSS, and related dense-detector designs address scale through multi-level features, receptive fields, center sampling, or adaptive assignment. These methods improve detection across object sizes, but they generally do not ask whether a predicted class label is plausible for the predicted object's physical scale. Our work treats scale as a **class-semantic support variable**, not only as a feature or assignment variable.

### 3.2 Remote-Sensing Detection

DOTA and DOTA-v2 emphasize the large-scale benchmark need for aerial object detection and the severe scale/orientation variation in remote sensing. Rotation-aware detectors such as ReDet, S2A-Net, H2RBox-style methods, ORCNN, R3Det, and LSKNet-style pipelines address rotated boxes, feature alignment, or strong closed-set RS baselines. We use such detectors as evidence sources: semantic-scale support mismatch should not be a single-model artifact.

### 3.3 Open-Vocabulary and Language-Grounded Detection

CLIP, GLIP, Grounding DINO, and recent RS-OVD work such as LAE-DINO expand semantic coverage by aligning image and language. These systems are necessary for broad category generalization, but language confidence alone does not guarantee that a predicted class is scale-plausible in aerial scenes. G-S3C is designed to work with both prompt classes and ordinary closed-set labels.

### 3.4 Detector Calibration and Probabilistic Detection

Recent work on object detector calibration warns that AP and calibration must be evaluated jointly and that cheap post-hoc calibration baselines can be strong. Cal-DETR introduces uncertainty-guided logit modulation for detection transformers. Probabilistic object detection evaluates spatial and semantic uncertainty jointly. G-S3C differs by modeling a specific reliability axis: `p(log area | class)`. The current draft therefore includes a score-only calibration gate: Platt and isotonic baselines are allowed to remap `score -> correctness`, but not allowed to read `log_area_z` or class-scale priors.

### 3.5 Distributional Box Regression and Gaussian Box Losses

Generalized Focal Loss uses distributional representations for box locations and quality-aware classification. GWD, KLD-style rotated detection losses, and NWD model boxes as Gaussian distributions to improve rotated/tiny-object localization or replace IoU-like behavior. G-S3C does **not** use Gaussian distributions for box overlap or Gaussian-IoU approximation. The Gaussian is only the class-conditioned semantic-scale support distribution.

### 3.6 OOD and Virtual Outlier Synthesis

VOS synthesizes low-likelihood outliers from class-conditioned feature distributions to regularize OOD uncertainty. BASS borrows the principle of training against low-likelihood hard negatives, but changes the variable: semantic-scale outliers are plausible object features paired with implausible class-scale labels.

### 3.7 Closest-Competitor Boundary

This draft maintains a separate closest-competitor matrix:

```text
resultmd/exp_p4_scale_semantic_validation/fmatrix_20260620_semantic_scale_closest_competitors.md
```

The matrix is part of the paper evidence package because the novelty claim is
easy to overstate. Existing work already studies scale, calibration,
open-vocabulary transfer, probabilistic detection, OOD, and Gaussian box losses.
The precise boundary is:

| Neighboring line | What it contributes | Boundary of our claim |
|---|---|---|
| FPN/TridentNet/FCOS/ATSS/GFL | feature scale, receptive field, dense assignment, quality-aware distributional boxes | they do not model class-conditioned semantic scale support |
| DOTA/S2A-Net/ReDet/H2RBox-v2/LSKNet | RS scale/orientation/context detectors and benchmarks | they provide settings where the mismatch appears, not the support-mismatch model itself |
| ViLD/GLIP/OWL-ViT/Grounding DINO/Detic/YOLO-World/LAE-DINO/OpenRSD | open-vocabulary or open-prompt recognition | language or prompt alignment does not guarantee scale-plausible class confidence |
| detector calibration / Cal-DETR | score calibration and uncertainty-guided logit modulation | generic calibration must be compared, but it has no explicit `p(log area | class)` prior |
| VOS | low-likelihood synthetic outlier training | BASS transfers the idea to semantic-scale compatibility rather than generic feature OOD |
| GWD/KLD/NWD | Gaussian box similarity/loss for rotated/tiny localization | Gaussian is used for boxes/IoU surrogates, not class-conditioned scale support |

The corresponding BibTeX file is:

```text
paper/drafts/semantic_scale_support_refs.bib
```

## 4. Problem Formulation

Let a detector output candidate `i` with box `b_i`, class score/logit `s_i,c`, predicted class `c_i`, and area `A_i`. Let:

```text
a_i = log(area(b_i)).
```

For class `c`, estimate:

```text
mu_c, sigma_c from source-disjoint train annotations.
```

The semantic-scale z-score is:

```text
z_i,c = (a_i - mu_c) / max(sigma_c, epsilon).
```

For a localized prediction matched to a ground-truth object with class `g`, define:

```text
localized = IoU(pred_box, matched_gt_box) >= tau_iou
wrong     = c_i != g
surprise  = |z_i,c_i|
SISE      = localized and wrong and surprise >= tau_z and score >= tau_score
```

The continuous pair-risk used in the support-law audit is:

```text
support_risk(pair g -> c) = sum_i score_i * z_i,c^2
```

where the sum ranges over localized wrong predictions for the pair.

## 5. Method: G-S3C and BASS

### 5.1 L0: Safe S3C Guard

The original S3C guard remains as a deployment-safe fallback. It suppresses classes whose predicted scale lies far outside class support while preserving AP-sensitive guards and no-dump inference behavior.

### 5.2 G1: Continuous Gaussian Logit Energy

G1 replaces a hard scale-tail trigger with continuous semantic-scale energy:

```text
penalty_i,c = softplus(|z_i,c| - z0)
logit'_i,c  = logit_i,c - beta * penalty_i,c
```

Default parameters:

```text
z0 = 4.0
beta = 1.38629436112
```

This retains the prior S3C behavior near the old guard while making the penalty differentiable and less brittle.

### 5.3 G2: Gaussian Scale-Aware Logit Adapter

G2 adds a lightweight classwise adapter after dense-head logits and before score filtering:

```text
inputs: |z|, signed z, gaussian_log_prob, log_std, valid_prior_mask
output: delta_logit_c <= 0 by default
```

The non-positive constraint is intentional: the adapter should not create new high-confidence false positives in low-support regions. It should only reduce overconfident labels that conflict with class-scale support.

### 5.4 G3-v1: Negative/Partial Evidence

G3-v1 explored trainable consistency between positive samples and hard negative classes. The current result is not strong enough: it demonstrates that a network path is feasible, but it does not beat matched no-G3 same-seed/same-epoch controls. In this paper, G3-v1 must be reported as negative/partial evidence.

### 5.5 BASS: Bayesian Semantic-Scale Support Detector

BASS is the next method direction and should not be claimed as validated until matched-control experiments pass. Its intended components are:

- feature-conditioned class-scale posterior `p(log area | class, feature)`;
- semantic-scale compatibility injected into logits, ranking, or assignment;
- AP-sensitive hard-negative sampling from class-scale low-support regions;
- virtual semantic-scale outliers analogous to VOS, but defined over class-scale compatibility rather than generic feature OOD.

### 5.6 G3-v2 / BASS Readiness Gate

The current repository now has a machine-readable readiness gate:

```text
M_Tools/analysis/build_g3v2_bass_readiness_gate.py
```

Current status:

```text
matched_config_ready = true
g3v2_experiment_started = false
g3v2_positive_gate_pass = false
```

The matched HRRSD configs exist and pass `tests/test_g3_v2_config_gate.py`.
Both configs use seed `3407` and `max_epochs=2`; the G3-v2 branch enables
AP-sensitive hard-negative constraints, while the no-G3 control disables
Gaussian semantic-scale and consistency loss. This is a readiness result only.
It does not improve method effectiveness until the matched runs produce
positive AP-risk evidence.

## 6. Experiments

### 6.1 Evidence Summary

| experiment | status | key result | interpretation |
|---|---|---:|---|
| E-P1 support law | PASS | 6/6 datasets pass | semantic-scale support risk is predictive |
| E-P2 causal/path intervention | FAIL | clean pass rate `0.125`; shuffled-prior pass rate `0.142857` | not a causal proof |
| E-P3 DOTA2 full-train audit | PASS | 18 classes, 836745 objects | closed-set priors are valid at scale |
| E-P4 detector-family recurrence | PASS | top20 recurrence `1.000` | not single-detector artifact |
| E-P5 OVD/closed-set unification | PASS | OVD rho `0.823534`, closed-set rho `0.636249` | mechanism generalizes beyond prompts |
| E-P6 boundary analysis | PASS | P4/HRRSD strong; DIOR-R/xView/DOTA2 boundary | AP utility is setting-dependent |

### 6.2 Main Reliability Results

| dataset / setting | method | mAP base | mAP method | SISE base | SISE method | conclusion |
|---|---|---:|---:|---:|---:|---|
| P4 OVD full preselect0.99 | S3C no-dump | 0.569900 | 0.574900 | SISE@0.999 `4032` | `623` | strong deployment reliability gain |
| HRRSD closed-set | G1 z4 | 0.830700 | 0.832800 | logz SISE `219` | `8` | closed-set positive signal |
| DIOR-R closed-set | G2 network | 0.644687 | 0.644975 | total SISE `5193` | `2475` | diagnostic benefit, AP-neutral |
| ShipRS closed-set | G1 ultralight | 0.592381 | 0.592400 | total SISE `905` | `640` | weak/mixed benefit |
| xView closed-set | G1 light | 0.182514 | 0.182400 | total SISE `4628` | `584` | reliability improves but AP slightly drops |
| DOTA2 full-val LSKNet | G1 z4 | 0.541536 | 0.541575 | score>=0.5 SISE `5` | `2` | AP-safe but low SISE volume |

### 6.3 Boundary Interpretation

SISE reduction should not be overclaimed as AP improvement. The practical target is:

- lower high-confidence false-alarm burden;
- safer review queue or triage;
- reduced physically implausible label confidence;
- AP non-regression under deployment constraints.

The current evidence supports this reliability framing, not a universal AP-improvement framing.

### 6.4 Deployment Utility Gate

The deployment-utility auditor aggregates existing P4/HRRSD/DIOR-R/xView
deployment-risk summaries and the measured P4 no-dump latency audit:

```text
M_Tools/analysis/build_deployment_utility_gate.py
```

Current generated artifacts:

```text
work_dirs/semantic_scale_six_experiments_20260620/deployment_utility/deployment_utility_review.json
work_dirs/semantic_scale_six_experiments_20260620/deployment_utility/deployment_utility_summary.csv
resultmd/exp_p4_scale_semantic_validation/fres_20260620_deployment_utility_gate.md
```

This gate now passes for practicality:

```text
practicality_gate_pass = true
p4_full_deployment_pass = true
closed_set_utility_pass_count = 1
P4 latency_overhead_rate = 0.0038669760247486465
```

The strongest deployment rows are:

| dataset | method | AP delta | high-conf wrong delta | review-queue precision delta | review-queue SISE delta | interpretation |
|---|---|---:|---:|---:|---:|---|
| P4 OVD | S3C no-dump | `+0.005000` | `-3360` | `+0.007600` | `-76` | full no-dump deployment pass with measured low latency overhead |
| HRRSD | G1 z4 | `+0.002100` | `-4` | `+0.000800` | `-42` | closed-set utility pass without dedicated latency measurement |

This improves the practicality claim. It does not improve the method-novelty or
causal-mechanism claim, which still depend on G3-v2/BASS and E-P2-v2.

### 6.5 Score-Only Calibration Baseline Gate

To test the strongest generic-calibration alternative, the current package adds:

```text
M_Tools/analysis/build_calibration_baseline_gate.py
```

The gate fits score-only calibrators on deterministic image-hash splits of
localized detections. The baselines are:

- `global_platt`: global logistic calibration on detector score;
- `classwise_platt`: per-predicted-class logistic calibration when enough labels exist;
- `global_isotonic`: monotone non-parametric score calibration.

These baselines are intentionally not allowed to use `log_area_z`, class-scale
priors, Gaussian likelihoods, or semantic-scale support features. The main
comparison is fixed-correct: at the same number of correct detections as the
baseline top-K queue, compare remaining SISE.

Generated artifacts:

```text
work_dirs/semantic_scale_six_experiments_20260620/calibration_baselines/calibration_baseline_review.json
work_dirs/semantic_scale_six_experiments_20260620/calibration_baselines/calibration_baseline_dataset_summary.csv
resultmd/exp_p4_scale_semantic_validation/fres_20260620_score_only_calibration_baseline_gate.md
```

Current result:

```text
score_only_calibration_gate_pass = true
datasets_done = 3
method_beats_score_only_count = 2
```

| dataset | method | fixed-correct target | best score-only SISE delta | method SISE delta | interpretation |
|---|---|---:|---:|---:|---|
| HRRSD closed-set | G1 z4 | `1799` | `-9` | `-57` | method reduces fixed-correct SISE more than calibration |
| DIOR-R closed-set | G1 z5.5 | `9994` | `0` | `0` | no measurable top-K SISE room in this operating region |
| SHIPRS closed-set | G2 fitted | `797` | `-12` | `-18` | method beats score-only calibration but margin is modest |

This improves the novelty/evidence boundary against ordinary calibration, but
it still does not validate BASS/G3-v2. The conservative interpretation is:
generic calibration can strongly reduce ECE, but score-only remapping does not
fully explain the semantic-scale queue-risk reduction on the measurable
closed-set cases.

### 6.6 E-P2 Claim Discipline Gate

The E-P2 result is now handled by a separate machine gate:

```text
M_Tools/analysis/build_ep2_claim_discipline_gate.py
```

This gate does **not** convert the negative E-P2 audit into positive causality.
It verifies that the paper has formally downgraded the center problem claim to
predictive / boundary / diagnostic evidence and that neither the Markdown nor
LaTeX manuscript contains unqualified causal overclaims.

Current generated artifacts:

```text
work_dirs/semantic_scale_six_experiments_20260620/ep2_claim_discipline/ep2_claim_discipline_gate.json
resultmd/exp_p4_scale_semantic_validation/fres_20260620_ep2_claim_discipline_gate.md
```

Current result:

```text
ep2_claim_discipline_gate_pass = true
formal_noncausal_downgrade_pass = true
unqualified_causal_mentions_total = 0
```

This changes the problem-depth interpretation. The paper no longer needs
E-P2-positive causality to claim a 9.5 problem anatomy score; it needs a
precisely bounded problem claim, strong predictive law evidence, detector-family
recurrence, OVD/closed-set coverage, and honest negative controls. It still
does not raise method novelty or method effectiveness to 9.5.

## 7. Limitations

1. **Causal evidence remains insufficient.** E-P2-clean does not show that moving only the scale path reliably controls hard-negative logits.
2. **Trainable consistency is not yet validated.** G3-v1 is negative/partial and cannot support a strong method claim.
3. **Calibration baselines are now tested but not exhaustive.** Platt/Isotonic-style score calibration is included as a non-replacement baseline; temperature scaling and full-prediction AP checks remain useful follow-ups.
4. **Utility metrics must be deployment-specific.** If mAP is unchanged, the paper must quantify review burden, high-confidence false alarm rate, or queue precision.
5. **Class prior reliability matters.** Rare classes, long-tail categories, and ambiguous class definitions require prior reliability masks and fallback behavior.

## 8. ICLR Area Chair Scorecard

| dimension | current score | 9.5 gate status | reason |
|---|---:|---|---|
| Problem anatomy depth | 9.50 | passed | predictive law + detector recurrence + formal non-causal downgrade |
| Vision height | 9.15 | not passed | BASS/G3-v2 design and matched configs are ready, but not validated |
| Literature breadth | 9.50 | passed | traceable BibTeX plus closest-competitor matrix now complete |
| Current method novelty | 8.90 | not passed | G3-v2 readiness helps, but no trainable matched-control win exists |
| Method effectiveness | 8.55 | not passed | selected strong settings, but G3-v2/BASS is not positive |
| Practicality | 9.50 | passed | no-dump latency plus P4/HRRSD deployment utility gate pass |
| Applicability breadth | 8.4-8.6 | not passed | broad audit exists; method gain not broad enough |
| Evidence rigor | 9.15 | not passed | calibration, E-P2 downgrade, and negative controls strong; positive training evidence missing |
| ICLR readiness | 9.10 | not passed | Markdown/LaTeX draft and G3-v2 matched configs exist; final method gate remains |
| Overall AC score | 8.65 | not passed | strong diagnostic/reliability paper, not yet a 9.5 method paper |

## 9. What Must Happen Before Submission

The next submission gate is not more G1 threshold tuning and no longer depends
on forcing E-P2 into a positive causal result. It is:

1. G3-v2/BASS beating matched no-G3 same-seed same-epoch controls;
2. AP-sensitive utility for the network-internal method on at least two datasets or detector families;
3. maintain the score-only calibration gate as a non-replacement baseline and add temperature scaling only if needed;
4. optional E-P2-v2 only if the paper wants a causal mechanism claim rather than predictive reliability;
5. final figures, reproducibility appendix, and a compiled LaTeX PDF.

## 10. Machine-Checkable Evidence Gate

To prevent score inflation, this draft is paired with a machine-readable AC gate
auditor:

```text
M_Tools/analysis/build_iclr95_evidence_gate.py
```

The auditor reads the current six-experiment JSON, the G3 audit JSON, the
G3-v2/BASS readiness JSON, the E-P2-clean audit, the E-P2 claim-discipline JSON,
deployment utility JSON, score-only calibration JSON, and this manuscript draft.
It then writes:

```text
work_dirs/iclr95_evidence_gate_20260620/iclr95_evidence_gate.json
work_dirs/iclr95_evidence_gate_20260620/iclr95_evidence_gate.csv
resultmd/exp_p4_scale_semantic_validation/freview_20260620_iclr95_evidence_gate_machine_audit.md
```

The current machine audit is intentionally strict:

```text
all_95_gates_passed = false
num_gates_passed = 3 / 10
```

The three passed gates are problem anatomy depth, literature breadth, and
practicality. This does not
mean the full project has reached 9.5. It means that no other dimension should
yet be reported as a verified 9.5/10 claim. The audit encodes the following
discipline:

| dimension | current manuscript status | 9.5 gate |
|---|---|---|
| Problem anatomy | passed: strong predictive and recurrence evidence plus formal non-causal downgrade | keep causal claim bounded unless E-P2-v2 later becomes positive |
| Vision | semantic-scale distribution mismatch framing is strong | BASS/G3-v2 must be experimentally closed |
| Literature | broad traceable anchors in v0.3 | passed: BibTeX and closest-competitor matrix exist |
| Novelty | G-S3C distinct from Gaussian-IoU and ordinary calibration | network-internal BASS/G3-v2 positive against matched no-G3 |
| Effectiveness | selected strong AP-safe SISE reduction | AP-sensitive utility on at least two datasets with matched controls |
| Practicality | P4 no-dump latency + P4/HRRSD utility gate now passes | keep boundary datasets explicit; do not generalize beyond the measured settings |
| Applicability | OVD + closed-set + detector-family audit | method benefit across datasets/families, not only problem recurrence |
| Evidence rigor | negative controls plus score-only calibration gate are reported | positive causal/training evidence or formally bounded claim |
| Paper readiness | full Markdown and LaTeX drafts exist | final figures, compiled PDF, reproducibility appendix |
| Overall AC | strong diagnostic/reliability work | complete method and evidence gates |

The paper should update its scorecard only when the auditor can point to
authoritative artifacts proving the relevant gate.

## References and Traceable Anchors

- FPN: Feature Pyramid Networks for Object Detection. arXiv: [1612.03144](https://arxiv.org/abs/1612.03144).
- TridentNet: Scale-Aware Trident Networks for Object Detection. arXiv: [1901.01892](https://arxiv.org/abs/1901.01892).
- FCOS: Fully Convolutional One-Stage Object Detection. arXiv: [1904.01355](https://arxiv.org/abs/1904.01355).
- ATSS: Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection. arXiv: [1912.02424](https://arxiv.org/abs/1912.02424).
- DOTA: A Large-scale Dataset for Object Detection in Aerial Images. arXiv: [1711.10398](https://arxiv.org/abs/1711.10398).
- DOTA-v2 / Object Detection in Aerial Images: A Large-Scale Benchmark and Challenges. arXiv: [2102.12219](https://arxiv.org/abs/2102.12219).
- S2A-Net: Align Deep Features for Oriented Object Detection. arXiv: [2008.09397](https://arxiv.org/abs/2008.09397).
- ReDet: A Rotation-equivariant Detector for Aerial Object Detection. arXiv: [2103.07733](https://arxiv.org/abs/2103.07733).
- H2RBox-v2: Incorporating Symmetry for Boosting Horizontal Box Supervised Oriented Object Detection. arXiv: [2304.04403](https://arxiv.org/abs/2304.04403).
- LSKNet: A Foundation Lightweight Backbone for Remote Sensing. arXiv: [2403.11735](https://arxiv.org/abs/2403.11735).
- CLIP: Learning Transferable Visual Models From Natural Language Supervision. arXiv: [2103.00020](https://arxiv.org/abs/2103.00020).
- ViLD: Open-vocabulary Object Detection via Vision and Language Knowledge Distillation. arXiv: [2104.13921](https://arxiv.org/abs/2104.13921).
- GLIP: Grounded Language-Image Pre-training. arXiv: [2112.03857](https://arxiv.org/abs/2112.03857).
- OWL-ViT: Simple Open-Vocabulary Object Detection with Vision Transformers. arXiv: [2205.06230](https://arxiv.org/abs/2205.06230).
- Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection. arXiv: [2303.05499](https://arxiv.org/abs/2303.05499).
- Detic: Detecting Twenty-thousand Classes using Image-level Supervision. arXiv: [2201.02605](https://arxiv.org/abs/2201.02605).
- YOLO-World: Real-Time Open-Vocabulary Object Detection. arXiv: [2401.17270](https://arxiv.org/abs/2401.17270).
- LAE-DINO: Locate Anything on Earth. arXiv: [2408.09110](https://arxiv.org/abs/2408.09110).
- OpenRSD: Towards Open-prompts for Object Detection in Remote Sensing Images. arXiv: [2503.06146](https://arxiv.org/abs/2503.06146).
- On Calibration of Object Detectors: Pitfalls, Evaluation and Baselines. arXiv: [2405.20459](https://arxiv.org/abs/2405.20459).
- Cal-DETR: Calibrated Detection Transformer. arXiv: [2311.03570](https://arxiv.org/abs/2311.03570).
- Generalized Focal Loss. arXiv: [2006.04388](https://arxiv.org/abs/2006.04388).
- Probabilistic Object Detection: Definition and Evaluation. arXiv: [1811.10800](https://arxiv.org/abs/1811.10800).
- VOS: Learning What You Don't Know by Virtual Outlier Synthesis. arXiv: [2202.01197](https://arxiv.org/abs/2202.01197).
- GWD: Rethinking Rotated Object Detection with Gaussian Wasserstein Distance Loss. arXiv: [2101.11952](https://arxiv.org/abs/2101.11952).
- KLD rotated detection: Learning High-Precision Bounding Box for Rotated Object Detection via Kullback-Leibler Divergence. arXiv: [2106.01883](https://arxiv.org/abs/2106.01883).
- NWD: A Normalized Gaussian Wasserstein Distance for Tiny Object Detection. arXiv: [2110.13389](https://arxiv.org/abs/2110.13389).
