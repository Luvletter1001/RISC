# Semantic-Scale Support Mismatch in Remote Sensing Detection

Draft version: 2026-06-20 v0.1  
Status: evidence-safe initial manuscript draft, not submission-ready.

## Abstract

Remote-sensing object detectors operate in scenes where object categories often
have strong physical scale regularities: small vehicles, courts, ships, harbors,
and aircraft occupy distinct but sometimes overlapping log-area regimes. We
study a failure mode in which a detector localizes an object but assigns a
high-confidence class label whose class-conditional scale support is implausible.
We call this failure semantic-scale support mismatch and instantiate it through
Scale-Inconsistent Semantic Error (SISE): a localized wrong-label detection whose
predicted box area lies in a low-support region under the predicted class.

Across open-vocabulary and closed-set remote-sensing settings, a continuous
Gaussian support-risk feature over `log(area) | class` predicts pair-level
high-confidence risk. In our current audit, 6/6 datasets pass a support-risk
law gate, OVD and closed-set domains show Spearman correlations of `0.823534`
and `0.636249`, and DOTA2 high-risk pairs recur across detector families with
top-20 recurrence `1.000`. We further show that the phenomenon has deployment
value but also clear boundaries: some settings show large high-confidence SISE
reduction with AP non-regression, whereas DIOR-R, xView, and a stricter E-P2
target-object path intervention reveal that not every scale-semantic signal is
causal or AP-relevant.

We introduce G-S3C, a Gaussian semantic-scale support calibration family that
uses class-conditional log-area priors to modulate class logits or scores without
using Gaussian approximations to IoU. We also outline BASS, a trainable Bayesian
semantic-scale support detector that would couple class/text semantics, dense
features, predicted scale, assignment, ranking, and virtual semantic-scale
outliers. Current evidence supports G-S3C as an AP-safe reliability layer in
selected settings and supports BASS as the next method direction, but does not
yet prove a strong causal training method. This draft therefore positions the
work as a rigorous diagnostic and reliability-calibration study, with explicit
negative evidence and matched-control gates for future method claims.

## 1. Introduction

Object detection in aerial and remote-sensing images is shaped by extreme
variation in scale, orientation, density, and viewpoint. DOTA and DOTA-v2
formalize this challenge at benchmark scale, with large numbers of oriented
objects and severe scale/orientation variation. Modern detectors address many
of these issues through feature pyramids, scale-aware assignment, oriented box
representations, and open-vocabulary language grounding. However, these
mechanisms do not directly answer a different reliability question:

> When a detector predicts a class label for a localized object, is the
> predicted object scale plausible for that class?

This question is not equivalent to localization quality. A detector may produce
a high-overlap box around a true object and still assign a class whose semantic
scale support is physically implausible. In our motivating examples, small
vehicle-sized objects are labeled as tennis courts, planes, or ships with
near-saturated confidence. Conversely, some wrong-label pairs such as
`harbor -> ship` are contextual or semantic-overlap errors rather than scale
support failures. A useful reliability method must distinguish these families.

We therefore study semantic-scale support mismatch. For each class, we estimate
a class-conditional log-area distribution from source-disjoint training
annotations. For a predicted box with area `A`, we compute its log-area
surprise under the predicted class. If a localized detection is wrong and its
predicted class has low scale support for that area, we count it as SISE. This
simple diagnostic reveals that standard detector confidence often fails to
represent class-scale plausibility.

Our current evidence suggests three central findings:

1. **Semantic-scale support mismatch is predictable.** A continuous Gaussian
   support-risk feature over `log(area) | class` predicts pair-level risk across
   OVD and closed-set RS detectors.
2. **The problem is broad but not uniform.** Strong high-confidence deployment
   risk appears in P4 OVD and HRRSD; DOTA2 shows AP non-regression but weaker
   SISE volume; DIOR-R and xView expose boundary cases where raw SISE reduction
   does not imply AP gain.
3. **Causal and trainable claims need stronger evidence.** Our stricter E-P2
   target-object path audit fails against shuffled-prior controls, and G3-v1
   consistency does not beat matched no-G3 controls. These negative results
   shape the method and paper claims.

### Contributions

This draft makes four evidence-safe contributions:

1. **Problem definition.** We define semantic-scale support mismatch and SISE
   for localized high-confidence class errors in RS detection.
2. **Predictive audit.** We show a continuous class-conditional Gaussian
   support-risk law across OVD and closed-set settings, with detector-family
   recurrence on DOTA2.
3. **Practical calibration.** We present G-S3C, a no-dump/no-train or
   lightweight Gaussian semantic-scale calibration family that reduces
   high-confidence SISE with AP non-regression in selected settings.
4. **Boundary and next method.** We report negative E-P2 and G3-v1 evidence and
   use it to motivate BASS, a trainable Bayesian semantic-scale support detector
   whose final claims require matched-control validation.

## 2. Related Work

### Multi-scale detection and assignment

Feature Pyramid Networks, TridentNet, FCOS, ATSS, and related dense-detector
mechanisms address scale through feature representation, receptive fields, or
positive/negative sample assignment. These methods improve detection over
multi-scale objects, but they generally do not model whether a predicted class
label is semantically plausible for the predicted object scale. Our work treats
scale not only as a localization or feature-level variable, but as a
class-semantic support variable.

### Remote-sensing oriented object detection

DOTA and DOTA-v2 highlight the scale and orientation variation that makes aerial
object detection difficult. ReDet, S2A-Net, H2RBox-style methods, ORCNN, R3Det,
and LSKNet-style detectors address rotation, feature alignment, or strong RS
baselines. We use such detector families not as direct competitors to our
calibration layer, but as evidence that high-risk semantic-scale pairs recur
across architectures.

### Detection calibration and probabilistic detection

Object detector calibration work shows that AP and calibration must be evaluated
jointly and that cheap post-hoc calibrators can be strong baselines. Cal-DETR
uses uncertainty-guided logit modulation for detection transformers. Generalized
Focal Loss couples localization quality and classification through distributional
representations. Probabilistic object detection evaluates semantic and spatial
uncertainty jointly. Our contribution is orthogonal: we model
`log(area) | class` as semantic-scale support and ask whether class confidence is
physically plausible for the predicted object scale.

### Gaussian box losses and tiny-object metrics

GWD, KLD-style rotated detection losses, and NWD model bounding boxes as
Gaussian distributions to improve rotated/tiny-object localization or replace
IoU-like metrics. G-S3C and BASS do not use Gaussian distributions for box
overlap or IoU approximation. Gaussian is used only for class-conditional
semantic-scale support: `p(log area | class)`.

### OOD and virtual outlier synthesis

VOS synthesizes virtual outliers from low-likelihood regions of class-conditional
feature distributions to regularize OOD uncertainty. BASS adopts the spirit of
low-likelihood hard negatives but changes the variable: semantic-scale outliers
are plausible object features paired with implausible class-scale labels.

### Open-vocabulary detection and language grounding

GLIP, OWL-ViT, Grounding DINO, and RS-OVD systems show the value of
language-conditioned detection. These models expand vocabulary and semantic
coverage, but language alignment alone does not guarantee that a predicted class
is scale-plausible in remote-sensing scenes. Our method is designed as a
reliability layer for both open-vocabulary prompt classes and closed-set labels.

## 3. Problem Formulation

Let a detector output a box `b_i`, class score/logit for class `c`, and predicted
area `A_i`. We use:

```text
a_i = log(area(b_i)).
```

For every class `c`, estimate a source-disjoint Gaussian support model:

```text
p(a | c) = Normal(mu_c, sigma_c^2).
```

The semantic-scale z score is:

```text
z_i,c = (a_i - mu_c) / sigma_c.
```

For a localized prediction matched to ground-truth class `g`, we define:

```text
localized       = IoU(pred_box, matched_gt_box) > 0.7
wrong           = pred_class != gt_class
scale_surprise  = |z_i,pred_class|
SISE_tau        = localized and wrong and scale_surprise >= tau_z and score >= tau_score
```

We use continuous risk predictors for pair-level analysis:

```text
support_risk(c_gt -> c_pred) = sum_i score_i * z_i,c_pred^2
```

This continuous form avoids the sparsity of a hard `z>=4` threshold and better
predicts pair-level high-confidence risk.

## 4. Method

### 4.1 G-S3C: Gaussian Semantic-Scale Support Calibration

G-S3C estimates class-conditional log-area support from source-disjoint training
annotations and applies it to detector scores or logits.

#### G1: Continuous Gaussian Logit Energy

For each candidate and class:

```text
penalty_i,c = softplus(|z_i,c| - z0)
logit'_i,c  = logit_i,c - beta * penalty_i,c
```

Default values:

```text
z0   = 4.0
beta = log(1 / 0.25)
```

This replaces hard rejection with continuous energy. In no-train mode, it can be
applied during inference and is designed to preserve AP while reducing
high-confidence semantic-scale outliers.

#### G2: Gaussian Scale-Aware Logit Adapter

G2 uses features such as:

```text
|z_i,c|, signed z_i,c, Gaussian log probability, log sigma_c, valid_prior_mask
```

to predict a per-class logit delta. A safe deployment setting constrains deltas
to be non-positive, avoiding new high-confidence false positives. This retains
G1's reliability behavior while allowing more flexible class-dependent response.

#### G3: Dense-head Consistency Loss

G3 aims to move semantic-scale consistency inside training. For assigned
positive samples, it penalizes high hard-negative logits when the hard-negative
class is scale-incompatible and the ground-truth class is scale-compatible. The
current G3-v1 result is negative/partial: it demonstrates a feasible network
path, but it does not beat matched no-G3 controls. G3-v2 therefore adds
AP-sensitive hard-negative gating:

```text
min_hardneg_logit = 0.0
max_gt_abs_z      = 3.0
```

and must be evaluated only against same-seed, same-epoch no-G3 controls.

### 4.2 BASS: Bayesian Semantic-Scale Support Detector

G-S3C is practical but still resembles structured calibration. BASS is the
proposed next-stage method that would make semantic-scale support a trainable
detector variable.

For dense candidate `i` and class `c`, define:

```text
h_i: dense feature
b_i: predicted box
a_i = log(area(b_i))
t_c: class text/name embedding or closed-set class embedding
d: domain/dataset id
```

#### Hierarchical semantic-scale prior

```text
p_phi(a | c, d) = Normal(mu_cd, sigma_cd^2)
mu_cd, log sigma_cd = g_phi(t_c, domain_embedding_d) + empirical_posterior(c,d)
```

This supports both closed-set classes and OVD prompt classes. Rare or weakly
observed classes receive larger uncertainty rather than hard suppression.

#### Feature-conditioned posterior compatibility

The detector predicts:

```text
q_theta(a | h_i, c) = Normal(m_theta(h_i,c), s_theta(h_i,c)^2)
```

and computes compatibility:

```text
kappa_i,c =
  log Integral q_theta(a | h_i,c) p_phi(a | c,d) da
  - log Integral q_theta(a | h_i,c) p_global(a | d) da
```

`kappa_i,c` then enters assignment, logit energy, ranking, and uncertainty.

#### Semantic-scale virtual outliers

For a true candidate `(h_i, b_i, c_gt)`, construct hard negative class labels
`c_bad` for which `p_phi(a_i | c_bad,d)` is low. Train the detector to assign
high uncertainty and low energy to `(h_i, b_i, c_bad)`.

#### Assignment and ranking

Assignment can be modulated by stopped-gradient compatibility:

```text
task_align_i,c = cls_i,c^alpha * IoU_i,c^beta * exp(tau * stopgrad(kappa_i,c))
```

Inference logit:

```text
logit'_i,c = logit_i,c + alpha_k * kappa_i,c - beta_u * u_i,c
```

To avoid simply suppressing useful detections, BASS requires rank-preserving
distillation or AP-sensitive ranking loss against a matched baseline.

## 5. Experiments

### 5.1 Datasets and settings

Current evidence covers:

- P4 OVD on DOTA2-style tiles;
- DOTA2 full-val LSKNet / ORCNN-style predictions;
- HRRSD closed-set internal RTMDet-style experiments;
- DIOR-R, ShipRS, and xView boundary audits;
- detector-family recurrence across LSKNet/ORCNN-style, ORCNN-R50,
  R3Det-KFIoU-R50, and ReDet-Re50.

All future DOTA2 experiments should use absolute dataset roots under:

```text
/data1/zcy/datasets
```

and should not rely on repository-local `data/` paths.

### 5.2 Metrics

We report:

- `mAP` / `AP50`;
- per-class AP when available;
- localized wrong count at IoU `>0.7`;
- `SISE@0.9`, `SISE@0.99`, `SISE@0.999`;
- pair-level SISE for impossible-scale pairs;
- side-effect audit on localized correct detections;
- review-queue or TopK precision where AP is insensitive;
- strict causal/path gate for E-P2-style interventions.

### 5.3 Problem-law results

The support-risk law passes 6/6 dataset settings.

| dataset | domain | pairs | rho_z2_vs_p0199 | pass |
|---|---|---:|---:|---|
| P4 OVD full preselect0.99 | ovd | 112 | 0.823534 | yes |
| HRRSD full | closed_set | 87 | 0.779312 | yes |
| DIOR-R full | closed_set | 282 | 0.713196 | yes |
| ShipRS full | closed_set | 1123 | 0.665872 | yes |
| xView full | closed_set | 1229 | 0.601800 | yes |
| DOTA2 full-val LSKNet | closed_set | 72 | 0.628130 | yes |

Domain-level:

| domain | pairs | rho_z2_vs_p0199 | rho_z2_vs_logz |
|---|---:|---:|---:|
| closed_set | 2793 | 0.636249 | 0.378054 |
| ovd | 112 | 0.823534 | 0.699177 |

### 5.4 Detector-family recurrence

DOTA2 high-risk pairs recur across detector families. The top-20 recurrence rate
from LSKNet support-risk seed pairs is `1.000`. Examples include:

| pair | families_present | present_families |
|---|---:|---|
| small-vehicle->large-vehicle | 4 | LSKNet/ORCNN-style; ORCNN-R50; R3Det-KFIoU-R50; ReDet-Re50 |
| ship->large-vehicle | 4 | LSKNet/ORCNN-style; ORCNN-R50; R3Det-KFIoU-R50; ReDet-Re50 |
| large-vehicle->small-vehicle | 4 | LSKNet/ORCNN-style; ORCNN-R50; R3Det-KFIoU-R50; ReDet-Re50 |
| small-vehicle->tennis-court | 4 | LSKNet/ORCNN-style; ORCNN-R50; R3Det-KFIoU-R50; ReDet-Re50 |

This supports the claim that the phenomenon is not tied to one detector family.

### 5.5 Calibration results

Representative current results:

| setting | baseline | method | mAP base | mAP method | SISE base | SISE method | conclusion |
|---|---|---|---:|---:|---:|---:|---|
| P4 OVD full preselect0.99 | baseline | S3C no-dump | 0.569900 | 0.574900 | 4032 @0.999 | 623 @0.999 | strong deployment signal |
| HRRSD full | baseline | G1 z4 | 0.830700 | 0.832800 | 219 logz | 8 logz | closed-set positive |
| DOTA2 full-val LSKNet | baseline | G1 z4 | 0.541536 | 0.541575 | 5 @score>=0.5 | 2 @score>=0.5 | AP non-regression, weak SISE volume |
| DIOR-R full | baseline | G2 network | 0.644687 | 0.644975 | 5193 total | 2475 total | raw SISE reduction, AP nearly unchanged |
| xView full | baseline | G1 light | 0.182514 | 0.182400 | 4628 total | 584 total | boundary: AP/ranking risk |

Interpretation:

G-S3C is useful when the target metric is high-confidence deployment reliability
or review-queue burden. It should not be claimed as a universal mAP enhancer.

### 5.6 Negative causal/path evidence

The older E9 scale-swap analysis did not establish causality:

```text
complete_cases = 673
scale_prior_sensitive_cases = 1
```

The newer E-P2-clean target-object audit also fails:

```text
clean pass rate        = 0.125
shuffled_prior rate    = 0.142857
specificity_gap        = -0.017857
gate_pass              = false
```

Therefore, this draft does not claim that semantic-scale path movement alone
causally raises hard-negative logits. It claims that semantic-scale support is a
predictive reliability signal with clear boundary conditions.

### 5.7 G3-v1 and G3-v2 status

G3-v1 is `negative_partial`: it demonstrates that dense-head consistency can be
implemented, but current variants do not beat matched no-G3 controls. G3-v2 is
prepared as an AP-sensitive hard-negative consistency objective, with matched
no-G3 same-seed/same-epoch control. No positive G3-v2 result is claimed in this
draft.

## 6. Discussion

### Why mAP can stay unchanged while SISE drops

SISE focuses on localized wrong high-confidence predictions whose scale support
is implausible. Many such detections may sit outside AP-sensitive ranking
regions or be redundant after NMS. Thus, reducing SISE without changing mAP does
not make the problem meaningless. It means the paper must target reliability,
deployment burden, review-queue precision, and calibration rather than claiming
universal AP gains.

### Boundary cases

Scale support should not fix every semantic error. Contextual confusions such as
`harbor -> ship` and sibling confusions such as `small-vehicle <-> large-vehicle`
must be reported separately. This boundary is a strength if stated clearly: it
prevents the method from being evaluated against errors it is not designed to
solve.

### Why not Gaussian IoU?

Gaussian is used only for class-conditional log-area support. We do not model
or approximate rotated box overlap using Gaussian distance. This distinguishes
the work from GWD/KLD/NWD-style box similarity methods.

## 7. Limitations

1. **Causal evidence is not closed.** E-P2-clean currently fails against
   shuffled-prior controls.
2. **Network-level training is not yet positive.** G3-v1/density variants do not
   beat matched no-G3 controls; G3-v2 remains to be validated.
3. **Effectiveness is setting-dependent.** P4 and HRRSD are strong, while
   DIOR-R/xView/DOTA2 show boundary behavior.
4. **Calibration baselines need completion.** A final paper should compare or
   discuss Platt/Isotonic detector calibrators and detection ECE-style metrics.
5. **Prior quality matters.** Source-disjoint priors are mandatory; eval/test
   priors are diagnostic only and must not be used for main results.

## 8. Conclusion

We identify semantic-scale support mismatch as a measurable reliability problem
in remote-sensing detection. The evidence shows that class-conditional Gaussian
log-area support predicts high-risk semantic errors across open-vocabulary and
closed-set settings, and that practical calibration can reduce high-confidence
SISE with AP non-regression in selected deployments. The current evidence does
not yet support a strong causal training-method claim. The most defensible paper
framing is therefore a rigorous diagnostic and reliability-calibration study,
with BASS/G3-v2 positioned as the next trainable extension requiring matched
control validation.

## References

- DOTA-v2: Object Detection in Aerial Images: A Large-Scale Benchmark and
  Challenges. https://arxiv.org/abs/2102.12219
- Object detector calibration: On Calibration of Object Detectors: Pitfalls,
  Evaluation and Baselines. https://arxiv.org/abs/2405.20459
- Cal-DETR: Calibrated Detection Transformer. https://arxiv.org/abs/2311.03570
- VOS: Learning What You Don't Know by Virtual Outlier Synthesis.
  https://arxiv.org/abs/2202.01197
- Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for
  Dense Object Detection. https://arxiv.org/abs/2006.04388
- Probabilistic Object Detection: Definition and Evaluation.
  https://arxiv.org/abs/1811.10800
- Gaussian Wasserstein Distance for rotated object detection.
  https://arxiv.org/abs/2101.11952
- Normalized Gaussian Wasserstein Distance for tiny object detection.
  https://arxiv.org/abs/2110.13389

## Appendix A. Evidence Artifacts

Key local evidence files:

```text
resultmd/fres_20260620_semantic_scale_six_experiments.md
resultmd/exp_p4_scale_semantic_validation/fres_20260620_ep2clean_target_iou_filter_8case_audit.md
resultmd/exp_p4_scale_semantic_validation/fplan_20260620_ep2clean_g3v2_parallel_protocol.md
resultmd/exp_p4_scale_semantic_validation/freview_20260620_iclr_ac_system_score_and_paper_readiness.md
work_dirs/semantic_scale_six_experiments_20260620/six_experiment_review.json
```

## Appendix B. Submission Readiness Checklist

| item | status |
|---|---|
| Problem law across datasets | done |
| Detector-family recurrence | done |
| Source-disjoint prior protocol | done for current audits |
| E-P2 causal gate | negative |
| G3-v2 matched-control training | not yet complete |
| Calibration baselines | not yet complete |
| Full formal bibliography | not yet complete |
| Final ICLR LaTeX conversion | not yet complete |

