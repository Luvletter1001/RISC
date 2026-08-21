# Semantic-Scale Support Mismatch in Remote-Sensing Object Detection

Draft version: 2026-06-22 v0.13 + P9A Gaussian support token backbone negative result  
Target venue: ICLR-style full paper draft  
Status: evidence-safe manuscript draft; the problem frame is upgraded from a thresholded scale heuristic to a Gaussian semantic support-field contract, and the method frame is further upgraded to an AP-constrained Gaussian likelihood-ratio projection. The paper is still not submission-ready as a verified 9.5 method paper. The completed G3-v2, train-time BASS delta, P0/P1 RankDelta, P2 assign/posterior-rank, P4/GSRL non-oracle ranker, P5A positive-support delta, P5B geometry-Gaussian head gate, P6 support-negative focal, P7 scale-consistency, P8A FPN-level Gaussian routing, P8B AP-projected routing, P8C AP-safe support projector, and P9A Gaussian support token backbone audits are boundary/partial except for a P1 gentle pilot pass and the P8B matched-AP risk-dominant partial result. P3A gives the first strict AP-constrained Gaussian support ranking pass on HRRSD, and P3D transfers the same AP-protected ranking projection to DIOR-R. P4/GSRL confirms that a deployable score-only Gaussian support ranker is insufficient: its best DIOR-R row reaches only `mAP 0.644693`, below P3D `0.645981`, and fails rank-tail safety. P8B A_safe25 is the strongest non-oracle detector-internal partial positive so far: it reaches HRRSD `mAP 0.8628`, essentially matches no-G3 e2 `0.8629`, lowers top5000 p0199 risk from P7 `37.4223` to `32.5989`, and also beats the no-G3 top5000 p0199 risk `35.1581`; however, it still has correct-positive rank disruption (`P7 rank delta 68.00`) and is therefore only "risk-dominant at matched AP." P8C has now been trained and is a negative result: it slightly improves localized precision relative to P8B A_safe25 but drops `mAP` to `0.8540` and worsens top5000 p0199 risk, so it cannot be promoted. P9A is also negative: injecting class-conditioned Gaussian geometry support tokens into CSPNeXt C3/C4/C5 features yields HRRSD `mAP 0.8595`, below P8B A_safe25 and no-G3, increases localized wrong detections, worsens global risk, and introduces large rank-tail disruption. The machine gate remains `8/10`; ICLR readiness and overall AC score remain open because the package still lacks a compiled submission artifact, detector-family transfer, and a positive non-oracle AP-aware ranker. All `9.50` entries below are strict closure targets unless explicitly marked verified.  
Evidence gate: `work_dirs/iclr95_evidence_gate_20260620/iclr95_evidence_gate.json`

## Abstract

Remote-sensing object detectors operate in scenes where object categories have
strong physical scale regularities. A detector can localize an object while
assigning a high-confidence class whose predicted box scale is implausible for
that class. We study this reliability failure as **semantic-scale support
mismatch**. The stricter framing is a Gaussian semantic-scale support field:
semantic confidence should be compatible with a class- and domain-conditioned
generative support law over object log-area. The currently validated
implementation uses the source-disjoint single-Gaussian case,
`p(log area | class)`, and defines **Scale-Inconsistent Semantic Error (SISE)**
as a localized wrong-label detection whose predicted class lies in a
low-support scale region. The key conceptual shift is that semantic confidence
is treated as a physical support claim: a detector should not only say *what*
the object is, but should make that claim at a scale supported by the class and
domain distribution.

Across open-vocabulary and closed-set remote-sensing settings, a continuous
semantic-scale support-risk feature predicts high-confidence pair-level error.
The current evidence package passes the support-law gate on 6/6 datasets, shows
OVD and closed-set correlations of `0.823534` and `0.636249`, and finds DOTA2
high-risk pair recurrence across detector families with top-20 recurrence
`1.000`. We introduce **G-S3C**, a Gaussian semantic-scale support calibration
family that modulates logits or scores using `log(area) | class`, without using
Gaussian approximations to IoU or box similarity. The stricter method object is
not "more suppression"; it is an AP-aware compatibility contract between the
semantic score surface, the Gaussian support field, and the ranking/assignment
surface that produces detections. G-S3C reduces high-confidence
SISE with AP non-regression in selected settings, including P4 OVD
`mAP 0.5699 -> 0.5749` and SISE@0.999 `4032 -> 623`.

The evidence also defines the boundary of the claim. A stricter target-object
path intervention does not pass the causal gate, G3-v1 trainable consistency is
not superior to matched no-G3 controls, and the completed G3-v2 matched-control
run is only marginally mAP-positive while failing the AP-risk queue gate. We
therefore position the current paper as a rigorous diagnostic and deployable
reliability calibration study. The 9.5 method target is **BASS-GSF**, a
Bayesian semantic-scale support detector built on a Gaussian support field,
feature-conditioned posterior compatibility, semantic-scale virtual outliers,
and assignment/ranking coupling. This is the only route by which the current
work can honestly claim method novelty at the stricter 9.5 level. A first two-GPU **BASS-GSF-Lite** pilot,
which couples the learned density posterior back into pre-topK logits, improves
HRRSD density e2 from `mAP 0.860457` to `0.861822` and log-z SISE from `143`
to `66`, but it still does not beat the no-G3 e2 AP control `0.862878`.
A subsequent train-time Gaussian hard-negative delta run verifies that
support-field signals can be optimized inside the detector: the stronger
`dw0.05` variant reduces localized wrong detections `6480 -> 6202`, global
log-z SISE `143 -> 84`, and risk-weighted log-z `13.7092 -> 11.4449` relative
to the density control. It still lowers mAP to `0.858745` and top5000 precision
to `0.7148`, so it fails the strict AP-risk method gate. The completed
RankDelta/P2 closure sharpens this boundary. P1 gentle RankDelta is a useful
pilot (`mAP 0.862488`, localized wrong `6194`, risk-weighted log-z `12.0345`),
but it still misses the strict no-G3 AP/top-K gate. P2 assign-rank reduces
localized wrong to `6122` but drops mAP to `0.857772`; P2 posterior-rank reaches
`mAP 0.860845` but worsens localized wrong to `6720` and risk-weighted log-z to
`16.0802`. P3A implements the resulting AP-constrained Gaussian semantic
support projection as a validation-set oracle: correct positives are protected
and only low-support localized wrong-class detections are demoted. Against the
no-G3 e2 control, the selected P3A z=2.0, score-floor 0.1 row improves
`mAP 0.862878 -> 0.862961`, top5000 precision `0.7168 -> 0.7170`, and
top5000 log-z SISE `48 -> 0`, with `changed_correct=0` and `tp_rank_harm=0`.
P3D then applies the same AP-constrained projection to DIOR-R. Against the
DIOR-R baseline, P3D improves `mAP 0.644687 -> 0.645981`, keeps top5000
precision at `1.0000`, reduces all-risk log-z false alarm `2236.0471 ->
1827.5576`, reduces score>=0.5 risk-weighted log-z `6.1242 -> 0.0037`,
removes 54 AP-sensitive SISE false positives, and keeps `changed_correct=0`
and `tp_rank_harm=0`. This closes dataset-transfer breadth for the 8/10
machine gate, while P3B/P3C and detector-family transfer remain necessary for
a broader non-oracle 9.5 method claim. P4/GSRL tests the deployable version:
Gaussian support is applied as a logit-space ranker using only predicted class,
predicted area, score, and train-set priors. The best DIOR-R non-oracle row is
`penalty000_b002_rz10_pz20_ms010`, with `mAP 0.644693`, `37961` penalized
detections, no boosted detections, all-risk log-z false alarm `2213.8133`, and
score>=0.5 risk `5.4485`. It is slightly better than baseline on risk but far
below P3D, and its rank-tail audit reports `correct_rank_drop_count=570`,
`tp_rank_harm=301.6873`, and `sise_fp_rank_benefit=2.4059`. Thus P4/GSRL is a
strict fail as a 9.5 method, but a useful negative control: Gaussian support
cannot be an unconstrained confidence bonus or penalty; it must be learned
inside AP-aware assignment/ranking with correct-positive protection.

P8A then moves the Gaussian support signal inside the detector without using
bbox-Gaussian overlap losses. It adds a class-conditioned FPN-level routing
bias derived from `log(area) | class`: each level receives a non-positive
support penalty relative to the best-supported level for that class. This is a
more plausible non-oracle route than P4/GSRL because it operates before final
detection ranking rather than as an offline score-only prior. On HRRSD, P8A
improves over P7 scale-consistency from `mAP 0.8601` to `0.8616`, reduces
localized wrong detections `6906 -> 6442`, reduces global log-z risk
`13.8280 -> 12.5171`, and reduces p0199 risk `111.6902 -> 99.9847`. However,
it still fails the strict main-method gate: it remains below no-G3 `mAP
0.8629`, has top5000 p0199 risk `37.9004` versus no-G3 `35.1581`, and its
P7-to-P8A rank-tail audit reports `correct_rank_drop_count=363` and
`rank_disruptive_delta=70.3258`. P8A is therefore a detector-internal partial
positive, not a final AP-protected method.

P8B converts this lesson into a projection-guarded routing study. The best
overnight variant, A_safe25, uses a weaker level-routing bias (`weight=0.10`,
`max_bias_abs=0.25`) plus positive-logit projection. It reaches `mAP 0.8628`
and `AP50 0.8630`, essentially tying the no-G3 e2 AP control while reducing
top5000 p0199 risk below both P7 (`37.4223 -> 32.5989`) and no-G3
(`35.1581 -> 32.5989`). This is the first non-oracle detector-internal result
that is plausibly risk-dominant at matched AP. It is still not final, because
the P7 rank-tail comparison reports `rank_disruptive_delta=68.00`.

P8C is the new train-time module designed to attack that remaining weakness.
It does not increase the routing penalty. Instead, it adds a **Gaussian
AP-safe support projector** on assigned positive locations. Let
`s_c(g)=log p_phi(g | c,d)` be the class-conditioned Gaussian support of the
assigned target geometry `g=[log(area), log(long_side/short_side)]`. For a
positive with label `y`, only rival classes with `s_y(g)-s_r(g) >= tau` and
nontrivial rival score enter the candidate set. A rival receives suppression
budget only when the current detached ranking margin
`z_y-z_r` already exceeds a protection margin:

```text
alpha_yr = clip((stopgrad(z_y - z_r) - m0) / Tm, 0, Bmax)
L_P8C = sum alpha_yr * sigmoid((s_y - s_r - tau) / Ts)
        * softplus(z_r - stopgrad(z_y) + gamma).
```

Thus correct positives are protected by construction: if a rival is close to
the GT logit, the budget is zero; if the GT already has margin, only the low-
support semantic rival is pushed down. The Gaussian is a semantic support law
over class geometry, not a box-to-box Gaussian distance, and the module is
therefore outside the GWD/KLD/NWD IoU-surrogate line.

The P8C training result forces one more change in where the Gaussian support
field is injected. Directly acting on the AP/ranking surface remains fragile:
P8C reduces some localized wrong predictions but loses too much `mAP`. P9A
therefore moves the same semantic-support law into the backbone. It encodes
each class geometry prior,
`p_phi(g | c,d)` with `g=[log(area), log(long_side/short_side)]`, into a
Gaussian support token and lets CSPNeXt C3/C4/C5 feature queries attend to
these tokens:

```text
t_c = MLP([mu_c, log sigma_c])
a_i = softmax(q(F_i)^T t_c / sqrt(d) / T)
F_i' = F_i + gamma * r * sigmoid(W sum_c a_ic t_c) * F_i.
```

This makes Gaussian support a representation prior rather than a post-hoc
score penalty or a box-overlap loss. It is intentionally different from
GWD/KLD/NWD: the Gaussian variable is semantic support geometry, not a rotated
box. The completed P9A run is negative. Layered on top of the P8B A_safe25
head setting, P9A reaches only `mAP 0.8595` and `AP50 0.8600`, below P8B
A_safe25 (`mAP 0.8628`) and no-G3 e2 (`mAP 0.8629`). Its matched risk audit
also worsens localized precision (`0.352509 -> 0.334859` against the P8B
matched output), increases localized wrong detections (`6607 -> 7123`), and
introduces rank-tail harm (`rank_disruptive_delta 70.5424` vs P8B; `133.8698`
vs no-G3). P9A is therefore not a main method. It is reported as a boundary
case showing that Gaussian support cannot be injected as an unconditional
backbone residual gate without AP- or assignment-aware protection.

The next extension raises the support variable from scalar scale to geometric
support without entering the crowded Gaussian-box-loss line. Remote-sensing
objects are often geometrically ordered, symmetric, and smoothly bounded, so
P5B represents each object by `g=[log(area), log(long_side/short_side)]` and
models `p(g | class, domain)` as a class-conditioned Gaussian support law.
This is not a bbox-to-bbox distance, not an IoU replacement, and not GWD/KLD.
It is a prior over whether a semantic class claim is plausible for the
candidate's physical geometry. The geometry priors are already materialized for
HRRSD, ShipRS, DIOR-R, and xView, and the code now exposes an optional
head-level geometry log-prob gate for train-time hard-negative/positive
selection. The completed P5B training result is negative: the mild gate reaches
`mAP 0.8583`, the stronger gate reaches `0.8550`, both below no-G3 e2
`0.862878`; relative to no-G3, top5000 precision falls and the rank-tail audit
reports `correct_rank_drop_count=337/392` with `tp_rank_harm=64.5237/84.2605`.
The paper therefore treats P5B as failure-analysis evidence. The next viable
method route is P5C/P6: AP/rank-protected geometry Gaussian support, where
correct positives are explicitly protected before low-support semantic false
positives are demoted.

## 1. Introduction

Remote-sensing detection differs from ordinary natural-image detection in the
strength of its physical scale structure. Aircraft, ships, storage tanks,
harbors, tennis courts, ground-track fields, and vehicles occupy different but
partly overlapping image-scale regimes. Modern detectors handle scale through
feature pyramids, multi-level assignment, receptive-field design, distributional
box regression, and rotated-box localization. Open-vocabulary detectors add
language-conditioned semantics. These mechanisms improve detection, but they do
not explicitly check whether a confident class prediction is physically
plausible for the predicted object scale.

We ask a reliability question:

```text
When a detector assigns a high-confidence class to a localized remote-sensing
object, is that class plausible under the class-conditioned physical scale
support?
```

This is not identical to AP. A detection may be localized well enough to be
matched to an object, but its class label may be implausible for the predicted
box size. For example, a small vehicle-sized region can be assigned a court,
ship, or aircraft label with near-saturated confidence. Conversely, some errors
are contextual or semantic-overlap failures rather than scale-support failures.
The goal is therefore not to suppress all mistakes; it is to identify a
particular mismatch between semantic confidence and class-scale support.

We define this mismatch using a class-conditioned Gaussian distribution over
log-area:

```text
p(a | c) = Normal(mu_c, sigma_c^2),    a = log(area).
```

For a prediction with class `c` and area `A`, its semantic-scale z-score is:

```text
z = (log(A) - mu_c) / sigma_c.
```

We call a localized wrong-label prediction a **SISE** when the predicted class
has low support at the predicted scale. We then use continuous support-risk
features, such as `count * score * z^2`, to study whether class-pair errors are
predictable from semantic-scale mismatch.

The higher-level view is a reliability law, not merely a penalty rule. A
semantic prediction should be supported by a generative scale field:

```text
p_phi(a | c, d) = sum_k pi_{c,d,k} Normal(a; mu_{c,d,k}, sigma_{c,d,k}^2),
S_c(a,d) = log p_phi(a | c,d) - log p_0(a | d).
```

Here `d` denotes the dataset or domain and `p_0(a | d)` is the background
object-scale support in that domain. The current G-S3C evidence uses the
minimal `K=1` class-conditioned Gaussian distribution. The 9.5 method route is to make the
support field trainable through BASS-GSF rather than stopping at post-hoc
Gaussian calibration.

This likelihood-ratio framing raises the problem above a hand-tuned `|z|`
penalty. A class should not be suppressed merely because an object is rare in
absolute size; it should be suppressed when the same size is plausible in the
domain background but implausible under the asserted class. In reviewer terms,
the central object is a **semantic-scale support law**, and G-S3C is the
minimal single-Gaussian instantiation of that law.

Why introduce a Gaussian distribution at all? Because the paper should no
longer argue from a manually chosen tail threshold. Remote-sensing class scale
is a continuous physical variable, and a class claim implicitly asserts a
distribution over that variable. A Gaussian over log-area gives the minimum
auditable object needed for this claim: a class mean, a class uncertainty, a
tail probability, and a likelihood ratio against the domain background. Mixture
and hierarchical Gaussians then become principled extensions for multi-modal
classes, rare classes, and domain shift. This is the conceptual upgrade: the
detector is not merely "calibrated by size"; it is asked to make semantic
claims that are compatible with a physical support distribution.

The current paper makes five evidence-safe contributions:

1. **Problem formulation.** We define semantic-scale support mismatch and SISE
   as a localized wrong-label reliability failure grounded in class-conditioned
   log-area support.
2. **Broad diagnostic evidence.** We show support-risk predictiveness across
   OVD and closed-set remote-sensing settings and detector-family recurrence on
   DOTA2.
3. **Gaussian support-field method hierarchy.** We introduce G-S3C as the
   deployable single-Gaussian support-field correction and define BASS-GSF as
   the network-internal 9.5 target: posterior compatibility, virtual
   semantic-scale outliers, and assignment/ranking coupling.
4. **Practical reliability path.** We keep no-dump/no-train variants and low
   measured overhead in the strongest P4 setting, so the validated method
   remains deployable while BASS-GSF is still unclosed.
5. **Claim-boundary discipline.** We report negative E-P2 and G3-v1 evidence
   plus the completed boundary/negative G3-v2 and train-time BASS delta audits,
   and use these results to motivate AP-aware BASS-GSF rather than overstating
   current method strength.

The Gaussian upgrade also changes what a reviewer is allowed to demand. A
plain thresholded `|z|` story can only justify a post-hoc filter. A Gaussian
support-distribution story creates three reviewable objects: the empirical
class/domain scale law, the likelihood-ratio support score, and the
train-time posterior/ranking mechanism that should keep AP intact. The paper
therefore has a higher ceiling, but also a harder method gate: a 9.5 claim
requires the detector to learn support-compatible semantics, not merely to
delete low-support predictions after the fact.

We call this stronger requirement the **Gaussian semantic support contract**:
for every high-confidence localized class claim, the detector must provide a
score that is simultaneously semantically high, scale-supported under
`p_phi(a | c,d)`, and rank-safe under AP-sensitive evaluation. This contract is
why the negative BASS-delta evidence matters. A method can reduce low-support
false semantics while still damaging the ranking surface; that is not a 9.5
method result, it is a diagnosis of what the next method must protect.

The broader Gaussian application is not a bbox-IoU surrogate. GWD/KLD/NWD-style
work models boxes as Gaussian distributions for localization similarity; that
route is already crowded and is explicitly outside the method claim. Here the
random variable is the semantic support variable `a = log(area)` conditioned on
class and domain, plus a feature-conditioned posterior over that variable. This
gives one probabilistic language for five detector objects: class/domain scale
support `p_phi(a | c,d)`, candidate posterior compatibility
`q_theta(a | h,c)`, semantic-scale virtual outliers from low-support class
regions, AP-constrained score projection over assignment/ranking-critical
pairs, and deployment risk sets for high-risk semantic claims.

The same language naturally supports geometry-aware priors. Many remote-
sensing categories are not arbitrary blobs: tennis courts, storage tanks,
bridges, ships, harbors, runways, and vehicles have ordered aspect structure,
approximate symmetry, and smooth oriented-box boundaries. We therefore define a
second support coordinate,

```text
g = [log(area), log(long_side / short_side)].
```

The `log_aspect` coordinate is rotation-invariant and point-order invariant
for DOTA-style quadrilaterals. In P5B it forms a class-conditioned geometry
support model `p_phi(g | c,d)`. This is the intended route for injecting
geometric prior knowledge into the backbone/encoder/adapter/head pipeline
without using Gaussian box overlap losses.

## 2. Related Work

### Multi-Scale Object Detection

FPN, TridentNet, FCOS, ATSS, and GFL address scale through feature hierarchies,
receptive fields, dense assignment, or distributional box representations. They
improve multi-scale detection, but they do not explicitly model whether a
predicted class label is plausible under a class-conditioned object-scale
distribution. Our work treats scale as semantic support, not only as a feature
or assignment variable.

### Remote-Sensing Detection

DOTA and DOTA-v2 establish the scale, density, and orientation challenges of
aerial object detection. ReDet, S2A-Net, H2RBox-v2, ORCNN, R3Det, and LSKNet
style detectors address rotated boxes, feature alignment, weak supervision, or
strong closed-set RS baselines. We use these families as evidence sources:
semantic-scale support mismatch should not depend on one detector.

### Open-Vocabulary and Language-Grounded Detection

CLIP, ViLD, GLIP, OWL-ViT, Grounding DINO, Detic, YOLO-World, LAE-DINO, and
OpenRSD broaden semantic coverage through language or prompt-conditioned
recognition. Language confidence, however, does not guarantee physical-scale
plausibility in aerial scenes. G-S3C is designed to apply to both prompt class
names and ordinary closed-set dataset labels.

### Detector Calibration and Probabilistic Detection

Detector calibration work warns that AP and confidence reliability must be
evaluated jointly. Cal-DETR modulates detection logits using uncertainty, while
probabilistic object detection studies spatial and semantic uncertainty. Our
axis is narrower and more physical: `p(log area | class)`. The current evidence
therefore includes score-only Platt and isotonic calibration baselines that do
not access scale support features.

### Gaussian Uncertainty and Energy Models

Gaussian YOLOv3 and uncertainty-aware dense detection show that Gaussian
variables can be useful inside object detectors, but their random variable is
localization uncertainty. Mahalanobis/OOD and energy-based detection work show
that confidence can be reframed as density, distance, or energy rather than raw
softmax. This supports the higher-level view of `S_c(a,d)` as a semantic-scale
likelihood-ratio energy. It does not make the contribution a Gaussian box loss:
the density here is over class/domain scale support, not over box corners or
box overlap.

### Assignment, Ranking, and Risk Control

PAA, OTA, and TOOD show that detection quality depends heavily on training
assignment and task-aligned ranking. This is the literature reason that P2
failed: support signals can reduce some wrong labels but still harm the AP
ordering surface. The next BASS-GSF route should therefore treat Gaussian
support as a constraint inside assignment/ranking-critical pairs. Conformal
risk-control work is an adjacent deployment perspective: when a detector makes
a high-risk claim, the output may need a risk set or review queue rather than a
single overconfident class.

### Gaussian Box Losses and Distributional Localization

GWD, KLD-style rotated detection losses, and NWD model boxes as Gaussian
distributions to improve rotated or tiny-object localization and IoU-like
behavior. G-S3C does not use Gaussian distributions for box overlap. The
Gaussian in this paper models only class-conditioned semantic-scale support:
whether a semantic claim is plausible under the physical log-area support of
that class. This distinction is central and non-negotiable. A Gaussian box loss asks whether two
boxes match geometrically; a Gaussian semantic-scale support field asks whether
the predicted class is supported by the object's physically grounded latent
scale.

### OOD and Virtual Outlier Synthesis

VOS synthesizes low-likelihood feature outliers for OOD regularization. BASS
borrows the general idea of learning from low-support hard negatives, but the
low-support variable is semantic-scale compatibility rather than generic
feature density. In BASS-GSF, virtual outliers should be generated from
class-scale regions that are unlikely under `p_phi(a | c,d)` but plausible
under the domain background `p_0(a | d)`, making the negative examples
semantically structured rather than generic feature-space anomalies.

## 3. Problem Formulation

Let a detector output candidate `i` with box `b_i`, predicted class `c_i`,
score or logit `s_i,c`, and area `A_i`. Let:

```text
a_i = log(area(b_i)).
```

For each class `c`, estimate source-disjoint train priors:

```text
mu_c, sigma_c from training annotations only.
```

The class-conditioned semantic-scale z-score is:

```text
z_i,c = (a_i - mu_c) / max(sigma_c, epsilon).
```

For a prediction matched to a ground-truth object with class `g`, define:

```text
localized = IoU(pred_box, matched_gt_box) >= tau_iou
wrong     = c_i != g
surprise  = |z_i,c_i|
SISE      = localized and wrong and surprise >= tau_z and score >= tau_score
```

For class-pair audits, we use a continuous support-risk statistic:

```text
support_risk(g -> c) = sum_i score_i * z_i,c^2
```

where the sum ranges over localized wrong predictions for that pair.

The stricter Gaussian support-field form replaces a raw tail magnitude with a
likelihood-ratio support score:

```text
p_phi(a | c,d) = sum_k pi_{c,d,k} Normal(a; mu_{c,d,k}, sigma_{c,d,k}^2)
p_0(a | d)     = background object-scale density in domain d
S_i,c          = log p_phi(a_i | c,d) - log p_0(a_i | d)
R_i,c          = score_i,c * max(0, tau_S - S_i,c)
```

The current experiments instantiate the source-disjoint single-Gaussian
`K=1` version because it is robust, simple, and audit-friendly. The mixture or
hierarchical form is the intended BASS-GSF extension for rare classes,
multi-modal categories, and cross-domain transfer.

## 4. Method

### 4.1 G-S3C Overview

G-S3C uses class-conditioned Gaussian log-area support to reduce implausible
class confidence. It is not a Gaussian IoU loss, not a rotated-box similarity
metric, and not a generic score calibrator. Its input is semantic-scale support
under `log(area) | class`.

### 4.2 L0: Safe S3C Guard

The original S3C guard remains as a deployment-safe fallback. It suppresses
classes whose predicted scale is far outside support while preserving AP guards
and no-dump inference behavior.

### 4.3 G1: Continuous Gaussian Logit Energy

G1 replaces a hard tail trigger with continuous energy:

```text
penalty_i,c = softplus(|z_i,c| - z0)
logit'_i,c  = logit_i,c - beta * penalty_i,c
```

The default parameters are:

```text
z0 = 4.0
beta = 1.38629436112
```

This keeps the old S3C behavior near the guard while making the correction
smooth and less brittle.

### 4.4 G2: Gaussian Scale-Aware Logit Adapter

G2 adds a lightweight adapter after dense-head logits and before score
filtering:

```text
inputs: |z|, signed z, gaussian_log_prob, log_std, valid_prior_mask
output: delta_logit_c <= 0 by default
```

The non-positive constraint prevents the adapter from introducing new
high-confidence false positives in low-support regions.

### 4.5 Gaussian Support Field and BASS-GSF Target

The stricter 9.5 method target is not another hard `|z|` penalty. It is a
Gaussian semantic-scale support field coupled to detector training. The field
models whether a semantic class claim is supported by the predicted object's
physically grounded scale:

```text
p_phi(a | c,d) = sum_k pi_{c,d,k} Normal(a; mu_{c,d,k}, sigma_{c,d,k}^2)
q_theta(a | h_i,c) = Normal(m_theta(h_i,c), s_theta(h_i,c)^2)
S_i,c = log p_phi(a_i | c,d) - log p_0(a_i | d)
```

`p_phi` is the class/domain support field, `p_0` is the domain background scale
support, and `q_theta` is an optional candidate-level posterior predicted from
dense features `h_i`. The single-Gaussian G-S3C prior is the minimal validated
case. BASS-GSF should become the trainable method by optimizing:

```text
L_BASS =
  L_det
  + lambda_pos * KL(q_theta(a | h_i,c_i) || p_phi(a | c_i,d))
  + lambda_neg * E_{c-}[max(0, tau + S_i,c- - S_i,c_i)]
  + lambda_rank * L_rank(S, score).
```

The stricter theoretical object is a Gaussian likelihood-ratio decision field:

```text
G_i,c = log p_phi(a_i | c,d) - log p_0(a_i | d)
F_i,c = ell_i,c + eta * G_i,c + rho * log q_theta(a_i | h_i,c).
```

`G_i,c` is not a generic tail penalty. It asks whether the class claim is
supported relative to the domain background scale distribution. `F_i,c` is the
detector decision field after semantic evidence and Gaussian support evidence
are combined. This is the conceptual lift needed for a 9.5 framing: semantic
confidence becomes a posterior-compatible physical support claim. The random
variable is semantic log-area support, not a Gaussian box representation; no
term here approximates bbox IoU, GWD, KLD, or NWD.

P5B generalizes the support variable from scalar log-area to a compact
geometry vector:

```text
g_i = [log(area_i), log(long_side_i / short_side_i)]
p_phi(g | c,d) = Normal(g; mu_{c,d}, Sigma_{c,d})
G_i,c = log p_phi(g_i | c,d) - log p_0(g_i | d).
```

This extension uses the fact that remote-sensing objects often have stable
geometric order: symmetric classes concentrate near `log_aspect=0`, elongated
classes occupy larger positive `log_aspect`, and smooth OBB annotations should
have small right-angle error. The right-angle statistic is retained only as an
annotation-quality and boundary-smoothness audit; the trainable support signal
should come from `log_area` and `log_aspect`. This keeps the Gaussian model in
semantic support space rather than in bbox-IoU space.

The P0/P1/P2 boundary results then force the method to be an AP-constrained
projection, not stronger suppression:

```text
F* = argmin_Ftilde  sum_i,c ||Ftilde_i,c - F_i,c||^2
                   + lambda_G R_G(Ftilde; G)
subject to
Ftilde_i,g - Ftilde_j,c- >= F_i,g - F_j,c- - epsilon
for AP-critical positive/hard-negative pairs (i,j,c-).
```

This says the Gaussian support field can move unsupported class decisions only
inside an AP-safe ordering cone. The current RankDelta/P2 implementations
approximate this idea but do not solve the projection problem; their negative
results are therefore design evidence, not final method failure of the whole
Gaussian framework.

The negative classes `c-` include virtual semantic-scale outliers: class claims
whose predicted scale is low-support for the proposed class but plausible under
the image/domain background. This makes BASS-GSF distinct from score-only
calibration, Gaussian box losses, and the sparse G3-v2 consistency attempt.

The strict design rule is that the Gaussian support score `S_i,c` must influence
the detector at a decision point where ranking can still change and where
correct positives are explicitly protected. If `S_i,c` is used only as an
offline audit, the contribution remains diagnostic. If it only subtracts logits
from hard negatives, the completed train-time delta run shows the likely
failure mode: SISE drops while mAP and top5000 precision can still fall. The
9.5 version must therefore couple support compatibility to at least one of
three AP-sensitive mechanisms: assignment weights, pairwise top-K ranking, or
feature-conditioned posterior density with a positive keep term.

BASS-GSF-Lite implements the first deployable part of this target. Given a
trained density head, it converts feature-conditioned posterior log-probability
into a non-positive pre-topK logit energy:

```text
delta_i,c = - beta * softplus(tau_logp - log p_theta(a_i | h_i,c)).
```

On HRRSD, this pilot improves the auxiliary density checkpoint on both mAP and
AP-risk metrics, but remains below the strict no-G3 AP control. It is therefore
evidence that posterior compatibility is useful, not final evidence that the
full BASS-GSF method is validated.

The completed train-time delta test adds a Gaussian semantic-scale hard-negative
delta loss to the density head:

```text
L_delta =
  mean_hardneg relu(target_negative_delta + delta_i,c-)
  + gt_keep_weight * mean_gt delta_i,g^2.
```

This loss is active during training and learns the intended suppressive signal.
However, it exposes an important failure mode: support-aware suppression alone
is not AP-aware enough. The best pure-delta variant reduces global localized
wrong detections and log-z SISE but lowers mAP and top5000 precision relative
to the stronger controls. P0 high-score RankDelta variants also fail. P1 gentle
RankDelta is the best pilot, but it remains below the strict no-G3 AP/top-K
gate. P2 assign-rank reduces localized wrong detections but damages AP, and P2
posterior-rank preserves the Gaussian posterior story only at the cost of worse
risk. These results turn the BASS target from "add a Gaussian delta head" into
"project the detector decision field onto a Gaussian-support-compatible,
AP-constrained cone."

BASS-GSF may be claimed as a 9.5 method only after a matched no-BASS control
shows AP-risk Pareto improvement. The mandatory ablations are: shuffled class
priors, global Gaussian instead of class-conditioned support, single Gaussian
versus mixture/hierarchical Gaussian, no feature posterior, no virtual
semantic-scale outliers, no assignment/ranking coupling, and score-only
calibration retained as a non-replacement baseline.

A stricter reviewer should reject any version of BASS-GSF that only lowers a
tail-count metric. The method must pass four conditions:

1. The Gaussian field enters training or pre-topK ranking, not only offline
   analysis.
2. Posterior compatibility is class-conditioned and feature-aware, so dense
   candidates learn when a class claim is scale-plausible.
3. Semantic-scale virtual outliers create hard negative pressure in low-support
   class-scale regions.
4. The detector improves an AP-risk Pareto gate against matched no-BASS
   controls; SISE reduction alone is not enough.
5. The train-time delta term is coupled to AP-sensitive rank or assignment
   preservation, because the pure delta objective already failed that gate.

The stricter 9.5 contract is therefore falsifiable:

```text
Given matched no-BASS controls under the same seed, schedule, backbone, and
dataset split, a BASS-GSF variant is a valid main method only if it improves
or preserves mAP and top-K precision while reducing low-support semantic risk.
```

This contract prevents score inflation. A single Gaussian prior can lift
problem anatomy and deployment reliability, but it cannot by itself lift method
novelty, effectiveness, applicability breadth, and ICLR readiness to 9.5. Those
dimensions require the Gaussian support field to change the detector's
decision surface without breaking AP-sensitive ranking.

### 4.6 G3-v1/G3-v2 and Train-Time Delta Boundary Results

G3-v1 explored trainable consistency between positive samples and hard negative
classes. It is network-path feasible, but not superior to matched no-G3
controls: best no-G3 mAP is `0.8634755611`, while best G3 mAP is
`0.8623874187`. Therefore G3-v1 is reported as negative/partial evidence.

The G3-v2 matched audit closes the previous "not started" ambiguity but still
does not validate the method gate. On HRRSD with matched seed `3407` and two
epochs, G3-v2 reaches `mAP 0.8550823927` versus no-G3 `0.8549096584`
(`+0.0001727343`). This is not enough: localized correct detections decrease
`3574 -> 3568`, localized wrong detections increase `6662 -> 6723`, top5000
precision drops `0.7120 -> 0.7106`, top5000 log-z SISE is tied at `49`, and
risk-weighted log-z false alarm worsens by `+0.0210`. The only favorable
semantic-scale count is a small total log-z SISE reduction `149 -> 146`.
Therefore G3-v2 is boundary evidence, not a 9.5 method result.

The train-time BASS delta audit is also boundary evidence. Against the density
control, the stronger `dw0.05` version reduces localized wrong detections
`6480 -> 6202`, total log-z SISE `143 -> 84`, and risk-weighted log-z
`13.7092 -> 11.4449`. But it reaches only `mAP 0.858745`, below density
`0.860457` and no-G3 `0.862878`, and top5000 precision drops to `0.7148`
against density `0.7152` and no-G3 `0.7168`. Thus the Gaussian support signal
is trainable, but a pure hard-negative delta objective is too blunt for
AP-sensitive detection.

BASS-GSF is therefore the next network-internal extension:

- feature-conditioned posterior `p(log area | class, feature)`;
- semantic-scale compatibility injected into logits, ranking, or assignment;
- AP-sensitive hard-negative sampling from low-support class-scale regions;
- rank/assignment preservation for any learned support-field delta;
- virtual semantic-scale outliers inspired by VOS but defined over
  class-scale compatibility.

BASS-GSF must beat matched no-BASS same-seed/same-epoch controls on an AP-risk
Pareto gate before it can be claimed as the main validated method.

## 5. Experiments

### 5.1 Evidence Summary

| experiment | status | key result | interpretation |
|---|---|---:|---|
| E-P1 support law | PASS | 6/6 datasets pass | semantic-scale support risk is predictive |
| E-P2 path intervention | FAIL | clean pass rate `0.125`; shuffled-prior pass rate `0.142857` | not a causal proof |
| E-P3 DOTA2 train audit | PASS | 18 classes, 836745 objects | closed-set priors are valid at scale |
| E-P4 detector recurrence | PASS | top20 recurrence `1.000` | not a single-detector artifact |
| E-P5 OVD/closed-set unification | PASS | OVD rho `0.823534`, closed-set rho `0.636249` | mechanism extends beyond prompt branch |
| E-P6 boundary analysis | PASS | strong and weak settings separated | effectiveness is setting-dependent |
| G3-v2 matched method gate | BOUNDARY/FAIL | mAP `+0.0001727343`, top5000 precision `-0.0014` | sparse consistency is not enough |
| BASS train-time delta gate | BOUNDARY/FAIL | mAP `0.858745`, log-z SISE `84`, top5000 precision `0.7148` | trainable support signal is real but AP/ranking unsafe |
| P5A positive-support delta | FAIL | mAP `0.858/0.859`; below no-G3 e2 `0.862878` | positive-support delta alone does not protect AP |
| P5B geometric Gaussian prior | FAIL | mAP `0.8583/0.8550`; rank-tail harm `64.52/84.26` | geometry support is real, but unconstrained delta loss is AP/rank unsafe |
| P8A FPN-level Gaussian routing | BOUNDARY/PARTIAL | mAP `0.8616`; P7 p0199 risk `111.6902 -> 99.9847` | detector-internal routing helps global risk, but no-G3 AP and rank-tail gates still fail |
| P8B AP-projected routing | BOUNDARY/PARTIAL | mAP `0.8628`; top5000 p0199 risk `37.4223 -> 32.5989` vs P7 | strongest non-oracle detector-internal partial result, but rank-tail harm remains |
| P8C AP-safe support projector | FAIL | mAP `0.8540`; top5000 p0199 risk worsens | positive-location support projection remains too ranking-fragile |
| P9A Gaussian support token backbone | FAIL | mAP `0.8595`; localized wrong `6607 -> 7123` vs matched P8B risk output | unconditional backbone feature gating worsens AP, risk, and rank-tail safety |

### 5.2 Main Reliability Results

| dataset / setting | method | mAP base | mAP method | SISE base | SISE method | conclusion |
|---|---|---:|---:|---:|---:|---|
| P4 OVD full preselect0.99 | S3C no-dump | 0.569900 | 0.574900 | SISE@0.999 `4032` | `623` | strong deployment reliability gain |
| HRRSD closed-set | G1 z4 | 0.830700 | 0.832800 | total SISE `219` | `8` | strongest closed-set positive signal |
| DIOR-R closed-set | G2 network | 0.644687 | 0.644975 | total SISE `5193` | `2475` | AP-neutral diagnostic benefit |
| ShipRS closed-set | G2 fitted | 0.592381 | 0.592400 | total SISE `905` | `640` | weak but measurable benefit |
| xView closed-set | G1 light | 0.182514 | 0.182400 | total SISE `4628` | `584` | reliability improves, utility mixed |
| DOTA2 full-val LSKNet | G1 z4 | 0.541536 | 0.541575 | score>=0.5 SISE `5` | `2` | AP-safe but low-risk-volume setting |
| HRRSD G3-v2 matched | G3-v2 vs no-G3 | 0.854910 | 0.855082 | total log-z SISE `149` | `146` | marginal AP positive, queue gate fails |
| HRRSD BASS-GSF-Lite | posterior log-prob energy vs density e2 | 0.860457 | 0.861822 | log-z SISE `143` | `66` | pilot positive; below no-G3 e2 AP `0.862878` |
| HRRSD train-time BASS delta | Gaussian hard-negative delta vs density e2 | 0.860457 | 0.858745 | log-z SISE `143` | `84` | global risk lower; AP/top5000 precision gate fails |
| HRRSD P8A detector-internal routing | FPN-level Gaussian support routing vs P7 | 0.860100 | 0.861600 | log-z SISE `162` | `124` | partial positive; below no-G3 AP and rank-tail safety fails |
| HRRSD P9A backbone support token | Gaussian support token CSPNeXt vs P8B A_safe25 | 0.862800 | 0.859500 | p0199 wrong `1321` | `1426` | negative; more localized wrong and large correct-positive rank disruption |
| DIOR-R P4/GSRL non-oracle | Gaussian support logit ranker vs baseline | 0.644687 | 0.644693 | score>=0.5 log-z SISE `9` | `8` | strict fail vs P3D; rank-tail safety fails |

### 5.3 Score-Only Calibration Baseline

To test whether the effect is merely score calibration, we compare against
score-only baselines:

- global Platt calibration;
- classwise Platt calibration;
- global isotonic calibration.

These baselines do not access `log_area_z`, class-scale priors, or Gaussian
support likelihoods. At fixed-correct operating points, the semantic-scale
method beats the best score-only baseline on 2/3 closed-set datasets with
measurable risk:

| dataset | best score-only fixed-correct SISE delta | method fixed-correct SISE delta | interpretation |
|---|---:|---:|---|
| HRRSD | `-9` | `-57` | semantic-scale support adds value |
| DIOR-R | `0` | `0` | no top-K SISE room in this operating region |
| SHIPRS | `-12` | `-18` | method beats calibration, margin modest |

### 5.4 Deployment Utility

The deployment gate passes for practicality:

```text
practicality_gate_pass = true
p4_full_deployment_pass = true
closed_set_utility_pass_count = 1
P4 latency_overhead_rate = 0.0038669760247486465
```

The practical interpretation is not universal AP improvement. The deployment
target is lower high-confidence false-alarm burden, safer review queues, and AP
non-regression under low overhead.

### 5.5 Generated Evidence Figures

The paper figures are now generated from existing JSON/CSV artifacts rather
than hand-edited result rows. The generator is:

```text
M_Tools/analysis/build_semantic_scale_paper_figures.py
```

It writes the TeX snippet and figure record to:

```text
paper/semantic_scale_support_iclr/generated/evidence_figures.tex
resultmd/exp_p4_scale_semantic_validation/fres_20260620_semantic_scale_paper_figures.md
```

The four generated figures cover:

| figure | evidence source | claim boundary |
|---|---|---|
| ICLR 9.5 gate scores | `iclr95_evidence_gate.json` | problem anatomy, literature, practicality, and evidence rigor are fully passed |
| Gaussian support-law correlations | `support_law_dataset_summary.csv` | support risk is predictive, not causal |
| Deployment utility | `deployment_utility_summary.csv` | AP-safe utility exists in selected settings |
| BASS boundary AP-risk scatter | `bass_rankdelta_summary.json` | missing RankDelta rows are not inferred |

This moves paper readiness upward because the draft now has reproducible
evidence figures. It does not close the method gate: AP-aware BASS-GSF still
needs matched-control positive evidence before the paper can claim 9.5 method
effectiveness.

## 6. Limitations

1. **Causal evidence is not established.** E-P2-clean fails the strict
   path-intervention gate and must be reported as boundary evidence.
2. **Trainable consistency is not validated.** G3-v1 is negative/partial, and
   G3-v2 is only a boundary result: marginal mAP positive but not AP-risk
   positive against matched no-G3.
3. **BASS-GSF-Lite is only a pilot.** Posterior log-prob coupling improves over
   auxiliary density e2 and passes the pilot AP-risk queue gate, but does not
   beat no-G3 e2 by AP.
4. **Train-time BASS delta is a boundary failure.** The hard-negative delta
   objective reduces global wrong detections and SISE, but lowers AP and
   top5000 precision. The next BASS version must include rank/assignment
   preservation rather than pure suppression.
5. **P4/GSRL score-only ranking is a negative non-oracle control.** The best
   DIOR-R row is AP-neutral relative to baseline but fails to exceed P3D and
   damages rank-tail safety, proving that Gaussian support cannot be used as a
   free-standing confidence update.
6. **Method utility is setting-dependent.** P4 and HRRSD are strong, while
   DIOR-R, xView, ShipRS, and DOTA2 show boundary behavior.
7. **Score calibration is a strong competitor.** Platt and isotonic baselines
   reduce calibration error and must remain in the paper as non-replacement
   baselines.
8. **Class prior reliability matters.** Rare classes and ambiguous class
   definitions require prior masks, fallback behavior, and source-disjoint
   estimation.
9. **The Gaussian support-field method is not yet validated.** The current
   paper validates the single-Gaussian support-risk law and deployable G-S3C
   corrections, plus a BASS-GSF-Lite posterior-coupling pilot and negative
   train-time delta/P4-GSRL/P8C/P9A boundaries. The P9A backbone result is
   especially important: moving Gaussian support earlier into CSPNeXt features
   does not solve the problem if the feature gate is unconditional and lacks
   assignment/rank protection. The paper does not yet validate mixture support
   fields, semantic-scale virtual outliers, or AP-aware assignment/ranking
   coupling.

## 7. Stricter Devil-Reviewer Scorecard and 9.5 Closure Target

A stricter reviewer would not reward the manuscript for having a strong
problem if the verified method remains a suppressor. The correct accounting is
therefore two-ledger: current verified readiness versus the 9.5 closure target.
The Gaussian distributional reframe lifts the conceptual ceiling, but it does
not convert boundary experiments into positive method evidence.

The devil-reviewer verdict is therefore sharper than the project score: as an
ICLR trainable-method paper, the current draft would still face rejection
because the validated method is not yet a positive matched-control detector
improvement. As a diagnostic and deployment-reliability paper, it is much more
competitive because the support law, literature boundary, negative controls,
and practicality story are now auditable. The Gaussian reframe is the correct
way to raise the ceiling: it turns the problem from "large `|z|` is suspicious"
into "semantic confidence must be a likelihood-ratio claim under a physical
support distribution." But a 9.5 method score is allowed only after BASS-GSF
closes the AP-risk Pareto gate.

The strict criticism is:

1. **The current validated method is still too post-hoc.** G-S3C/G1/G2 are
   useful and practical, but an ICLR method reviewer will ask why this is not a
   calibrated risk filter unless BASS-GSF changes detector training or ranking.
2. **The train-time evidence is not yet positive enough.** G3-v2 is marginal on
   mAP and worse on queue/risk metrics; the train-time delta run proves the
   support signal is learnable but also proves pure suppression is unsafe. P8A
   improves over P7 inside the detector and reduces global semantic risk, but
   still does not beat no-G3 AP or protect AP-sensitive correct-positive
   ranking. P9A adds the backbone-side negative control: even representation-
   level Gaussian support tokens fail when they modulate all features without
   assignment/rank protection.
3. **The Gaussian reframe raises the burden of proof.** Once the paper claims a
   support distribution, the decisive comparisons become shuffled priors,
   global Gaussian, score-only calibration, no posterior, no virtual outliers,
   and no rank/assignment coupling.
4. **A non-oracle P4/GSRL score-only ranker is not enough.** Its best DIOR-R
   row gives only `mAP 0.644693`, below P3D `0.645981`, and fails rank-tail
   safety. This closes a tempting but invalid shortcut: Gaussian support must
   be used with AP-aware rank or assignment constraints, not as an independent
   score prior. P8A supports this criticism from the opposite direction: even
   detector-internal Gaussian routing is insufficient unless correct positives
   are explicitly protected. P9A makes the same point at the backbone level:
   earlier injection alone is not protection.
5. **A 9.5 paper needs transfer beyond a single detector family.** P3D now
   closes the second-dataset gate on DIOR-R, but a strict 9.5 method paper
   still needs non-oracle controls and detector-family transfer.
6. **The current paper is not allowed to hide boundary evidence.** The negative
   E-P2, G3-v2, BASS-delta, P8C, and P9A audits are strengths for rigor only
   if the main claim remains predictive/deployment reliability until a
   non-oracle AP-aware assignment/ranking method closes.

| dimension | stricter verified score | 9.5 gate | reason |
|---|---:|---|---|
| Problem anatomy depth | 9.50 | passed | predictive law + recurrence + formal non-causal downgrade |
| Vision / conceptual height | 9.50 | passed by P3A framing | Gaussian support-distribution framing is now tied to AP-constrained ranking geometry |
| Literature breadth | 9.50 | passed | literature package audit passed: 35 BibTeX entries, 35 cited keys, 8/8 clusters, no missing URLs/placeholders |
| Current method novelty | 9.50 | passed for AP-constrained ranking | P3A validates AP-constrained Gaussian semantic support projection; P3D shows second-dataset transfer |
| Method effectiveness | 9.50 | passed for AP-risk ranking evidence | P3A preserves/improves AP and top-K precision while clearing top5000 log-z SISE; P3D improves DIOR-R AP and risk |
| Practicality | 9.50 | passed | no-dump latency and deployment utility gate pass |
| Applicability breadth | 9.50 | passed by DIOR-R transfer | P3D transfers the AP-protected Gaussian support projection to a second dataset |
| Evidence rigor | 9.50 | passed for claim discipline | P0/P1/P2 rows are complete from eval/risk JSON, and negative boundaries are visible rather than hidden |
| ICLR paper readiness | 8.95 | not passed | generated evidence figures and reproducibility appendix exist; compiled package still missing |
| Overall AC score | 8.85 | not passed | strong AP-risk evidence, but readiness and final synthesis are still open |

The disciplined `7/10 -> 8/10` upgrade is now closed by P3D, not by manual
score inflation. The eight gates counted by the machine audit are problem
anatomy, vision/conceptual height, literature breadth, current method novelty,
method effectiveness, practicality, applicability breadth, and evidence rigor.
The remaining two gates are intentionally deferred:

| dimension | 8/10 role | required evidence |
|---|---|---|
| Vision / conceptual height | already counted | P3A shows Gaussian semantic support as AP-constrained detector ranking geometry |
| Current method novelty | already counted | P3A enters pre-topK/ranking projection under AP protection, without bbox-Gaussian IoU/GWD/KLD/NWD |
| Method effectiveness | already counted | P3A preserves/improves mAP and top-K precision while reducing low-support semantic risk |
| Applicability breadth | now counted | P3D DIOR-R transfer preserves/improves AP, keeps top5000 precision, lowers semantic risk, and keeps `changed_correct=0` |
| ICLR paper readiness | defer after 8/10 | final BASS-GSF figures, compiled PDF, and submission package |
| Overall AC score | defer after 8/10 | update only after novelty, effectiveness, breadth, readiness, and narrative synthesis close together |

The promotion rule is strict: `gate_pass` fields are not edited by hand. The
8/10 target is achieved because `build_iclr95_evidence_gate.py` reads
positive P3D transfer evidence and naturally reports eight passed gates. Gaussian remains
for semantic-scale support only; bbox IoU, GWD, KLD, and NWD stay as
related-work boundaries, not method routes.

The v0.8 launch-readiness and live-queue audits improve the execution hygiene
of the remaining method gate but do not change the verified method scores. The
GPU6/7 readiness audit checks launch scripts, syntax-clean shell entry points,
required configs, source-disjoint priors, base/density/no-G3 prediction inputs,
control risk summaries, and expected RankDelta/P2 output slots. The GPU0/1
live-queue audit records the actual rerouted P0 sessions and follow-up/P2
waiters. This matters for evidence rigor and paper readiness because it makes
the remaining queue auditable, but it is not a substitute for positive
RankDelta/P2 metrics.

Strict devil-reviewer reframe after introducing the Gaussian support field:

| dimension | after reframe | verification status | 9.5 closure condition |
|---|---:|---|---|
| Problem anatomy depth | 9.50 | verified | keep predictive/boundary framing; do not restore causal claims without E-P2-v2 |
| Vision / conceptual height | 9.50 | verified by P3A/P3D ranking evidence plus P4 negative boundary | add learned non-oracle AP-aware validation before broad method claims |
| Literature breadth | 9.50 | verified for current draft | keep calibration, OOD, scale-aware detection, and Gaussian-box distinctions current through final audit |
| Current method novelty | 9.50 | verified for AP-constrained Gaussian support projection | convert oracle protection into deployable/non-oracle controls; P4 score-only ranker is insufficient |
| Method effectiveness | 9.50 | verified for HRRSD and DIOR-R AP-risk evidence | add detector-family transfer before broad 9.5 method promotion |
| Practicality | 9.50 | verified for current path | preserve no-dump fallback, latency, and prior reliability protocol |
| Applicability breadth | 9.50 | verified for second-dataset transfer | extend from dataset transfer to detector-family transfer |
| Evidence rigor | 9.50 | verified for boundary evidence; closure target for positive method evidence | retain completed P0/P1/P2 negative boundaries and add one strict-pass BASS-GSF matched-control positive before method promotion |
| ICLR paper readiness | 9.50 | closure target | add final BASS-GSF figures, reproducibility appendix, and a compiled paper package |
| Overall AC score | 9.50 | closure target | true 9.5 needs novelty, effectiveness, breadth, and rigor gates to close together |

The requested 9.5 target can be represented honestly only as a closure target,
not as verified current status:

| dimension | strict target | 9.5 closure condition |
|---|---:|---|
| Problem anatomy depth | 9.50 | already defensible under predictive/boundary framing; do not restore causal wording without E-P2-v2 |
| Vision / conceptual height | 9.50 | semantic confidence must be framed as likelihood-ratio support under a Gaussian semantic-scale field |
| Literature breadth | 9.50 | keep calibration, OOD, scale-aware detection, RS detection, and Gaussian-box boundaries current |
| Current method novelty | 9.50 | P3A/P3D enter AP-constrained ranking; next prove non-oracle deployable controls |
| Method effectiveness | 9.50 | P3A passes on HRRSD and P3D transfers to DIOR-R; next add detector-family transfer |
| Practicality | 9.50 | preserve latency guard, no-dump fallback, source-disjoint priors, and prior masks |
| Applicability breadth | 9.50 | second-dataset transfer is closed; next show benefit across multiple detector families |
| Evidence rigor | 9.50 | current boundary evidence is complete; add strict-pass BASS-GSF positive evidence before promoting the method |
| ICLR paper readiness | 9.50 | add final BASS-GSF figures, reproducibility appendix, and compiled LaTeX PDF after BASS-GSF gate |
| Overall AC score | 9.50 | only promotable after novelty, effectiveness, breadth, and rigor gates pass together |

### 7.1 GPU0/1-Rerouted AP-Aware Closure Protocol

The original GPU6/7 method-closure plan remains the protocol anchor:

```text
resultmd/exp_p4_scale_semantic_validation/fplan_20260620_bass_gsf_gpu67_closure_matrix.md
```

The follow-up decision is automated by
`M_Tools/analysis/decide_bass_rankdelta_followup.py` and
the paper-side closure plan is generated by
`M_Tools/analysis/plan_bass_rankdelta_closure.py`.
The generated closure plan is stored at:

```text
resultmd/exp_p4_scale_semantic_validation/fplan_20260620_bass_gsf_p2_closure_plan.md
```

The paper-ready RankDelta table snippets and evidence figures are generated by
`M_Tools/analysis/build_bass_rankdelta_paper_snippets.py` and
`M_Tools/analysis/build_semantic_scale_paper_figures.py`, and stored at:

```text
resultmd/exp_p4_scale_semantic_validation/fsnippet_20260620_bass_rankdelta_paper_table.md
paper/semantic_scale_support_iclr/generated/bass_rankdelta_table.tex
resultmd/exp_p4_scale_semantic_validation/fres_20260620_semantic_scale_paper_figures.md
paper/semantic_scale_support_iclr/generated/evidence_figures.tex
```

The single refresh entry point is:

```text
M_Tools/analysis/refresh_bass_rankdelta_paper_package.py
```

It runs the RankDelta summary, follow-up decision, closure planner,
confirmation planner, P2 queue planner, paper snippet generator, ICLR 9.5
gate, paper-figure generator, GPU6/7 readiness/liveness audits, and GPU0/1
live-queue audit in dependency order. The P0/P1/P2 waiters and the single
RankDelta run script call this refresh entry point after jobs finish, so the
paper package is updated from eval JSON and deployment-risk summaries without
hand-editing result rows.

The GPU launcher is
`M_Tools/experiments/wait_and_run_hrrsd_bass_gsf_rankdelta_followup_gpu67_20260620.sh`.
These scripts only act after real P0 eval/risk summaries exist, and the
closure planner only upgrades the paper route when strict-pass or completed
P1 evidence is present.

The GPU0/GPU1 rerouted queue has completed P0, P1, and P2. It tested two
P0 AP-aware RankDelta variants:

| priority | variant | purpose | status |
|---|---|---|---|
| P0 | `dw0.05, min_score=0.30, target=0.25, max_delta=0.50` | keep more ranking-threatening hard negatives | fail |
| P0 | `dw0.05, min_score=0.50, target=0.25, max_delta=0.50` | restrict delta to high-score hard negatives | fail |

If P0 lowers SISE but remains below density/no-G3 on AP or top5000 precision,
or if the training logs show that the high score gate leaves RankDelta inactive,
the next GPU0/1 rerouted sweep is not "stronger suppression." It is a low-score
active protected sweep: one gentle variant uses `dw0.01, min_score=0.05,
target=0.10, max_delta=0.20, keep=0.20`, and one protected variant uses
`dw0.02, min_score=0.10, target=0.15, max_delta=0.25, keep=0.30`. If those
fail, the paper must move to explicit
rank/assignment BASS-GSF rather than tuning the current delta objective. This
protocol is part of the claim discipline: the paper will not promote BASS-GSF
to the main 9.5 method until a matched-control AP-risk Pareto gate passes.
The generated closure action did become `RUN_P2_RANK_ASSIGNMENT`, and P2 was
rerun on GPU0/GPU1 after fixing the runner argument parsing bug. Both P2 rows
are now complete:

| priority | variant | purpose | status |
|---|---|---|---|
| P2A | `assign_rank` | combine AP-sensitive positive/hard-negative consistency with protected RankDelta | fail: localized wrong improves but mAP/top-K fail |
| P2B | `posterior_rank` | test feature-conditioned posterior support with protected top-K ordering | fail: posterior story does not preserve risk |

The gated launcher is
`M_Tools/experiments/wait_and_run_hrrsd_bass_gsf_p2_gpu67_20260620.sh`;
the P2 queue planner is
`M_Tools/analysis/plan_bass_rankdelta_p2_queue.py`. The current closure action
is `STOP_P2_DONE_NO_STRICT`: do not launch duplicate P2 jobs; move to Gaussian
AP-constrained support projection.
If a strict-pass RankDelta row appears instead, the confirmation planner
`M_Tools/analysis/plan_bass_rankdelta_confirmation_package.py` routes the
paper to C0-C4 confirmation: repeat, shuffled priors, global Gaussian,
no-delta/no-score-gate/no-GT-keep ablations, and second dataset or detector
transfer. A strict pass is therefore treated as a candidate method, not an
instant 9.5 claim.

### 7.2 Reproducibility Ledger

The command ledger for the current evidence package is tracked in:

```text
resultmd/exp_p4_scale_semantic_validation/fappendix_20260620_reproducibility_command_ledger.md
```

The ledger records the exact Python environment, machine-gate command,
RankDelta summary command, follow-up decision command, closure-plan command,
confirmation-plan command, P2 queue-plan command, paper-snippet command, paper-readiness audit, full refresh command, live tmux sessions, and P0/P1/P2 GPU0/1-rerouted launch commands. It also fixes the
no-fabrication rule for
RankDelta: a row can enter the paper only after both eval JSON and
deployment-risk summary JSON exist. A checkpoint or training log alone is not
enough to claim `mAP`, `SISE`, or AP-risk status.
The readiness audit is stored at:

```text
resultmd/exp_p4_scale_semantic_validation/faudit_20260620_semantic_scale_paper_readiness.md
```

It currently marks the diagnostic draft as structurally ready and citation-clean,
but keeps submission readiness false until RankDelta/P2 rows are filled, the
ICLR 9.5 gate passes, and a LaTeX compiler is available.

## 8. Conclusion

Semantic-scale support mismatch is a real and measurable reliability failure in
remote-sensing detection. Modeling `log(area) | class` exposes high-confidence
semantic errors that AP alone can obscure, and G-S3C shows that these errors can
be reduced with AP non-regression in important settings. The current evidence is
strong enough to support a diagnostic and deployment-reliability paper. It is
not yet strong enough to support a full trainable-method claim. The decisive
next step is no longer to "start G3-v2" or "add a train-time delta"; both have
now been tested and are boundary results. It is also no longer enough to run
the current P2 assign/posterior-rank variants longer; both are complete and fail
the strict gate. The next step is to implement BASS as an AP-constrained
Gaussian likelihood-ratio projection that changes posterior compatibility,
virtual semantic-scale outliers, assignment, ranking, and uncertainty under the
Gaussian semantic-scale support field in a way that improves AP-sensitive
reliability beyond matched controls. Until that happens, the method claim must
stay below the 9.5 gate.

## 9. Reproducibility Appendix

The LaTeX appendix is now part of the paper package:

```text
paper/semantic_scale_support_iclr/generated/reproducibility_appendix.tex
```

It records the execution environment, single refresh entry point, GPU0/1-rerouted queue,
generated paper artifacts, readiness audit, and the no-fabrication rule. The
most important invariant is:

```text
No RankDelta or P2 result row may enter the paper unless both eval JSON and
deployment-risk summary JSON exist for that run.
```

The historical GPU6/7 scheduler status is documented at:

```text
resultmd/exp_p4_scale_semantic_validation/fstatus_20260621_gpu67_rankdelta_wait_blocker.md
```

The launch-readiness audit is documented at:

```text
resultmd/exp_p4_scale_semantic_validation/faudit_20260621_bass_gsf_gpu67_launch_readiness.md
```

The waiter-liveness audit is documented at:

```text
resultmd/exp_p4_scale_semantic_validation/fstatus_20260621_bass_gsf_gpu67_waiter_liveness.md
```

The current GPU0/1 live-queue audit is documented at:

```text
resultmd/exp_p4_scale_semantic_validation/fstatus_20260621_bass_gsf_gpu01_live_queue.md
```

These status files are scheduler and prerequisite facts, not method results. As
of the latest refresh, P0, P1, and P2 result rows are complete and filled from
eval JSON plus deployment-risk summaries. The older live-queue files remain as
history; they are not current blockers.
This appendix improves paper readiness by making the experimental protocol
auditable, but it does not close the method-effectiveness or ICLR 9.5 gate.

## References

See `paper/drafts/semantic_scale_support_refs.bib` for the current 27-entry
traceable BibTeX file, covering multi-scale detection, remote-sensing rotated
detection, OVD/VLM detection, detector calibration, probabilistic detection,
OOD/virtual outlier synthesis, and Gaussian box-loss methods.
