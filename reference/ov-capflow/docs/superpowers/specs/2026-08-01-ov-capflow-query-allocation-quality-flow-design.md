# OV-CapFlow Unified Query Allocation–Quality Flow Design

## 1. Status, decision, and target

**Working title:** *OV-CapFlow: Calibrated Query Flow for Open-Vocabulary
Oriented Set Prediction*

**Target:** ICLR 2027, with the venue date treated as milestone-relative until
the official submission schedule is published.

The user approved **Approach A: Unified Query Allocation–Quality Flow** on
2026-08-01. This document freezes the scientific and engineering design for
review before an implementation plan is written. It does not authorize model
changes or GPU training by itself.

The project is no longer a sequence of small adapters. Its one research
question is:

> In fixed-capacity, all-row, NMS-free open-vocabulary oriented set
> prediction, can every query's spatial allocation, validity, localization
> quality, and semantic compatibility be governed by one structured object,
> so that the resulting predictions are comparable across images, classes,
> scales, and densities?

The proposed answer is a layer-wise, query-to-image **evidence coupling**. The
same coupling moves query centers toward image-supported locations, sends
unsupported queries to a null sink, supplies the validity used during
training, and calibrates the final all-query score. Query count, query
identity, output row order, class vocabulary, and the strict inference mouth
remain unchanged.

This is a hypothesis, not a result. The design contains early falsification
gates intended to stop the whole direction before expensive implementation
or training if the required image evidence is absent.

## 2. Why the project is changing direction

### 2.1 Authoritative T7 substrate

The canonical substrate is T7 Epoch 24:

- generic GroundingDINO+BERT initialization, not a remote-sensing parent;
- 600 fixed learned content queries with 5-D rotated references;
- six decoder layers;
- three independently matched groups plus DN during training;
- exactly one class, score, and rotated box per query at inference;
- no global/proposal top-k, NMS, rotated NMS, dense head, RPN, or RoI mouth.

Its raw DOTA-v2 validation trajectory was:

| Endpoint | mAP / official AP50 |
|---|---:|
| E1 | `0.4168` |
| E6 | `0.5537 / 0.5540` |
| E12 | `0.6043 / 0.6040` |
| E18 best | `0.6081 / 0.6080` |
| E24 canonical | `0.6064053488 / 0.6060` |

The E24 checkpoint is
`work_dirs/dotav2_cleanstart/full24e_grouped_scale1024_rare4x_seed20260716_gpu89_b2/epoch_24.pth`
with SHA256
`a4f2661e6c1645b08f296dfb2bbebfd76afe8c366d6152dbbc333bbf840af6b8`.
Its canonical raw dump contains 13,833 images, 8,299,800 prediction rows, and
243,632 GT objects.

The observed E24 decomposition is the factual starting point:

- base14 mAP `0.654018`, novel4 mAP `0.439762`; the novel4 result is not a
  valid zero-shot claim because the training protocol does not enforce a
  strict class holdout;
- fixed-recall oracle mAP `0.782828`, leaving `0.176423` headroom; reaching
  mAP `0.7000` would consume 53.05% of that headroom;
- TP score--assigned-IoU Spearman correlation is only `0.365425`;
- geometry misses are 76,741 (`31.499%` of GT), semantic misses are 2,440
  (about `1.002%`), ownership misses are 11, and reachable GT is `67.495%`;
- AP-support FP shares are localization/background `64.634%`, empty-query
  `17.623%`, duplicate `15.302%`, and semantic `2.441%`;
- `<=8 px` objects contribute `76.366%` geometry miss, scenes with more than
  600 GT contribute `82.912%`, and small-vehicle AP is `0.319549` with
  `43.736%` geometry miss;
- among Dense400 geometry misses, only `22.256%` have a nearby final decoder
  center; `77.744%` do not. Re-ranking existing final queries therefore
  cannot solve the main bottleneck;
- the previously audited T/E center sources recovered at most `14.51%` and
  `21.07%`, below the pre-registered `25%` source gate, so A0 was correctly
  stopped.

These numbers point to one coupled problem: image evidence is not allocated to
enough useful query locations, and prediction confidence is not sufficiently
aligned with the localization quality that emerges. A score-only adapter
cannot create missing centers; a center-only adapter does not make all 600
rows comparable.

### 2.2 Closed branches

The following branches are closed and must not be revived, scaled, stacked,
or rescued by extra epochs, coefficients, seeds, or post-hoc endpoint choice:

| Branch | Decisive result | Decision |
|---|---|---|
| A1 quality target | endpoint candidate `0.4643` vs control `0.4691`; novel4 `0.2445` vs `0.3675` | discard |
| D11 | endpoint `0.4521` vs `0.4660`; delta `-0.0139`, novel4 delta `-0.0605` | discard |
| D12 | non-finite Hungarian input at E2; no valid paired endpoint | discard |
| D13-N | candidate `0.838305831` vs control `0.847486973`; delta `-0.009181142` | discard |

D13-N used a 1,600-train/400-validation proxy; it was not a full 13,833-image
raw evaluation. Both arms completed 12 epochs and 160 updates per epoch,
1,920 updates per arm. Its best E2 excursion was only `+0.0004875`, was not the
fixed endpoint, and cannot be selected. D13-N is therefore formally negative.
The earlier suggestion to scale its existence residual is withdrawn.

The balanced helper is also closed: it shrinks the classification loss by
about `64.1x` and has sparse/NaN failure modes. Fixed-1280 training,
Hausdorff/Chamfer matching, extending the same recipe beyond E24, and a
semantic-false-positive mainline are closed or demoted by existing evidence.

### 2.3 Assets that remain valid

The new direction reuses rather than replaces:

- the T7 E24 checkpoint, canonical raw dump, frozen analyzer, and diagnostic
  definitions;
- Dense400 assets and the established geometry-miss audit;
- the strict fixed-Q inference mouth and its five Stage-0 checks: identity,
  protocol, finiteness, parent freeze, and zero-step equality;
- the HRSC parent-preserving residual result, but only as evidence that a
  zero-initialized route can preserve a parent;
- scale-1024 plus rare4x as the current training substrate, without claiming
  a factorized benefit for either choice.

## 3. Claim boundary and related-work position

The paper must not claim “first end-to-end OVD,” “first fixed-query/no-NMS
OVD,” “first aerial or rotated OVD,” “first rotated DETR without NMS,” “first
query allocation,” or “first existence/quality calibration.” Those regions
are already occupied by prior work including:

- [OV-DETR](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/1248_ECCV_2022_paper.php)
  and [OWL-ViT](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/7529_ECCV_2022_paper.php)
  for end-to-end open-vocabulary detection;
- [CastDet](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/11741_ECCV_2024_paper.php),
  [orientation adaptation](https://arxiv.org/abs/2411.02057),
  [LAE-DINO](https://ojs.aaai.org/index.php/AAAI/article/view/32672), and
  [OpenRSD](https://openaccess.thecvf.com/content/ICCV2025/html/Huang_OpenRSD_Towards_Open-prompts_for_Object_Detection_in_Remote_Sensing_Images_ICCV_2025_paper.html)
  for aerial/open-vocabulary or oriented remote-sensing detection;
- [AO2-DETR](https://arxiv.org/abs/2205.12785),
  [D2Q-DETR](https://arxiv.org/abs/2303.00542), and
  [RHINO](https://openaccess.thecvf.com/content/WACV2025/html/Lee_Hausdorff_Distance_Matching_with_Adaptive_Query_Denoising_for_Rotated_Detection_WACV_2025_paper.html)
  for NMS-free rotated set prediction and matching;
- [DQ-DETR](https://arxiv.org/abs/2404.03507),
  [Dome-DETR](https://arxiv.org/abs/2505.05741), and
  [PaQ-DETR](https://arxiv.org/abs/2603.06917) for query allocation or
  query-population mechanisms;
- [Rank-DETR](https://openreview.net/forum?id=WUott1ZvRj),
  [Cascade-DETR](https://openaccess.thecvf.com/content/ICCV2023/html/Ye_Cascade-DETR_Delving_into_High-Quality_Universal_Object_Detection_ICCV_2023_paper.html),
  [OV-DQUO](https://ojs.aaai.org/index.php/AAAI/article/view/32836), and
  [ProCal](https://arxiv.org/abs/2607.01759) for ranking, quality-aware
  training, unknown/objectness modeling, or calibration.

The defensible gap is narrower and structural:

> Existing work does not establish that a single query-to-image evidence
> allocation can jointly govern spatial transport, null validity, training
> quality, and all-row score comparability under a fixed-Q, open-vocabulary,
> oriented, NMS-free contract.

This wording is a literature-backed gap to test, not a priority claim. Before
submission it must be refreshed against contemporary papers, especially
methods published after this design's 2026-08-01 search cutoff. The official
[ICLR 2027 OpenReview page](https://openreview.net/group?id=ICLR.cc%2F2027%2FConference)
exists, but its schedule must not be inferred until official dates appear.

The method boundary to defend experimentally is:

| Prior-work family | What it already establishes | What this design must add without claiming priority |
|---|---|---|
| DQ-DETR, Dome-DETR, PaQ-DETR | query allocation, density/population adaptation, or query specialization | keep Q fixed and make one inference-time evidence coupling jointly control null mass, center transport, training quality, and final ranking |
| Rank-DETR, Cascade-DETR | localization-aware classification/ranking and high-quality DETR training | derive quality from the same spatial allocation that can create better centers, rather than attach an independent quality score |
| OV-DQUO, ProCal and related OVD calibration | unknown/objectness handling and cross-class/open-vocabulary calibration | preserve the strict oriented Q600 all-row mouth and connect calibration to query-to-image spatial transport |
| AO2-DETR, D2Q-DETR, RHINO and related rotated DETRs | end-to-end oriented set prediction and improved matching/refinement | add text-conditioned open-vocabulary evidence allocation and cross-image/class validity under the same oriented set contract |
| CastDet, LAE-DINO, OpenRSD and oriented GroundingDINO variants | practical aerial or oriented open-vocabulary detection | isolate a fixed-capacity, proposal-free, NMS-free mechanism and verify it with all rows rather than a selected detection mouth |

Every distinction in this table is provisional until reproduced from the
final papers and code at submission time. The paper should lead with the
measurable contract and evidence, not with categorical statements about what
another method “cannot” do.

## 4. Scientific hypotheses and falsifiers

### H1: recoverable evidence exists before the final centers

A frozen T7 encoder contains spatially localized, text-compatible evidence
for a materially larger fraction of E24 geometry misses than the failed T/E
sources.

**Falsifier:** on a held-out proxy, the pre-registered frozen-evidence audit
cannot recover at least 25% of canonical geometry misses, does not materially
exceed the `21.07%` prior source ceiling, or performs similarly after spatial
or semantic shuffling. If falsified, stop Approach A before production model
code.

### H2: allocation and quality are the same latent decision

A query that receives coherent non-null spatial evidence should both move
toward that evidence and receive higher localization-validity mass. A query
that lacks coherent evidence should retain its identity but route mass to the
null sink.

**Falsifier:** a shared coupling cannot improve center reachability and
score--IoU alignment together at the fixed proxy endpoint, or one improvement
requires a separate head/target that no longer derives from the same
coupling.

### H3: the gain survives the strict mouth

The coupled flow improves accuracy without deleting, sorting, or selecting
queries and without NMS, proposal generation, a dense detection head, or a
changed vocabulary.

**Falsifier:** gains disappear when every one of the 600 rows is evaluated in
original query order, or require any forbidden inference operation.

### H4: improvement is not a base-class or density-only trade

The same mechanism improves the dominant geometry/ranking failure while
preserving open-vocabulary behavior across object scale, orientation, aspect
ratio, and image density.

**Falsifier:** the fixed endpoint collapses novel-class performance, merely
trades sparse for dense scenes, or worsens empty/duplicate/localization FP
regions beyond the registered tolerance.

## 5. One mathematical object

### 5.1 State and evidence

At decoder layer `l`, fixed query `i` has state

```text
z_i^l = (h_i^l, r_i^l, v_i^l)
```

where `h_i^l` is the content representation, `r_i^l` is the existing 5-D
oriented reference, and `v_i^l` is validity derived from allocation. Query
identity is the persistent row index `i`; the method never creates, deletes,
sorts, or replaces rows.

Use one pre-registered coarse encoder scale with `S` spatial tokens
`x_j`, normalized token centers `p_j`, and existing text-conditioned semantic
evidence `g_j`. Add one learnable null sink indexed by `0`. There is no dense
detection output, proposal list, hard token selection, or density/count head.

At every decoder layer, a single parameter-shared operator produces

```text
Pi^l in R_+^{Q x (S+1)}
```

where each row describes how one fixed query allocates evidence across the
`S` spatial tokens and the null sink. Parameters of this operator are shared
across all decoder layers; only the input state changes.

### 5.2 Coupling operator

The reference formulation is a row-constrained, column-soft entropic
transport problem:

```text
Pi^l = argmin_{Pi >= 0}
         <Pi, C^l>
       + epsilon * sum_ij Pi_ij * (log(Pi_ij + eps) - 1)
       + tau * KL((Pi^T 1), b(x))
       subject to Pi 1 = 1_Q.
```

`b(x)` is a bounded spatial-plus-null evidence prior formed directly from the
chosen encoder evidence. It is not a predicted object count. The soft column
penalty creates competition for image evidence; the null column absorbs
unsupported rows. Row normalization makes the non-null mass interpretable
without changing Q.

For spatial token `j`, the cost must contain exactly three kinds of evidence:

```text
C_ij^l = content_cost(h_i^l, x_j)
       + lambda_p * oriented_spatial_cost(r_i^l, p_j)
       + lambda_s * text_compatibility_cost(h_i^l, x_j, g_j).
```

The null cost is a bounded learned scalar or bounded query-conditioned value
inside the same operator. It must not be a separate existence classifier.
The implementation plan may choose the simplest stable parameterization of
these terms after the source-feasibility audit, but it may not add another
prediction branch.

The solver must use float32 log-domain updates, fixed iteration count, fixed
temperature, deterministic reductions, and explicit finite checks. An
ordinary normalized soft coupling is permissible for the first prototype
only if it retains all registered invariants—row normalization, a null sink,
soft spatial competition, and the same downstream quantities—and the
equivalence is documented before results are observed. A GT-assignment UOT
method is not the claimed contribution: this operator couples inference-time
queries to image evidence rather than predictions to GT.

### 5.3 Validity and barycenter

The coupling defines both quantities; neither gets an independent head:

```text
v_i^l = sum_{j=1..S} Pi_ij^l                 in [0, 1]
c_i^l = sum_{j=1..S} Pi_ij^l * p_j / (v_i^l + eps).
```

`v_i^l` is the only query-validity variable. `c_i^l` is the evidence
barycenter conditional on non-null mass. For near-null queries the center
transport is multiplied by validity and therefore vanishes continuously.

The initial method changes center only. Width, height, and angle remain the
parent outputs. Extent/angle may later be expressed as moments of the same
coupling, but only after the center gate passes; they are not part of the
initial implementation and may not become new heads.

### 5.4 Center transport

After the existing parent update at layer `l`, apply a bounded residual toward
the barycenter:

```text
delta_i^l = v_i^l * clip(c_i^l - center(r_parent,i^{l+1}), radius_l)
center(r_i^{l+1}) = center(r_parent,i^{l+1}) + alpha_center^l * delta_i^l.
```

The residual is direct; it does not require another MLP. `alpha_center^l` is a
zero-initialized bounded gate. The parent width, height, angle, content path,
and row identity are unchanged. Coordinate transforms, padding masks, and
reference conventions must reuse the existing decoder implementation.

### 5.5 Training use of the same validity

The existing grouped Hungarian assignment remains the ownership mechanism.
When enabled by a fixed, pre-registered `lambda_match`, its query-quality term
uses `-log(v_i + eps)` in the existing cost. Because it is constant over GT
columns for a given query, it changes which queries are eligible for
ownership without encoding a class or GT identity. Assignment remains
non-differentiable as in the parent, so `lambda_match` is not presented as a
learnable gate. It is zero in equality mode and follows one deterministic,
pre-registered warm-up to its frozen candidate value during training.

After assignment, define the detached natural-prevalence target

```text
y_i = detached_rotated_IoU(pred_i, gt_assigned_i)  if matched
      0                                             otherwise.
```

Train `v_i` with one per-query proper soft-label objective averaged over all
queries and groups. There is no 1:1 matched/unmatched balancing, no positive
oversampling inside the loss, and no separately learned objectness target.
The objective must report positive, negative, empty-image, and all-matched
strata for diagnosis while its optimization value remains the natural
all-query mean. Parent losses and three-group matching otherwise remain
unchanged.

Validity supervision applies to the same `Pi` that produces center transport.
This is the essential difference from D13-N: it is not a detached,
after-argmax local BCE head added only to the score.

### 5.6 All-row readout

The parent still selects the class label for each query. The same final-layer
validity then supplies a centered log-space calibration:

```text
s_i,final = s_i,parent
          + alpha_score * log(v_i + eps).
```

`alpha_score` is a zero-initialized bounded gate. This route changes scores
only, after unchanged class argmax and before the existing clamp/exp. It
cannot change class labels or boxes at readout time. The validity route is
therefore suppressive when the gate is nonnegative; no validation-fitted
neutral constant or second reference model is introduced.

All 600 rows are returned in their original query order. There is no
threshold, top-k, query pruning, reordering, NMS, or hidden proposal mouth.

## 6. End-to-end data flow

### 6.1 Training

```text
image + prompts
  -> unchanged image/text encoders
  -> one registered coarse encoder evidence scale
  -> fixed Q600 parent query states
  -> shared coupling Pi^l(query state, spatial evidence, null), l=1..6
       -> v_i^l: null/non-null validity
       -> c_i^l: spatial barycenter
  -> zero-gated center residual inside normal decoder refinement
  -> unchanged class/5-D box predictions
  -> existing three-group Hungarian assignment
       + zero-gated validity cost from the same Pi
  -> existing parent losses
       + natural-prevalence proper validity loss on the same Pi
```

DN rows must remain explicitly separated from the 3 x 600 matching rows.
Whether DN rows receive coupling supervision is fixed to **no** for the first
implementation: DN stays on its parent route to reduce causal ambiguity.

### 6.2 Inference

```text
image + prompts
  -> unchanged encoders
  -> fixed Q600 decoder
  -> shared coupling and zero-gated center transport at each layer
  -> unchanged per-query class argmax and oriented box
  -> final Pi validity as zero-gated log-score calibration
  -> 600 labels + 600 scores + 600 boxes, original order
```

The method is therefore a set predictor throughout; the evidence grid is an
internal continuous support, not a second detection mouth.

## 7. Parent equivalence and numerical contract

With `alpha_center = alpha_score = lambda_match = 0` and the new validity loss
weight zero, the candidate must reproduce the T7 parent exactly:

- byte-identical selected labels;
- `torch.equal` scores and rotated boxes;
- identical row order and exactly 600 predictions per image;
- identical parent losses and assignments under the zero route;
- no unexpected or shape-mismatched checkpoint keys;
- every parent parameter frozen in the Stage-0 equality test.

The coupling may be computed during a zero-route diagnostic, but it must not
consume random numbers that alter the parent path. Its constructor and fixed
solver are deterministic. Empty images, ordinary images, and the known
1,223-GT stress image must produce finite costs, couplings, barycenters,
losses, gradients, scores, and boxes. Near-zero validity uses explicit epsilon
and mask-safe reductions; NaN replacement is not an accepted remedy.

Training is invalid if any of the following occurs:

- non-finite input to Hungarian assignment;
- non-finite coupling, loss, gradient, or optimizer state;
- a forbidden inference operation appears in the call graph;
- a parent tensor changes while the Stage-0 freeze contract is active;
- score/label/box equality fails at zero gates;
- any output row is added, removed, or reordered.

## 8. Preflight before production implementation

No production module is written until a frozen-evidence source audit passes.
This audit uses the E24 parent, train-only fitting if a probe is necessary,
and a held-out proxy that is not used to tune thresholds or choose endpoints.

For each image, expose only the proposed coarse encoder scale and its existing
text-conditioned evidence. Rank or softly allocate at most the equivalent of
Q600 spatial evidence mass without GT. Measure recovery of the canonical E24
geometry-miss set using the same center-nearness definition as the frozen
Dense400 audit. Report:

- overall geometry-miss evidence reachability;
- `<=8 px`, small-vehicle, `>600 GT/image`, base14, and novel4 strata;
- spatial-shuffle, semantic-shuffle, and uniform-evidence placebos;
- comparison with the frozen T/E ceilings of `14.51%` and `21.07%`;
- evidence concentration on empty images and already-reachable GT.

The source passes only if all are true at the pre-registered endpoint:

1. overall geometry-miss reachability is at least `25%`;
2. it exceeds the best prior source ceiling by at least `3` percentage points;
3. both spatial and semantic shuffles lose at least `5` percentage points of
   reachability;
4. novel4 reachability does not trail base14 by more than `10` percentage
   points;
5. the result is finite and reproducible from a frozen manifest and command.

If this gate fails, Approach A closes. The response is a new scientific
direction, not a larger model, more epochs, another evidence scale chosen on
validation, or a new adapter.

## 9. Experimental gates

### Phase 0: governance reset

After this design is reviewed, update `.lab/results.tsv`, `progress.md`,
`task_plan.md`, and the project README so that D13-N is recorded as negative
and this design becomes the only active branch. These files are deliberately
not modified in the design-only commit.

Freeze before training:

- dataset manifests and hashes;
- class prompt/order and base/novel definitions;
- checkpoint and analyzer hashes;
- endpoint, optimizer, scheduler, batch, seed, topology, and precision;
- exact gate arithmetic and comparison tolerances;
- source scale, coupling iterations, temperature, and all initial
  coefficients.

### Phase 1: evidence-source feasibility

Run only the audit in Section 8. It may use one GPU for forward extraction or
a small frozen-feature probe. It is not a training result and must not be
reported as model improvement. Failure terminates the method family.

### Phase 2: five Stage-0 checks

The minimal integration must pass:

1. **identity:** T7 checkpoint, prompts, class order, query order, and Q600;
2. **protocol:** no top-k, thresholding, sorting, NMS, dense/RoI/RPN mouth, or
   proposal selection;
3. **finite:** empty, normal, and 1,223-GT stress cases including backward;
4. **parent freeze:** only pre-registered coupling/gate parameters trainable;
5. **zero-step equality:** exact scores, labels, boxes, order, parent loss, and
   assignment at zero route.

No training starts until all five pass.

### Phase 3: one paired proxy experiment

Run exactly one fixed-endpoint control/candidate pair on the frozen
1,600-train/400-validation proxy. Both arms load E24 and freeze every parent
tensor. The control instantiates the same coupling route but keeps all output
gates and its loss weight zero; the candidate trains only the registered
coupling/gate parameters. Launcher, data order, topology, and evaluation are
otherwise identical. If ten GPUs remain available, use two simultaneous
five-GPU jobs: GPUs `0-4` control and `5-9` candidate. GPU availability does
not waive any preceding gate.

Promotion requires all of:

- candidate minus control mAP/AP50 at least `+0.010`;
- geometry-miss rate decreases by at least `2.0` absolute percentage points;
- TP score--assigned-IoU Spearman improves by at least `+0.05`;
- novel4 mAP does not decline and base14 declines by no more than `0.005`;
- each AP-support FP rate—empty, duplicate, and
  localization/background, normalized by the frozen analyzer's registered
  denominator rather than compared as compositional shares—worsens by no
  more than `5%` relative;
- all strict mouth, identity, and finiteness checks pass.

The endpoint is fixed before launch. Intermediate peaks cannot promote the
candidate. A failed pair closes the family; there is no coefficient, epoch,
seed, or component-stack rescue.

### Phase 4: canonical raw evaluation

Only a proxy pass permits full raw DOTA-v2 evaluation. Train one frozen-recipe
candidate with all E24 parent tensors still frozen and compare it with the
canonical E24 parent on all 13,833 images and 8,299,800 rows. Ten GPUs may be
used for this candidate after the run manifest and Stage-0 evidence are
archived. Joint parent fine-tuning is a separate paper-scale generalization
question and cannot replace this causal raw gate.

Raw promotion requires:

- mAP/AP50 improvement at least `+0.005` over `0.6064053488 / 0.6060`;
- base14 and novel4 mAP each decline by no more than `0.001`;
- AP75, geometry-miss rate, and score--IoU correlation improve in the same
  direction;
- no registered density, size, angle, or aspect-ratio stratum suffers a
  material unreported collapse;
- strict Q600 all-row protocol and raw-dump integrity pass.

### Phase 5: paper evidence package

Only a raw pass unlocks broader evidence:

- three seeds for the complete method and parent;
- a true base/novel class-holdout protocol with absent-prompt FPR and leakage
  audit—the existing novel4 split is never relabeled as zero-shot;
- DOTA-v2 plus at least one second oriented dataset such as DIOR-R, HRSC2016,
  or STAR, selected before results;
- fair comparisons against relevant available methods, including
  CastDet/oriented GroundingDINO/OpenRSD and rotated-set/query-quality methods
  such as RHINO, OrientedFormer, PaQ-DETR, or Rank-DETR where protocol and code
  permit;
- calibration and ranking metrics across images and classes, not only AP;
- stratification by density, size, angle, aspect ratio, base/novel status, and
  empty images;
- runtime, memory, solver iterations, parameter count, and convergence cost;
- all methods evaluated through their declared mouth, with OV-CapFlow always
  retaining the strict no-NMS Q600 mouth.

## 10. Ablations that test unity, not module accumulation

The paper-facing ablation is defined at interfaces of the same coupling:

| Variant | Center transport | Validity training/matching | Score use | Purpose |
|---|---:|---:|---:|---|
| Parent | no | no | no | canonical reference |
| Flow-to-center | yes | coupling validity only as required to learn allocation | no | tests spatial consequence |
| Flow-to-score | no | yes | yes | tests quality consequence |
| Unified flow | yes | yes | yes | tests shared-object hypothesis |
| Shuffled flow | yes | yes | yes | falsifies image-evidence explanation |

All variants use the same `Pi` operator and parameter budget. They are not
independent modules to be stacked. The main claim requires the unified flow to
beat both single-interface variants at a pre-registered endpoint; otherwise
the “one structured object” interpretation is unsupported even if one
variant gains AP.

Additional ablations are limited to solver iteration count and removal of one
cost term at a time. Extent/angle moments, extra encoder scales, dynamic Q,
dense proposals, auxiliary quality heads, and loss-balancing variants are out
of scope for the first paper branch.

## 11. Paper evidence matrix

| Intended claim | Required evidence | Claim fails when |
|---|---|---|
| Frozen image evidence can reach missing geometry | held-out source audit, T/E comparison, shuffle placebos | Section 8 gate fails |
| One coupling improves allocation and quality | center reachability, geometry misses, score--IoU correlation, interface ablations | only one axis improves or separate heads are needed |
| Improvement survives strict set prediction | Q600 row audit, call-graph protocol check, raw dump hashes | top-k/NMS/filtering is required |
| Open-vocabulary behavior is preserved | true holdout, leakage audit, absent-prompt FPR, base/novel AP | gains rely on seen-class leakage or novel collapse |
| Benefit is structurally general | second OBB dataset, density/size/angle strata, three seeds | gain is one-dataset/one-stratum noise |
| Practical cost is acceptable | parameters, FLOPs/runtime, memory, solver sensitivity | transport cost dominates benefit |

The abstract and introduction may only contain claims whose corresponding row
passes. Negative or mixed strata must remain visible in the main paper or
appendix according to their relevance; they cannot be hidden by aggregate AP.

## 12. Engineering scope and commit budget

The implementation must be a small adaptation of the existing framework:

- at most one new production component for the shared coupling;
- integration inside the existing decoder/head interfaces;
- at most two thin experiment configs;
- at most two focused test files;
- at most three concentrated commits before the first paired proxy launch;
- reuse the existing fixed-Q evaluator, dump analyzer, launcher, queue guard,
  and five core checks;
- no new experiment orchestration framework, audit platform, result database,
  generalized plugin system, or parallel model family.

D13 accumulated about 43 commits and roughly 18,000 changed lines around a
257-parameter intervention. That process is not the template for this work.
If the proposed change cannot reach zero-step verification within the scope
above, stop for design review rather than building infrastructure.

The expected implementation sequence after a separately approved plan is:

1. governance correction and frozen source audit;
2. minimal coupling operator plus unit tests;
3. existing-decoder integration and five Stage-0 checks;
4. one paired proxy launch;
5. only on promotion, one canonical raw candidate;
6. only on raw promotion, paper-scale evidence.

## 13. Failure handling and stop rules

- A non-finite solver or Hungarian event invalidates that run; it is not
  replaced by clipping after the fact. One root-cause repair may be reviewed
  if it preserves the registered method. A second scientific recipe is not
  silently substituted.
- An infrastructure failure may be resumed only from a verified identical
  checkpoint and sampler state. A scientific metric failure is not an
  infrastructure failure.
- An OOM may reduce micro-batch with mathematically matched accumulation only
  if both arms are restarted under the same frozen effective batch.
- No early peak, cherry-picked seed, changed class subset, unofficial mouth,
  or proxy-only result can promote a branch.
- No GPU is occupied merely because it is free. The source and Stage-0 gates
  determine when the authorized ten-GPU window is scientifically useful.
- If a gate fails, record the negative result, archive exact evidence, and
  close the family before proposing a different hypothesis.

## 14. Milestone-relative schedule

Because the official ICLR 2027 deadline was not available at design time,
execution is organized by gates rather than invented calendar dates:

| Milestone | Deliverable | Exit condition |
|---|---|---|
| M0 | reviewed design and governance reset | one active hypothesis |
| M1 | frozen source-feasibility report | all five source gates pass |
| M2 | minimal integration | all five Stage-0 checks pass |
| M3 | paired proxy endpoint | every proxy promotion gate passes |
| M4 | canonical raw result | every raw promotion gate passes |
| M5 | holdout + second-dataset + three-seed package | claim matrix complete |
| M6 | paper draft and internal review | no unsupported claim or protocol gap |

Once official dates are published, M0--M6 receive calendar deadlines with at
least one full internal-review cycle before submission. Scope is reduced if
time is short; evidence gates are not relaxed.

## 15. Alternatives considered

### B. Quality/calibration-only continuation

Rejected because D13-N was negative and final-center availability is only
`22.256%` for Dense400 geometry misses. Re-ranking cannot create missing
spatial support, and quality-aware detection/calibration already has strong
prior art.

### C. Dynamic query count or density-capacity prediction

Rejected for this branch because it breaks the clearest fixed-Q scientific
contract, adds a count/density subsystem, and confounds allocation with
capacity. Scenes with more than 600 GT remain an acknowledged ceiling.

### D. Dense proposal or RoI hybrid

Rejected because it changes the inference mouth, weakens the end-to-end set
prediction claim, and makes gains inseparable from a second detector.

### E. Continue T/E, D11, D12, D13, or stack their pieces

Rejected by their registered failures. Combining failed components would
increase degrees of freedom without restoring causal evidence.

### F. Immediately scale to ten-GPU full training

Rejected as the first step. The decisive unknown is whether the proposed
encoder source contains usable evidence. That can be falsified cheaply before
production integration; only passed gates justify the authorized GPUs.

## 16. Known risks and required honesty

1. **Evidence-source risk:** the encoder may not contain enough localized
   open-vocabulary evidence at the chosen scale. The preflight is designed to
   terminate quickly if so.
2. **Transport prior-art risk:** optimal-transport assignment is not novel by
   itself. The claim concerns one inference-time query-to-image coupling under
   the strict mouth and requires empirical support for all four interfaces.
3. **Solver risk:** crowded images may make column competition expensive or
   unstable. Fixed-iteration log-domain behavior and stress tests are gates.
4. **Validity circularity:** validity affects matching and is supervised by
   the resulting assignment. Zero-gated introduction, detached assignment,
   and the flow-to-center/flow-to-score ablations must reveal collapse.
5. **Small-object resolution risk:** one coarse scale may be inadequate for
   `<=8 px` objects. Failure closes the initial hypothesis; it does not
   automatically license multi-scale module growth.
6. **Novel-class risk:** the current novel4 number is not zero-shot evidence.
   A strict holdout and leakage audit are mandatory before an OVD claim.
7. **Capacity ceiling:** Q600 cannot recall more than 600 objects. The method
   aims to allocate the fixed budget better, not deny this ceiling.
8. **Title risk:** “Calibrated Query Flow” is a working name and must be
   checked against subsequent literature before submission.

## 17. Approval boundary

Approval of Approach A selected the research direction. Approval of this
document will freeze its hypotheses, mathematical interfaces, gates, scope,
and stop rules. Only then may a separate implementation plan name concrete
files, tests, commands, resource assignments, and rollback points.

Until that review occurs:

- D13-N remains closed;
- no model/config code is changed;
- no source probe, proxy, or raw training is launched;
- stale governance files are acknowledged but intentionally untouched;
- available GPUs remain idle unless needed by already authorized unrelated
  work.
