# OV-CapFlow Architecture Specification

## Objective

Build a strict end-to-end open-vocabulary oriented detector on top of the
public CastDet `main_obb` codebase. The research target is narrower and more
defensible than "the first open-vocabulary rotated DETR": fixed matching
queries, direct rotated boxes, no inference NMS, no encoder-proposal top-k, no
decoder survivor pruning, and no dense detection head used at inference.

## Upstream provenance

- Engineering shell: VisionXLab/CastDet, branch `main_obb`, commit
  `e0f28f053d6aa8262de6920b85b4b63fae365f67`.
- Rotated geometry: the RHINO implementation already vendored by CastDet.
- Open-vocabulary path: CastDet's `RotatedGroundingDINO` implementation.
- Later semantic initialization donor: LAE-DINO checkpoints and its
  remote-sensing vocabulary/prompt construction, imported only after a clean
  baseline is reproducible.

RHINO files carry CC BY-NC 4.0 terms. The combined research code therefore
must be treated as academic/non-commercial unless the relevant rights holders
grant broader permission.

## Non-negotiable inference invariants

1. A configured number `Q` of matching queries enters every decoder layer.
2. Initial query content and 5-D spatial references are learned parameters;
   they are not selected from encoder tokens.
3. Every matching query produces one `(score, label, cx, cy, w, h, angle)`
   record. Class selection is per query; global query-class top-k is forbidden.
4. No NMS, rotated NMS, min-area rectangle conversion, or dense detector head
   participates in raw-patch inference.
5. Denoising queries and auxiliary one-to-many supervision are training-only.
6. Standard DOTA crop merging is reported separately from the strict raw
   full-validation protocol and is never called strict end-to-end inference.

## Architecture

### Fixed native query state

Each matching slot owns persistent content `q_native` and a learned 5-D
reference logit. Reference logits are sigmoid-normalized before decoding. A
deterministic low-discrepancy initialization spreads centers across the image,
uses conservative initial sizes, and initializes the angle at zero. Parameters
remain learnable after initialization.

### Semantic-preserving evidence fusion

For decoder layer `l`, image/text cross-attention produces transported
evidence `e_transport`. It is fused without overwriting query identity:

`q_fused = q_native + tanh(gate(q_native, e_transport)) * P(e_transport)`.

`P` starts as identity (or orthogonal in the explicit ablation), and the gate
starts at exactly zero. The first forward pass therefore reproduces native
query semantics rather than injecting random evidence.

### Continuous density capacity

Every matching query predicts continuous capacity `c_i in (0, 1)`. A global
density token predicts expected scene count `N_hat`. Capacity scales the
amount of transported evidence and is trained with:

- query capacity calibration on Hungarian matched/unmatched slots;
- global mass consistency between `sum(c_i)` and ground-truth count;
- global density regression between `N_hat` and ground-truth count;
- a null-reservoir calibration term for empty and background-heavy patches.

Capacity is never converted into a survivor mask or top-k list. All queries
remain in the decoder and raw prediction set.

### Rotated matching and regression

The first substrate keeps RHINO's direct 5-D box refinement and Gaussian/KLD
matching. The planned geometry ablation compares KLD against corner Hausdorff
matching without changing inference. Angle representation remains direct and
does not call `minAreaRect`.

## Delivery phases

1. Strict substrate: fixed learned query/reference initialization and one
   label per query prediction, with regression tests forbidding both top-k
   sites present in the upstream code.
2. Semantic state: native/evidence separation with identity and orthogonal
   adapter initializations.
3. Capacity flow: continuous query capacities, global density token, mass and
   null calibration losses.
4. DOTA2 integration: all 18 classes, raw full-val 13,833 mouth, no NMS, plus
   a separately labelled standard crop-merge mouth.
5. Open-vocabulary evidence: base/novel splits, harmonic mean, rare-class
   recall, duplicate rate, empty-tile false positives, count MAE, latency and
   memory.

## Required ablations

- upstream Oriented GroundingDINO;
- fixed learned references only;
- native semantics without transported evidence;
- transported evidence without native residual;
- identity versus orthogonal evidence adapter;
- capacity without mass loss;
- capacity with mass loss;
- null reservoir and airport/null calibration;
- KLD versus Hausdorff matching;
- strict all-query prediction versus upstream global top-k prediction.

## 2026-07-14 strict-substrate verification

- Checkout: `/data1/zcy/OV-CapFlow` on `codex/ov-capflow`.
- Upstream base: CastDet `main_obb` at
  `e0f28f053d6aa8262de6920b85b4b63fae365f67`.
- Interpreter: `/data/zcy/anaconda3/envs/mmdet/bin/python`.
- Versions: Python 3.8.19, PyTorch 1.12.1+cu113, MMCV 2.1.0,
  MMEngine 0.10.4, MMDetection 3.3.0, local MMRotate 1.0.0rc1.
- Editable installation used `pip install -e . --no-deps`; pytest was the
  only package added to the existing environment.
- Focused result: 9 tests passed.
- Forbidden-selection scan over executable OV-CapFlow Python files found no
  `torch.topk`, method `.topk`, NMS, multiclass NMS, or `minAreaRect` calls.
- Active now: learned fixed content/reference queries, no encoder-proposal
  ranking in `OVCapFlow.pre_decoder`, and one class decision per query in
  `OVCapFlowHead`.
- Implemented after the strict-substrate record: identity/orthogonal evidence
  adapters, a zero-initialized semantic gate, matching-suffix-only fusion,
  continuous per-query capacity, masked global density prediction, and
  capacity-mass/density-count losses.
- Still not active: explicit null-reservoir calibration and DOTA2 18-class
  experiment configuration.
- Verification boundary: semantic/capacity modules and the custom decoder
  constructor have unit coverage, but no real image/annotation batch has run.
  Therefore this record does not claim successful training or convergence.
