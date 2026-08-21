# OV-CapFlow Three-Figure Paper Pipeline Design

## 1. Purpose

Produce three publication-ready methodology figures for OV-CapFlow. The set
must give a reviewer a correct mental model in under one minute while keeping
the current scientific evidence boundary explicit:

1. the validated T7 strict end-to-end detector;
2. the complete OV-CapFlow architecture, including implemented mechanisms
   that are not enabled in the current best T7 recipe;
3. a mathematical zoom-in of the parent-preserving semantic-capacity flow.

The figures target the clean vector style commonly used in CVPR, ICCV, and
ECCV papers. They are separate artifacts with one shared visual grammar, not
three panels forced into one oversized canvas.

## 2. Scientific accuracy boundary

### 2.1 Validated T7 path

The first figure may present the following as active and verified:

- a `1024 x 1024` aerial image and category prompts;
- Swin-T multi-scale visual features and BERT text tokens;
- cross-modal GroundingDINO-style transformer encoding;
- `Q = 600` learned content queries and learned 5-D rotated references;
- six rotated decoder layers;
- training-only three-group matching-query supervision and denoising queries;
- inference using one primary `Q = 600` query set;
- text-token similarity classification and direct 5-D rotated-box regression;
- one class decision and one rotated box for every matching query;
- no encoder-proposal ranking, global query-class top-k, NMS, rotated NMS,
  dense proposal head, or RoI inference head.

Scale-1024 training and rare-positive exposure are optimization/data choices,
not inference modules. They may appear in a small training annotation but may
not be drawn as learned architectural blocks.

### 2.2 Implemented candidates pending causal validation

The complete-method figure may include the following implemented mechanisms:

- parent-preserving semantic residual fusion;
- continuous per-query capacity;
- global scene-density prediction;
- explicit null-reservoir calibration;
- capacity-mass, density-count, and null-calibration supervision.

At the current evidence stage, these mechanisms must use an amber dashed
outline and the legend `implemented candidate; excluded from current best T7`.
They must not be visually encoded as established causes of the current T7 AP.
When a future matched experiment validates a mechanism, its outline can be
changed from dashed amber to solid orange without changing the layout.

### 2.3 Open-vocabulary wording

The figures may show text-prompt scoring and vocabulary-conditioned class
readout. They may not claim proven unseen-category or zero-shot generalization.
No SOTA, first-of-its-kind, or AP70 statement appears in these methodology
figures.

## 3. Shared visual system

### 3.1 Canvas and export

- Final width: `178 mm`, suitable for a two-column spanning figure.
- Figure 1 height: `58-64 mm`.
- Figure 2 height: `62-68 mm`.
- Figure 3 height: `54-60 mm`.
- Primary editable source: diagrams.net/draw.io XML.
- Publication exports: vector PDF and SVG.
- Review export: PNG at 300 dpi.
- White background; no drop shadows, gradients, 3-D effects, textures, or
  decorative chart elements.

All labels remain live vector text in SVG/PDF. No screenshot or rasterized text
is accepted. The aerial input and rotated-output examples are simplified
vector glyphs rather than embedded bitmap photographs.

### 3.2 Typography

- Paper-matched serif family for final export: TeX Gyre Termes, with Times New
  Roman as a portable fallback.
- Module labels: `8.5-9.0 pt` at final insertion size.
- Stage labels and panel markers: `9.5-10.0 pt`.
- Mathematical symbols follow LaTeX italic conventions.
- Bold is reserved for the method name, panel labels, and at most one key
  mechanism per figure.
- Figure titles are not embedded in the artwork; the LaTeX caption supplies
  the title and claim.

### 3.3 Colour and line encoding

Use an Okabe-Ito-compatible palette with pale fills:

| semantic role | stroke | fill | secondary encoding |
|---|---|---|---|
| visual/image path | `#0072B2` | `#E7F4FA` | solid line |
| text/vocabulary path | `#009E73` | `#E7F7F2` | solid line |
| learned/native queries | `#7A5195` | `#F0EAF6` | solid line |
| OV-CapFlow mechanism | `#E69F00` | `#FFF4D6` | solid or dashed by evidence state |
| rotated output | `#D55E00` | `#FCEAE4` | solid line |
| training-only path | `#7A828B` | `#F1F3F5` | dotted arrow/light-grey enclosure |

Primary block strokes are `0.85 pt`; arrows are `0.9-1.0 pt`. Candidate
mechanisms use a `3.0 pt on / 2.0 pt off` dashed pattern at final size.
Training-only supervision uses a dotted line. Colour is never the sole carrier
of meaning.

### 3.4 Shape and spacing rules

- Use a strict left-to-right grid.
- Stage groups have square or `1.5 mm` corner radii, not web-style pills.
- Maintain at least `3 mm` whitespace between unrelated blocks.
- Arrows do not cross.
- Tensor labels sit above arrows only when necessary.
- Repeated decoder layers use one detailed block plus a bracket labelled
  `x6`, avoiding six duplicated boxes.
- Each visible module name matches terminology used in the methodology text.

## 4. Figure 1: Validated strict T7 pipeline

### 4.1 Story

The detector is a strict fixed-set, vocabulary-conditioned oriented detector:
learned query/reference pairs directly produce an ordered set of rotated
predictions without proposal selection or NMS.

### 4.2 Layout

Use one horizontal main path with four stages:

1. **Inputs**
   - a compact vector aerial-tile glyph;
   - a text strip such as `plane . ship . bridge . ...`.
2. **Multimodal encoding**
   - `Swin-T` above and `BERT` below;
   - both enter `Cross-modal Encoder`.
3. **Strict query decoding**
   - `Fixed Content Queries (Q=600)`;
   - `Learned 5-D Rotated References`;
   - both enter `Rotated Decoder x6` together with encoded memory.
4. **All-query readout**
   - `Text Similarity` and `Direct 5-D RBox` as two compact heads;
   - a vector output tile containing several rotated boxes;
   - the label `600 ordered predictions`.

A thin grey lower lane is labelled `Training only`. It contains `3 shared
query groups`, `DN queries`, and `independent Hungarian O2O losses`. A vertical
dotted arrow returns supervision to the decoder. The main inference path is
not split into three groups.

At the far right, a compact three-item contract reads:

`No proposal top-k | No global top-k | No NMS`.

The contract is text with small line icons, not three large boxes.

### 4.3 Caption draft

**OV-CapFlow instantiates strict vocabulary-conditioned oriented detection as
a fixed-set transformer.** Aerial features and text tokens are encoded jointly,
while 600 learned content queries with learned 5-D rotated references are
decoded directly into one class decision and one rotated box per query.
Grouped matching and denoising queries are used only during training; inference
retains one ordered Q600 set without proposal ranking, global top-k selection,
or NMS.

## 5. Figure 2: Complete OV-CapFlow architecture

### 5.1 Story

The complete method adds continuous semantic and capacity control to the
strict fixed-query substrate without sorting, pruning, or replacing queries.

### 5.2 Layout

Use three visual zones:

1. **Validated substrate, left**
   - compact `Visual-Text Memory` token stack;
   - compact `Immutable Native Queries` stack;
   - pale colours and reduced visual weight.
2. **OV-CapFlow decoder layer, centre**
   - one large block labelled `Semantic-Capacity Flow Layer` with an `x6`
     bracket;
   - the internal baseline `Parent Decoder` produces transported evidence
     `q_parent`;
   - `Capacity Head` predicts per-query `c_i`;
   - pooled memory enters `Scene Density Head` and predicts `N_hat`;
   - `Parent-preserving Semantic Residual` combines `q_native`, `q_parent`,
     and `c_i`;
   - the layer output feeds the next layer without query removal.
3. **Fixed-set calibration, right**
   - `Null Reservoir` receives the final Q600 states;
   - `Per-query Class + 5-D RBox` produces all Q600 rows.

Training-only capacity-mass, density-count, and null-calibration losses occupy
a narrow lower strip. Their arrows are dotted and terminate at the associated
heads. The inference route stays entirely in the upper main path.

Candidate modules use dashed amber outlines at the present evidence stage.
The legend contains only two entries: `validated substrate` and `implemented
candidate; pending matched validation`.

### 5.3 Caption draft

**The complete OV-CapFlow architecture regulates semantic transport and scene
capacity while retaining every query row.** Each decoder layer treats the
unchanged parent output as transported evidence, predicts continuous per-query
capacity and global scene density, and applies a parent-preserving native-query
residual. The final null reservoir calibrates background-heavy predictions;
mass, density, and null losses are training-only. Dashed modules are
implemented candidates that are not enabled in the current best T7 recipe.

## 6. Figure 3: Parent-preserving semantic-capacity flow

### 6.1 Story

The key mechanism begins as the exact parent detector and learns only a
capacity-controlled correction toward or away from persistent native query
identity.

### 6.2 Layout

Use three panels labelled `(a)`, `(b)`, and `(c)`:

### Panel (a): State separation

- `q_native`: persistent query identity, purple.
- `q_parent`: transported image-text evidence produced by the unchanged parent
  layer, blue.
- subtraction node computing `q_native - q_parent`.

### Panel (b): Capacity-controlled residual

Typeset the equation centrally:

`q_out = q_parent + c_i tanh(G[q_native, q_parent]) P(q_native - q_parent)`.

Show `Adapter P`, `Zero-initialized Gate G`, and `Capacity c_i` as three small
operators around the equation rather than three large pipeline stages.

### Panel (c): Guarantees

- initialization: `G = 0 => q_out = q_parent`;
- zero capacity: `c_i = 0 => q_out = q_parent`;
- inference: `continuous weighting; all Q queries retained`.

Use one curved return arrow from `q_out` to the next decoder layer. Do not show
null calibration in this mechanism figure; it belongs to Figure 2.

### 6.3 Caption draft

**The zero-initialized semantic-capacity residual preserves the parent decoder
exactly before learning.** The immutable native query provides a persistent
identity direction, the parent output supplies transported visual-text
evidence, and continuous capacity scales the learned gated correction. A zero
gate or zero capacity recovers the parent output exactly, while inference keeps
all queries rather than converting capacity into a pruning decision.

## 7. Artifact layout

Create the following tracked files:

```text
docs/figures/ov_capflow/
  ov_capflow_t7_pipeline.drawio
  ov_capflow_t7_pipeline.svg
  ov_capflow_t7_pipeline.pdf
  ov_capflow_t7_pipeline.png
  ov_capflow_full_architecture.drawio
  ov_capflow_full_architecture.svg
  ov_capflow_full_architecture.pdf
  ov_capflow_full_architecture.png
  ov_capflow_semantic_capacity_flow.drawio
  ov_capflow_semantic_capacity_flow.svg
  ov_capflow_semantic_capacity_flow.pdf
  ov_capflow_semantic_capacity_flow.png
  README.md
```

The README records canvas dimensions, font fallbacks, colour tokens, export
commands, and the evidence-status convention.

## 8. Acceptance criteria

The figure set is complete only when all checks pass:

1. every architecture label maps to an implemented class, active T7 config,
   or explicitly marked candidate mechanism;
2. Figure 1 contains no semantic-capacity or null module falsely attributed to
   T7;
3. Figure 2 visually distinguishes the validated substrate from candidates;
4. Figure 3 uses the code-consistent parent-preserving equation;
5. training-only query groups and DN queries do not appear on the inference
   path;
6. the output is one class and one direct 5-D rotated box per query;
7. no NMS, proposal top-k, global query-class top-k, or survivor-pruning block
   appears in the forward path;
8. SVG and PDF remain vector at 800% zoom;
9. all labels remain at least 8 pt after insertion at `178 mm` width;
10. figures remain understandable in greyscale through line-style encoding;
11. arrows do not cross and text does not overlap at the final canvas size;
12. each caption is self-contained and begins with the figure's main claim;
13. PNG previews match the SVG/PDF geometry and are exported at 300 dpi;
14. no unlicensed icon, external logo, or embedded raster asset is used.

## 9. Figure-designer quality audit

- Figure type: solution overview plus mechanism supporting figure.
- Paradigm: left-to-right pipeline for Figure 1; architecture-with-zoom for
  Figure 2; three-panel mechanism decomposition for Figure 3.
- Vector format: required.
- Post-scaling font size: at least 8 pt.
- Colour-blind safety: Okabe-Ito-compatible palette plus line-style encoding.
- Self-contained captions: drafted in Sections 4-6.
- Axis integrity: not applicable; these are architecture figures.
- Chartjunk: prohibited.
- Integrity result: pass, provided candidate status remains explicit until
  matched experimental validation changes that status.
