# DOTA-v2 Q600 Canonical Dump Analyzer Design

## 1. Status and purpose

This design was approved on 2026-07-22 as approach 3: a focused analysis
module with a thin command-line wrapper. It turns the one-off D139 diagnostic
preflight into a deterministic, CPU-only analyzer for the canonical T7 Epoch
24 prediction dump.

The analyzer must answer three experiment-routing questions without changing
the model or starting training:

1. Does the current route reproduce the P0148 context-to-small-vehicle failure
   pattern while preserving a measurable P0682 true-small-vehicle baseline?
2. Are tiny and dense misses caused mainly by geometric unreachability,
   semantic assignment, same-label ownership, or the fixed Q600 capacity
   ceiling?
3. How much raw AP loss is associated with score--IoU ordering, null/existence
   calibration, duplicate predictions, or low-ranked all-query tail rows?

The output is diagnostic evidence, not automatic authorization for another
training run. Candidate selection remains a separate scientific gate.

## 2. Frozen scientific contract

Canonical mode is valid only when all of the following hold:

- DOTA-v2.0 raw validation mouth: 13,833 images, including empty-GT tiles;
- 18 classes in the order declared by the canonical config;
- base/novel split read from that config: base14 and novel4;
- exactly 600 prediction rows per image and 8,299,800 rows in total;
- rotated boxes in `(cx, cy, w, h, angle)` form;
- one selected class and one score per query row;
- rotated IoU threshold 0.5 and VOC07 11-point AP;
- no score thresholding, row deletion, NMS, rotated NMS, global top-k,
  max-per-image truncation, dense head, or RoI head;
- CPU-only analysis with one PyTorch/OpenMP thread;
- the dump, checkpoint, config, and official metric source are read-only.

The CLI may expose smaller record/query/class counts only behind an explicit
`--allow-noncanonical` flag for unit tests and historical preflight replay.
Every report must state `canonical=true` or `canonical=false`; a noncanonical
report cannot satisfy the E24 gate.

## 3. Chosen architecture

Use two implementation files and one focused test file:

```text
projects/OVCapFlow/tools/
  dotav2_q600_diagnostics.py
  analyze_dotav2_q600_dump.py
tests/test_projects/ov_capflow/
  test_dotav2_q600_diagnostics.py
```

`dotav2_q600_diagnostics.py` contains side-effect-free calculation functions,
small typed records, aggregation logic, and report serializers. It does not
parse process arguments or write files.

`analyze_dotav2_q600_dump.py` is the thin orchestration layer. It validates
the environment and inputs, reuses `validate_dotav2_q600_dump.py` for
CPU-remapped loading and fixed-row integrity, calls the core functions, and
atomically publishes the report bundle only after every gate passes.

This boundary follows the existing standalone-tool convention under
`projects/OVCapFlow/tools/` while keeping evaluator and decomposition logic
directly unit-testable.

## 4. Inputs and command contract

Canonical invocation requires:

```text
positional dump path
--config                 canonical non-resume config
--checkpoint             exact checkpoint used to make the dump
--official-metrics-json  same-dump MMEngine JSON containing dota/mAP and dota/AP50
--training-metrics-json  live T7 scalar JSON containing the Epoch 24 validation row
--training-step 24
--output-dir             new diagnostic directory
```

Optional diagnostic controls are limited to:

```text
--iou-threshold          default 0.5; changing it makes the run noncanonical
--sample-size            default 500000 for all-row calibration sampling
--sample-seed            default 20260722
--max-error-examples     default 100 per error type
--case-dump              optional two-tile Q600 diagnostic dump
--case-manifest          required only when --case-dump is supplied
--allow-noncanonical     required for synthetic/historical replays
```

The config is loaded but no registry object, model, dataloader, CUDA context,
or optimizer is built. The analyzer reads `classes`, `base_classes`, and
`novel_classes` from the resolved config and rejects missing, duplicated, or
overlapping class definitions.

The official metric reader scans JSON lines and selects the requested record:

- the same-dump metric source must contain exactly one applicable row with
  `dota/mAP` and `dota/AP50`;
- the training source must contain exactly one row with `step=24` and those
  keys;
- `dota/mAP` is retained at full JSON precision;
- `dota/AP50` is the evaluator's three-decimal `round(mean_ap, 3)` value.

The checkpoint is loaded on CPU before the dump. The analyzer extracts exactly
one state-dict key ending in
`query_initializer.reference_embedding.weight`, checks shape `(600, 5)`,
applies sigmoid, then releases the rest of the checkpoint before loading the
large pickle. Zero or multiple suffix matches are fatal.

P0148 and P0682 are historical DOTA1 crop diagnostics and do not occur in the
DOTA-v2 raw-validation mouth. They must never be required members of the
13,833-record canonical dump. If a separately authorized same-checkpoint
two-tile Q600 dump is supplied, its manifest is exactly:

```json
{
  "context_false_sv": ["P0148__1024__651___0"],
  "true_sv_safety": ["P0682__1024__553___0"]
}
```

Every listed ID must occur exactly once in the auxiliary case dump. The
manifest contains no threshold or expected result, so it cannot silently
encode a favourable outcome. Absence of the auxiliary dump yields
`case_gate=not_run`: it does not invalidate canonical E24 metric/decomposition
evidence, but the complete three-question routing package remains incomplete.
Generating the auxiliary inference dump is outside this analyzer design and
requires its own reviewed run plan.

## 5. Data loading and integrity stage

The CLI first hashes all provenance files with SHA256, records byte sizes, and
checks that `CUDA_VISIBLE_DEVICES` is unset or the empty string. It sets
PyTorch intra-op and inter-op threads to one. A visible CUDA device is a canonical-mode failure
even though the implementation never calls `.cuda()`.

Dump loading reuses `CpuUnpickler`, `load_cpu`, `as_tensor`, and
`validate_records` from `validate_dotav2_q600_dump.py`. CUDA-tagged serialized
storages are remapped to CPU without changing other pickle classes. The
analyzer adds these checks:

- GT labels have the same range and integer semantics as prediction labels;
- GT and prediction widths/heights are positive;
- optional ignored instances are finite and are passed through evaluator
  assignment exactly; absence means an empty ignored set;
- every query receives its immutable `query_id` from row position 0--599;
- total GT and ignored-GT counts are recorded;
- when an auxiliary case dump is supplied, both fixed manifest IDs are present
  exactly once in that dump; their status is not inferred from the canonical
  dump.

No result file is written during this stage.

## 6. Exact evaluator reconstruction

The reconstruction deliberately mirrors
`mmrotate.evaluation.functional.mean_ap`:

1. Split every image's 600 rows by predicted class without deleting any row.
2. For each image and class, compute CPU `box_iou_rotated` against same-class
   GT and ignored GT.
3. Use the evaluator's `np.argsort(-scores)` ordering and greedy one-GT
   coverage rule to assign TP/FP.
4. Concatenate image results for each class, apply the same global descending
   score order, and compute cumulative precision/recall.
5. Call MMDetection's `average_precision(..., mode='11points')` on those exact
   curves.
6. Average the 18 class AP values using the same inclusion rule as
   `eval_rbbox_map`.

The run passes only when:

- reconstructed exact mAP differs from same-dump `dota/mAP` by at most `1e-7`;
- `round(reconstructed_mAP, 3)` equals same-dump `dota/AP50`;
- same-dump and Epoch-24 training mAP differ by at most `5e-4`, and their
  rounded AP50 values agree; a difference above `1e-7` is retained as an
  explicit replay-delta warning even when the cross-run gate passes;
- detection counts, GT counts, TP counts, class count, and row count are
  internally consistent.

There is no fallback to rounded log text. A parity failure stops all causal
interpretation and publishes only a failure record in the temporary working
directory.

## 7. Ground-truth reachability decomposition

The analyzer computes one mutually exclusive state for every non-ignored GT:

| state | exact definition |
|---|---|
| `geometry_miss` | maximum rotated IoU over all 600 rows is below 0.5 |
| `semantic_miss` | an any-label row reaches IoU 0.5, but no same-label row does |
| `ownership_miss` | a same-label row reaches IoU 0.5, but this GT is not the best-overlap same-label GT of any qualifying row |
| `evaluator_reachable` | at least one qualifying same-label row owns this GT as its best overlap; the highest-score such row is its maximum-recall witness |

For each GT, retain only compact witness fields in memory: image ID, GT index,
class, state, best-any query/IoU, best-same-label query/IoU, reachable-witness
query/score/IoU, and image GT count. This distinguishes reachability from the
score-greedy evaluator TP assignment; candidate multiplicity is never called
recall competition unless it produces an `ownership_miss`.

The same pass reports:

- number of same-label IoU>=0.5 candidates per GT;
- candidate excess `sum(max(candidate_count - 1, 0))`;
- query IDs contributing reachable witnesses;
- global and per-class fixed-Q capacity lower bound
  `sum(max(number_of_GT_in_image - 600, 0))`;
- the corresponding optimistic Q600 recall ceiling;
- capacity excess separately from observed geometry misses.

## 8. Evaluator FP taxonomy and AP-sensitive regions

Every prediction row receives exactly one evaluator outcome. Canonical DOTA-v2
uses `diff_thr=100`, so its ordinary 0/1/2 difficulty values do not create
ignored GT; the sixth outcome preserves evaluator fidelity for noncanonical
fixtures that do contain ignored instances:

| outcome | definition |
|---|---|
| `tp` | official score-greedy same-class match to a previously uncovered GT |
| `duplicate_fp` | same-class IoU>=0.5 match to an already covered GT |
| `semantic_fp` | not a same-class match, but IoU>=0.5 to a non-ignored GT of another class |
| `localization_background_fp` | nonempty-GT image with no IoU>=0.5 GT explanation |
| `empty_tile_fp` | image has no non-ignored GT |
| `ignored_prediction` | official assignment overlaps an ignored same-class GT and therefore contributes neither TP nor FP |

Counts are reported for three independent class-wise score regions:

1. all rows;
2. through each class's last TP;
3. the VOC07 AP-support prefix.

For an 11-point recall threshold, its supporting rank is the earliest rank
that attains the maximum precision among rows whose recall reaches that
threshold. Thresholds not reached contribute zero and no rank. A class's
AP-support endpoint is the maximum of its supporting ranks. This definition is
tested directly against the 11-point AP calculation; prefixes from different
classes are summed only for reporting and never treated as a global score
threshold.

The fixed-recall perfect-ranking oracle keeps boxes, labels, TP assignments,
and maximum recall unchanged, places all current TPs before FPs within each
class, and recomputes VOC07 AP. Report current AP, oracle AP, headroom, AP70
gap, and the fraction of headroom required globally and per class group.

## 9. Calibration, query, and refinement diagnostics

Quality diagnostics include:

- exact Spearman correlation between evaluator-TP score and assigned rotated
  IoU, globally and per class when at least two nonconstant pairs exist;
- score bins `[0,.05), [.05,.10), [.10,.25), [.25,.50), [.50,.75), [.75,1]`
  with row count, TP precision, same-label hit@0.5, and mean/binned IoU;
- deterministic all-row sample formed by taking the lowest `sample_size`
  stable-hash keys of `(sample_seed, image_id, query_id)`, never by
  traversal-order RNG;
- per-query counts for labels, TP, each FP type, reachable witnesses,
  AP-support appearances, score quantiles, and effective-query count defined
  as `exp(Shannon entropy)` over normalized query contribution counts;
- all-row label allocation and AP-support label allocation as separate tables.

Reference refinement uses the extracted checkpoint prior for the same query
row. Canonical DOTA crops use the frozen 1024-pixel patch coordinate system:
normalized reference x/w components are multiplied by 1024, y/h components
are multiplied by 1024, and normalized reference angle is multiplied by
`pi`. Center refinement is Euclidean pixels; scale refinement is
`exp(abs(log(sqrt(final_area / prior_area))))`; aspect refinement applies the
same symmetric log-ratio to `max(w,h)/min(w,h)`; angle refinement is the
smallest absolute difference modulo `pi`. For each GT
witness, report prior-to-final center distance, multiplicative scale change,
aspect change, and periodic angle change. These are associations, not causal
effects.

Required strata are fixed in the implementation:

- GT sqrt-area pixels: `<=8`, `(8,16]`, `(16,32]`, `(32,64]`, `>64`;
- GT/image: `1-10`, `11-50`, `51-100`, `101-200`, `201-400`, `401-600`,
  `>600`;
- aspect ratio: `<=1.5`, `(1.5,3]`, `(3,5]`, `>5`;
- absolute periodic angle: six 15-degree bins over `[0,90]`;
- class, base14/novel4, and small-vehicle size-by-density cross-strata.

Every stratum includes GT count before any rate. Empty and low-support strata
are emitted with counts and null rates, not silently dropped.

## 10. P0148/P0682 safety evidence

When the optional auxiliary case dump is provided, emit an exact row for each
of its two manifest images containing:

- image ID and case group;
- GT count and small-vehicle GT count;
- number of predicted small-vehicle rows;
- top-1, top-5, and top-20 small-vehicle scores;
- small-vehicle evaluator TP, duplicate, semantic, localization/background,
  and AP-support counts;
- best same-label IoU and query ID for every true small-vehicle GT;
- count of true small vehicles that are geometry, semantic, ownership, or
  reachable;
- the highest-scoring false-small-vehicle witness and its nearest GT label/IoU.

P0148 is therefore a cross-dataset context-false-positive diagnostic and P0682
a cross-dataset true-positive safety diagnostic. Neither enters DOTA-v2 mAP,
base/novel summaries, or the 13,833-record integrity count. The first E24
report does not claim that an intervention is safe. Future candidate case
dumps must use the same manifest and schema; comparison is performed from the
two immutable report bundles rather than by changing a threshold in this
analyzer.

## 11. Output bundle

The analyzer writes to a temporary sibling directory and renames it to the
requested output directory only after all canonical gates pass:

```text
diagnostics.json       complete versioned machine-readable summary
per_class.csv          AP, recall, oracle, GT states, FP regions, correlations
strata.csv             size/density/aspect/angle/base-novel/SV cross-strata
per_query.csv          600 query-row attribution summary
case_studies.csv       optional exact P0148/P0682 evidence, or not-run status
error_examples.csv     deterministically capped witnesses for audit
report.md              concise scientific interpretation and decision inputs
manifest.json          hashes, sizes, command, environment, schema version
```

The manifest hashes the other seven payload files. Its own SHA256 is recorded
by the experiment ledger after publication, avoiding a recursive self-hash.

`diagnostics.json` starts with `schema_version=1`. JSON uses finite numbers or
`null`; NaN and Infinity are forbidden. CSV row order is fixed by class order,
stratum definition, query ID, and image ID. The Markdown report is generated
only from the JSON summary and contains:

1. integrity and parity gates;
2. exact raw mAP/AP50 and AP70 gap;
3. macro AP-support/ranking evidence;
4. micro tiny/dense/capacity evidence;
5. base/novel evidence and the optional P0148/P0682 case gate;
6. a factual decision-input matrix.

The report may say which preregistered conditions are supported. It may not
label a mechanism causal, authorize E25+, claim AP70, or create a new model
candidate automatically.

## 12. Failure handling

Expected user/data failures raise a dedicated diagnostic exception and produce
exit code 2 with a one-line error. Unexpected exceptions retain a traceback
and exit nonzero. Fatal conditions include:

- canonical mouth, class order, query shape, finiteness, or case-ID mismatch;
- visible CUDA in canonical mode;
- missing/ambiguous reference state key;
- missing or ambiguous official metric row;
- exact mAP or same-route validation parity failure;
- nonpositive boxes, invalid labels, or duplicate image IDs;
- any attempt to overwrite an existing output directory;
- nonfinite output statistics or incomplete atomic publication.

Temporary files are preserved with a `.failed` suffix for forensic inspection,
but no `diagnostics.json` with `canonical=true` is published on failure.

## 13. Test strategy

Implementation follows red-green-refactor. Tests use tiny synthetic CPU
records with known rotated boxes and scores; no mock substitutes for the core
IoU/evaluator path.

Required unit and component tests are:

1. exact two-image evaluator reconstruction and VOC07 AP;
2. all four GT states, including a genuine ownership miss;
3. all six prediction outcomes, including empty-tile, duplicate FP, and an
   ignored prediction;
4. AP-support endpoint with a precision-envelope tie;
5. perfect-ranking oracle at fixed recall;
6. Q600 capacity lower bound and optimistic ceiling;
7. size, density, aspect, angle, base/novel, and SV cross-binning boundaries;
8. stable sample identity under reversed record order;
9. query IDs remain original row positions after class-wise sorting;
10. checkpoint reference suffix extraction: success, missing, and ambiguous;
11. metric JSON selection and exact/rounded parity failures;
12. optional P0148/P0682 case-dump separation, manifest completeness, and
    case summaries;
13. canonical/noncanonical flag behavior;
14. atomic output success and no canonical publication on failure;
15. deterministic JSON/CSV content across two runs.

After focused tests pass, run the existing OV-CapFlow test subset to guard the
validator and queue. Then perform one manual integration replay on the D143
13,833x600 dump. It must reproduce, within stated numeric tolerance:

- 13,833 records and 8,299,800 rows;
- exact mAP absolute error no greater than `1e-7`;
- exactly 159,129 reachable GT, with geometry/semantic rates within 0.01
  percentage point of 33.787%/0.893%;
- evaluator-TP score--IoU Spearman within `1e-4` of 0.2778;
- AP-support FP shares about 14.89% duplicate, 3.23% semantic,
  65.36% localization/background, and 16.52% empty-tile, each within 0.01
  percentage point.

D143 is a noncanonical regression fixture and never becomes E24 evidence.

## 14. Acceptance criteria

The analyzer implementation is complete only when:

- every required test first failed for the intended reason and now passes;
- the existing validator/queue/audit focused suite still passes;
- the D143 full integration replay matches the recorded decomposition;
- the canonical E24 run, once its dump exists, passes every integrity and
  evaluator-parity gate;
- all eight output files are deterministic and finite; the manifest hashes
  the other seven, and the experiment ledger hashes the manifest;
- no GPU process, model forward, optimizer step, training change, prediction
  filtering, or post-processing is introduced;
- the report supplies enough evidence to choose at most one next causal route
  under the existing preregistration, without making that choice implicitly.

## 15. Explicit exclusions

This work does not implement a loss, head, matcher, initializer, prompt sweep,
new training config, candidate model, visualization dashboard, database,
distributed analyzer, or E18 canonical re-dump. It does not increase Q, run
NMS/top-k, generate the auxiliary P0148/P0682 inference dump, or infer missing
case IDs. Those activities require separate evidence and authorization after
the E24 report.
