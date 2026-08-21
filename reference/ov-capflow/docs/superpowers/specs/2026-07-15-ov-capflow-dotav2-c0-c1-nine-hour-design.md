# OV-CapFlow DOTA-v2.0 C0/C1 Nine-Hour Causal Validation Design

## Objective

Use physical GPUs 4, 5, 6, and 7 to produce a trustworthy, same-code
DOTA-v2.0 comparison between the strict native-only C0 substrate and the
repaired parent-preserving C1 semantic fusion. The primary evaluation mouth is
the unfiltered raw split with 13,833 validation patches. The work is complete
only when the data mouth, parent checkpoint, strict inference contract,
training intervention, independent evaluation, mediator metrics, and written
decision are all verified and recorded.

This stage transfers one mechanism from HRSC. It does not add balanced
classification, null reservoir, density capacity, query-count changes,
geometry changes, teacher labels, pseudo labels, or longer training.

## Why This Experiment

HRSC seeds 20260712 and 20260713 both passed the pre-registered H3 gate for
the transported-parent residual fusion. Their AP50 values were 0.6290 and
0.6210 versus C0 at 0.5870. The next unresolved question is whether that same
mechanism improves a fixed-query, 18-class DOTA-v2.0 substrate under the
strict raw-patch evaluator.

Historical OpenSetFlow and P134B measurements are useful external context,
but they do not establish a causal delta in the current OV-CapFlow codebase.
This experiment therefore rebuilds the comparison in one repository and one
evaluation mouth.

## Experimental Pair

### C0: strict native-only control

- Build the current-code DOTA-v2.0 strict parent with one full training epoch
  from the repository's declared Swin/BERT initialization.
- Disable semantic fusion, balanced reduction, null reservoir, and density
  capacity.
- Emit every fixed matching query directly, with no NMS, top-k survivor
  selection, dense inference head, or minimum-area-rectangle conversion.
- Save, hash, and replay the complete raw-13,833 validation mouth to establish
  the C0 control checkpoint.

### C1: repaired parent-preserving fusion

- Load the identical C0 checkpoint.
- Use the verified residual form:

  ```text
  native_residual = q_native - q_parent
  q_fused = q_parent + tanh(g(q_native, q_parent)) * P(native_residual)
  ```

- Freeze the parent and train only the six decoder layers' semantic-fusion
  parameters for one epoch.
- Keep balanced reduction, null reservoir, density capacity, and auxiliary
  mechanisms disabled.
- Use seed 20260712.

The fixed query count is inherited from the checkpoint-compatible DOTA2
parent. Historical P134B used Q=200; if the recovered checkpoint's effective
config proves a different value, both C0 and C1 must use that parent-native
value. Query count is never tuned in this experiment.

## Authoritative Data Mouth

The formal mouth contract is:

- train: 47,294 patches;
- validation: 13,833 patches;
- 18 DOTA-v2.0 classes in the repository's canonical order;
- `filter_empty_gt=False` for raw validation;
- identical image, annotation, angle, prompt, and class mappings for C0/C1.

The currently visible 20,995/10,833 split is a different mouth and is invalid
for this experiment. It may not be relabeled as raw13833. The exact split must
be located or regenerated deterministically from source data, with a manifest
recording counts, empty-tile count, class order, paths, and preparation
command.

## Parent and Initialization Contract

The DOTA2 parent must have a concrete checkpoint file, SHA256, effective
config, training provenance, fixed query count, and raw-mouth replay. The
current repository has no checkpoint compatible with its Swin, six-layer
OV-CapFlow topology, so this experiment reproducibly builds C0 for one epoch
before starting C1. A metric reported in historical Markdown is not a
checkpoint.

Historical P134B is retained only as an external reference. Its recovered
checkpoint uses a ResNet backbone, two decoder layers, and a custom P15 head;
it cannot be loaded as the parent of the current Swin, six-layer OV-CapFlow
model. Cross-topology partial loading is forbidden.

Before C1 training, the parent-equivalence audit must prove that C0 and
zero-update C1 load the same checkpoint and produce exact equality for decoder
states, references, class scores, rotated boxes, prediction scores, labels,
and boxes on the same real DOTA2 batch. Maximum absolute difference must be
0.0, including non-finite-safe comparison. Missing keys must be limited to the
expected new semantic-fusion parameters; invalid missing and unexpected keys
must be empty.

If the locally built C0 checkpoint is incomplete or fails its raw replay, C1
does not start. Work continues on reproducibly rebuilding C0 rather than
substituting the HRSC checkpoint or a historical metric.

## Dynamic Batch Safety

DOTA2 dense patches can make DN self-attention exceed memory. Both arms use
the same deterministic, distributed-aware DN query-budget batch sampler.
Before formal training it must prove:

- every one of the 47,294 training samples appears exactly once per epoch;
- no duplicate or missing indices;
- ordering is identical across C0/C1 for the same seed;
- actual batch size and DN query count are logged for every update;
- shrinking depends only on a predeclared DN/query memory budget, not class,
  loss, or model prediction;
- distributed ranks receive disjoint shards with the same global coverage;
- minimum batch size and shrink count are written to machine-readable JSON.

An OOM caused by an invalid fixed-batch launcher is an engineering failure,
not evidence against C1.

## GPU and Execution Layout

- Parent phase: GPUs 4, 5, 6, and 7 jointly build the one-epoch C0 checkpoint.
- Gate phase: GPUs 4 and 5 run the complete C0 parent replay while GPUs 6 and
  7 concurrently run the complete zero-update C1 replay from that checkpoint.
- Adapter phase: GPUs 4, 5, 6, and 7 jointly run the one-epoch C1 adapter
  training after both full replays pass.
- Revalidation phase: GPUs 4 and 5 and GPUs 6 and 7 form two independent
  two-GPU lanes that re-evaluate the same trained checkpoint into separate
  output directories.
- Both lanes use the same Python environment, code commit, dataset manifest,
  evaluator, prompt/class bank, fixed query count, and checkpoint bytes.
- GPU reassignment is allowed only after a lane finishes; every command and
  physical GPU mapping is recorded.

Useful GPU saturation is an execution objective, but never overrides a
science gate. Idle GPUs may run independent required audits or revalidation;
they may not be filled with unregistered C2/C3 arms or altered configurations.

C0 is trained once to create the current-code parent, then frozen. The causal
question is whether one epoch of the small C1 adapter improves that fixed C0
parent without changing its initialization function.

## Nine-Hour Operating Schedule

### T+0:00 to T+1:30 — asset truth gate

- Locate or regenerate the exact 47,294/13,833 mouth.
- Locate and hash the DOTA2 parent checkpoint and its config.
- Locate, port, or implement the dynamic DN budget sampler.
- Record the selected Python/CUDA/MMEngine/MMDetection/MMRotate stack.

### T+1:30 to T+2:30 — config and engineering gate

- Add C0/C1 configs and differential tests.
- Run portable tests and real-batch forward/backward.
- Run checkpoint mapping, parent equivalence, strict-inference, and sampler
  coverage audits.
- Commit all formal-run code before using the GPUs.

### T+2:30 to T+4:30 — build the current-code C0 parent

- Train strict native-only C0 for one epoch on all four GPUs.
- Save and hash its checkpoint and verify finite optimization and sampler
  coverage.

### T+4:30 to T+5:30 — matched initialization gate

- Replay trained C0 on GPUs 4–5 while replaying zero-update C1 from the same
  checkpoint on GPUs 6–7.
- Require exact parent tensors and identical complete raw-mouth metrics.

### T+5:30 to T+7:30 — train the C1 adapter

- Train C1 for one epoch on all four GPUs.
- Monitor loss, gradients, memory, sampler shrink events, and process health.
- Run raw-13,833 validation without changing the pre-registered config.

### T+7:30 to T+9:00 — independent evidence and decision

- Re-evaluate the C1 checkpoint independently and concurrently on GPU pairs
  4–5 and 6–7.
- Compute detection, empty-tile, coverage, duplicate, and gate diagnostics.
- Verify checkpoint SHA and artifact completeness.
- Write the final Chinese report and update the experiment ledger.

The schedule is a resource target, not permission to weaken a gate. If asset
restoration consumes more time, the formal run starts only after the same
gates pass, and the goal remains open until the requested comparison is
actually complete.

## Pre-Registered Decision Rules

The primary metric is raw-13,833 `dota/mAP`; AP50 is reported as a secondary
metric when supported by the evaluator.

- **Promote:** C1 minus C0 mAP is at least +0.003 with strict inference intact.
- **Park with mechanism signal:** mAP drop is no worse than 0.001 and at least
  two of empty foreground mass, GT coverage, duplicate extras/GT, gate gap,
  or rare/dense-class recall materially improve without recall falling more
  than 0.005.
- **Positive but under gate:** delta is between 0 and +0.003; record as a
  trend, not promotion.
- **Stop exact recipe:** mAP drops by more than 0.010 and no mediator improves.
- **Do not promote:** any other valid finite outcome that does not satisfy the
  promote, park, positive-under-gate, or stop-exact conditions. Record it
  without starting an unregistered follow-up recipe.
- **Invalid engineering:** wrong mouth, incomplete coverage, checkpoint
  mismatch, intervention overwrite, non-finite behavior, strict-audit failure,
  incomplete validation, or launcher/sampler error.

Absolute raw milestones such as 0.50 or 0.60 remain context only and do not
replace the matched C1-minus-C0 decision.

"Materially improve" is fixed before the run as follows: empty foreground
score mass decreases by at least 1% relative to C0; GT coverage increases by
at least 0.001 absolute; duplicate extras/GT decreases by at least 1%
relative; or the C1-only gate gap (`unmatched - matched`) is no greater than
-0.005. The park recall floor is implemented as GT-coverage delta >= -0.005.
Rare/dense-class recall does not count toward the park gate in this run unless
a deterministic class-frequency manifest and recall calculator are committed
before result inspection.

## Required Evidence

### Detection

- raw mAP and AP50 when available;
- 18-class AP, recall, and prediction count;
- image count, GT count, detection count, and empty-tile count;
- best and final checkpoint metrics if they differ.

### Mechanism

- GT coverage and duplicate extras/GT;
- matched and unmatched gate means and gate gap;
- empty-tile foreground score mass;
- per-class/density failure pattern;
- fusion gradient norms and trained parameter list.

### Engineering

- strict all-query audit and forbidden-call list;
- parent-equivalence JSON;
- checkpoint-load JSON before and after training;
- sampler coverage/shrink JSON;
- environment versions, commit, command, GPU map, duration, peak memory;
- checkpoint and prediction artifact SHA256 values.

## Failure Handling

- Missing exact data or checkpoint: restore the asset; do not substitute a
  different mouth or parent.
- Sampler coverage failure: fix and rerun coverage tests before GPU training.
- Parent-equivalence failure: classify as engineering invalid and diagnose
  config/load path before training.
- OOM: verify sampler behavior and memory budget; rerun the unchanged science
  configuration after fixing only the launcher/sampler.
- NaN/non-finite: stop the affected lane, preserve logs, locate the first bad
  iteration, and do not report downstream metrics as valid.
- Evaluation interruption: resume or rerun the complete raw mouth into a new
  output directory; never combine partial metrics.

## Artifacts and Records

Tracked artifacts use these roles:

```text
docs/project_history/exp_20260715_dotav2_parent_preserving_causal/
  fplan_dotav2_parent_preserving_causal_zh.md
  flog_dotav2_parent_preserving_causal_zh.md
  faudit_dotav2_parent_preserving_causal_zh.md
  fres_dotav2_parent_preserving_causal_zh.md
```

Machine-readable outputs live under separate C0 and C1 work directories and
include mouth, load, equivalence, strict, sampler, metrics, mediator, and hash
records. `.lab` receives experiment genealogy, hypothesis, raw values,
duration, status, and the next decision.

## Completion Definition

The experiment is complete only when:

1. the 47,294/13,833 mouth is proven and recorded;
2. the one-epoch current-code C0 parent checkpoint and SHA are proven;
3. C0 training and full raw replay are complete;
4. zero-update C1 is exactly parent-equivalent;
5. the one-epoch C1 intervention is finite and sampler-complete;
6. training-time and independent raw evaluation agree;
7. required mediator and strict audits exist;
8. a pre-registered promote/park/stop/invalid decision is written;
9. tests and artifact consistency checks pass;
10. the final report is committed and merged into the research baseline.
