# OV-CapFlow Parent-Preserving Fusion Design

## 1. Purpose

Repair the HRSC C1 semantic-fusion arm so that enabling a freshly initialized
fusion module does not change the established C0 detector function. The first
deliverable is exact zero-update equivalence to C0. Only after that gate passes
may the repaired arm receive a five-epoch HRSC training budget.

This iteration remains on HRSC2016. It does not change C2 balanced
classification, C3 null calibration, strict all-query inference, query count,
geometry losses, data settings, or the DOTA2 transfer decision.

## 2. Confirmed root cause

C0 passes each decoder layer's transported image/text-attended query to the
next layer. The original C1 fusion computes

```text
q_out = q_native + gate * adapter(q_parent)
```

with a zero-initialized gate. Its initial output is therefore `q_native`, not
the C0 layer output `q_parent`. Because the immutable initial query is supplied
to every decoder layer, C1 repeatedly replaces the parent trajectory after
each layer. A minimal reproduction gives a zero gate and exact equality to the
native query while differing from the C0 output.

The problem is structural rather than evaluator drift: three C0 evaluations
and the post-matrix replay all returned `AP50=0.5870`, whereas the original C1
returned `AP50=0.5580`.

## 3. Selected architecture

### 3.1 Parent-preserving semantic residual

For each matching query at decoder layer `l`, define:

```text
delta_native = q_native - q_parent
residual     = P(delta_native)
gate         = tanh(G(concat(q_native, q_parent)))
q_out        = q_parent + gate * residual
```

`q_parent` is the output produced by the unchanged C0 decoder layer. `P` is a
channel projection initialized to identity, and `G` is a per-query,
per-channel gate whose weight and bias are initialized to zero.

At initialization, `gate` is exactly zero, so `q_out` is elementwise equal to
`q_parent`. Once training opens the gate, a positive value moves a component
toward the persistent native semantic state, a negative value moves it away,
and zero retains the parent trajectory. This keeps the original query-identity
hypothesis while making the intervention a true residual on the working C0
function.

If continuous capacity is supplied in a later experiment, it multiplies the
gate after the `tanh`, as before. Capacity remains disabled in this C1 arm.

### 3.2 Query boundaries

The fusion applies only to the matching-query suffix. Training-only denoising
queries retain the transported C0 output without an added arithmetic path.
When semantic fusion is disabled, the existing helper continues to return the
transported tensor and no gate state.

No query is sorted, removed, thresholded, or duplicated. Strict inference
continues to emit exactly 600 rows per HRSC image without top-k, NMS, rotated
NMS, multiclass NMS, or minimum-area-rectangle fallback.

## 4. Components and artifacts

### 4.1 Fusion module

`SemanticEvidenceFusion` keeps its public three-input interface so decoder
integration remains localized. Its implementation and documentation change
from native-base fusion to parent-base residual correction. Shape and capacity
validation remain fail-closed.

### 4.2 Decoder integration

`apply_matching_query_interventions` continues to split the denoising prefix
from the matching suffix. It passes the immutable native state and current
parent layer output to the repaired fusion module, then concatenates the
unchanged prefix with the repaired matching suffix.

### 4.3 HRSC configuration

A new parent-preserving C1 config inherits the C0 HRSC settings, loads the same
C0 epoch-10 checkpoint, freezes the parent detector, and trains only the six
decoder layers' semantic-fusion parameters. It uses a new work directory so
the discarded original C1 artifacts are never overwritten.

### 4.4 Equivalence audit

An opt-in real-HRSC audit loads the same C0 checkpoint into C0 and the repaired
C1, evaluates the same preprocessed batch with both models in evaluation mode,
and compares decoder states, classification scores, and rotated-box outputs.
It reports the maximum absolute difference for each state and exits nonzero if
any parent-visible output differs.

## 5. Test-first implementation

The first changed test replaces the old, incorrect contract that a zero gate
must return the native query. The new test requires a zero gate and exact
equality to the transported parent query. It must fail against the current
implementation for the expected reason before production code changes.

Additional tests cover:

- an open positive gate moving an identity-projected output toward the native
  query;
- an open negative gate moving it in the opposite direction;
- zero capacity preserving the parent query even when the gate is open;
- an unchanged denoising prefix and exact parent equality for the matching
  suffix at zero initialization;
- construction of the new HRSC configuration with only semantic-fusion
  parameters trainable;
- real-checkpoint, real-batch C0/C1 output equivalence.

All focused OV-CapFlow tests must remain green after the change.

## 6. Experimental gates

### Gate E0: isolated function equivalence

- Gate tensor is exactly zero at initialization.
- Fusion output is elementwise equal to `q_parent`.
- The matching suffix helper returns the complete C0 tensor exactly.

Failure stops the iteration before checkpoint or GPU work.

### Gate E1: real-checkpoint equivalence

Using the C0 epoch-10 checkpoint and the same real HRSC batch:

- decoder-state maximum absolute difference is `0`;
- classification-score maximum absolute difference is `0`;
- rotated-box maximum absolute difference is `0`;
- both models emit the same 600 query rows.

Failure stops the iteration before full-dataset evaluation.

### Gate E2: zero-update full replay

The repaired C1 must reproduce C0 on the HRSC test set:

- `dota/AP50 = 0.5870`;
- recall `= 0.941` at the report precision;
- 600 predictions per image;
- no forbidden selection or post-processing calls.

The run is invalid if it misses this gate; it is not rescued by training.

### Gate E3: five-epoch HRSC training

Only after E0-E2 pass, train the repaired C1 for five epochs from the same C0
checkpoint and seed `20260712`. The parent remains frozen. Record AP50, recall,
GT coverage, duplicate extras per GT, gate statistics, runtime, peak memory,
loss, checkpoint hash, and strict-row audit.

Promotion uses the existing rule:

- preferred: AP50 improvement of at least `0.002` over C0R; or
- tie path: AP50 within `0.001`, recall loss no greater than `0.005`, and at
  least two preregistered mediator improvements.

Otherwise the repaired arm is discarded as a model candidate, while the
parent-preserving implementation and equivalence audit may remain as verified
engineering infrastructure.

## 7. Experiment accounting and stopping

The completed HRSC matrix remains immutable history in `.lab/`. This repair is
recorded as the next experiment on a new research branch derived from the
current completed state. Repository-changing experiments are committed before
measurement, and all results are logged before any discard decision.

C2, C3, density, query-count changes, geometry changes, teacher labels, and
DOTA2 are outside scope. No second seed or ten-epoch continuation is launched
unless the repaired C1 first passes the existing promotion gate.

## 8. Deliverables

1. Parent-preserving semantic residual with focused regression tests.
2. A non-overwriting HRSC C1 configuration.
3. Real-checkpoint C0/C1 equivalence audit output.
4. Zero-update HRSC replay result and strict inference audit.
5. If eligible, a five-epoch HRSC result with mediator diagnostics.
6. Updated `.lab` ledger and a concise Chinese experiment report.
