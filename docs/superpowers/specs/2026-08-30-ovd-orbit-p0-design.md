# OVD Object-Orbit P0 Design

## Objective

Build a read-only, deterministic evidence exporter for the Object-Orbit Semantic Reliability P0. The first implementation targets the frozen OpenRSD parent only. It must export full vocabulary score vectors tied to exact source carriers, validate the C4/identity input contract, and write a machine-readable manifest without launching training or changing detector outputs.

## Scope boundary

- This worktree owns the protocol, OpenRSD exporter, manifest schema, and CPU tests.
- It does not train, evaluate AP, allocate GPUs, alter model weights, or implement the conditional vocabulary-margin method.
- It does not modify CastDet. CastDet requires a separate worktree because its model interface and strict 16B/4N adapter are an independent subsystem.
- P0148's full source scene is excluded at manifest build time.

## Architecture

```text
sealed C4 inputs / model hook output
       -> exact source-carrier normalizer
       -> full-score record validator
       -> oracle/full-mouth manifest builder
       -> JSONL records + receipt.json
```

`ovd_orbit_p0.py` is pure CPU validation and aggregation logic. It accepts already-captured score fields and does not import MMEngine/MMRotate. `ovd_orbit_p0_export.py` is the only OpenRSD-facing adapter; it records full score vectors before final class selection, preserves `(level,row)` source identity, and delegates all serialization validation to the pure module. The CLI performs a dry-run manifest build and refuses unsupported views, class dimensions, duplicate carrier conflicts, mixed prompt hashes, or P0148 scene IDs.

## Invariants

1. Only `rot000`, `rot090`, `rot180`, and `rot270` are primary C4 views.
2. Scores are finite float32 `[C]` vectors and are never collapsed to top-1 scores.
3. Source identity is exact `(level,row)`, never a score/order/text-similarity heuristic.
4. Duplicate records with identical source geometry/scores collapse; conflicting duplicates fail closed.
5. Identity-repeat records must be represented explicitly; later G1 determines the numeric noise envelope.
6. Exporter mode never mutates model outputs, decoder state, checkpoint, or dataset inputs.
7. Every receipt records input hashes, prompt/vocabulary hash, exclusion count, class dimension, and terminal status.

## Verification

- Pure tests cover C4 validation, score shape/dtype/finite checks, duplicate handling, scene-level P0148 exclusion, vocabulary hash consistency, manifest determinism, and fail-closed errors.
- Exporter tests use a tiny fake tensor-producing head and verify full vectors plus exact source identities are retained without changing its returned tensor.
- The first live path is dry-run only and must not invoke `model.test_step` until its preflight gate is separately approved.
