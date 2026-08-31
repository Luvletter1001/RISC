# P0 Progress

## 2026-08-30

- Created the approved P0 design, implementation plan, and OpenRSD-only integration contract.
- Implemented the pure C4 full-score carrier contract, tensor normalizer, receipt writer, and CPU-only dry-run CLI.
- TDD evidence: record-contract import failure, exporter import failure, invalid score-matrix failure, receipt-writer import failure, and direct-CLI import failure were each observed before the corresponding implementation.
- Environment note: tests importing `M_Tools` must run from `framework/openrsd`; repository-root pytest path invocation cannot import that namespace.
- Runtime note: use `PYTHONPYCACHEPREFIX=/tmp/ovd_orbit_p0_pyc` because this worktree's Python cache directory is read-only to subprocesses.
- At this 2026-08-30 stage, scientific status was
  `P0_BLOCKED_NO_FULL_LOGIT_EXPORT`: E0 validated the contract only and did not
  attach a live model hook, run a model, use a GPU, or produce a scientific
  result. The 2026-08-31 entry supersedes this hook-status limitation.

## 2026-08-31

- Added and CPU-verified the default-off OpenRSD P0 live-hook seam: raw native
  foreground-logit `[N,C]` vectors are the canonical `scores` field, optional
  post-calibration `[N,C]` vectors are named `calibrated_scores`, and aligned
  decoded `[N,5]` boxes retain exact `(level,row)` source identities.
- The sink transaction rolls back active records and supports immediate recovery
  of a just-committed image when `end_image()` raises; no partial image remains
  serializable in the fake-head tests.
- Status is `E0_LIVE_HOOK_READY_NO_FORWARD`, not a scientific result. No actual
  model forward, checkpoint, dataset, GPU, or AP calculation was performed. No
  receipt based on real project assets was published, and no live-P0 receipt
  was published in this implementation task.

### G0 input-seal implementation boundary (2026-08-31)

- Added and CPU-tested the G0 builder as infrastructure only. It validates the
  candidate-scene plan and canonical object inventory, validates declared
  candidate image and annotation file bytes, and deterministically constructs
  C4 view records including the identity repeats `rot000_a` and `rot000_b`.
- Completed protocol-authority validation for the no-forward builder: it seals
  ordered text embedding hashes, the oracle mouth adapter and carrier source
  identity schema, and the canonical digest of the exact P0-v1 threshold
  bundle. This CPU-only synthetic work uses no GPU and computes no P0 metric.
- `P0_INPUT_FAIL_STOP` has two fail-closed forms: a malformed/hash/asset
  mismatch produces a minimal failure package, while a valid-but-below-G0-scope
  input produces a full diagnostic package. The CLI exits 2 in both forms, and
  neither form authorizes a forward. Only an actual successful builder run can
  produce `G0_INPUTS_SEALED_NO_FORWARD`. The tests cover synthetic inputs and
  CLI boundaries, not real project assets.
- No authority package was generated from real project assets in this implementation task.
- No actual model run, no checkpoint deserialization, no dataset iterator, no
  GPU use, and no P0 metric computation occurred. No receipt based on real
  project assets was published, and no live-P0 receipt was published in this
  implementation task. This entry does not change the historical E0 status
  into a real G0 result, and it does not assert G1, G2, G3, an effect, AP, or a
  paper result.

### P0-A diagnostic inventory preparation (2026-08-31)

- Real read-only conversion completed at
  `docs/provenance/ovd_orbit_p0a_inventory_20260831/` with receipt status
  `P0A_DIAGNOSTIC_INVENTORY_READY_NO_FORWARD`.
- The source plan SHA-256 is
  `0b2c190bfa7231cb19cbf746aaf4612e898f9417263ac383087c2746e69abe35`.
  It yielded 80 C4 diagnostic scenes, excluded `P0148`, and verified the bytes
  of all 80 selected annotation files.
- The full-vocabulary diagnostic inventory contains 2666 retained GT objects;
  68 difficulty-2 DOTA annotations were recorded and excluded by the sealed
  adapter rule. This is neither a base/novel split nor a strict OVD result.
- 未执行模型前向、checkpoint 加载、GPU、AP 或 P0 指标计算。
- 下一步仅可依据这份 GT 几何诊断提出独立的正常尺寸/孤立阈值 authority；它不授权
  strict G0、`OV_gap`、`DID_closed`、`DID_shift` 或任何论文现象结论。
