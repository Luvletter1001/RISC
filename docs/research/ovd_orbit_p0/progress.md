# E0 Progress

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
  model forward, checkpoint, dataset, GPU, AP calculation, or receipt
  publication was performed.
