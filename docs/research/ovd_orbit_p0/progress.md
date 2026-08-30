# E0 Progress

## 2026-08-30

- Created the approved P0 design, implementation plan, and OpenRSD-only integration contract.
- Implemented the pure C4 full-score carrier contract, tensor normalizer, receipt writer, and CPU-only dry-run CLI.
- TDD evidence: record-contract import failure, exporter import failure, invalid score-matrix failure, receipt-writer import failure, and direct-CLI import failure were each observed before the corresponding implementation.
- Environment note: tests importing `M_Tools` must run from `framework/openrsd`; repository-root pytest path invocation cannot import that namespace.
- Runtime note: use `PYTHONPYCACHEPREFIX=/tmp/ovd_orbit_p0_pyc` because this worktree's Python cache directory is read-only to subprocesses.
- Scientific status remains `P0_BLOCKED_NO_FULL_LOGIT_EXPORT`: E0 validates the contract only and does not attach a live model hook, run a model, use a GPU, or produce a scientific result.
